#include "density.h"

using namespace cmp;

density::density(){

};

density::density(density const &d) {
    m_model = d.m_model;
    m_err_mean = d.m_err_mean;
    m_log_prior_hpar = d.m_log_prior_hpar;
    m_log_prior_par = d.m_log_prior_par;
    m_err_kernel = d.m_err_kernel;
    m_err_kernel_derivatives = d.m_err_kernel_derivatives;
    m_dim_hpar = d.m_dim_hpar;
    m_dim_par = d.m_dim_par;
    m_lb_hpar = d.m_lb_hpar;
    m_ub_hpar = d.m_ub_hpar;
    m_lb_par = d.m_lb_par;
    m_ub_par = d.m_ub_par;
    m_grid = d.m_grid;
    m_x_obs = d.m_x_obs;
    m_y_obs = d.m_y_obs;
    m_par_samples = d.m_par_samples;
    m_hpar_samples = d.m_hpar_samples;
};

density::density(doe const &grid) {

    //set the grid
    m_grid = grid;

    //set lower and upper bounds 
    m_lb_par = m_grid.m_lb;
    m_ub_par = m_grid.m_ub;

    m_dim_par = m_lb_par.size();
}

void density::set_new_doe(doe const &grid) {
    // set the grid
    m_grid = grid;

    //set lower and upper bounds
    m_lb_par = m_grid.m_lb;
    m_ub_par = m_grid.m_ub;
    m_dim_par = m_lb_par.size();
}

void density::set_obs(std::vector<vector_t> const &x_obs, vector_t const &y_obs)
{
    if (x_obs.size() != y_obs.size()) {
        spdlog::error("x_obs and y_obs sizes do not match!\nx_obs has size {0:d} and y_obs has size {1:d}",x_obs.size(),y_obs.size());
    } else {
        m_x_obs = x_obs;
        m_y_obs = y_obs;
    }
}

matrix_t density::covariance(vector_t const &hpar) const {
    
    //Number of points
    int n_pts = m_x_obs.size();
    
    //Computation of the kernel matrix
    matrix_t kernel_mat = matrix_t::Zero(n_pts, n_pts);
    for (int i = 0; i < n_pts; i++) {
        for (int j = i; j < n_pts; j++) {
        
            kernel_mat(i, j) = m_err_kernel(m_x_obs[i], m_x_obs[j], hpar);

            //Kernel matrix is symmetric
            if (i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat;
}

vector_t density::residuals(vector_t const &par) const {
    return m_y_obs - m_model(m_x_obs,par);
}

double density::loglikelihood(vector_t const &res, Eigen::LDLT<matrix_t> const &ldlt) const {
    
    // Solve linear system
    vector_t alpha = ldlt.solve(res);
    
    return -0.5 * res.dot(alpha) - 0.5 * (ldlt.vectorD().array().log()).sum() - 0.5 * res.size() * log(2 * M_PI);
}

double density::loglikelihood(vector_t const &par, vector_t const &hpar) const {
    
    // Compute the covariance matrix
    matrix_t cov_mat = covariance(hpar);

    //Compute the residuals
    vector_t res = residuals(par);
    
    // Cholesky decomposition
    Eigen::LDLT<matrix_t> ldlt(cov_mat);

    //Evaluate the loglikelihood by calling the overloaded version
    return loglikelihood(res, ldlt);
}

vector_t density::hpar_KOH(vector_t const &hpars_guess, double int_const ,double ftol_rel) const {

    //begin optimization
    auto begin = std::chrono::steady_clock::now();

    vector_t x0 = hpars_guess;

    //Create the data structure
    auto data = std::make_pair(this, &int_const);

    //Perform the optmiziation
    opt_routine(opt_fun_KOH, &data, x0, m_lb_hpar, m_ub_hpar, ftol_rel,nlopt::LN_SBPLX);

    // end optimization
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

    //print info
    spdlog::info("KOH optimisation over. Elapsed: {0} seconds", duration);
    return x0;
}

bool density::in_bounds_par(vector_t const &par) const {

    for (int i = 0; i < m_dim_par; i++) {
        if (par(i) < m_lb_par(i) || par(i) > m_ub_par(i)) {
            return false;
        }
    }
    return true;
}

bool density::in_bounds_hpar(vector_t const &hpar) const
{

    for (int i = 0; i < m_dim_hpar; i++) {
        if (hpar(i) < m_lb_hpar(i) || hpar(i) > m_ub_hpar(i)) {
            return false;
        }
    }
    return true;
}

std::vector<vector_t> density::pred_calibrated_model(std::vector<vector_t> const &x_pts, double confidence) const {

    // Verify that the confidence interval is between [0,1]
    if ( !(confidence>0) || !(confidence<1)) {
        spdlog::error("The confidence interval must be between 0 and 1. ");
        spdlog::info("Computation will be carried out with 95\% confidence");
        confidence = 0.95;
    } 
    double alfa = 1.0-confidence;

    int n_pts = x_pts.size();
    vector_t mean = vector_t::Zero(n_pts);

    //prepare the container
    matrix_t model_values(x_pts.size(), m_par_samples.size());
    
    for (int i = 0; i < m_par_samples.size(); i++) {
        
        // extract the parameter - hyperparameter sample
        vector_t par = m_par_samples[i];

        // evaluate the model
        vector_t sample = m_model(x_pts, par);

        //save the sample and update the mean
        model_values.col(i) = sample;
        mean += sample;
    }

    //rescale the mean
    mean /= m_par_samples.size();

    //create the two quantiles
    vector_t quantile_low(x_pts.size());
    vector_t quantile_up(x_pts.size());

    // Each colum holds a sample of the evaluated model 
    // To compute the confidence intervals we need to sort the rows and extract the low and upper quantiles
    for (int i = 0; i < model_values.rows(); i++) {

        auto row_i = vxd_to_v(model_values.row(i).transpose());
        std::sort(row_i.begin(), row_i.end());
        
        quantile_low(i) = row_i.at( static_cast<int>(0.5*alfa*row_i.size()) );
        quantile_up(i) = row_i.at( static_cast<int>((1-0.5*alfa)*row_i.size()) );
    }
    

    return std::vector<vector_t>{mean, quantile_low, quantile_up};
}

std::vector<vector_t> cmp::density::pred_corrected_model(std::vector<vector_t> const &x_pts) const {

    //prepare data
    vector_t pred_mean = vector_t::Zero(x_pts.size());
    matrix_t expected_of_var = matrix_t::Zero(x_pts.size(), x_pts.size()); //expected value of the variance (conditional variance)
    matrix_t second_moment_of_expected = matrix_t::Zero(x_pts.size(), x_pts.size()); //second moment of the expected value (conditional mean)

    for (int i = 0; i < m_par_samples.size(); i++) {
        
        // extract parameters and hyperparameters
        vector_t par = m_par_samples[i];
        vector_t hpar = m_hpar_samples[i];
        
        // evaluate the prediction using contribution from the model and the gp error 
        vector_t prediction = m_model(x_pts, par) + gp_cond_mean(x_pts, par, hpar);
        
        //update mean
        pred_mean += prediction;

        // the second moment of the expection 
        second_moment_of_expected += prediction * prediction.transpose();
        
        // update the mean of the variance
        expected_of_var += gp_cond_var(x_pts,par,hpar);
    }

    // divide by the number of samples
    pred_mean /= static_cast<double>(m_par_samples.size());
    second_moment_of_expected /= static_cast<double>(m_par_samples.size());
    expected_of_var /= static_cast<double>(m_par_samples.size());
    
    // compute the total variance using the law of total variance
    // note: keep only the diagonal terms
    vector_t var = second_moment_of_expected.diagonal() - (pred_mean*pred_mean.transpose()).diagonal();       // var(E(y_pred | par))
    var += expected_of_var.diagonal();                                                                        // E(var(y_pred | par))
    
    return std::vector<vector_t>{pred_mean, var};
}

vector_t density::gp_cond_mean(std::vector<vector_t> const &x_pts, vector_t const &par, vector_t const &hpar) const {
    
    // create the complete residual vector
    vector_t res(x_pts.size());
    res = m_y_obs - m_model(m_x_obs, par) - m_err_mean(m_x_obs, hpar);
    
    // compute the error covariance matrix
    matrix_t err_cov_mat = covariance(hpar);

    // compute the cholesky decomposition
    Eigen::LDLT<matrix_t> ldlt(err_cov_mat);

    // compute the k_star matrix (appears in the gp predictive equations) 
    // this is defined as the kernel evaluated at (x_pred, x_obj)
    matrix_t k_star = matrix_t::Zero(x_pts.size(), m_x_obs.size());
    for (int i = 0; i < k_star.rows(); i++) {
        for (int j = 0; j < k_star.cols(); j++) {
            
            k_star(i, j) = m_err_kernel(x_pts[i], m_x_obs[j], hpar);
        }
    }
    
    // compute the mean vector, using the gp predictive equations
    return m_err_mean(x_pts, hpar) + k_star * ldlt.solve(res);
}

matrix_t density::gp_cond_var(std::vector<vector_t> const &x_pts, vector_t const &par, vector_t const &hpar) const {

    // create the complete residual vector
    vector_t res(x_pts.size());
    res = m_y_obs - m_model(m_x_obs, par) - m_err_mean(m_x_obs, hpar);
    
    // compute the error covariance matrix
    matrix_t err_cov_mat = covariance(hpar);

    // compute the cholesky decomposition
    Eigen::LDLT<matrix_t> ldlt(err_cov_mat);

    // compute the k_star matrix (appears in the gp predictive equations) 
    // this is defined as the kernel evaluated at (x_pred, x_obj)
    matrix_t k_star = matrix_t::Zero(x_pts.size(), m_x_obs.size());
    for (int i = 0; i < k_star.rows(); i++) {
        for (int j = 0; j < k_star.cols(); j++) {
            
            k_star(i, j) = m_err_kernel(x_pts[i], m_x_obs[j], hpar);
        }
    }

    // compute the kernel matrix at prediction points
    matrix_t k_xy = matrix_t::Zero(x_pts.size(), x_pts.size());
    for (int i = 0; i < k_xy.rows(); i++) {
        for (int j = 0; j < k_xy.cols(); j++) {
            
            k_xy(i, j) = m_err_kernel(x_pts[i], x_pts[j], hpar);
        }
    }

    matrix_t gp_var = k_xy - k_star * ldlt.solve(k_star.transpose());
    return gp_var;
}

vector_t density::draw_gp_sample(std::vector<vector_t> const &x_pts, vector_t const &par, vector_t const &hpar, std::default_random_engine &rng) const {
    
    std::normal_distribution<double> dist_n(0, 1);
    
    // compute mean and covariance of the gp
    vector_t mean = gp_cond_mean(x_pts, par, hpar);
    matrix_t cov = gp_cond_var(x_pts, par, hpar);
    
    // Setup the solver (note the Cholesky decomposition cannot be used as the covariance is positive SEMIdefite)
    Eigen::SelfAdjointEigenSolver<matrix_t> solver(cov);
    vector_t eigen_val = solver.eigenvalues();

    // Compute the sqrt of the eigenvalue matrix
    for (unsigned i = 0; i < eigen_val.rows(); i++)
        eigen_val(i) = sqrt(fabs(eigen_val(i)));
    
    // Extract a uniform sample and multiply it by the sqrt of the eigenvalues
    vector_t sample(mean.size());
    for (int i = 0; i < sample.size(); i++)
        sample(i) = dist_n(rng) * eigen_val(i);
    
    // A single random sample ~ N(mean, cov)
    return mean + solver.eigenvectors() * sample;
}

vector_t density::draw_corrected_model_sample(std::vector<vector_t> const &x_pts, std::default_random_engine &rng) const {
    

    std::uniform_int_distribution<int> u_sample(0, m_par_samples.size() - 1);
    int index = u_sample(rng);

    return m_model(x_pts, m_par_samples[index]) + draw_gp_sample(x_pts, m_par_samples[index], m_hpar_samples[index], rng);
}