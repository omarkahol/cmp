#include "density.h"
#include "optimization.h"

using namespace cmp;

density::density(grid *grid) {

    //set the grid
    m_grid = grid;

    //set lower and upper bounds 
    m_lb_par = m_grid->m_lb;
    m_ub_par = m_grid->m_ub;

}

void density::set_obs(const std::vector<vector_t> &x_obs, const std::vector<double>&y_obs) {
    
    if (x_obs.size() != y_obs.size()) {
        spdlog::error("x_obs and y_obs sizes do not match!\nx_obs has size {0:d} and y_obs has size {1:d}",x_obs.size(),y_obs.size());
    } else {
        m_x_obs = x_obs;
        m_y_obs = y_obs;
    }

    // Set also the observations for the model error gp
    //m_model_error->set_obs(x_obs,y_obs);

}

vector_t density::evaluate_model(const std::vector<vector_t> &x_pts, vector_t const &par) const{
    vector_t y(x_pts.size());

    for (size_t i=0; i<x_pts.size(); i++) {
        y(i) = m_model(x_pts[i],par);
    }

    return y;
}

vector_t density::residuals(vector_t const &par) const {
    vector_t res(m_y_obs.size());

    for(size_t i=0; i<m_y_obs.size(); i++) {
        res(i) = m_y_obs[i]-m_model(m_x_obs[i],par);
    }
    return res;
}

double density::loglikelihood(vector_t const &res, Eigen::LDLT<matrix_t> const &ldlt) const {
    
    // Solve linear system
    vector_t alpha = ldlt.solve(res);
    
    return -0.5 * res.dot(alpha) - 0.5 * (ldlt.vectorD().array().log()).sum() - 0.5 * res.size() * log(2 * M_PI);
}

double density::loglikelihood(vector_t const &par, vector_t const &hpar) const {
    
    // Compute the covariance matrix
    matrix_t cov_mat = m_model_error->covariance(hpar);

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
    opt_routine(opt_fun_KOH, &data, x0, m_model_error->get_lb_par(), m_model_error->get_ub_par(), ftol_rel,nlopt::LN_SBPLX);

    // end optimization
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

    //print info
    spdlog::info("KOH optimisation over. Elapsed: {0} seconds", duration);
    return x0;
}

matrix_t density::pred_calibrated_model(const std::vector<vector_t> &x_pts, double confidence) const {

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
    matrix_t model_values(n_pts, m_par_samples.size());
    
    for (int i = 0; i < m_par_samples.size(); i++) {
        
        // extract the parameter - hyperparameter sample
        vector_t par = m_par_samples[i];

        // evaluate the model
        vector_t sample = evaluate_model(x_pts, par);

        //save the sample and update the mean
        model_values.col(i) = sample;
        mean += sample;
    }

    //rescale the mean
    mean /= m_par_samples.size();

    
    /*
    Note, computing the confidence interval
    */
    matrix_t return_matrix(n_pts,3);
    for (int i = 0; i < n_pts; i++) {

        // Perform the sort
        auto row_i = vxd_to_v(model_values.row(i).transpose());
        std::sort(row_i.begin(), row_i.end());

        // Fill the matrix
        return_matrix(i,0) = mean(i); // mean
        return_matrix(i,1) = row_i.at( static_cast<int>(0.5*alfa*row_i.size()) ); // lower quantile
        return_matrix(i,2) = row_i.at( static_cast<int>((1-0.5*alfa)*row_i.size()) ); // upper quantile
        
    }
    
    return return_matrix;
}

matrix_t cmp::density::pred_corrected_model(const std::vector<vector_t> &x_pts) const {

    // Prepare data
    vector_t pred_mean = vector_t::Zero(x_pts.size());
    vector_t expected_of_var = vector_t::Zero(x_pts.size()); //expected value of the variance (conditional variance)
    matrix_t second_moment_of_expected = matrix_t::Zero(x_pts.size(), x_pts.size()); //second moment of the expected value (conditional mean)

    for (int i = 0; i < m_par_samples.size(); i++) {
        
        // extract parameters and hyperparameters
        vector_t par = m_par_samples[i];
        vector_t hpar = m_hpar_samples[i];

        // evaluate the matrices and the residuals
        vector_t res = residuals(par);
        matrix_t cov = m_model_error->covariance(hpar);
        auto cov_inv = Eigen::LDLT<matrix_t>(cov);

        matrix_t model_error_pred=m_model_error->predict(x_pts,hpar,cov_inv,res);
        
        // evaluate the prediction using contribution from the model and the gp error 
        vector_t prediction = evaluate_model(x_pts, par) + model_error_pred.col(0);
        
        //update mean
        pred_mean += prediction;

        // the second moment of the expection 
        second_moment_of_expected += prediction * prediction.transpose();
        
        // update the mean of the variance
        expected_of_var += model_error_pred.col(1);
    }

    // divide by the number of samples
    pred_mean /= static_cast<double>(m_par_samples.size());
    second_moment_of_expected /= static_cast<double>(m_par_samples.size());
    expected_of_var /= static_cast<double>(m_par_samples.size());
    
    // compute the total variance using the law of total variance
    // note: keep only the diagonal terms
    vector_t var = (second_moment_of_expected - (pred_mean*pred_mean.transpose())).diagonal();                // var(E(y_pred | par))
    var += expected_of_var;                                                                                   // E(var(y_pred | par))
    

    // Return data
    matrix_t mean_var(x_pts.size(),2);
    mean_var.col(0) = pred_mean;
    mean_var.col(1) = var;
    return mean_var;
}

vector_t density::draw_corrected_model_sample(const std::vector<vector_t> &x_pts, std::default_random_engine &rng) const {
    
    std::uniform_int_distribution<int> u_sample(0, m_par_samples.size() - 1);
    int index = u_sample(rng);

    return evaluate_model(x_pts, m_par_samples[index]) + m_model_error->draw_sample(x_pts, m_hpar_samples[index], rng);
}

vector_t density::hpar_opt(vector_t const &par, vector_t x0, double ftol_rel) const {

    //compute the residuals
    vector_t res = residuals(par);

    //construct the data type
    auto data = std::make_pair(&res, this);

    opt_routine(opt_fun_cmp, &data, x0, m_model_error->get_lb_par(), m_model_error->get_ub_par(), ftol_rel, nlopt::LN_SBPLX);
    return x0;
}

vector_t density::hpar_opt_grad(vector_t const &par, vector_t x0, double ftol_rel) const {
    
    //compute the reisduals
    vector_t res = residuals(par);

    //construct the data type
    auto data = std::make_pair(&res, this);

    int fin = opt_routine(opt_fun_cmp_grad, &data, x0, m_model_error->get_lb_par(), m_model_error->get_ub_par(), ftol_rel, nlopt::LD_TNEWTON_PRECOND_RESTART);
    return x0;
}


double density::loglikelihood_gradient(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res, const int &i) const{
    
    // Useful quantities (see Rasmussen for reference)
    vector_t alpha = cov_inv.solve(res);
    matrix_t alpha_alpha_t = alpha * alpha.transpose();
    matrix_t covariance_grad = m_model_error->covariance_gradient(hpar,i);

        
    return 0.5 * (alpha_alpha_t * covariance_grad - cov_inv.solve(covariance_grad)).trace();
}


double density::loglikelihood_hessian(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res, const int &l, const int &k) const {

    // Required quantities for the computation
    vector_t alpha = cov_inv.solve(res);
    matrix_t alpha_alpha_t = alpha * alpha.transpose();

    //Evaluate the gradients and the hessian in a single run instead of calling the three functions (maybe more efficient?)
    // matrix_t covariance_grad_l = covariance_gradient(hpar,l);
    // matrix_t covariance_grad_k = covariance_gradient(hpar,k);
    // matrix_t covariance_hess = covariance_hessian(hpar,l,k);
    
    size_t n_obs = res.size();
    matrix_t covariance_grad_l = matrix_t::Zero(n_obs, n_obs);
    matrix_t covariance_grad_k = matrix_t::Zero(n_obs, n_obs);
    matrix_t covariance_hess = matrix_t::Zero(n_obs, n_obs);

    // Fill the various matrices
    for (int i = 0; i < n_obs; i++) {
        for (int j = i; j < n_obs; j++) {
            
            covariance_hess(i, j)   =   m_model_error->kernel_hessian(m_x_obs.at(i), m_x_obs.at(j), hpar, l, k);
            covariance_grad_l(i, j) =   m_model_error->kernel_gradient(m_x_obs.at(i), m_x_obs.at(j), hpar, l);
            covariance_grad_k(i, j) =   m_model_error->kernel_gradient(m_x_obs.at(i), m_x_obs.at(j), hpar, k);
            
            // The tensors must be symmetric
            if (i != j) {
                covariance_hess(j, i)   = covariance_hess(i,j);
                covariance_grad_l(j, i) = covariance_grad_l(i,j);
                covariance_grad_k(j, i) = covariance_grad_k(i,j);
            }
        }
    }

    // Compute the remaining symbols
    auto a_l  = cov_inv.solve(covariance_grad_l);
    auto a_k  = cov_inv.solve(covariance_grad_k);
    auto sym_tens = 0.5*((a_l*alpha_alpha_t) + (a_l*alpha_alpha_t).transpose());

    // Compute the 4 contributions to the hessian (I think that taking the trace 4 times is faster than adding and then taking the trace(?))
    double H1 = (alpha_alpha_t*covariance_hess).trace();
    double H2 = (cov_inv.solve(covariance_hess)).trace();
    double H3 = (sym_tens*covariance_grad_k).trace();
    double H4 = (a_l*a_k).trace();

    // return the weighted sum of the contributions
    return 0.5*H1 - 0.5*H2 - H3 + 0.5*H4;
}

double density::log_cmp_correction(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res) {
    
    int n_hpar = hpar.size();
    
    // S contains the hessian of the posterior.
    matrix_t S = matrix_t::Zero(n_hpar,n_hpar);

    for(int i=0; i<n_hpar; i++) {
        for (int j=i; j<n_hpar; j++) {

            //Contribution from likelihood and prior
            S(i,j) = -loglikelihood_hessian(hpar,cov_inv,res,i,j) - m_model_error->logprior_hessian(hpar,i,j);

            // The tensor is symmetric
            if (i != j) {
                S(j,i) = S(i,j);
            }
        }
    }

    //Compute the LDL factorization and compute the determinant
    Eigen::LDLT<matrix_t> ldlt(S);
    return -0.5*(ldlt.vectorD().array().abs().log()).sum();

}