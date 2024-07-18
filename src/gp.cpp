#include "gp.h"
#include "optimization.h"
using namespace cmp;


void gp::set_obs(const std::vector<vector_t> &x_obs, const std::vector<double>&y_obs) {
    if (x_obs.size() != y_obs.size()) {
        spdlog::error("x_obs and y_obs sizes do not match!\nx_obs has size {0:d} and y_obs has size {1:d}",x_obs.size(),y_obs.size());
    } else {
        m_x_obs = x_obs;
        m_y_obs = y_obs;
    }
}

matrix_t gp::covariance(const vector_t &par) const{
    
    //Number of points
    int n_pts = m_x_obs.size();
    
    //Computation of the kernel matrix
    matrix_t kernel_mat = matrix_t::Zero(n_pts, n_pts);
    for (int i = 0; i < n_pts; i++) {
        for (int j = i; j < n_pts; j++) {
        
            kernel_mat(i, j) = m_kernel(m_x_obs.at(i), m_x_obs.at(j), par);

            //Kernel matrix is symmetric
            if (i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat;
}

vector_t gp::evaluate_mean(const std::vector<vector_t> &x_pts, const vector_t &par) const{
    vector_t y(x_pts.size());

    for (size_t i=0; i<x_pts.size(); i++) {
        y(i) = m_mean(x_pts.at(i),par);
    }

    return y;
}

vector_t cmp::gp::evaluate_mean_gradient(const std::vector<vector_t> &x_pts, const vector_t &par, int i) const {
    
    vector_t y(x_pts.size());
    for (size_t j=0; j<x_pts.size(); j++) {
        y(j) = m_mean_gradient(x_pts.at(j), par, i);
    }

    return y;
}

vector_t gp::residual(const vector_t &par) const{
    
    vector_t res(m_y_obs.size());
    for (size_t i=0; i<m_y_obs.size(); i++) {
        res(i) = m_y_obs[i]-m_mean(m_x_obs[i],par);
    }
    return res;
}

double gp::loglikelihood(vector_t const &res, Eigen::LDLT<matrix_t> const &ldlt) const {
    
    // Solve linear system
    vector_t alpha = ldlt.solve(res);
    
    return -0.5 * res.dot(alpha) - 0.5 * (ldlt.vectorD().array().log()).sum() - 0.5 * res.size() * log(2 * M_PI);
}

vector_t gp::par_opt(vector_t x0, double ftol_rel, nlopt::algorithm alg) const {
    opt_routine(opt_fun_gp, (void*)this, x0, m_lb_par, m_ub_par, ftol_rel, alg);
    return x0;
}

vector_t gp::par_opt_loo(vector_t x0, double ftol_rel) const {
    opt_routine(opt_fun_gp_loo, (void*)this, x0, m_lb_par, m_ub_par, ftol_rel, nlopt::LN_SBPLX);
    return x0;
}

double gp::prediction_mean(const vector_t &x, const vector_t &par, const vector_t &alpha) const {
    
    int n_obs = m_x_obs.size();

    double k_star_dot_alpha = 0.0;
    for(int i=0; i<n_obs; i++) {
        k_star_dot_alpha += m_kernel(x,m_x_obs.at(i),par)*alpha(i);
    }

    return m_mean(x,par) + k_star_dot_alpha;
}

matrix_t gp::predict(const std::vector<vector_t> &x_pts, const vector_t &par, const Eigen::LDLT<matrix_t> &ldlt, const vector_t &res) const{
    
    // Define the two kernel evaluation matrices
    size_t n_obs = m_x_obs.size();
    size_t n_pred = x_pts.size();
    matrix_t k_star = matrix_t::Zero(n_pred, n_obs);
    vector_t k_xy = vector_t::Zero(n_pred);

    // Fill the matrices
    for (size_t i = 0; i < n_pred; i++) {
        k_xy(i) = m_kernel(x_pts.at(i), x_pts.at(i), par);
        for (size_t j = 0; j < n_obs; j++) {
            k_star(i, j) = m_kernel(x_pts.at(i), m_x_obs.at(j), par);
        }
    }

    // Define the matrix 
    matrix_t mean_var(n_pred,2);
    mean_var.col(0) = evaluate_mean(x_pts, par) + k_star * ldlt.solve(res);
    mean_var.col(1) = k_xy - (k_star * ldlt.solve(k_star.transpose())).diagonal();

    return mean_var;

}

matrix_t gp::predict(const std::vector<vector_t> &x_pts, const vector_t &par) const{
    // Compute the residuals and the covariance matrix
    auto cov_mat = this->covariance(par);
    auto res = this->residual(par);
    auto ldlt = Eigen::LDLT<matrix_t>(cov_mat);

    // Call overloaded version of the function
    return gp::predict(x_pts,par,ldlt,res);
}


vector_t gp::draw_sample(const std::vector<vector_t> &x_pts, vector_t const &par, std::default_random_engine &rng) const {

    std::normal_distribution<double> dist_n(0, 1);
    size_t n_obs = m_x_obs.size();
    size_t n_pts = x_pts.size();

    // Compute the residuals and the covariance matrix
    matrix_t cov(n_obs,n_obs);
    vector_t res(n_obs);

    for(size_t i=0; i<n_obs; i++) {
        res(i) = m_y_obs[i] - m_mean(m_x_obs[i],par);
        for(size_t j=i; j<n_obs;j++) {
            cov(i,j) = m_kernel(m_x_obs.at(i),m_x_obs.at(j),par);

            if(i!=j) {
                cov(j,i) = cov(i,j);
            }
        }
    }
    auto ldlt = cov.ldlt();

    // Compute the matrices for the predictive mean and variance
    matrix_t k_star(n_pts, n_obs);
    matrix_t k_xy(n_pts, n_pts);
    vector_t gp_mean (n_pts);

    for (size_t i = 0; i < n_pts; i++) {

        gp_mean(i) = m_mean(x_pts.at(i),par);
        
        // Fill k_star
        for (size_t j = 0; j < n_obs; j++) {
            k_star(i, j) = m_kernel(x_pts.at(i), m_x_obs.at(j), par);
        }
        
        // Fill k_xy
        for (size_t j = i; j < n_pts; j++) {
            k_xy(i, j) = m_kernel(x_pts.at(i), x_pts.at(j), par);

            // Tensor must be symmetric
            if (i!=j){
                k_xy(j,i) = k_xy(i,j);
            }
        }

    }

    // Use GP predictive equations for the evaluation of the mean and variance
    vector_t prediction_mean = gp_mean + k_star*ldlt.solve(res);
    matrix_t prediction_var = k_xy - k_star * ldlt.solve(k_star.transpose());
    
    // Setup the solver and compute the eigen-decomposition of the covariance matrix
    Eigen::SelfAdjointEigenSolver<matrix_t> solver(prediction_var);
    vector_t eigen_val = solver.eigenvalues();

    // Compute the sqrt of the eigenvalue matrix
    for (unsigned i = 0; i < eigen_val.rows(); i++)
        eigen_val(i) = sqrt(fabs(eigen_val(i)));
    
    // Extract a uniform sample and multiply it by the sqrt of the eigenvalues
    vector_t sample(prediction_mean.size());
    for (int i = 0; i < sample.size(); i++)
        sample(i) = dist_n(rng) * eigen_val(i);
    
    // A single random sample ~ N(mean, cov)
    return prediction_mean + solver.eigenvectors() * sample;
}


matrix_t gp::covariance_gradient(const vector_t &par, const int &n) const{
    size_t n_obs = m_x_obs.size();

    // Initialize the kernel derivative
    matrix_t k_der = matrix_t::Zero(n_obs, n_obs);
    
    // Fill the matrix that contains the kernel derivative
    for (int i = 0; i < n_obs; i++) {
        for (int j = i; j < n_obs; j++) {
            
            k_der(i, j) = m_kernel_gradient(m_x_obs.at(i), m_x_obs.at(j), par, n);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

matrix_t gp::covariance_hessian(const vector_t &par, const int &l, const int &k) const {
    size_t n_obs = m_x_obs.size();

    // Initialize the kernel hessian
    matrix_t k_der = matrix_t::Zero(n_obs, n_obs);
    
    // Fill the matrix that contains the kernel hessian
    for (int i = 0; i < n_obs; i++) {
        for (int j = i; j < n_obs; j++) {
            
            k_der(i, j) = m_kernel_hessian(m_x_obs.at(i), m_x_obs.at(j), par, l, k);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

matrix_t cmp::gp::reduced_covariance_matrix(const std::vector<vector_t> x_pts, const vector_t &par) {

    // Sizes of the matrices
    size_t n_obs = m_x_obs.size();
    size_t n_pts = x_pts.size();

    // Get the initial covariance
    matrix_t k = matrix_t::Identity(n_obs+1,n_obs+1);
    matrix_t k_star = matrix_t::Zero(n_pts, n_obs+1);
    matrix_t k_xy_diag = vector_t::Zero(x_pts.size());

    // Fill the initial part of the covariance
    for (size_t i=0; i<n_obs; i++) {
        for(size_t j=i; j<n_obs; j++) {
            
            k(i,j) = m_kernel(m_x_obs[i],m_x_obs[j],par);
            
            // Kernel is symmetric
            if (i != j) {
                k(j,i) = k(i,j);
            }
        }
    }
    
    // Fill k_star and k_xy
    for(size_t i=0; i<n_pts; i++) {
        for (size_t j=0; j<n_obs; j++) {
            k_star(i,j) = m_kernel(x_pts[i],m_x_obs[j],par);
        }

        k_xy_diag(i) = m_kernel(x_pts[i],x_pts[i],par);
    }

    // This is the initial predictive variance
    vector_t var_0 = k_xy_diag - (k_star * k.ldlt().solve(k_star.transpose())).diagonal();
    
    // Now we compute the actual matrix
    matrix_t vij = matrix_t::Zero(n_pts,n_pts);
    for(size_t i=0; i<n_pts; i++) {

        vector_t x_sel = x_pts[i];
        
        // Fill the last row of the covariance matrix
        for (size_t j=0; j<n_obs; j++) {
            k(n_obs,j) = m_kernel(m_x_obs[j],x_sel,par);
            k(j,n_obs) = k(n_obs,j);
        }
        k(n_obs,n_obs) = m_kernel(x_sel,x_sel,par);

        // Fill the last colum of the k_star matrix
        for (size_t j=0; j<n_pts; j++) {
            k_star(j,n_obs) = m_kernel(x_pts[j],x_sel,par);
        }

        // Compute the new variance
        vector_t var_i = k_xy_diag - (k_star * k.ldlt().solve(k_star.transpose())).diagonal();
        vij.row(i).array() = var_0.array()-var_i.array();
    }

    return vij;
}

double cmp::gp::loglikelihood_gradient(const vector_t &res, const Eigen::LDLT<matrix_t> &ldlt, const vector_t &par, const int &n) const {
    
    // Solve linear system
    vector_t alpha = ldlt.solve(res);
    
    // Compute the gradient
    vector_t grad = vector_t::Zero(par.size());
    for (int i = 0; i < par.size(); i++) {
        
        // Compute the derivative of the covariance matrix
        matrix_t k_der = covariance_gradient(par, i);

        // Compute the derivative of the mean
        vector_t mean_der = evaluate_mean_gradient(m_x_obs, par, i);
        
        // Kernel contribution
        grad(i) = 0.5 * (alpha*alpha.transpose()*k_der - ldlt.solve(k_der)).trace();

        // Mean contribution
        grad(i) += res.transpose()*ldlt.solve(mean_der);
    }
    
    return grad(n);
}