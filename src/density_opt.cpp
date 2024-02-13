#include "density.h"

using namespace cmp; 

density_opt::density_opt(density const &d) : density(d) {
    
    m_par_samples.clear();
    m_hpar_samples.clear();
}

vector_t density_opt::hpar_opt(vector_t const &par, vector_t x0, double ftol_rel) const {

    //compute the reisduals
    vector_t res = m_y_obs - m_model(m_x_obs, par);

    //construct the data type
    auto data = std::make_pair(&res, this);

    opt_routine(opt_fun_fmp, &data, x0, m_lb_hpar, m_ub_hpar, ftol_rel, nlopt::LN_SBPLX);
    return x0;
}

vector_t density_opt::hpar_opt_grad(vector_t const &par, vector_t x0, double ftol_rel) const {
    
    //compute the reisduals
    vector_t res = m_y_obs - m_model(m_x_obs, par);

    //construct the data type
    auto data = std::make_pair(&res, this);

    int fin = opt_routine(opt_fun_fmp_grad, &data, x0, m_lb_hpar, m_ub_hpar, ftol_rel, nlopt::LD_TNEWTON_PRECOND_RESTART);
    return x0;
}

void cmp::density_opt::set_beta(double beta) {
    m_beta = beta;
}

double cmp::density_opt::get_beta() const{
    return m_beta;
}

matrix_t density_opt::covariance_gradient(const vector_t &hpar, const int &n) const{
    size_t n_obs = m_x_obs.size();

    // Initialize the kernel derivative
    matrix_t k_der = matrix_t::Zero(n_obs, n_obs);
    
    // Fill the matrix that contains the kernel derivative
    for (int i = 0; i < n_obs; i++) {
        for (int j = i; j < n_obs; j++) {
            
            k_der(i, j) = m_err_kernel_gradient(m_x_obs[i], m_x_obs[j], hpar, n);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

matrix_t density_opt::covariance_hessian(const vector_t &hpar, const int &l, const int &k) const {
    size_t n_obs = m_x_obs.size();

    // Initialize the kernel hessian
    matrix_t k_der = matrix_t::Zero(n_obs, n_obs);
    
    // Fill the matrix that contains the kernel hessian
    for (int i = 0; i < n_obs; i++) {
        for (int j = i; j < n_obs; j++) {
            
            k_der(i, j) = m_err_kernel_hessian(m_x_obs[i], m_x_obs[j], hpar, l, k);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}


double density_opt::loglikelihood_gradient(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res, const int &i) const{
    
    // Useful quantities (see Rasmussen for reference)
    vector_t alpha = cov_inv.solve(res);
    matrix_t alpha_alpha_t = alpha * alpha.transpose();
    matrix_t covariance_grad = covariance_gradient(hpar,i);

        
    return 0.5 * (alpha_alpha_t * covariance_grad - cov_inv.solve(covariance_grad)).trace();
}


double density_opt::loglikelihood_hessian(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res, const int &l, const int &k) const {

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
            
            covariance_hess(i, j) = m_err_kernel_hessian(m_x_obs[i], m_x_obs[j], hpar, l, k);
            covariance_grad_l(i, j) = m_err_kernel_gradient(m_x_obs[i], m_x_obs[j], hpar, l);
            covariance_grad_k(i, j) = m_err_kernel_gradient(m_x_obs[i], m_x_obs[j], hpar, k);
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

double density_opt::log_cmp_correction(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res) {
    // S contains the hessian of the posterior.
    matrix_t S = matrix_t::Zero(m_dim_hpar,m_dim_hpar);

    for(int i=0; i<m_dim_hpar; i++) {
        for (int j=i; j<m_dim_hpar; j++) {

            //Contribution from likelihood and prior
            S(i,j) = -loglikelihood_hessian(hpar,cov_inv,res,i,j) - logprior_hpar_hessian(hpar,i,j);

            // the tensor is symmetric
            if (i != j) {
                S(j,i) = S(i,j);
            }
        }
    }

    //Compute the LDL factorization and compute the determinant
    Eigen::LDLT<matrix_t> ldlt(S);
    return -0.5*(ldlt.vectorD().array().abs().log()).sum();

}
