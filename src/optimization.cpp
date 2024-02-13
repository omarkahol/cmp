#include "density.h"

using namespace cmp;

double cmp::opt_routine(nlopt::vfunc opt_func, void *data_ptr, vector_t &x0, const vector_t &lb, const vector_t &ub, double ftol_rel, nlopt::algorithm alg) {
    
    //Convert vector_t to std::vector
    std::vector<double> x0_v = vxd_to_v(x0);
    std::vector<double> lb_v = vxd_to_v(lb);
    std::vector<double> ub_v = vxd_to_v(ub);

    //Initialize the algorithm
    nlopt::opt local_opt(alg, x0.size());

    //Setup properties
    local_opt.set_max_objective(opt_func, data_ptr);
    local_opt.set_ftol_rel(ftol_rel);
    local_opt.set_lower_bounds(lb_v);
    local_opt.set_upper_bounds(ub_v);

    //Trye the optimization
    double f_val{};
    try {
        local_opt.optimize(x0_v, f_val);
    } catch (const std::runtime_error &err) {
        spdlog::error("Local optimization failed. Keeping initial value.");
    }
    
    //Convert back to vector_t
    x0 = v_to_vxd(x0_v);
    return f_val;
}

double cmp::opt_fun_fmp(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {
    
    // The data type contains the residuals and the density_opt class
    auto data = (std::pair<vector_t*, const density_opt*> *) data_bit;
    
    auto res = data->first;
    auto d_opt = data->second;
    
    // Convert the hyperparameters from std::vector to vector_t
    vector_t hpar = v_to_vxd(x);

    //Compute the covariance matrix
    matrix_t k_mat = d_opt->covariance(hpar);

    //compute the Cholesky decomposition and retrieve the function
    Eigen::LDLT<matrix_t> ldlt(k_mat);
    double ll = d_opt->loglikelihood(*res, ldlt);
    double lp = d_opt->logprior_hpar(hpar);
    
    return d_opt->get_beta()*ll + lp;
};

double cmp::opt_fun_fmp_grad(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {
    
    // The data type contains the residuals and the density_opt class
    auto data = (std::pair<vector_t*, const density_opt*> *) data_bit;
    
    auto res = data->first;
    auto d_opt = data->second;

    // Convert the hyperparameters from std::vector to vector_t
    vector_t hpar = v_to_vxd(x);

    //Compute the covariance matrix
    matrix_t k_mat = d_opt->covariance(hpar);

    //compute the Cholesky decomposition and retrieve the function
    Eigen::LDLT<matrix_t> ldlt(k_mat);
    double ll = d_opt->loglikelihood(*res, ldlt);
    double lp = d_opt->logprior_hpar(hpar);
    
    // Update the gradient
    if (!(grad.size() == 0)) {

        // Update each component of the gradient
        for (int n = 0; n < hpar.size(); n++) {

            // Evaluate the contribution of the likelihood and the contribution of the prior
            grad[n] = d_opt->loglikelihood_gradient(hpar,ldlt,*res,n) + d_opt->logprior_hpar_gradient(hpar, n);

            /*
            NOTE -- This needs to be fixed AS SOON AS POSSIBLE. Right now I will do a transformation to transform the gradient wrt the hpar to the gradient wrt the log-hpar. 
            This will have to be specified by the user as the Jacobian of the transformation.
            */
           grad[n] = grad[n]*exp(hpar[n]);
        }
    }
    return ll + lp;
};


double cmp::opt_fun_par(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {
    const density_opt *d_opt = (const density_opt *) data_bit;
    vector_t pars = v_to_vxd(x);
    return 0;
};


double cmp::opt_fun_KOH(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {

    // Cast the void* to the correct type
    auto data = static_cast<std::pair<density*, double>*>(data_bit);
    auto d_opt = data->first;
    auto int_const = data->second;

    // Convert the hyperparameters from std::vector to vector_t
    vector_t hpar = v_to_vxd(x);

    // Retrieve the observation locations
    const std::vector<vector_t> *x_obs = d_opt->get_x_obs();

    //Compute the covariance matrix
    matrix_t k_mat = d_opt->covariance(hpar);

    //Cholesky decomposition of the kernel matrix
    Eigen::LDLT<matrix_t> ldlt(k_mat);

    //Iterate through every point of the grid
    size_t n_par = d_opt->get_grid()->size(); //number of points
    double integral{};
    for(size_t i=0; i<n_par; i++) {

        // Model parameters
        vector_t par = d_opt->get_grid()->at(i);

        // Residual
        vector_t res = *d_opt->get_y_obs() - d_opt->evaluate_model(*x_obs,par) - d_opt->error_mean(*x_obs, hpar); 
        
        // sum and check for potential numerical errors
        integral += exp(d_opt->loglikelihood(res,ldlt) + d_opt->logprior_par(par) + d_opt->logprior_hpar(hpar));

        if (std::isinf(integral)) {
            spdlog::error("KOH optimization failed! The inetgral became too large. Try increasing the value of the integration constant!");
        }
    }

    return log(integral);
};