#include "gp.h"
#include "optimization.h"
#include "utils.h"
#include "finite_diff.h"
#include "distribution.h"

using namespace cmp;

double cmp::opt_routine(nlopt::vfunc opt_func, void *data_ptr, Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, double ftol_rel, nlopt::algorithm alg) {
    
    //Convert Eigen::VectorXd to std::vector
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

    //Try the optimization
    double f_val{0.0};
    try {
        local_opt.optimize(x0_v, f_val);
    } catch (const std::runtime_error &err) {
        spdlog::error("Local optimization failed. Keeping initial value.");
    }
    
    //Convert back to Eigen::VectorXd
    x0 = v_to_vxd(x0_v);
    return f_val;
}

Eigen::VectorXd cmp::arg_max(const score_t &score, const Eigen::VectorXd &par_0, const Eigen::VectorXd &par_lb, const Eigen::VectorXd &par_ub, const double &tol){

    Eigen::VectorXd par_opt = par_0;

    std::pair<const score_t &, void *> my_pair(score,nullptr);
    void *p_data = (void *) &my_pair;
    
    auto my_fun = [](const std::vector<double> &x, std::vector<double> &grad, void *data)  {
        std::pair<const score_t &, void *> *my_pair = (std::pair<const score_t &, void *>*)data;
        Eigen::VectorXd x_v = v_to_vxd(x);
        return my_pair->first(x_v);
    };

    // Perform the optimization
    double fmap = cmp::opt_routine(my_fun,p_data,par_opt,par_lb,par_ub,tol,nlopt::LN_SBPLX);

    return par_opt;
}

double cmp::opt_fun_gp_mle(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {
    
    // The data type contains the gp class
    auto my_gp = static_cast<const gp*>(data_bit);
    
    // Convert the hyperparameters from std::vector to Eigen::VectorXd
    Eigen::VectorXd par = v_to_vxd(x);
    Eigen::LDLT<Eigen::MatrixXd> cov_ldlt = my_gp->covariance(par).ldlt();

    // Evaluate the log-likelihood and the log-prior
    double ll = cmp::multivariate_normal_distribution::log_pdf(my_gp->residual(par),cov_ldlt);

    // Update the gradient (note that the gradient is optional)
    if (grad.size() != 0) {
        
        // Update each component of the gradient
        for (int n = 0; n < par.size(); n++) {
            auto cov_gradient = my_gp->covariance_grad(par,n);
            auto mean_gradient = my_gp->mean_grad(par,n);
            grad[n] = cmp::multivariate_normal_distribution::log_pdf_gradient(my_gp->residual(par),cov_ldlt,cov_gradient,mean_gradient);
        }
    }
    return ll;
}

double cmp::opt_fun_gp_map(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {

    // The data type contains the gp class
    auto my_gp = static_cast<const gp*>(data_bit);
    Eigen::VectorXd par = v_to_vxd(x);
    
    // Call the MLE function
    double ll = opt_fun_gp_mle(x,grad,data_bit);

    // Update the gradient (note that the gradient is optional)
    if (grad.size() != 0) {
        
        // Update each component of the gradient
        for (int n = 0; n < par.size(); n++) {
            grad[n] += my_gp->log_prior_grad(par,n);
        }
    }
    return ll+my_gp->log_prior(par);
}

double cmp::opt_fun_gp_mloo(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {
    
    // The data type contains the gp class
    auto my_gp = (const gp*) data_bit;
    Eigen::VectorXd par = v_to_vxd(x);
    Eigen::LDLT<Eigen::MatrixXd> cov_ldlt = my_gp->covariance(par).ldlt();

    //compute the inverse of the covariance matrix
    size_t n_obs = my_gp->get_size();
    Eigen::MatrixXd k_inv = cov_ldlt.solve(Eigen::MatrixXd::Identity(n_obs,n_obs));

    //compute the residuals
    Eigen::VectorXd res = my_gp->residual(par);

    //compute the leave-one-out residuals
    Eigen::VectorXd loo_res = Eigen::VectorXd::Zero(res.size());
    loo_res = k_inv*res;
    loo_res.array() /= k_inv.diagonal().array();
    
    // compute the leave-one-out variance
    Eigen::VectorXd loo_var = Eigen::VectorXd::Zero(res.size());
    loo_var.array() = 1.0/k_inv.diagonal().array();

    // Compute the leave-one-out log-likelihood
    Eigen::VectorXd loo_ll = Eigen::VectorXd::Zero(res.size());
    loo_ll.array() = - 0.5 * loo_var.array().log() - 0.5 * loo_res.array().square()/loo_var.array();

    // Update the gradient (note that the gradient is optional)
    if (grad.size() != 0) {
        throw std::runtime_error("Gradient computation is not implemented for the leave-one-out log-likelihood");
    }

    return loo_ll.sum();
}

double cmp::opt_fun_gp_mloop(const std::vector<double> &x, std::vector<double> &grad, void *data_bit) {

    // The data type contains the gp class
    auto my_gp = (const gp*) data_bit;
    Eigen::VectorXd par = v_to_vxd(x);
    

    // Call the MLOO function
    double ll = opt_fun_gp_mloo(x,grad,data_bit);

    return ll+my_gp->log_prior(par);

}