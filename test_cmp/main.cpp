#define NDEBUG 

#include <cmp_defines.h>
#include <mcmc.h>
#include <grid.h>
#include <density.h>
#include <kernel.h>
#include <finite_diff.h>
#include <distribution.h>
#include <optimization.h>

using namespace cmp;


// Model to calibrate
double model(const double &x, const vector_t &par) {
    double V_hat = x-1;
    return par(0)*tanh(par(1)*V_hat/par(0));
}

double model_v(const vector_t &x, const vector_t &par) {
    double V_hat = x(0)-1;
    return par(0)*tanh(par(1)*V_hat/par(0));
}

// Kernel
double err_kernel(const vector_t & x, const vector_t &y, const vector_t & hpar) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return white_noise_kernel(x,y,sigma_e) + squared_kernel(x,y,sigma_k,l);
}

// kernel gradient, i is the component required
double err_kernel_gradient(const vector_t & x, const vector_t &y, const vector_t & hpar, const int &i) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return white_noise_kernel_grad(x,y,sigma_e,i)+squared_kernel_grad(x,y,sigma_k,l,i-1);
}

// kernel hessian, i and j are the row and colum respectively
double err_kernel_hessian(const vector_t & x, const vector_t &y, const vector_t & hpar, const int &i, const int &j) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return white_noise_kernel_hess(x,y,sigma_e,i,j)+squared_kernel_hess(x,y,sigma_k,l,i-1,j-1);
}

int main() {

    // Initialize the rng
    std::default_random_engine rng(140998);

    //Read the data vector
    std::ifstream file_obs("data.csv");
    auto obs = read_vector(file_obs);

    std::vector<double> x_obs(obs.size());
    std::vector<double> y_obs(obs.size());

    for (int i=0; i<obs.size(); i++) {
        x_obs[i] = obs[i](0);
        y_obs[i] = obs[i](1);
    }


    // Two parameters
    int dim_par = 2;

    // set up the mcmc initial values
    vector_t par_0(dim_par);
    par_0 << 0.39, 0.35;

    matrix_t cov_prop = matrix_t::Identity(dim_par, dim_par)*0.5;

    // uniform prior on the parameters
    auto logprior_par = [](const vector_t & par) {
            return 0;
    };

    // model error guassian process
    auto err_mean = [](vector_t x, vector_t const &hpar) {
        return 0;
    };


    /*
    *   In this section we initialize the model error object
    */

    // priors for the hyperparameters
    cmp::inverse_gamma_distribution prior_se(3,0.04,rng);
    cmp::inverse_gamma_distribution prior_sk(3,0.04,rng);
    cmp::inverse_gamma_distribution prior_l(5,1.2,rng);


    // log prior
    auto log_prior_hpar = [&prior_se,&prior_sk,&prior_l](const vector_t & hpar) {
        return prior_se.log_pdf(exp(hpar(0)))+prior_sk.log_pdf(exp(hpar(1)))+prior_l.log_pdf(exp(hpar(2)));
    };

    auto log_prior_hpar_hessian = [&prior_se,&prior_sk,&prior_l](const vector_t & hpar, const int &i, const int &j) {
        if (i==0 && j==0)
            return prior_se.dd_log_pdf(exp(hpar(0)));
        else if (i==1 && j==1)
            return prior_sk.dd_log_pdf(exp(hpar(1)));
        else if (i==2 && j==2)
            return prior_l.dd_log_pdf(exp(hpar(2)));
        else
            return 0.0;
    };

    // Create the model error
    gp model_error;
    model_error.set_mean(err_mean);
    model_error.set_kernel(err_kernel);
    model_error.set_kernel_gradient(err_kernel_gradient);
    model_error.set_kernel_hessian(err_kernel_hessian);
    model_error.set_par_bounds(-5*vector_t::Ones(3),2*vector_t::Ones(3));
    model_error.set_logprior(log_prior_hpar);
    model_error.set_logprior_hessian(log_prior_hpar_hessian);
    model_error.set_obs(v_to_vvxd(x_obs),y_obs);


    // Here we create a function that computes the residuals
    auto compute_residuals = [&model_error, &y_obs, &x_obs](const vector_t & par) {
        vector_t res = vector_t::Zero(y_obs.size());
        for (int i = 0; i < y_obs.size(); i++) {
            res(i) = y_obs[i] - model(x_obs[i], par);
        }
        return res;
    };

    vector_t hpar_0(3);
    hpar_0 << -4.77, -4.62, -1.40;

    // density
    density d;
    d.set_model(model_v);
    d.set_model_error(&model_error);
    d.set_obs(v_to_vvxd(x_obs),y_obs);
    d.set_log_prior_par(logprior_par);

    auto cmp_score_d = [&d,&model_error,&hpar_0](const vector_t & par) {
        vector_t hpar_opt = d.hpar_opt(par,hpar_0,1e-15);
        double corr = d.log_cmp_correction(par,hpar_opt);
        return d.loglikelihood(par,hpar_opt)+model_error.logprior(hpar_opt)+corr;
    };

    // create the score function
    auto cmp_score = [&compute_residuals,&model_error,&x_obs,&hpar_0](const vector_t & par) {
        
        auto res = compute_residuals(par);

        // Compute the optimal hyperparameters
        auto hpar_fun = [&model_error,&res](const vector_t & hpar) {
            auto cov_ldlt = model_error.covariance(hpar).ldlt();
            return cmp::multivariate_normal_distribution::log_pdf(res,cov_ldlt) + model_error.logprior(hpar);
        };
        
        // Compute the optimal hyperparameters
        auto hpar_opt = cmp::arg_max(hpar_fun,hpar_0,-5*cmp::vector_t::Ones(3),5*cmp::vector_t::Ones(3),1e-15);

        auto cov_ldlt = model_error.covariance(hpar_opt).ldlt();

        // Compute the correction factor
        cmp::matrix_t S_theta = cmp::matrix_t::Zero(3,3);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = i; j < 3; j++) {
                S_theta(i,j) = -model_error.logprior_hessian(hpar_opt,i,j)-cmp::multivariate_normal_distribution::log_pdf_hessian(res,cov_ldlt,model_error.covariance_gradient(hpar_opt,i),model_error.covariance_gradient(hpar_opt,j),model_error.covariance_hessian(hpar_opt,i,j));

                if (i!=j) {
                    S_theta(j,i) = S_theta(i,j);
                }
            }
        }

        // Compute the correction factor
        Eigen::LDLT<matrix_t> ldlt(S_theta);
        double corr = -0.5*(ldlt.vectorD().array().abs().log()).sum();
        
        return  cmp::multivariate_normal_distribution::log_pdf(res,cov_ldlt)+model_error.logprior(hpar_opt)+corr;
    };

    // Set up a chain
    mcmc chain(dim_par);
    chain.seed(par_0,cmp_score_d(par_0));

    // Set up the proposal
    cmp::multivariate_normal_distribution proposal(vector_t::Zero(2),cov_prop.ldlt(), rng);
    chain.init(&proposal);

    // Parameters for the MCMC
    size_t n_burn = 10'000;
    size_t n_samples = 10'000;
    size_t n_skip = 1;

    // Run the chain
    for (size_t i = 0; i < n_burn; i++) {
        chain.step(cmp_score_d, true, 0.1);

        if (i % 1000 == 0 && i > 0) {
            proposal.set_cov_ldlt(chain.get_adapted_cov());
        }
    }
    chain.info();

    // Save the samples
    std::vector<vector_t> samples(n_samples);

    for (size_t i = 0; i < n_samples; i++) {
        samples[i] = chain.get_par();

        // Subsample
        for (size_t j = 0; j < n_skip; j++) {
            chain.step(cmp_score_d, true, 0.1);
        }
    }
    chain.info();
    // Save the samples
    std::ofstream file("par_samples.csv");
    cmp::write_vector(samples,file);

    return 0;
}