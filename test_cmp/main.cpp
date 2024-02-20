#define NDEBUG 

#include <cmp_defines.h>
#include <mcmc.h>
#include <io.h>
#include <doe.h>
#include <density.h>
#include <pdf.h>
#include <kernel.h>

using namespace cmp;


// Model to calibrate
vector_t model(std::vector<vector_t> x_pts, vector_t par) {

    vector_t eval(x_pts.size());
    double x;
    for (int i=0; i< x_pts.size(); i++) {
        x = x_pts[i](0);
        eval(i) =  par(0)*tanh(par(1)*(x-1)/par(0));
    }

    return eval;
}

// Kernel
double err_kernel(const vector_t & x, const vector_t &y, const vector_t & hpar) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return squared_kernel(x,y,sigma_k,l) + white_noise_kernel(x,y,sigma_e);
}

// kernel gradient, i is the component required
double err_kernel_gradient(const vector_t & x, const vector_t &y, const vector_t & hpar, const int &i) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return white_noise_kernel_grad(x,y,sigma_e,i) + squared_kernel_grad(x,y,sigma_k,l,i-1);
}

// kernel hessian, i and j are the row and colum respectively
double err_kernel_hessian(const vector_t & x, const vector_t &y, const vector_t & hpar, const int &i, const int &j) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return white_noise_kernel_hess(x,y,sigma_e,i,j)+squared_kernel_hess(x,y,sigma_k,l,i-1,j-1);
}

// hyperparameter prior
double logprior_hpar(const vector_t & hpar) {
        return log_inv_gamma_pdf(exp(hpar(0)),3,0.04)+log_inv_gamma_pdf(exp(hpar(1)),3,0.04)+log_inv_gamma_pdf(exp(hpar(2)),5,1.2);
}

// prior hessian
double logprior_hpar_hessian(const vector_t &hpar, const int &i, const int &j) {
        if (i==0 && j==0)
            return dd_log_inv_gamma_pdf(exp(hpar(0)),3,0.04);
        else if (i==1 && j==1)
            return dd_log_inv_gamma_pdf(exp(hpar(1)),3,0.04);
        else if (i==2 && j==2)
            return dd_log_inv_gamma_pdf(exp(hpar(2)),5,1.2);
        else
            return 0.;
}

int main() {

    // Initialize the rng
    std::default_random_engine rng(1409);

    //Read the data vector
    std::ifstream file_obs("data.csv");
    auto obs = read_vector(file_obs);

    std::vector<vector_t> x_obs(obs.size());
    vector_t y_obs(obs.size());

    for (int i=0; i<obs.size(); i++) {
        vector_t x(1);
        x << obs[i](0);
        x_obs[i] = x;

        y_obs(i) = obs[i](1);
    }


    // Two parameters
    int dim_par = 2;

    // set up the mcmc initial values
    vector_t par_0(dim_par);
    par_0 << 0.39, 0.35;

    matrix_t cov_prop(dim_par, dim_par);
    cov_prop << 0.005, 0, 0, 0.005;

    // uniform prior on the parameters
    auto logprior_par = [](const vector_t & par) {
            return 0;
    };

    // hyperparameters, s_err, s_ker, l
    int dim_hpar = 3;

    vector_t lb_hpar(dim_hpar);
    vector_t ub_hpar(dim_hpar);

    lb_hpar << 1e-7, 1e-7, 1e-7;
    ub_hpar << 100, 100, 100;

    lb_hpar.array() = lb_hpar.array().log();
    ub_hpar.array() = ub_hpar.array().log();

    // propose a sensible guess
    vector_t hpar_guess(dim_hpar);
    hpar_guess << 0.0043, 0.0010, 1;
    hpar_guess.array() = hpar_guess.array().log();

    // model error guassian process
    auto err_mean = [](std::vector<vector_t> const &x_pts, vector_t const &hpar) {
        vector_t b = vector_t::Zero(x_pts.size());
        return b;
    };

    // Define prediction points and save them
    int num_pred = 100;
    std::vector<vector_t> x_pred(num_pred);
    double x_min = 1;
    double x_max = 5;
    for (int i = 0; i < num_pred; i++) {
        vector_t tmp(1);
        tmp << x_min + (x_max-x_min) * double(i) / double(num_pred-1);
        x_pred[i] = tmp;
    }

    std::ofstream file_x_pred("x_pred.csv");
    write_vector(x_pred,file_x_pred);


    //create density
    density main_density;

    // set up the main properties
    main_density.set_model(model);
    main_density.set_err_mean(err_mean);
    main_density.set_err_kernel(err_kernel);
    main_density.set_log_prior_par(logprior_par);
    main_density.set_obs(x_obs,y_obs);
    main_density.set_hpar_bounds(lb_hpar,ub_hpar);
    main_density.set_log_prior_hpar(logprior_hpar);

    density_opt d_opt(main_density);
    d_opt.set_err_kernel_gradient(err_kernel_gradient);
    d_opt.set_err_kernel_hessian(err_kernel_hessian);
    d_opt.set_logprior_hpar_hessian(logprior_hpar_hessian);

    // create the getter function for the hyperparameters. It is an optimization
    get_hpar_t get_hpar = [&d_opt](const vector_t &par, const vector_t &hpar) {
        return d_opt.hpar_opt(par,hpar,1e-4);
    };

    // create the score function
    score_t score = [&d_opt](const vector_t & par, const vector_t & hpar) {
        auto res = d_opt.residuals(par);
        auto cov = d_opt.covariance(hpar);
        Eigen::LDLT<matrix_t> cov_inv(cov);

        return d_opt.loglikelihood(res,cov_inv) + d_opt.logprior_par(par) + d_opt.logprior_hpar(hpar) + d_opt.log_cmp_correction(hpar,cov_inv,res);
    };

    in_bounds_t in_bounds = [&d_opt](const vector_t & par) {
        return true;
    };

    // setup mcmc chain
    mcmc_chain chain(cov_prop,par_0,hpar_guess,score,get_hpar,in_bounds,rng);

    // burn initial samples
    int burn_in = 10000;
    for(int i=0; i<burn_in; i++) {
        chain.step();
        chain.update();
    }
    chain.adapt_cov();
    chain.info();
    chain.reset();

    // sample burned chain
    int steps = 10000;
    std::vector<vector_t> pars(steps);
    std::vector<vector_t> hpars(steps);
    int skip = 0;

    // do steps
    for (int i=0; i<steps; i++) {
        chain.step();
        chain.update();
        
        //save
        pars[i] = chain.get_par();
        hpars[i] = chain.get_hpar();

        //burn some steps
        for (int j=0; j<skip; j++) {
            chain.step();
            chain.update();
        }
    }
    chain.info();


    //set the samples
    main_density.set_par_samples(pars);
    main_density.set_hpar_samples(hpars);

    //write samples
    std::ofstream file_par_samples("par_samples.csv");
    write_vector(pars,file_par_samples);

    std::ofstream file_hpar_samples("hpar_samples.csv");
    write_vector(hpars,file_hpar_samples);

    //write predictions
    std::ofstream file_cal_pred("pred.csv");
    auto pred_cal  = main_density.pred_calibrated_model(x_pred,0.95);
    auto pred_corr = main_density.pred_corrected_model(x_pred);
    
    for(int i=0; i<x_pred.size(); i++) {
        file_cal_pred << x_pred[i](0) << " " << pred_cal[0](i) << " " << pred_cal[1](i) << " " << pred_cal[2](i) << " " << pred_corr[0](i) << " " << 2*sqrt(pred_corr[1](i)) <<'\n';
    }

    // Diagnostics
    auto sc_diagnosis = single_chain_diagnosis(pars);
    std::cout << "Correlation length: \n" << sc_diagnosis.first << ".\nEffective sample size: " << sc_diagnosis.second << std::endl;
}