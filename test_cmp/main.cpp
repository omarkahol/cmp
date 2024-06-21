#define NDEBUG 

#include <cmp_defines.h>
#include <mcmc.h>
#include <io.h>
#include <grid.h>
#include <density.h>
#include <pdf.h>
#include <kernel.h>
#include <finite_diff.h>

using namespace cmp;


// Model to calibrate
double model(const vector_t &x, const vector_t &par) {
    double V_hat = x(0)-1;
    return par(0)*tanh(par(1)*V_hat/par(0));
}

// Kernel
double err_kernel(const vector_t & x, const vector_t &y, const vector_t & hpar) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        return white_noise_kernel(x,y,sigma_e) + matern_52_kernel(x,y,sigma_k,l);
}

// kernel gradient, i is the component required
double err_kernel_gradient(const vector_t & x, const vector_t &y, const vector_t & hpar, const int &i) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        vector_t hpars_0(2);
        hpars_0 << sigma_k,l;

        auto fun = [&x,&y](const vector_t &par){
            return matern_52_kernel(x,y,par(0),par(1));
        };

        return white_noise_kernel_grad(x,y,sigma_e,i)+fd_gradient(hpars_0,fun,i-1);
}

// kernel hessian, i and j are the row and colum respectively
double err_kernel_hessian(const vector_t & x, const vector_t &y, const vector_t & hpar, const int &i, const int &j) {
        double sigma_e = exp(hpar(0));
        double sigma_k = exp(hpar(1));
        double l = exp(hpar(2));

        vector_t hpars_0(2);
        hpars_0 << sigma_k,l;

        auto fun = [&x,&y](const vector_t &par){
            return matern_52_kernel(x,y,par(0),par(1));
        };

        return white_noise_kernel_hess(x,y,sigma_e,i,j)+fd_hessian(hpars_0,fun,i-1,j-1);
}

// hyperparameter prior
double logprior_hpar(const vector_t & hpar) {
        return log_inv_gamma_pdf(exp(hpar(0)),3,0.04)+log_inv_gamma_pdf(exp(hpar(1)),3,0.04)+log_inv_gamma_pdf(exp(hpar(2)),5,1.2);
}

// prior hessian
double logprior_hpar_hessian(const vector_t &hpar, const int &i, const int &j) {
    return dd_log_inv_gamma_pdf(exp(hpar(0)),3,0.04,i,j) + dd_log_inv_gamma_pdf(exp(hpar(0)),3,0.04,i-1,j-1)
        + dd_log_inv_gamma_pdf(exp(hpar(2)),5,1.2,i-2,j-2);
}

int main() {

    // Initialize the rng
    std::default_random_engine rng(1409);

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

    lb_hpar << -100, -100, -100;
    ub_hpar <<  100,  100,  100;

    // propose a sensible guess
    vector_t hpar_0(dim_hpar);
    hpar_0 << -4.77, -4.62, -1.40;

    // model error guassian process
    auto err_mean = [](vector_t x, vector_t const &hpar) {
        return 0;
    };

    // Define prediction points and save them
    int num_pred = 100;
    std::vector<double> x_pred(num_pred);
    double x_min = 1;
    double x_max = 5;
    for (int i = 0; i < num_pred; i++) {
        x_pred[i] = x_min + (x_max-x_min) * double(i) / double(num_pred-1);
    }


    //Create the model error
    gp model_error;
    model_error.set_mean(err_mean);
    model_error.set_kernel(err_kernel);
    model_error.set_kernel_gradient(err_kernel_gradient);
    model_error.set_kernel_hessian(err_kernel_hessian);

    model_error.set_logprior(logprior_hpar);
    model_error.set_logprior_hessian(logprior_hpar_hessian);
    model_error.set_par_bounds(lb_hpar,ub_hpar);
    model_error.set_obs(v_to_vvxd(x_obs),y_obs);


    //create density
    density main_density;

    // set up the main properties
    main_density.set_model(model);
    main_density.set_model_error(&model_error);
    main_density.set_log_prior_par(logprior_par);
    main_density.set_obs(v_to_vvxd(x_obs),y_obs);

    //calculate the KOH value for the hyper-parameters
    vector_t lb_par(2), ub_par(2); // set the lower and upper bounds
    lb_par << 0.,0.;
    ub_par << 1.,1.;

    std::vector<vector_t> int_points = qmc_halton_grid(lb_par,ub_par,100'000); // generate a QMC grid in the bounds

    vector_t hpar_KOH = main_density.hpar_KOH(hpar_0,int_points,20,1E-5);
    std::cout << "Hyperparameters KOH: \n" << hpar_KOH << std::endl;

    // create the score function
    auto cmp_score = [&main_density,&model_error,&x_obs](const vector_t & par, const vector_t & hpar) {
        
        auto res = main_density.residuals(par);
        auto cov = model_error.covariance(hpar);
        Eigen::LDLT<matrix_t> cov_inv(cov);

        return main_density.loglikelihood(res,cov_inv) + main_density.logprior_par(par) + main_density.model_error()->logprior(hpar);
    };

    // Setup mcmc chain
    mcmc chain(dim_par,rng);
    chain.seed(cov_prop,par_0);
    vector_t par_prop(dim_par);
    vector_t hpar_prop(dim_hpar);
    double score = 0.0;
    bool accept;

    // burn initial samples
    int burn_in = 10000;
    for(int i=0; i<burn_in; i++) {

        // Propose
        par_prop = chain.propose();
        hpar_prop = main_density.hpar_opt(par_prop,hpar_0,1e-4);
        
        // Evaluate
        score = cmp_score(par_prop,hpar_prop);

        // Check if to accept
        accept=chain.accept(par_prop,score);
        if (accept) {
            hpar_0 = hpar_prop;
        }

        // Update mean and covariance
        chain.update();
    }

    // Adapt and
    chain.adapt_cov();
    chain.info();
    chain.reset();

    // sample burned chain
    int steps = 10000;
    std::vector<vector_t> pars(steps);
    std::vector<vector_t> hpars(steps);

    // do steps
    for (int i=0; i<steps; i++) {
        // Propose
        par_prop = chain.propose();
        hpar_prop = main_density.hpar_opt(par_prop,hpar_0,1e-4);
        
        // Evaluate
        score = cmp_score(par_prop,hpar_prop);

        // Check if to accept
        accept = chain.accept(par_prop,score);
        chain.update();
        if (accept) {
            hpar_0 = hpar_prop;
        }
        pars[i] = chain.get_par();
        hpars[i] = hpar_0;
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
    auto pred_cal  = main_density.pred_calibrated_model(v_to_vvxd(x_pred),0.95);
    auto pred_corr = main_density.pred_corrected_model(v_to_vvxd(x_pred));
    
    for(int i=0; i<x_pred.size(); i++) {
        file_cal_pred << x_pred[i] << " " << pred_cal(i,0) << " " << pred_cal(i,1) << " " << pred_cal(i,2) << " " << pred_corr(i,0) << " " << 2*sqrt(pred_corr(i,1)) <<'\n';
    }

    // Diagnostics
    auto sc_diagnosis = single_chain_diagnosis(pars);
    std::cout << "Correlation length: \n" << sc_diagnosis.first << ".\nEffective sample size: " << sc_diagnosis.second << std::endl;
}