#include <cmp_defines.h>
#include <gp.h>
#include <kernel.h>
#include <utils.h>
#include <omp.h>

using namespace cmp;

int main() {

    // Initialize the rng
    std::default_random_engine rng(14091998);
    std::normal_distribution<double> dist_n(0,1);

    // Number of training Points
    int n_pts = 1000;

    // Generate observations y(x)=sin(x) + 1
    std::vector<double> x_obs(n_pts);
    std::vector<double> y_obs(n_pts);

    for (int i=0; i<n_pts; i++) {
        double x = i/(n_pts-1.0);
        x_obs[i] = x+0.01*dist_n(rng);
        y_obs[i] = std::sin(2*M_PI*x)+1;
    }

    // Define GP-kernel
    auto kernel = [](vector_t x, vector_t y, vector_t hpar){
        return squared_kernel(x,y,exp(hpar(1)),exp(hpar(2)))+white_noise_kernel(x,y,1e-5);
    };

    auto kernel_grad = [](vector_t x, vector_t y, vector_t hpar, int i){
        return squared_kernel_grad(x,y,exp(hpar(1)),exp(hpar(2)),i-1);
    };

    // Define GP-mean
    auto mean = [](const vector_t &x, const vector_t &hpar){
        return hpar(0);
    };

    auto mean_grad = [](const vector_t &x, const vector_t &hpar, int i){
        return i==0 ? 1.0 : 0.0;
    };

    // Define log-prior
    cmp::normal_distribution prior(0,1,rng);
    auto logprior = [&prior](vector_t hpar) {
        return -2*hpar(1)+prior.log_pdf(hpar(2))+prior.log_pdf(hpar(0));
    };

    auto logprior_grad = [&prior](vector_t hpar, int i){
        if (i==0) {
            return prior.d_log_pdf(hpar(0));
        } else if (i==1) {
            return -2.0;
        } else {
            return prior.d_log_pdf(hpar(2));
        }
    };

    gp my_gp;
    my_gp.set_mean(mean);
    my_gp.set_kernel(kernel);
    my_gp.set_par_bounds(-2*vector_t::Ones(3),2*vector_t::Ones(3));
    my_gp.set_logprior(logprior);
    my_gp.set_obs(v_to_vvxd(x_obs),y_obs);

    my_gp.set_mean_gradient(mean_grad);
    my_gp.set_kernel_gradient(kernel_grad);
    my_gp.set_logprior_gradient(logprior_grad);

    vector_t par_guess(3);
    par_guess << 1.0,-2,-1;

    vector_t par_opt = my_gp.par_opt(par_guess,1E-5,nlopt::LD_LBFGS);
    std::cout << par_opt << std::endl;
    auto cov = my_gp.covariance(par_opt);
    auto res = my_gp.residual(par_opt);
    auto ldlt = Eigen::LDLT<matrix_t>(cov);

    // Number of prediction points
    int n_pts_pred = 100;

    // Generate observations y(x)=sin(x) + 1
    std::vector<vector_t> x_pred(n_pts_pred);
    double x_min = -1;
    double x_max = 2;

    for (int i=0; i<n_pts_pred; i++) {
        vector_t xx(1);
        double x = x_min + (x_max-x_min)*i/(n_pts_pred-1.0) + 0.01*dist_n(rng);
        xx<<x;
        x_pred[i] = xx;
    }

    std::ofstream pred_file("pred.csv");
    matrix_t y_pred = my_gp.predict(x_pred,par_opt,ldlt,res);
    for (int i=0; i<n_pts_pred; i++) {
        y_pred(i,0) = my_gp.prediction_mean(x_pred[i],par_opt,ldlt.solve(res));
    }
    write_data(x_pred,y_pred,pred_file);

    std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd c2 = my_gp.compute_variance_reduction(x_pred,ldlt,par_opt);
    std::cout << c2.norm() << std::endl;
    std::chrono::high_resolution_clock::time_point t22 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t22-t11);
    std::cout << "Time to compute fast reduced covariance matrix: " << time_span2.count() << " seconds" << std::endl;



    return 0;



    
}