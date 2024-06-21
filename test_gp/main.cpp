#include <cmp_defines.h>
#include <gp.h>
#include <kernel.h>
#include <pdf.h>
#include <io.h>

using namespace cmp;

int main() {

    // Initialize the rng
    std::default_random_engine rng(14091998);

    // Number of training Points
    int n_pts = 10;

    // Generate observations y(x)=sin(x) + 1
    std::vector<double> x_obs(n_pts);
    std::vector<double> y_obs(n_pts);

    for (int i=0; i<n_pts; i++) {
        double x = i/(n_pts-1.0);
        x_obs[i] = x;
        y_obs[i] = std::sin(2*M_PI*x)+1;
    }

    // Define GP-kernel
    auto kernel = [](vector_t x, vector_t y, vector_t hpar){
        return squared_kernel(x,y,exp(hpar(1)),exp(hpar(2)))+white_noise_kernel(x,y,1e-5);
    };

    // Define GP-mean
    auto mean = [](const vector_t &x, const vector_t &hpar){
        return hpar(0);
    };

    // Define log-prior 
    auto logprior = [](vector_t hpar){
        return -2*hpar(1)+log_inv_gamma_pdf(exp(hpar(2)),1.5,0.25);
    };

    gp my_gp;
    my_gp.set_mean(mean);
    my_gp.set_kernel(kernel);
    my_gp.set_par_bounds(-2*vector_t::Ones(3),2*vector_t::Ones(3));
    my_gp.set_logprior(logprior);
    my_gp.set_obs(v_to_vvxd(x_obs),y_obs);

    vector_t par_guess(3);
    par_guess << 1.0,-2,-1;

    vector_t par_opt = my_gp.par_opt(par_guess,1E-5);
    std::cout << par_opt << std::endl;
    auto cov = my_gp.covariance(par_opt);
    auto res = my_gp.residual(par_opt);
    auto ldlt = Eigen::LDLT<matrix_t>(cov);

    // Number of prediction points
    int n_pts_pred = 1000;

    // Generate observations y(x)=sin(x) + 1
    std::vector<vector_t> x_pred(n_pts_pred);
    double x_min = -1;
    double x_max = 2;

    for (int i=0; i<n_pts_pred; i++) {
        vector_t xx(1);
        double x = x_min + (x_max-x_min)*i/(n_pts_pred-1.0);
        xx<<x;
        x_pred[i] = xx;
    }

    std::ofstream pred_file("pred.csv");
    matrix_t y_pred = my_gp.predict(x_pred,par_opt,ldlt,res);
    for (int i=0; i<n_pts_pred; i++) {
        y_pred(i,0) = my_gp.prediction_mean(x_pred[i],par_opt,ldlt.solve(res));
    }
    write_data(x_pred,y_pred,pred_file);


    return 0;



    
}