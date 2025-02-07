#include <cmp_defines.h>
#include <gp.h>
#include <kernel.h>
#include <utils.h>
#include <omp.h>
#include <multi_gp.h>

using namespace cmp;

int main() {

    // Initialize the rng
    std::default_random_engine rng(14091998);
    std::normal_distribution<double> dist_n(0,1);

    // Number of training Points
    int n_pts = 5;

    // Generate observations y(x)=sin(x) + 1
    std::vector<double> x_obs(n_pts);
    std::vector<Eigen::VectorXd> y_obs(n_pts,Eigen::VectorXd::Zero(3));

    for (int i=0; i<n_pts; i++) {
        double x = i/(n_pts-1.0);
        x_obs[i] = x;
        y_obs[i](0) = std::sin(2*M_PI*x)+1 + 0.001*dist_n(rng);
        y_obs[i](1) = std::cos(2*M_PI*x) + 0.001*dist_n(rng);
        y_obs[i](2) =  + std::sin(2*M_PI*x) +.001*dist_n(rng);
    }

    // Define GP-kernel
    auto kernel = [](Eigen::VectorXd x, Eigen::VectorXd y, Eigen::VectorXd hpar){
        return squared_kernel(x,y,exp(hpar(1)),exp(hpar(2)))+white_noise_kernel(x,y,1e-3);
    };

    auto kernel_grad = [](Eigen::VectorXd x, Eigen::VectorXd y, Eigen::VectorXd hpar, int i){
        return squared_kernel_grad(x,y,exp(hpar(1)),exp(hpar(2)),i-1);
    };

    // Define GP-mean
    auto mean = [](const Eigen::VectorXd &x, const Eigen::VectorXd &hpar){
        return hpar(0);
    };

    auto mean_grad = [](const Eigen::VectorXd &x, const Eigen::VectorXd &hpar, int i){
        return i==0 ? 1.0 : 0.0;
    };

    // Define log-prior
    cmp::normal_distribution prior(0,1);
    auto logprior = [&prior](Eigen::VectorXd hpar) {
        return -2*hpar(1)+prior.log_pdf(hpar(2))+prior.log_pdf(hpar(0));
    };

    auto logprior_grad = [&prior](Eigen::VectorXd hpar, int i){
        if (i==0) {
            return prior.d_log_pdf(hpar(0));
        } else if (i==1) {
            return -2.0;
        } else {
            return prior.d_log_pdf(hpar(2));
        }
    };

    // Create the multi_gp object
    cmp::multi_gp my_gp(2);

    // Set the observations
    cmp::standard_vector_scaler x_scaler(v_to_vvxd(x_obs));
    cmp::pca_scaler y_scaler(y_obs,2);

    my_gp.set_observations(&x_scaler,&y_scaler);
    my_gp.set_kernel(kernel);
    my_gp.set_mean(mean);
    my_gp.set_log_prior(logprior);
    my_gp.set_mean_grad(mean_grad);
    my_gp.set_kernel_grad(kernel_grad);
    my_gp.set_log_prior_grad(logprior_grad);


    // Optimize the hyperparameters
    Eigen::VectorXd one = Eigen::VectorXd::Ones(3);
    my_gp.fit(one,-2*one,2*one,cmp::MAP,nlopt::LN_SBPLX,1e-15);
    std::cout << my_gp[0].get_params().transpose() << std::endl;
    //std::cout << my_gp[1].get_params().transpose() << std::endl;

    // Predict the mean and the variance
    size_t n_pred = 100;

    std::ofstream file("pred.csv");

    for (size_t i=0; i<n_pred; i++) {
        Eigen::VectorXd x = double(i)/double(n_pred-1)*Eigen::VectorXd::Ones(1);
        Eigen::VectorXd y_pred = my_gp.predictive_mean(x);

        std::cout << y_pred.transpose() << std::endl;
        Eigen::VectorXd y_var = my_gp.predictive_var(x).diagonal();

        file << x(0) << " " << std::sin(2*M_PI*x(0))+1 << " " << std::cos(2*M_PI*x(0)) << " " << y_pred(0) << " " << y_pred(1) << " " << std::sqrt(y_var(0)) << " " << std::sqrt(y_var(1)) << std::endl;
    }
    file.close();






    


    return 0;



    
}