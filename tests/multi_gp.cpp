/**
 * @file multi_gp.cpp
 * @brief Example of using the multi_gp class to fit multiple outputs with Gaussian Processes
 * @author Omar Kahol
 */

// Standard includes
#include <iostream>
#include <matplotlibcpp.h>

// Eigen includes
#include <Eigen/Dense>

// CMP includes
#include <multi_gp.h>
#include <grid.h>

namespace plt = matplotlibcpp;

using namespace cmp;

int main() {

    // Initialize the rng for reproducibility and set other parameters
    std::default_random_engine rng(42);
    std::normal_distribution<double> dist_n(0, 1);
    double noise_std = 0.001;

    // Number of observations
    int n_obs = 50;

    // Generate 3D observations from a 1D input
    auto x_obs = cmp::grid::LinspacedGrid(cmp::ScalarZero, cmp::ScalarOne).construct(n_obs);
    Eigen::MatrixXd y_obs(n_obs, 3);
    for(int i = 0; i < n_obs; i++) {
        double x = x_obs(i, 0);
        y_obs(i, 0) = std::sin(2 * M_PI * x) + noise_std * dist_n(rng);
        y_obs(i, 1) = std::cos(2 * M_PI * x) + noise_std * dist_n(rng);

        // Make the third output dependent on the first two to show the power of PCA
        y_obs(i, 2) = -1 * y_obs(i, 0) + 2 * y_obs(i, 1) + 3 + noise_std * dist_n(rng);
    }

    // Make a test grid for plotting
    int n_test = 100;
    auto x_test = cmp::grid::LinspacedGrid(cmp::ScalarZero, cmp::ScalarOne).construct(n_test);

    // Define the kernel, mean and log-prior functions
    auto kernel = cmp::covariance::Constant::make(0) * cmp::covariance::SquaredExponential::make(1, -1);
    auto mean = cmp::mean::Constant::make(2);
    auto logprior = cmp::prior::FromDistribution::make(cmp::distribution::PowerLawDistribution::make(2), 0);

    // Scale the x observations
    cmp::scaler::StandardScaler x_scaler;
    auto x_obs_scaled = x_scaler.fit_transform(x_obs);

    // Scale the y observations using PCA to reduce dimensionality
    cmp::scaler::PCA y_scaler(2);
    auto y_obs_scaled = y_scaler.fit_transform(y_obs);


    /**
     * Now we create the multi GP object.
     * Each component is a GP with the same kernel, mean and log-prior functions.
     */
    cmp::gp::MultiOutputGaussianProcess gp(y_obs_scaled.cols(), kernel, mean, 0.1 * Eigen::VectorXd::Ones(3), 1e-5);


    // Now we fit the hyperparameters of each GP using MAP estimation
    Eigen::VectorXd lb(3);
    lb << 1e-2, 1e-1, -5;
    Eigen::VectorXd ub(3);
    ub << 5, 5, 5;
    gp.fit(x_obs_scaled, y_obs_scaled, lb, ub, cmp::gp::MLE, nlopt::LD_TNEWTON, 1e-5, true, logprior);

    // Sample the prior of the first GP and plot
    size_t n_samples = 10;
    plt::figure_size(1200, 400);
    plt::title("Prior samples of GP 1");

    // Sample the GP prior
    auto [mu_prior, cov_prior] = gp[0].predictMultiple(x_test, cmp::gp::PRIOR);
    auto prior_distribution = cmp::distribution::MultivariateNormalDistribution(mu_prior, cov_prior);

    for(size_t j = 0; j < n_samples; j++) {
        auto sample = prior_distribution.sample(rng);
        plt::plot(x_test.col(0), sample, "g-");
    }
    plt::xlabel("x");
    plt::ylabel("y");
    plt::show();

    // Print each GPs hyperparameters
    for(size_t i = 0; i < gp.size(); i++) {
        std::cout << "GP " << i << " hyperparameters: " << gp[i].getParameters().transpose() << std::endl;
    }
    std::cout << std::endl;

    Eigen::MatrixXd y_pred(n_test, 3);
    for(int i = 0; i < n_test; i++) {

        Eigen::VectorXd x_pred = x_scaler.transform(x_test.row(i));
        auto [y_pred_scaled, y_var] = gp.predict(x_pred);

        // Inverse transform the prediction
        y_pred.row(i) = y_scaler.inverseTransform(y_pred_scaled);
    }

    // Plot the results
    for(size_t i = 0; i < 3; i++) {
        plt::figure_size(1200, 400);
        plt::title("Output " + std::to_string(i + 1));
        plt::scatter(x_obs.col(0), y_obs.col(i), 10.0);
        plt::plot(x_test.col(0), y_pred.col(i));
        plt::xlabel("x");
        plt::ylabel("y");
        plt::show();
    }


    return 0;




}