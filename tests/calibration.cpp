/**
 * @file cmp.cpp
 * @brief Example of using the cmp library to perform MCMC sampling with model error
 * @author Omar Kahol
 */

#include <cmp_defines.h>
#include <mcmc.h>
#include <grid.h>
#include <kernel++.h>
#include <finite_diff.h>
#include <distribution.h>
#include <optimization.h>
#include <matplotlibcpp.h>

namespace matplotlibpp = matplotlibcpp;
namespace plt = matplotlibpp;

using namespace cmp;

int main() {

    // Initialize the RNG for reproducibility
    std::default_random_engine rng(42);

    // Generate simple synthetic data from a parabola: y = 0.8 x^2 + 0.5 x + 1 + noise
    size_t n_data = 30;
    Eigen::VectorXd x_data = Eigen::VectorXd::LinSpaced(n_data, -2.0, 2.0);
    Eigen::VectorXd y_data(n_data);
    std::normal_distribution<double> noise_dist(0.0, 0.08);
    for(size_t i = 0; i < n_data; ++i) {
        double x = x_data(i);
        y_data(i) = 0.8 * x * x + 0.5 * x + 1.0 + noise_dist(rng);
    }

    // Calibrated model (intentionally simple): y = model_scale * x^2
    auto model = [&](const double & x, const double & model_scale) {
        return model_scale * x * x;
    };

    auto compute_residuals = [&](const double & model_scale) {
        Eigen::VectorXd residuals(n_data);
        for(size_t i = 0; i < n_data; ++i) {
            double model_output = model(x_data(i), model_scale);
            residuals(i) = y_data(i) - model_output;
        }

        return residuals;
    };

    std::cout << "Initial residuals with model_scale=1.0: " << std::endl;
    std::cout << compute_residuals(1.0).transpose() << std::endl;


    // Define the model error
    cmp::gp::GaussianProcess model_error;

    auto kernel = cmp::covariance::Constant::make(0) * cmp::covariance::SquaredExponential::make(1, -1);
    auto mean = cmp::mean::Zero::make();

    double observation_noise = 1e-3;

    Eigen::VectorXd gp_hyper_guess(2);
    gp_hyper_guess << 0.2, 1.0;

    double model_scale_guess = 1.0;

    model_error.set(kernel, mean, gp_hyper_guess, observation_noise);
    model_error.condition(x_data, compute_residuals(model_scale_guess), true);

    auto compute_cov_matrix = [&](const Eigen::VectorXd & gp_hyper) {
        return model_error.covariance(gp_hyper);
    };

    // Define the log-posterior function
    auto prior_model_scale = cmp::distribution::NormalDistribution(1.0, 0.5);
    auto prior_log_sigma_f = cmp::distribution::NormalDistribution(std::log(0.2), 1.0);
    auto prior_log_length_scale = cmp::distribution::NormalDistribution(std::log(1.0), 1.0);
    auto log_posterior = [&](const Eigen::VectorXd & params) {

        double model_scale = params(0);
        double log_sigma_f = params(1);
        double log_length_scale = params(2);

        Eigen::VectorXd gp_hyper(2);
        gp_hyper << std::exp(log_sigma_f), std::exp(log_length_scale);

        Eigen::VectorXd residuals = compute_residuals(model_scale);
        Eigen::MatrixXd cov = compute_cov_matrix(gp_hyper);
        Eigen::LDLT<Eigen::MatrixXd> cov_ldlt = cov.ldlt();

        return cmp::distribution::MultivariateNormalDistribution::logPDF(residuals, cov_ldlt) +
               prior_model_scale.logPDF(model_scale) +
               prior_log_sigma_f.logPDF(log_sigma_f) +
               prior_log_length_scale.logPDF(log_length_scale);
    };


    // Initial starting point
    Eigen::VectorXd params(3);
    params(0) = model_scale_guess;
    params(1) = std::log(gp_hyper_guess(0));
    params(2) = std::log(gp_hyper_guess(1));


    // Make a proposal distribution
    Eigen::MatrixXd proposal_cov = 0.05 * Eigen::MatrixXd::Identity(3, 3);
    auto proposal = cmp::distribution::MultivariateNormalDistribution(params, proposal_cov.ldlt());

    cmp::mcmc::MarkovChain mcmc(&proposal, rng, log_posterior(params));

    size_t n_burnin = 2000;
    size_t n_samples = 4000;
    size_t n_thin = 5;

    for(size_t i = 0; i < n_burnin; ++i) {
        mcmc.step(log_posterior, {0.1});
    }

    mcmc.info();
    proposal.setLdltDecomposition(cmp::ldltDecomposition(mcmc.getAdaptedCovariance()));

    Eigen::MatrixXd samples(n_samples, 3);
    for(size_t i = 0; i < n_samples; ++i) {

        Eigen::VectorXd sample = mcmc.getCurrent();
        samples.row(i) = sample;

        for(size_t j = 0; j < n_thin; ++j) {
            mcmc.step(log_posterior, {0.1});
        }

    }
    std::cout << "Final estimate of model_scale: " << std::endl;
    std::cout << samples.col(0).mean() << " +/- " << std::
              sqrt((samples.col(0).array() - samples.col(0).mean()).square().sum() / (samples.rows() - 1)) << std::endl;


    // Plot posterior distributions
    std::vector<double> h_samples(n_samples);
    std::vector<double> sigma_f_samples(n_samples);
    std::vector<double> l_samples(n_samples);
    for(size_t i = 0; i < n_samples; ++i) {
        h_samples[i] = samples(i, 0);
        sigma_f_samples[i] = std::exp(samples(i, 1));
        l_samples[i] = std::exp(samples(i, 2));
    }

    plt::figure_size(700, 450);
    plt::hist(h_samples, 40, "blue", 0.85);
    plt::title("Posterior of model_scale");
    plt::xlabel("model_scale");
    plt::ylabel("Frequency");
    plt::grid(true);
    plt::show();

    plt::figure_size(700, 450);
    plt::hist(sigma_f_samples, 40, "green", 0.85);
    plt::title("Posterior of GP sigma_f");
    plt::xlabel("sigma_f");
    plt::grid(true);
    plt::show();

    plt::figure_size(700, 450);
    plt::hist(l_samples, 40, "red", 0.85);
    plt::title("Posterior of GP length_scale");
    plt::xlabel("length_scale");
    plt::grid(true);
    plt::show();

    // Plot data and mean calibrated model
    double model_scale_mean = samples.col(0).mean();
    std::vector<double> x_plot(n_data);
    std::vector<double> y_plot(n_data);
    std::vector<double> y_model(n_data);
    for(size_t i = 0; i < n_data; ++i) {
        x_plot[i] = x_data(i);
        y_plot[i] = y_data(i);
        y_model[i] = model(x_data(i), model_scale_mean);
    }
    plt::figure_size(800, 500);
    plt::scatter(x_plot, y_plot, 20.0, {{"color", "blue"}});
    plt::plot(x_plot, y_model, {{"color", "red"}, {"linewidth", "2.0"}});
    plt::title("Synthetic data and calibrated model");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::grid(true);
    plt::show();

    // Save the samples to a CSV file
    std::ofstream file("mcmc_samples.csv");
    if(file.is_open()) {
        file << "model_scale,sigma_f,length_scale" << std::endl;
        for(size_t i = 0; i < samples.rows(); ++i) {
            file << samples(i, 0) << "," << std::exp(samples(i, 1)) << "," << std::exp(samples(i, 2)) << std::endl;

        }
        file.close();
        std::cout << "Samples saved to mcmc_samples.csv" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }


    return 0;
}