/**
 * @file mcmc.cpp
 * @brief Test file for MCMC sampling methods, including DRAM.
 * @author Omar Kahol
 *
 */

// Standard library includes (with python for plotting)
#include <iostream>
#include <matplotlibcpp.h>

// CMP++ includes
#include <mcmc.h>
#include <distribution.h>
#include <statistics.h>

// Eigen includes
#include <Eigen/Dense>

const double gamma = 16;

// Define the log-PDF to sample from
double score(const Eigen::VectorXd &x) {
    return -gamma * pow(x.squaredNorm() - 1, 2);
}

namespace plt = matplotlibcpp;

int main() {

    // Init random number generator
    std::default_random_engine rng(42);

    // MCMC parameters
    size_t n_burn = 10'000;       // Number of burn-in samples
    size_t n_samples = 100'000;   // Number of samples to draw
    size_t n_subsample = 10;      // Subsampling factor

    // Initialize a proposal distribution, here a multivariate t-distribution with 1 degree of freedom (Cauchy distribution)
    Eigen::VectorXd xInit = Eigen::VectorXd::Zero(1);             // Initial position
    Eigen::MatrixXd covInit = Eigen::MatrixXd::Identity(1, 1);    // Initial covariance matrix
    cmp::distribution::MultivariateStudentDistribution proposal(xInit, covInit, 1);

    // Create the MCMC chain object
    cmp::mcmc::MarkovChain chain(&proposal, rng);

    // Burn-in phase, step with no adaptation
    for(size_t i = 0; i < n_burn; i++) {
        chain.step(score);
    }

    // Print info
    chain.info();

    // Set the proposal covariance to the adapted covariance
    proposal.setLdltDecomposition(chain.getAdaptedCovariance().ldlt());

    // Reset the chain (mean, cov, steps, accepts)
    chain.reset();

    // Sampling phase
    std::vector<double> samples(n_samples);        // Store the samples here
    for(size_t i = 0; i < n_samples; i++) {
        samples[i] = chain.getCurrent()(0);   // Get the current sample

        // Take n_subsample steps between each sample
        for(size_t j = 0; j < n_subsample; j++) {
            chain.step(score, {0.5, 0.25, 0.125});  // DRAM step with gamma=0.5, 0.25, and 0.125
        }
    }

    // Print info
    chain.info();

    // Plot the histogram of the samples
    plt::figure_size(800, 600);
    plt::hist(samples, 100, "blue", 0.9);
    plt::title("MCMC sampling using DRAM");
    plt::xlabel("x");
    plt::ylabel("Frequency");
    plt::grid(true);
    plt::show();

    std::cout << "-----------------------------------" << std::endl;

    // Compute the lagged correlation of the samples
    std::vector<int> lags = {0, 1, 2, 5, 10, 20, 50, 100};
    for(int lag : lags) {
        Eigen::MatrixXd correlationCoeff = cmp::statistics::laggedCorrelation(cmp::asEigen(samples), cmp::asEigen(samples), lag);
        std::cout << "Lagged correlation (lag=" << lag << "): " << std::endl;
        std::cout << correlationCoeff << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    // Compute the effective sample size and self correlation length
    auto [correlationLength, ess] = cmp::statistics::selfCorrelationLength(cmp::asEigen(samples));

    std::cout << "Self correlation length: \n" << correlationLength.transpose() << std::endl;
    std::cout << "Effective sample size: " << ess << std::endl;

    // Now we are interested in computing the self correlation using FFT
    auto corr = cmp::statistics::selfCrossCorrelationFFT(cmp::asEigen(samples));
    size_t nLags = 100; // Number of lags to plot
    std::vector<double> corr_00(nLags);
    for(size_t i = 0; i < nLags; ++i) {
        corr_00[i] = corr[i](0, 0);
    }

    // Plot the self-correlation using FFT
    plt::figure_size(800, 600);
    plt::plot(corr_00, {{"color", "red"}, {"linestyle", "-"}});
    plt::title("Self-correlation using FFT");
    plt::xlabel("Lag");
    plt::ylabel("Correlation");
    plt::grid(true);
    plt::show();

    return 0;
}




