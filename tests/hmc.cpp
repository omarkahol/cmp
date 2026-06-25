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
#include <hmc.h>
#include <distribution.h>
#include <statistics.h>

// Eigen includes
#include <Eigen/Dense>

const double gamma = 16;

// Define the log-PDF to sample from
double score(const Eigen::VectorXd &x) {
    return -gamma * pow(x.squaredNorm() - 1, 2);
}

Eigen::VectorXd gradient(const Eigen::VectorXd &x) {
    double r2 = x.squaredNorm();
    return -4 * gamma * (r2 - 1) * x;
}

namespace plt = matplotlibcpp;

int main() {

    // Init random number generator
    std::default_random_engine rng(42);

    // MCMC parameters
    size_t n_burn = 10'000;       // Number of burn-in samples
    size_t n_samples = 10'000;   // Number of samples to draw
    size_t n_subsample = 10;      // Subsampling factor


    // Create the MCMC chain object
    cmp::mcmc::HamiltonianMarkovChain chain(Eigen::VectorXd::Ones(2), rng, 0.5);

    // Burn-in phase, step with no adaptation
    for(size_t i = 0; i < n_burn; i++) {
        chain.step(score, gradient, true);
    }

    // Print info
    chain.info();

    // Reset the chain (mean, cov, steps, accepts)
    chain.reset();

    // Sampling phase
    std::vector<Eigen::VectorXd> samples(n_samples);        // Store the samples here
    for(size_t i = 0; i < n_samples; i++) {
        samples[i] = chain.getCurrent();   // Get the current sample

        // Take n_subsample steps between each sample
        for(size_t j = 0; j < n_subsample; j++) {
            chain.step(score, gradient, false);
        }
    }

    // Print info
    chain.info();

    // 1. Extract X and Y coordinates from the Eigen vectors
    std::vector<double> x_vals;
    std::vector<double> y_vals;

    x_vals.reserve(samples.size());
    y_vals.reserve(samples.size());

    for(const auto& sample : samples) {
        x_vals.push_back(sample(0)); // First dimension (X)
        y_vals.push_back(sample(1)); // Second dimension (Y)
    }

    // 2. Initialize the figure FIRST
    plt::figure_size(400, 400);
    plt::xlim(-2.0, 2.0);
    plt::ylim(-2.0, 2.0);
    plt::grid(true);
    plt::scatter(x_vals, y_vals, 1.0);

    // 5. Render
    plt::show();

    std::cout << "-----------------------------------" << std::endl;
    return 0;
}




