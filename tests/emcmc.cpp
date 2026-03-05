/**
 * @file emcmc.cpp
 * @brief Test file for EMCMC sampling methods.
 * @author Omar Kahol
 *
 */

// Standard library includes (with python for plotting)
#include <iostream>
#include <matplotlibcpp.h>

// CMP++ includes
#include <mcmc.h>
#include <distribution.h>

// Eigen includes
#include <Eigen/Dense>

// OmpenMP includes
#include <omp.h>


// Define the log-PDF to sample from
const double gamma = 8;
double score(const Eigen::VectorXd &x) {
    return -gamma * pow(x.dot(x) - 1, 2);
}

namespace plt = matplotlibcpp;

int main() {

    // Init random number generator
    std::default_random_engine rng(42);

    // EMCMC parameters
    size_t n_chains = 10;           // Number of chains
    size_t n_samples = 100'000;     // Number of samples to draw

    std::vector<Eigen::VectorXd> initialSamples(n_chains);
    std::vector<double> initialScores(n_chains);

    // Create a random starting point for each chain and evaluate its score
    cmp::distribution::NormalDistribution dist_n(0, 2);
    for(size_t i = 0; i < n_chains; i++) {
        initialSamples[i] = dist_n.sample(rng) * Eigen::VectorXd::Ones(1);
        initialScores[i] = score(initialSamples[i]);
    }

    // Create the EMCMC chain object
    cmp::mcmc::EvolutionaryMarkovChain emchain(initialSamples, initialScores, 1e-2);

    // Step the chains
    std::vector<double> samples(n_samples);        // Store the samples here
    double gamma;
    for(size_t i = 0; i < n_samples; i++) {

        // one chooses gamma = 2.38/sqrt(2*dim) for Gaussian targets, every 10th iteration we set gamma=1 to allow jumps between modes
        // gamma = 2.38/std::sqrt(2*dim); // Optimal scaling
        if(i % 10 == 0) {
            gamma = 1.0;
        } else {
            gamma = 2.38 / std::sqrt(2);
        }

        emchain.step(score, rng, gamma);
        samples[i] = emchain.getCurrent()[0](0); // Store the sample of the first chain only
    }

    // Print info
    std::cout << "EMCMC sampling completed." << std::endl;

    // Plot the histogram of the samples
    plt::figure_size(800, 600);
    plt::hist(samples, 100, "blue", 0.9);
    plt::title("EMCMC sampling");
    plt::xlabel("x");
    plt::ylabel("Frequency");
    plt::show();

    return 0;
}