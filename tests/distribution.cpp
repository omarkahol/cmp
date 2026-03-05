/**
 * @file distribution.cpp
 * @brief Implementation of various probability distributions.
 * @author Omar Kahol
 */

#include <cmp_defines.h>
#include <distribution.h>
#include <scaler.h>
#include <statistics.h>

// Matplotlib for plotting
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

int main() {

    // Initialize the RNG
    std::default_random_engine rng(42);

    // Define a 2D mean and covariance matrix
    Eigen::VectorXd mean(2);
    mean << 0.0, 0.0;
    Eigen::MatrixXd cov(2, 2);
    cov << 1.0, 0.8,
        0.8, 1.0;

    // Create the Multivariate Normal Distribution
    cmp::distribution::MultivariateNormalDistribution mvn(mean, cov);

    // Sample from the distribution
    size_t n_samples = 10'000;
    Eigen::MatrixXd samples(n_samples, 2);

    for(int i = 0; i < n_samples; i++) {
        samples.row(i) = mvn.sample(rng).transpose();
    }

    // Compute the self-correlation of the samples
    std::vector<int> lags = {0, 1, 2, 3, 4, 5};
    for(int lag : lags) {
        Eigen::MatrixXd correlationCoeff = cmp::statistics::laggedCorrelation(samples, samples, lag);
        std::cout << "Lagged correlation (lag=" << lag << "): " << std::endl;
        std::cout << correlationCoeff << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }


    // Plot the samples
    plt::figure();
    plt::scatter(samples.col(0), samples.col(1), 1.0);
    plt::title("Samples from 2D Multivariate Normal Distribution");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::axis("equal");
    plt::show();

    return 0;
}