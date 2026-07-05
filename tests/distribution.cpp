/**
 * @file distribution.cpp
 * @brief Implementation of various probability distributions.
 * @author Omar Kahol
 */

#include <cmp_defines.h>
#include <distribution.h>
#include <scaler.h>
#include <statistics.h>
#include <wasserstein.h>
#include <omp.h>

// Matplotlib for plotting
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

int main() {

    omp_set_num_threads(4); // Set the number of threads for OpenMP

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
    plt::save("/Users/omarkahol/opt/CMP++/Technical_Doc/images/distribution_mvn.pdf");
    plt::close();

    // Compute the sliced-Wasserstein distance between two sets of samples
    Eigen::MatrixXd samples_1 = samples.topRows(5000);
    Eigen::MatrixXd samples_2 = samples.bottomRows(5000);
    double p = 2.0; // Order of the Wasserstein distance
    int slices = 100; // Number of slices for the sliced-Wasserstein distance
    double sw_distance = cmp::slicedWassersteinDistance(samples_1, samples_2, p, slices);
    std::cout << "Sliced-Wasserstein distance (p=" << p << ", slices=" << slices << "): " << sw_distance << std::endl;

    return 0;
}