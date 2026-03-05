#include <kde.h>
#include <distribution.h>
#include <matplotlibcpp.h>
#include <kernel++.h>
int main() {

    // Initialize the RNG
    std::default_random_engine rng(42);

    // Create two multivariate normal distributions in 2D
    Eigen::Vector2d mu1;
    mu1 << -0.5, 0.5;
    Eigen::Vector2d mu2;
    mu2 << 0.5, -0.5;
    Eigen::Matrix2d cov1;
    cov1 << 0.5, 0.1, 0.1, 0.5;
    Eigen::Matrix2d cov2;
    cov2 << 0.5, -0.1, -0.1, 0.5;

    // Make a multivariate mixture distribution
    auto dist = cmp::distribution::MultivariateMixtureDistribution({
        std::make_shared<cmp::distribution::MultivariateNormalDistribution>(mu1, cov1),
        std::make_shared<cmp::distribution::MultivariateNormalDistribution>(mu2, cov2)},
    {0.3, 0.7});


    // Sample this distribution
    size_t nSamples = 1000;
    Eigen::MatrixXd samples(nSamples, 2);
    for(size_t i = 0; i < nSamples; i++) {
        samples.row(i) = dist.sample(rng);
    }

    // Plot the samples
    matplotlibcpp::figure_size(800, 600);
    matplotlibcpp::scatter(samples.col(0), samples.col(1), 10.0);
    matplotlibcpp::title("Samples from a Gaussian Mixture Model");
    matplotlibcpp::xlabel("X1");
    matplotlibcpp::ylabel("X2");
    matplotlibcpp::xlim(-3, 3);
    matplotlibcpp::ylim(-3, 3);
    matplotlibcpp::grid(true);
    matplotlibcpp::show();

    std::cout << "True density at (0,0): " << std::exp(dist.logPDF(Eigen::Vector2d(0, 0))) << std::endl;

    // First we estimate the bandwidth using Silverman's rule of thumb
    Eigen::Matrix2d sample_band = cmp::density::bandwidthSelectionRule(cmp::density::KDE::BandWidthSelectionMethod::SILVERMAN, samples);

    std::cout << "Estimated bandwidth (Silverman's rule):\n" << sample_band << std::endl;

    // Create the bandwidth and kernel objects
    auto bandwidth = std::make_shared<cmp::kernel::FullBandwidth>(sample_band);
    auto kernel = cmp::kernel::Gaussian::make();

    // Set up the KDE
    cmp::density::KDE kde;
    kde.set(bandwidth, kernel);
    kde.condition(samples, false);

    std::cout << "KDE is set up." << std::endl;

    std::cout << "KDE density at (0,0): " << kde.eval(Eigen::Vector2d(0, 0)) << std::endl;

    // Now we improve the bandwidth using likelihood cross-validation
    cmp::statistics::KFold kf(20, nSamples, true, 42);
    cmp::density::bandwidthOptimizationCrossValidation(kf, samples, kernel, bandwidth, -1, 1, nlopt::LD_TNEWTON, 1e-3);

    std::cout << "Optimized bandwidth:\n" << bandwidth->matrix() << std::endl;
    std::cout << "KDE density at (0,0) after optimization: " << kde.eval(Eigen::Vector2d(0, 0)) << std::endl;

    return 0;
}
