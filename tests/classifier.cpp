/**
 * @file classifier.cpp
 * @brief Implementation of various classifiers.
 * @author Omar Kahol
 */

#include <cmp_defines.h>
#include <cluster.h>
#include <classifier.h>
#include <distribution.h>

// Matplotlib for plotting
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;


#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>

// Function to generate "make_moons" dataset
std::pair<Eigen::MatrixXd, Eigen::VectorXs> make_moons(int n_samples, double noise, unsigned int seed = 42) {

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n_samples, 2);
    Eigen::VectorXs y = Eigen::VectorXs::Zero(n_samples);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::normal_distribution<double> noise_dist(0.0, noise);

    int half = n_samples / 2;

    // First moon
    for(int i = 0; i < half; ++i) {
        double t = dist(gen);
        X(i, 0) = std::cos(t) + noise_dist(gen);
        X(i, 1) = std::sin(t) + noise_dist(gen);
        y(i) = 0;
    }

    // Second moon
    for(int i = 0; i < n_samples - half; ++i) {
        double t = dist(gen);
        X(half + i, 0) = 1.0 - std::cos(t) + noise_dist(gen);
        X(half + i, 1) = -std::sin(t) - 0.5 + noise_dist(gen);
        y(half + i) = 1;
    }
    return {X, y};
}

int main() {

    // Initialize the RNG
    std::default_random_engine rng(42);

    // Create two multivariate normal distributions in 2D
    int n_samples_per_class = 100;
    auto [samples, labels] = make_moons(2 * n_samples_per_class, 0.1, 42);

    // Make a uniform grid in x to evaluate the classifier
    int x_db_size = 500;
    auto x_pts = Eigen::VectorXd::LinSpaced(x_db_size, -1, 2);

    // Plot the samples
    plt::figure();
    plt::scatter(samples.col(0).head(n_samples_per_class), samples.col(1).head(n_samples_per_class), 10.0, {{"label", "Class 0"}});
    plt::scatter(samples.col(0).tail(n_samples_per_class), samples.col(1).tail(n_samples_per_class), 10.0, {{"label", "Class 1"}});
    plt::title("Samples from Two Classes (2D Moons)");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::legend();
    plt::axis("equal");
    plt::show();

    /**
     * Function to find decision boundary points by evaluating the classifier.
     * Given a classifier and a point x, it find the coordinate y such that
     * P(class 1 | (x, y)) = prob
     */
    auto find_decision_boundary = [&](cmp::classifier::Classifier * clf, double x, double prob) {

        // Use binary search to find the correct y coordinate
        double low = -1.0;
        double high = 1.0;
        double y = 0.0;

        while(high - low > 1e-6) {
            y = (low + high) / 2.0;
            Eigen::Vector2d pt(x, y);
            std::vector<double> probs = clf->predictProbabilities(pt);
            if(probs[0] < prob) {
                low = y;
            } else {
                high = y;
            }
        }
        return y;
    };

    // Optimization, make a KFold object
    // 2 folds, 2*n_samples_per_class observations, shuffle
    cmp::statistics::KFold kf(2, 2 * n_samples_per_class, true);

    // Make a classifier --- KDE ---
    cmp::classifier::KDE kde;
    auto kernel = cmp::kernel::Gaussian::make();


    // Estimate the bandwidth using Silverman's rule of thumb
    Eigen::Matrix2d sample_band = cmp::density::bandwidthSelectionRule(cmp::density::KDE::BandWidthSelectionMethod::SILVERMAN, samples);
    auto bandwidth = cmp::kernel::FullBandwidth::make(sample_band);

    // Set the KDE classifier
    kde.set(kernel, bandwidth);
    kde.condition(samples, labels);
    std::cout << "Initial bandwidth:\n" << bandwidth->getParams().transpose() << std::endl;
    kde.fit(0.1, samples, labels, -1, 1.0, nlopt::LN_SBPLX, 1e-3);
    std::cout << "Optimized bandwidth:\n" << bandwidth->getParams().transpose() << std::endl;


    // Make a classifier --- SVM ---
    cmp::classifier::SVM svm;
    auto kernel_svm = cmp::covariance::SquaredExponential::make(0, -1);
    Eigen::VectorXd hpar(1);
    hpar(0) = 1.0;
    svm.set(kernel_svm, hpar, 100, 1e-3);
    svm.condition(samples, labels);

    // Make a vector of classifier  and their names
    std::vector<cmp::classifier::Classifier *> classifiers = {&kde, &svm};
    std::vector<std::string> classifier_names = {"KDE", "SVM"};

    for(size_t clf_idx = 0; clf_idx < classifiers.size(); ++clf_idx) {
        cmp::classifier::Classifier * clf = classifiers[clf_idx];
        std::string clf_name = classifier_names[clf_idx];

        // For each x point in the x_pts, find the y point where the probability is [0.05, 0.5, 0.95]
        Eigen::MatrixXd pts(x_db_size, 4);
        for(int i = 0; i < x_db_size; ++i) {
            double x = x_pts(i);
            double y_low = find_decision_boundary(clf, x, 0.25);
            double y_mid = find_decision_boundary(clf, x, 0.5);
            double y_high = find_decision_boundary(clf, x, 0.75);
            pts.row(i) = Eigen::Vector4d(x, y_low, y_mid, y_high);
        }

        // Plot the decision boundary
        plt::figure();
        plt::scatter(samples.col(0).head(n_samples_per_class), samples.col(1).head(n_samples_per_class), 10.0, {{"label", "Class 0"}});
        plt::scatter(samples.col(0).tail(n_samples_per_class), samples.col(1).tail(n_samples_per_class), 10.0, {{"label", "Class 1"}});

        // Plot the lines (dashed for 0.25 and 0.75, solid for 0.5) color red
        plt::plot(pts.col(0), pts.col(1), {{"linestyle", "dashed"}, {"color", "red"}});
        plt::plot(pts.col(0), pts.col(2), {{"linestyle", "solid"}, {"color", "red"}});
        plt::plot(pts.col(0), pts.col(3), {{"linestyle", "dashed"}, {"color", "red"}});
        plt::xlabel("X");
        plt::ylabel("Y");
        plt::axis("equal");

        plt::title(clf_name + " Classifier Decision Boundary");
        plt::show();
    }


    // Now I want to make a stupid 1D example
    Eigen::VectorXd x1d = Eigen::VectorXd::LinSpaced(20, 0, 1);
    Eigen::VectorXs y(x1d.size());
    for(int i = 0; i < x1d.size(); ++i) {
        if(x1d(i) < 0.5) {
            y(i) = 0;
        } else {
            y(i) = 1;
        }
    }

    return 0;

}
