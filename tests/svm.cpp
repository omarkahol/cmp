/**
 * @file svm_multiclass_test.cpp
 * @brief Multi-class SVM hyperparameter optimization and decision region visualization.
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
#include <vector>
#include <iomanip>
#include <algorithm>

/**
 * @brief Generates a multi-class noisy spiral dataset in 2D.
 */
std::pair<Eigen::MatrixXd, Eigen::VectorXs> make_spirals(int samples_per_class, int n_classes, double noise, unsigned int seed = 42) {
    const int total_samples = samples_per_class * n_classes;

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(total_samples, 2);
    Eigen::VectorXs y = Eigen::VectorXs::Zero(total_samples);

    std::mt19937 gen(seed);
    std::normal_distribution<double> noise_dist(0.0, noise);

    int row_idx = 0;
    for(int c = 0; c < n_classes; ++c) {
        for(int i = 0; i < samples_per_class; ++i) {
            // Radial distance along the spiral arm
            double r = static_cast<double>(i) / samples_per_class;
            // Angle with an angular offset per class plus a spiral twist
            double theta = c * (2.0 * M_PI / n_classes) + r * 5.0 + noise_dist(gen);

            // Scale out slightly to make regions clearer
            X(row_idx, 0) = 1.5 * r * std::cos(theta);
            X(row_idx, 1) = 1.5 * r * std::sin(theta);
            y(row_idx) = c;

            row_idx++;
        }
    }
    return {X, y};
}

int main() {
    // 1. Generate the dataset with 5 classes
    const int n_classes = 10;
    const int samples_per_class = 50;
    auto [samples, labels] = make_spirals(samples_per_class, n_classes, 0.1, 1234);

    // 2. Set up the Multiclass SVM
    cmp::classifier::SVM svm;
    auto kernel_svm = cmp::covariance::SquaredExponential::make(0, -1);

    Eigen::VectorXd hpar(1);
    hpar(0) = 0.5; // Initial lengthscale

    svm.set(kernel_svm, hpar, 10.0, 1e-3);
    svm.condition(samples, labels);

    // 3. Perform Hyperparameter Optimization via NLOPT
    std::cout << "--- Initializing Hyperparameter Tuning (" << n_classes << " Classes) ---" << std::endl;
    Eigen::VectorXd lb(2), ub(2);
    // Lower bound for C changed from 0.1 to 0.001
    lb << 0.05, 1.0;
    ub << 5.0,  500.0;
    svm.fit(samples, labels, lb, ub, nlopt::LN_SBPLX, 1e-4);
    svm.condition(samples, labels);

    // Compute the percentage of correctly classified samples after optimization
    int correct_predictions = 0;
    for(int i = 0; i < samples.rows(); ++i) {
        size_t pred_class = svm.predict(samples.row(i));
        if(pred_class == static_cast<size_t>(labels(i))) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / samples.rows() * 100.;
    std::cout << "Training Accuracy after Hyperparameter Optimization: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;

    auto [opt_hpar, opt_C] = svm.getHyperparameters();
    std::cout << "\n--- Optimization Complete ---" << std::endl;
    std::cout << "Optimized RBF Lengthscale: " << opt_hpar(0) << std::endl;
    std::cout << "Optimized Regularization C: " << opt_C << std::endl;

    // 4. Compute the Decision Regions over a Grid
    std::cout << "\nComputing decision boundaries for plotting... this may take a moment." << std::endl;

    double grid_min = -1.8;
    double grid_max =  1.8;
    double step = 0.04; // Lower this value for higher resolution (but slower rendering)

    // Store grid points grouped by their predicted class
    std::vector<std::vector<double>> grid_x(n_classes);
    std::vector<std::vector<double>> grid_y(n_classes);

    for(double x = grid_min; x <= grid_max; x += step) {
        for(double y = grid_min; y <= grid_max; y += step) {
            Eigen::Vector2d pt(x, y);
            std::vector<double> probs = svm.predictProbabilities(pt);

            // Find the class with the maximum probability
            auto max_it = std::max_element(probs.begin(), probs.end());
            int pred_class = std::distance(probs.begin(), max_it);

            grid_x[pred_class].push_back(x);
            grid_y[pred_class].push_back(y);
        }
    }

    // 5. Plotting
    plt::figure_size(800, 800);

    // Define a list of colors to use for the regions and points
    std::vector<std::string> colors = {"tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"};

    // Plot background decision regions
    // We use a very small marker size (2.0) instead of alpha transparency to avoid Python TypeErrors
    for(int c = 0; c < n_classes; ++c) {
        if(!grid_x[c].empty()) {
            plt::scatter(grid_x[c], grid_y[c], 10.0, {
                {"color", colors[c % colors.size()]},
                {"marker", "s"},
                {"edgecolors", "none"}
            });
        }
    }

    // Helper lambda to safely extract Eigen column segments into std::vector<double>
    auto extract_vec = [&](int col, int start, int length) {
        std::vector<double> v(length);
        for(int i = 0; i < length; ++i) {
            v[i] = samples(start + i, col);
        }
        return v;
    };

    // Plot foreground actual data points
    // We use a larger size (30.0) and a black edge to make them pop against the background
    for(int c = 0; c < n_classes; ++c) {
        int start_idx = c * samples_per_class;
        plt::scatter(extract_vec(0, start_idx, samples_per_class),
        extract_vec(1, start_idx, samples_per_class), 30.0, {
            {"color", colors[c % colors.size()]},
            {"edgecolors", "black"},
            {"label", "Class " + std::to_string(c)}
        });
    }

    plt::title("SVM Multi-class Decision Regions (" + std::to_string(n_classes) + " Classes)");
    plt::xlabel("X1");
    plt::ylabel("Y1");
    plt::legend();
    plt::xlim(grid_min, grid_max);
    plt::ylim(grid_min, grid_max);
    plt::save("/Users/omarkahol/opt/CMP++/Technical_Doc/images/svm_multiclass.pdf");
    plt::close();

    return 0;
}