/**
 * @file bayesian_optimization.cpp
 * @brief Test file for Bayesian optimization methods.
 * @author Omar Kahol
 *
 */

// Standard library includes (with python for plotting)
#include <iostream>
#include <matplotlibcpp.h>

// CMP++ includes
#include <gp.h>
#include <optimization.h>
#include <distribution.h>
#include <statistics.h>
#include <grid.h>

// Eigen includes
#include <Eigen/Dense>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

// Target function to optimize (1D)
double target_function(const Eigen::VectorXd &x) {
    return std::sin(3 * x[0]) + 0.5 * std::cos(5 * x[0]);
}


int main() {
    // Init random number generator
    std::default_random_engine rng(42);

    // Create the Gaussian Process object
    auto kernel = cmp::covariance::Constant::make(1) * cmp::covariance::SquaredExponential::make(1, -1);
    auto mean = cmp::mean::Constant::make(2);
    auto prior = cmp::prior::Uniform::make();

    cmp::gp::GaussianProcess gp;
    gp.set(kernel, mean, Eigen::VectorXd::Ones(3), 1e-5);

    // Create hyperparameter bounds
    Eigen::VectorXd lower_bounds(3);
    lower_bounds << 0.1, 0.1, -10;
    Eigen::VectorXd upper_bounds(3);
    upper_bounds << 10.0, 10.0, 10.0;

    // Create the first set of observations (x, y)
    double lb = -2.0;
    double ub = 2.0;
    size_t n_initial = 5;
    Eigen::MatrixXd X(5, 1);
    Eigen::VectorXd y(5);
    for(size_t i = 0; i < n_initial; ++i) {
        double x = lb + (ub - lb) * i / (n_initial - 1);
        X(i, 0) = x;
        y(i) = target_function(X.row(i));
    }

    // Create the functional to optimize
    auto objective_function = [&](Eigen::Ref<const Eigen::VectorXd> x) {

        // Predict the mean and variance at the new point
        auto [mean, variance] = gp.predict(x);

        // Use the Upper Confidence Bound (UCB) acquisition function
        double kappa = 2.0; // Exploration-exploitation trade-off parameter
        return mean + kappa * sqrt(variance);
    };

    gp.fit(X, y, lower_bounds, upper_bounds, cmp::gp::MLE, nlopt::algorithm::LN_SBPLX, 1e-6);

    // --- Configuration ---
    const double convergence_threshold = 1e-4;
    const int max_iterations = 50;
    double last_best_y = y.maxCoeff();
    bool converged = false;

    for(int i = 0; i < max_iterations && !converged; ++i) {
        // 1. Optimize Acquisition Function
        Eigen::VectorXd x_next(1);
        // Use multi-start for better global optimization
        x_next << lb + (ub - lb) * ((double) rand() / RAND_MAX); // Random starting point

        cmp::ObjectiveFunctor functor(objective_function);
        cmp::nlopt_max(functor, x_next, lb * cmp::ScalarOne, ub * cmp::ScalarOne, nlopt::algorithm::GN_DIRECT_L_RAND, 1e-6);

        // 2. Evaluate
        double y_next = target_function(x_next);

        // 3. Convergence Check
        // Stop if the improvement is negligible
        if(std::abs(y_next - last_best_y) < convergence_threshold) {
            std::cout << "Converged at iteration " << i + 1 << std::endl;
            converged = true;
        }
        last_best_y = std::max(last_best_y, y_next);

        // 4. Update Data and GP
        X.conservativeResize(X.rows() + 1, Eigen::NoChange);
        y.conservativeResize(y.size() + 1);
        X.row(X.rows() - 1) = x_next.transpose();
        y(y.size() - 1) = y_next;

        // Optional: Only re-fit hyperparameters every N steps to save time
        gp.fit(X, y, lower_bounds, upper_bounds, cmp::gp::MLE, nlopt::algorithm::LN_SBPLX, 1e-6);

        std::cout << "Iteration " << i + 1 << ": x=" << x_next[0] << ", y=" << y_next << std::endl;
    }

    size_t n_test = 1000;
    auto x_test = cmp::grid::LinspacedGrid(lb * cmp::ScalarOne, ub * cmp::ScalarOne).construct(n_test);

    // Now let's plot some prior samples
    auto [mu_prior, cov_prior] = gp.predictMultiple(x_test);
    auto prior_distribution = cmp::distribution::MultivariateNormalDistribution(mu_prior, cov_prior);

    size_t n_samples = 10;
    plt::figure_size(800, 600);
    for(size_t i = 0; i < n_samples; i++) {
        Eigen::VectorXd sample = prior_distribution.sample(rng);
        plt::plot(x_test, sample, "g-");
    }

    // Scatter the training data points
    plt::scatter(X, y, 50, {{"color", "red"}, {"label", "Observations"}});
    plt::legend();

    plt::title("Samples from the GP prior");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::save("/Users/omarkahol/opt/CMP++/Technical_Doc/images/bayesian_optimization.pdf");
    plt::close();


}