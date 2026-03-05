/**
 * @file gp.cpp
 * @brief Test file for Gaussian Process regression.
 * @author Omar Kahol

*/

// Standard includes
#include <iostream>
#include <matplotlibcpp.h>

// Include GP regression library
#include <gp.h>
#include <grid.h>
#include <statistics.h>

namespace plt = matplotlibcpp;

using namespace cmp;

class CovSum : public cmp::covariance::Covariance {
  private:
    const cmp::covariance::Covariance &cov1_;
    const cmp::covariance::Covariance &cov2_;

  public:
    CovSum(const cmp::covariance::Covariance &cov1,
           const cmp::covariance::Covariance &cov2)
        : cov1_(cov1), cov2_(cov2) {}

    double eval(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &par) const override {
        return cov1_.eval(x1, x2, par) + cov2_.eval(x1, x2, par);
    }

    double evalGradient(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &par, const size_t &i) const override {
        return cov1_.evalGradient(x1, x2, par, i) + cov2_.evalGradient(x1, x2, par, i);
    }

    double evalHessian(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &par, const size_t &i, const size_t &j) const override {
        return cov1_.evalHessian(x1, x2, par, i, j) + cov2_.evalHessian(x1, x2, par, i, j);
    }
};

int main() {

    // Seed the rng
    std::default_random_engine rng(42);
    std::normal_distribution<double> dist_n(0, 1);
    auto c1 = cmp::covariance::Constant(0);
    auto c2 = cmp::covariance::SquaredExponential(1, -1);
    CovSum cov_sum(c1, c2);
    std::cout << cov_sum.eval(Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), Eigen::VectorXd::Ones(3)) << std::endl;

    /**
     * Generate the synthetic data
     * y(x) = sin(2*pi*x) + 1
     */

    size_t n_obs = 10;
    double sigma_noise = 1e-3;

    // Make a grid in [0,1]
    Eigen::VectorXd x_obs = cmp::grid::LinspacedGrid(0 * cmp::ScalarOne, 1 * cmp::ScalarOne).construct(n_obs);
    Eigen::VectorXd y_obs = Eigen::VectorXd::Zero(n_obs);
    for(size_t i = 0; i < n_obs; i++) {
        y_obs(i) = std::sin(2.0 * M_PI * x_obs(i)) + 1.0 + sigma_noise * dist_n(rng);
    }

    // Make a test grid for plotting
    size_t n_test = 1000;
    auto x_test = cmp::grid::LinspacedGrid(0 * cmp::ScalarOne, 1 * cmp::ScalarOne).construct(n_test);

    /**
     * Define the GP covariance, mean function and hyperparameter prior
     *
     * KERNEL: k(x,x') = c * exp(-0.5*((x-x')/l)^2). This is classical RBF covariance function
     * MEAN: m(x) = a. This is a constant mean function
     * HYPERPARAMETERS: par = [c,l,a]. c: kernel amplitude, l: lengthscale, a: mean function value
     * PRIOR: p(c,l,a) = 1/c^2 * This prior penalizes small lengthscales and large amplitudes
     *
     * In the kernel and mean function definitions below, we use the index of the hyperparameter
     * in the hyperparameter vector par. For example, in the kernel definition below, the
     * amplitude c is par(0) and the lengthscale l is par(1). The mean function value a is par(2).
     * In the RBF kernel definition, we also use indexX=-1 to indicate that the kernel is defined
     * on the full input space (1D in this case). If we wanted to define the kernel on a specific
     * input dimension, we would use indexX=0 (for 1D input), indexX=1 (for 2D input) and so on.
     *
     * We use Maximum A Posteriori (MAP) estimation to fit the hyperparameters with nlopt's TNEWTON algorithm.
     * This requires the gradient and hessian of the log-posterior, which are computed automatically using the
     * definitions of the kernel, mean function and prior below.
     */
    auto kernel = cmp::covariance::Constant::make(0) * cmp::covariance::SquaredExponential::make(1, -1);
    auto mean = cmp::mean::Constant::make(2);
    Eigen::VectorXd param = 0.1 * Eigen::VectorXd::Ones(3); // Initial guess for the hyperparameters


    // Create the GP object and set it, for now the hyperparameters are just a guess
    cmp::gp::GaussianProcess gp;
    gp.set(kernel, mean, param, 1e-5);

    // Now let's plot some prior samples
    auto [mu_prior, cov_prior] = gp.predictMultiple(x_test);
    auto prior_distribution = cmp::distribution::MultivariateNormalDistribution(mu_prior, cov_prior);

    size_t n_samples = 10;
    plt::figure_size(800, 600);
    for(size_t i = 0; i < n_samples; i++) {
        Eigen::VectorXd sample = prior_distribution.sample(rng);
        plt::plot(x_test, sample, "g-");
    }
    plt::title("Samples from the GP prior");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::show();

    // Now we can condition the GP on the observations
    gp.condition(x_obs, y_obs, false);

    // Let's plot the some samples
    auto [mu_post, cov_post] = gp.predictMultiple(x_test);
    auto post_distribution = cmp::distribution::MultivariateNormalDistribution(mu_post, cov_post);

    plt::figure_size(800, 600);
    for(size_t i = 0; i < n_samples; i++) {
        Eigen::VectorXd sample = post_distribution.sample(rng);
        plt::plot(x_test, sample, "g-");
    }
    plt::scatter(x_obs, y_obs, 50.0, {{"c", "r"}, {"label", "Observations"}});
    plt::title("Samples from the GP posterior");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::show();

    // Let's evaluate the MSE using K-Fold Cross Validation (CV)
    size_t n_folds = 20;
    cmp::statistics::KFold kfold(n_obs, n_folds, true, 42);

    double mse = 0.0;
    for(const auto& [train_indices, test_indices] : kfold) {

        double local_mse = 0.0;
        Eigen::MatrixXd x_train = cmp::slice(x_obs, train_indices);
        Eigen::VectorXd y_train = cmp::slice(y_obs, train_indices);
        Eigen::MatrixXd x_test_cv = cmp::slice(x_obs, test_indices);
        Eigen::VectorXd y_test_cv = cmp::slice(y_obs, test_indices);

        // Set the training data (this does not retrain the GP, just sets the observations)
        gp.condition(x_train, y_train, true);

        // Compute the predictions for the test set
        for(size_t i = 0; i < x_test_cv.rows(); i++) {
            Eigen::VectorXd x = x_test_cv.row(i);
            auto [y_pred, y_var] = gp.predict(x);
            local_mse += (y_pred - y_test_cv(i)) * (y_pred - y_test_cv(i)) + y_var;
        }

        local_mse /= static_cast<double>(x_test_cv.rows());
        mse += local_mse;

    }
    std::cout << "Cross Validation MSE: " << mse / static_cast<double>(n_folds) << std::endl;


    /**
     * Now let's optimize the hyperparameters.
     * Many methods are available:
     *  - Maximum Likelihood Estimation (MLE)
     *  - Maximum A Posteriori (MAP)
     *  - Leave-One-Out Cross Validation (MLOO)
     *  - Leave-One-Out Cross Validation with prior (MLOOP)
     * We will use MLOOP here and nlopt's TNEWTON algorithm.
     */
    Eigen::VectorXd lb(3);    // Lower bounds for the hyperparameters
    lb << 1e-3, 1e-3, -5;

    Eigen::VectorXd ub(3);    // Upper bounds for the hyperparameters
    ub << 1e5, 1e5, 5;

    auto prior = cmp::prior::FromDistribution::make(cmp::distribution::PowerLawDistribution::make(2), 0); // 2 refers to the degree of the power-law, 0 to the par index

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    gp.fit(x_obs, y_obs, lb, ub, cmp::gp::LOO, nlopt::LD_TNEWTON, 1e-5, true, prior);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Time to optimize the hyperparameters: " << time_span.count() << " seconds" << std::endl;
    std::cout << "Optimized hyperparameters: " << gp.getParameters().transpose() << std::endl;

    // Now sample a few realizations from the GP posterior
    auto [mu_post_opt, cov_post_opt] = gp.predictMultiple(x_test);
    auto post_distribution_opt = cmp::distribution::MultivariateNormalDistribution(mu_post_opt, cov_post_opt);

    plt::figure_size(800, 600);
    for(size_t i = 0; i < n_samples; i++) {
        Eigen::VectorXd sample = post_distribution_opt.sample(rng);
        plt::plot(x_test, sample, "g-");
    }
    plt::scatter(x_obs, y_obs, 50.0, {{"c", "r"}, {"label", "Observations"}});
    plt::title("Samples from the optimized GP posterior");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::show();

    // Now we recompute the MSE using K-Fold Cross Validation (CV) with the optimized hyperparameters
    mse = 0.0;
    for(const auto& [train_indices, test_indices] : kfold) {
        double local_mse = 0.0;
        Eigen::MatrixXd x_train = cmp::slice(x_obs, train_indices);
        Eigen::VectorXd y_train = cmp::slice(y_obs, train_indices);
        Eigen::MatrixXd x_test_cv = cmp::slice(x_obs, test_indices);
        Eigen::VectorXd y_test_cv = cmp::slice(y_obs, test_indices);

        // Set the training data (this does not retrain the GP, just sets the observations)
        gp.condition(x_train, y_train, true);

        // Compute the predictions for the test set
        for(size_t i = 0; i < x_test_cv.rows(); i++) {
            Eigen::VectorXd x = x_test_cv.row(i);
            auto [y_pred, y_var] = gp.predict(x);
            local_mse += (y_pred - y_test_cv(i)) * (y_pred - y_test_cv(i)) + y_var;
        }

        local_mse /= static_cast<double>(x_test_cv.rows());
        mse += local_mse;

    }
    std::cout << "Cross Validation MSE after hyperparameter optimization: " << mse / static_cast<double>(n_folds) << std::endl;

    return 0;

}