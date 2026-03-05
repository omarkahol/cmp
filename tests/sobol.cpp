#include <sobol.h>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

/**
 * This is a class to store the Ishigami function and its known Sobol indices.
 * The Ishigami function is defined as:
 * f(x) = sin(x1) + a * sin^2(x2) + b * x3^4 * sin(x1)
 * where x1, x2, x3 are uniformly distributed in [-pi, pi].
 * The parameters a and b control the function's behavior.
 *
 * The Sobol indices are known analytically for this function from literature.
 * See Uncertainty Quantification: Theory, Implementation, and Applications by Ralph C. Smith (2006)
 */

class IshigamiFunction {
  private:
    double a_;
    double b_;

    double D_;
    double D1_;
    double D2_;
    double D3_;
    double D12_;
    double D13_;
    double D23_;
    double D123_;

  public:
    IshigamiFunction(double a, double b)
        : a_(a), b_(b) {

        // Precompute variances for Sobol indices
        D_ = std::pow(a_, 2) / 8 + b_ * std::pow(M_PI, 4) / 5 + b_ * b_ * std::pow(M_PI, 8) / 18 + 0.5;
        D1_ = b_ * std::pow(M_PI, 4) / 5 + b_ * b_ * std::pow(M_PI, 8) / 50 + 0.5;
        D2_ = a_ * a_ / 8;
        D3_ = 0.0;
        D12_ = 0.0;
        D13_ = b_ * b_ * std::pow(M_PI, 8) * 8 / 225;
        D23_ = 0.0;
        D123_ = 0.0;
    }

    double operator()(const Eigen::VectorXd &x) const {
        // x is assumed to be in [0, 1]^3, map to [-pi, pi]
        double x1 = x(0) * 2 * M_PI - M_PI;
        double x2 = x(1) * 2 * M_PI - M_PI;
        double x3 = x(2) * 2 * M_PI - M_PI;

        return std::sin(x1) + a_ * std::pow(std::sin(x2), 2) + b_ * std::pow(x3, 4) * std::sin(x1);
    }

    // Known Sobol indices for this function
    double getFirstOrderSobolIndex(size_t i) const {
        if(i == 0) return D1_ / D_; // S1
        if(i == 1) return D2_ / D_; // S2
        if(i == 2) return D3_ / D_; // S3
        throw std::out_of_range("Index out of range");
    }

    double getSecondOrderSobolIndex(size_t i, size_t j) const {
        if((i == 0 && j == 1) || (i == 1 && j == 0)) return D12_ / D_;   // S12
        if((i == 0 && j == 2) || (i == 2 && j == 0)) return D13_ / D_;   // S13
        if((i == 1 && j == 2) || (i == 2 && j == 1)) return D23_ / D_;   // S23
        throw std::out_of_range("Index out of range");
    }

    double getTotalOrderSobolIndex(size_t i) const {
        if(i == 0) return (D1_ + D12_ + D13_ + D123_) / D_;  // T1
        if(i == 1) return (D2_ + D12_ + D23_ + D123_) / D_;  // T2
        if(i == 2) return (D3_ + D13_ + D23_ + D123_) / D_;  // T3
        throw std::out_of_range("Index out of range");
    }
};

int main() {

    // Create an instance of the Ishigami function with standard parameters
    IshigamiFunction f(1, 0.1);


    size_t dim = 3;
    size_t n_points = 1000;

    // Here the problem is defined in [0, 1]^3
    Eigen::VectorXd lowerBound = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd upperBound = Eigen::VectorXd::Ones(dim);

    // Create SobolSaltelli instance
    cmp::sobol::SobolSaltelli ss;

    // We can generate the evaluation grid using Sobol sequencing (with scrambling and shuffling)
    // The grid will contain 2 * n_points + dim * n_points + (dim * (dim - 1) / 2) * n_points rows if secondOrder is true
    // The method already provides all the necessary points for Sobol analysis
    auto evalGrid = ss.evaluationGrid(n_points, std::make_shared<cmp::grid::ScrambledSobolGrid>(lowerBound, upperBound, 42), true);
    std::cout << "Evaluation grid shape: " << evalGrid.rows() << " x " << evalGrid.cols() << std::endl;

    // Now it is up to us to evaluate the function at these points
    Eigen::VectorXd Y(evalGrid.rows());
    for(size_t i = 0; i < evalGrid.rows(); ++i) {
        Y(i) = f(evalGrid.row(i));
    }


    /**
     * Now we can compute the Sobol indices from the function evaluations
     * The compute method returns the Sobol indices
     * The computeWithBootstrap method returns the Sobol indices along with variance estimates using bootstrap resampling
     * This allows us to estimate confidence intervals for the indices
     * We will use 30 bootstrap samples for this example
     *
     * If we do not want bootstrap estimates, we can simply call:
     * auto results = ss.compute(Y);
     * and access results.S, results.totalOrder, results.S2 directly
     *
     * Note that S2 will be empty if secondOrder was set to false when creating the evaluation grid. If secondOrder is true, S2 will contain the second-order indices:
     * S2[0] = S12, S2[1] = S13, S2[2] = S14, ..., S2[dim*(dim-1)/2 - 1] = S(d-1)(d)
     *
     */
    size_t nBootstrap = 30;
    auto [meanResults, varResults] = ss.computeWithBootstrap(Y, nBootstrap, 42);

    std::cout << "First order Sobol indices (with " << nBootstrap << " bootstrap samples):" << std::endl;
    for(size_t i = 0; i < dim; ++i) {
        double exactS = f.getFirstOrderSobolIndex(i);
        std::cout << "S" << (i + 1) << " = " << meanResults.firstOrder(i) << " +/- " << std::sqrt(varResults.firstOrder(i)) << " (exact: " << exactS << ")" << std::endl;
    }
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Second order Sobol indices (with " << nBootstrap << " bootstrap samples):" << std::endl;
    size_t count = 0;
    for(size_t i = 0; i < dim; ++i) {
        for(size_t j = i + 1; j < dim; ++j) {
            double exactS2 = f.getSecondOrderSobolIndex(i, j);
            std::cout << "S" << (i + 1) << (j + 1) << " = " << meanResults.secondOrder(count) << " +/- " << std::sqrt(varResults.secondOrder(count)) << " (exact: " << exactS2 << ")" << std::endl;
            count++;
        }
    }
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Total order Sobol indices (with " << nBootstrap << " bootstrap samples):" << std::endl;
    for(size_t i = 0; i < dim; ++i) {
        double exactT = f.getTotalOrderSobolIndex(i);
        std::cout << "totalOrder" << (i + 1) << " = " << meanResults.totalOrder(i) << " +/- " << std::sqrt(varResults.totalOrder(i)) << " (exact: " << exactT << ")" << std::endl;
    }

    plt::figure();
    plt::errorbar(std::vector<double> {1, 2, 3}, std::vector<double> {meanResults.firstOrder(0), meanResults.firstOrder(1), meanResults.firstOrder(2)},
    std::vector<double> {std::sqrt(varResults.firstOrder(0)), std::sqrt(varResults.firstOrder(1)), std::sqrt(varResults.firstOrder(2))},
    {{"fmt", "o"}, {"label", "First order"}});
    plt::errorbar(std::vector<double> {1, 2, 3}, std::vector<double> {meanResults.totalOrder(0), meanResults.totalOrder(1), meanResults.totalOrder(2)},
    std::vector<double> {std::sqrt(varResults.totalOrder(0)), std::sqrt(varResults.totalOrder(1)), std::sqrt(varResults.totalOrder(2))},
    {{"fmt", "o"}, {"label", "Total order"}});

    // Plot the second order indices too
    std::vector<double> S2x;
    std::vector<double> S2y;
    std::vector<double> S2err;
    count = 0;
    for(size_t i = 0; i < dim; ++i) {
        for(size_t j = i + 1; j < dim; ++j) {
            S2x.push_back(1.5 + count);
            S2y.push_back(meanResults.secondOrder(count));
            S2err.push_back(std::sqrt(varResults.secondOrder(count)));
            count++;
        }
    }
    plt::errorbar(S2x, S2y, S2err, {{"fmt", "o"}, {"label", "Second order"}});

    plt::xticks(std::vector<double> {1, 2, 3}, std::vector<std::string> {"X1", "X2", "X3"});
    plt::ylabel("Sobol index");
    plt::title("Estimated Sobol indices with bootstrap confidence intervals");
    plt::legend();
    plt::grid(true);
    plt::show();

    return 0;
}