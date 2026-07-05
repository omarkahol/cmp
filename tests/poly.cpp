#include <iostream>
#include <cmath>
#include <vector>

#include <poly.h>
#include <grid.h>
#include <matplotlibcpp.h>
#include <Eigen/Dense>
// #include <scaler.h> // No longer needed for manual scaling!

namespace plt = matplotlibcpp;

class IshigamiFunction {
  private:
    double a_;
    double b_;
    double D_, D1_, D2_, D3_, D12_, D13_, D23_, D123_;

  public:
    IshigamiFunction(double a, double b)
        : a_(a), b_(b) {
        D_ = std::pow(a_, 2) / 8 + b_ * std::pow(M_PI, 4) / 5 + b_ * b_ * std::pow(M_PI, 8) / 18 + 0.5;
        D1_ = b_ * std::pow(M_PI, 4) / 5 + b_ * b_ * std::pow(M_PI, 8) / 50 + 0.5;
        D2_ = a_ * a_ / 8;
        D3_ = 0.0;
        D12_ = 0.0;
        D13_ = b_ * b_ * std::pow(M_PI, 8) * 8 / 225;
        D23_ = 0.0;
        D123_ = 0.0;
    }

    // Now evaluates natively on [-pi, pi] without manual internal mappings
    double operator()(const Eigen::VectorXd &x) const {
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        return std::sin(x1) + a_ * std::pow(std::sin(x2), 2) + b_ * std::pow(x3, 4) * std::sin(x1);
    }

    double getFirstOrderSobolIndex(size_t i) const {
        if(i == 0) return D1_ / D_;
        if(i == 1) return D2_ / D_;
        if(i == 2) return D3_ / D_;
        return 0.0;
    }
};

int main() {
    // 1. Setup
    IshigamiFunction f(1.0, 0.01);
    size_t dim = 3;
    size_t total_degree = 8;

    // 3. Generate Grid directly in the physical domain [-pi, pi]^dim
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(dim, -M_PI);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(dim, M_PI);
    auto grid = cmp::grid::LatinHypercubeGrid(lower, upper).construct(1000);

    Eigen::VectorXd values(grid.rows());

    for(size_t i = 0; i < grid.rows(); ++i) {
        values(i) = f(grid.row(i).transpose());
    }

    // Fit using the physical grid; PCE maps to canonical internally
    cmp::PolynomialExpansion<cmp::LegendreBasis> pce;
    pce.set(dim, total_degree, lower, upper);
    //pce.project(f, 10);
    pce.fit(grid, values);

    // 5. Extract Analytical Sobol Indices
    Eigen::VectorXd pceMainIndices = pce.getSobolMainIndices();

    std::cout << "\n--- Main (First-Order) Sobol Indices ---" << std::endl;
    std::vector<double> exact_S(dim), pce_S(dim);

    for(size_t i = 0; i < dim; ++i) {
        exact_S[i] = f.getFirstOrderSobolIndex(i);
        pce_S[i] = pceMainIndices(i);
        std::cout << "S" << (i + 1) << " | Exact: " << exact_S[i]
                  << " | PCE Analytical: " << pce_S[i]
                  << " | Error: " << std::abs(exact_S[i] - pce_S[i]) << std::endl;
    }

    // 6. Plot the prediction in a 1D slice for X1
    // Fixing X2 and X3 at 0.0 (the center of [-pi, pi], equivalent to your old 0.5 mapped value)
    size_t num_points = 100;
    Eigen::VectorXd x1 = Eigen::VectorXd::LinSpaced(num_points, -M_PI, M_PI);
    Eigen::MatrixXd X(num_points, dim);
    X.col(0) = x1;

    for(size_t i = 1; i < dim; ++i) {
        X.col(i) = Eigen::VectorXd::Constant(num_points, 0.0);
    }

    Eigen::VectorXd y_exact(num_points), y_pce(num_points);
    for(size_t i = 0; i < num_points; ++i) {
        // No more physical-to-canonical mapping logic in the test space
        Eigen::VectorXd currentPoint = X.row(i).transpose();

        y_exact(i) = f(currentPoint);
        auto [mean, variance] = pce.predict(currentPoint);
        y_pce(i) = mean;
    }

    plt::figure();
    plt::plot(x1, y_exact, "r-", {{"label", "Exact"}});
    plt::plot(x1, y_pce, "b--", {{"label", "PCE Prediction"}});
    plt::xlabel("X1");
    plt::ylabel("f(X1, 0, 0)");
    plt::title("Ishigami Function Slice at X2=0, X3=0");
    plt::legend();
    plt::save("/Users/omarkahol/opt/CMP++/Technical_Doc/images/poly_slice.pdf");
    plt::close();


    // Compute the mean with Tensor Integration using the physical integration method
    auto integrator = cmp::TensorIntegrator<cmp::LegendreBasis>(dim, 10);
    double mean = integrator.integrate(f, lower, upper);
    std::cout << "\n--- Mean Computation ---" << std::endl;
    std::cout << "Mean (Tensor Integration): " << mean << std::endl;
    std::cout << "Mean (Analytical): " << pce.getAnalyticalMean() << std::endl; // Ishigami function has a mean of 0

    return 0;
}