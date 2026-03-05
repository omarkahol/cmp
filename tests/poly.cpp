/**
 * @file poly.cpp
 * @brief Test file for Polynomial Chaos Expansion (PCE) functionalities.
 * @author Omar Kahol
 */

#include <iostream>

#include <cmp_defines.h>
#include <poly.h>
#include <grid.h>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

int main() {

    // Make a Polynomial Expansion in 1D with total degree 2 using Legendre polynomials
    cmp::PolynomialExpansion pce1;
    pce1.set(1, 3, cmp::LegendrePolynomial::make());

    // Generate some samples and values from a sinusoidal function
    double freq = 1.0;
    double phase = 0.0;
    double amplitude = 1.0;
    double offset = 0.0;

    size_t n_samples_1d = 20;
    Eigen::MatrixXd data(n_samples_1d, 2);
    for(size_t i = 0; i < n_samples_1d; i++) {
        double x = 0 + 1.0 * i / (n_samples_1d - 1); // from 0 to 1
        data(i, 0) = x;
        data(i, 1) = amplitude * sin(2 * M_PI * freq * x + phase) + offset;
    }

    // Fit the polynomial expansion to the samples
    pce1.fit(data.leftCols(1), data.col(1));

    // Make a test grid and evaluate the PCE
    auto testGrid = Eigen::VectorXd::LinSpaced(100, 0.0, 1.0);

    std::vector<double> pceValuesVec;
    std::vector<double> pceLbVec;
    std::vector<double> pceUbVec;
    std::vector<double> gridX;
    for(size_t i = 0; i < testGrid.rows(); i++) {
        auto [mean, var] = pce1.predict(testGrid.row(i));
        double stddev = sqrt(var);
        double lb = mean - 1.96 * stddev;
        double ub = mean + 1.96 * stddev;
        pceLbVec.push_back(lb);
        pceUbVec.push_back(ub);
        pceValuesVec.push_back(mean);
        gridX.push_back(testGrid(i, 0));
    }

    // Plot the results
    plt::figure();
    plt::plot(gridX, pceValuesVec, "r-", {{"label", "PCE Approximation"}});
    plt::plot(gridX, pceLbVec, "g--", {{"label", "PCE Lower Bound"}});
    plt::plot(gridX, pceUbVec, "g--", {{"label", "PCE Upper Bound"}});

    // Plot the samples
    plt::plot(data.col(0), data.col(1), "bo", {{"label", "Samples"}});
    plt::title("1D Polynomial Chaos Expansion using Hermite Polynomials");
    plt::legend();
    plt::show();

    return 0;
}
