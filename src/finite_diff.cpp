#include <finite_diff.h>


std::vector<double> external_stencil(const cmp::accuracy &accuracy) {
    switch(accuracy) {
    case cmp::SECOND:
        return { { 1, -1 } };
    case cmp::FOURTH:
        return { { 1, -8, 8, -1 } };
    case cmp::SIXTH:
        return { { -1, 9, -45, 45, -9, 1 } };
    case cmp::EIGHTH:
        return { { 3, -32, 168, -672, 672, -168, 32, -3 } };
    default:
        throw std::invalid_argument("Invalid accuracy order. Using order 2");
    }
}

std::vector<double> internal_stencil(const cmp::accuracy &accuracy) {
    switch(accuracy) {
    case cmp::SECOND:
        return { { 1, -1 } };
    case cmp::FOURTH:
        return { { -2, -1, 1, 2 } };
    case cmp::SIXTH:
        return { { -3, -2, -1, 1, 2, 3 } };
    case cmp::EIGHTH:
        return { { -4, -3, -2, -1, 1, 2, 3, 4 } };
    default:
        throw std::invalid_argument("Invalid accuracy order. Using order 2");
    }
}

double denominator(const cmp::accuracy &accuracy) {
    switch(accuracy) {
    case cmp::SECOND:
        return 2;
    case cmp::FOURTH:
        return 12;
    case cmp::SIXTH:
        return 60;
    case cmp::EIGHTH:
        return 840;
    default:
        throw std::invalid_argument("Invalid accuracy order. Using order 2");
    }
}


double cmp::fd_gradient(const Eigen::VectorXd &x_0, const std::function<double(const Eigen::VectorXd&)> fun, const int &i, const cmp::accuracy accuracy, const double h) {

    // Check on the components
    if(i < 0 || i > x_0.size() - 1) {
        return 0.0;
    }

    const std::vector<double> external_coeffs = external_stencil(accuracy);
    const std::vector<double> internal_coeffs = internal_stencil(accuracy);

    double denom = denominator(accuracy) * h;
    denom *= denom;

    const int n_steps = external_coeffs.size();


    Eigen::VectorXd x_step = x_0;
    double grad = 0.0;

    for(int l = 0; l < n_steps; l++) {
        x_step[i] += internal_coeffs[l] * h;
        grad += external_coeffs[l] * fun(x_step);
        x_step[i] = x_0[i];
    }

    return grad / denom;
}


double cmp::fd_hessian(const Eigen::VectorXd &x_0, const std::function<double(const Eigen::VectorXd&)> fun, const int &i, const int &j, const cmp::accuracy accuracy, const double h) {

    // Check on the components
    if(i < 0 || i > x_0.size() - 1) {
        return 0.0;
    }
    if(j < 0 || j > x_0.size() - 1) {
        return 0.0;
    }

    const std::vector<double> external_coeffs = external_stencil(accuracy);
    const std::vector<double> internal_coeffs = internal_stencil(accuracy);
    double denom = denominator(accuracy) * h;
    denom *= denom;

    const int n_steps = external_coeffs.size();


    Eigen::VectorXd x_step = x_0;
    double hess = 0.0;

    for(int l = 0; l < n_steps; l++) {
        for(int k = 0; k < n_steps; k++) {

            x_step[i] += internal_coeffs[l] * h;
            x_step[j] += internal_coeffs[k] * h;

            hess += external_coeffs[l] * external_coeffs[k] * fun(x_step);

            x_step[j] = x_0[j];
            x_step[i] = x_0[i];
        }
    }
    return hess / denom;
}