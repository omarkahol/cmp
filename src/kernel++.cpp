#include "kernel++.h"

// Bandwidth implementations
// IsotropicBandwidth implementations (apply accepts a single difference vector)
Eigen::VectorXd cmp::kernel::IsotropicBandwidth::apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const {
    return (x1 - x2) / h_;
}

double cmp::kernel::IsotropicBandwidth::determinant() const {
    return 1.0 / std::pow(h_, dim_);
}

Eigen::VectorXd cmp::kernel::IsotropicBandwidth::gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t& i) const {
    return - (x1 - x2) / (h_ * h_);
}

double cmp::kernel::IsotropicBandwidth::gradientOfLogDeterminant(const size_t& i) const {
    // Compute the gradient of log(det) = -dim * log(h)
    return -static_cast<double>(dim_) / h_;
}

// DiagonalBandwidth implementations
Eigen::VectorXd cmp::kernel::DiagonalBandwidth::apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const {
    return (x1 - x2).cwiseQuotient(h_);
}

double cmp::kernel::DiagonalBandwidth::determinant() const {
    return 1 / det_;
}

Eigen::VectorXd cmp::kernel::DiagonalBandwidth::gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t& i) const {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(h_.size());
    grad(i) = - (x1(i) - x2(i)) / (h_(i) * h_(i));
    return grad;
}

double cmp::kernel::DiagonalBandwidth::gradientOfLogDeterminant(const size_t& i) const {
    // The determinant is the product of the elements of h so log(det) = sum(log(h_i))
    return - 1.0 / h_(i);
}

void cmp::kernel::DiagonalBandwidth::setFromVector(const Eigen::VectorXd &params) {
    if(params.size() != h_.size()) {
        throw std::invalid_argument("Diagonal bandwidth expects a parameter vector of size " + std::to_string(h_.size()));
    }
    if((params.array() <= 0).any()) {
        throw std::invalid_argument("All bandwidth elements must be positive.");
    }
    h_ = params;
    det_ = h_.prod();
}

Eigen::VectorXd cmp::kernel::DiagonalBandwidth::getParams() const {
    return h_;
}

// FullBandwidth implementations

double cmp::kernel::FullBandwidth::determinant() const {
    return 1 / det_;
}

std::pair<size_t, size_t> cmp::kernel::FullBandwidth::indexToRowCol(const size_t& i) const {
    size_t row = 0;
    size_t col = 0;
    size_t count = 0;
    for(row = 0; row < dim_; ++row) {
        for(col = 0; col <= row; ++col) {
            if(count == i) {
                return {row, col};
            }
            count++;
        }
    }
    throw std::out_of_range("Index out of range in FullBandwidth::indexToRowCol");
}

// FullBandwidth implementations (apply accepts two vectors and returns their difference scaled) by
// the inverse of the Cholesky factor
Eigen::VectorXd cmp::kernel::FullBandwidth::apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const {
    return L_.matrixL().solve(x1 - x2);
}

Eigen::VectorXd cmp::kernel::FullBandwidth::gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t& i) const {

    // Make a matrix whose only non-zero element is a 1 at (row, col) or (col, row) if row != col
    auto [row, col] = indexToRowCol(i);
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(dim_, dim_);
    E(row, col) = 1.0;
    if(row != col) {
        E(col, row) = 1.0;
    }

    return - L_.matrixL().solve(E * L_.matrixL().solve(x1 - x2));
}

double cmp::kernel::FullBandwidth::gradientOfLogDeterminant(const size_t& i) const {

    // Make a matrix whose only non-zero element is a 1 at (row, col) or (col, row) if row != col
    auto [row, col] = indexToRowCol(i);
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(dim_, dim_);
    E(row, col) = 1.0;
    if(row != col) {
        E(col, row) = 1.0;
    }

    // The gradient is the trace of -L^{-1} E
    Eigen::MatrixXd Linv_E = L_.matrixL().solve(E);
    return -Linv_E.trace();
};



void cmp::kernel::FullBandwidth::setFromVector(const Eigen::VectorXd &params) {
    size_t expected_size = dim_ * (dim_ + 1) / 2;
    if(params.size() != expected_size) {
        throw std::invalid_argument("Full bandwidth expects a parameter vector of size " + std::to_string(expected_size));
    }

    // Reconstruct the lower triangular matrix
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dim_, dim_);
    size_t count = 0;
    for(size_t row = 0; row < dim_; ++row) {
        for(size_t col = 0; col <= row; ++col) {
            L(row, col) = params(count);
            count++;
        }
    }

    // Recompute the LLT
    L_ = Eigen::LLT<Eigen::MatrixXd>(L * L.transpose());
    if(L_.info() != Eigen::Success) {
        throw std::invalid_argument("The provided matrix is not positive definite.");
    }

    // Compute determinant as product of diagonal entries of the Cholesky factor
    // directly to avoid hitting Eigen internals that trigger deprecated enum
    // bitwise operations.
    det_ = 1.0;
    for(size_t i = 0; i < static_cast<size_t>(L_.matrixL().rows()); ++i) {
        det_ *= L_.matrixL()(i, i);
    }
};

Eigen::VectorXd cmp::kernel::FullBandwidth::getParams() const {
    size_t param_size = dim_ * (dim_ + 1) / 2;
    Eigen::VectorXd params(param_size);
    size_t count = 0;
    for(size_t row = 0; row < dim_; ++row) {
        for(size_t col = 0; col <= row; ++col) {
            params(count) = L_.matrixL()(row, col);
            count++;
        }
    }
    return params;
}



// Gaussian kernel implementation
double cmp::kernel::Gaussian::eval(const Eigen::VectorXd& z_diff) const {
    double quadForm = z_diff.dot(z_diff);
    return std::exp(-0.5 * quadForm);
}

double cmp::kernel::Gaussian::normalizationConstant(const size_t& dim) const {
    return 1 / std::pow(2 * M_PI, 0.5 * dim);
}

double cmp::kernel::Gaussian::applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const {
    return - eval(z) * z.dot(grad_z_i);
}

double cmp::kernel::Epanechnikov::eval(const Eigen::VectorXd& z_diff) const {
    double quadForm = z_diff.dot(z_diff);
    if(quadForm >= 1.0) {
        return 0.0;
    }
    return (1.0 - quadForm);
}

double cmp::kernel::Epanechnikov::normalizationConstant(const size_t& dim) const {

    // Volume of the unit ball in d dimensions
    double volume = std::pow(M_PI, dim / 2.0) / std::tgamma(dim / 2.0 + 1.0);
    return 1.0 / (volume * std::pow(2.0, dim)); // 0.75^dim = 1/2^dim

}

double cmp::kernel::Epanechnikov::applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const {

    double quadForm = z.dot(z);
    if(quadForm >= 1.0) {
        return 0.0;
    }

    // Gradient of the kernel: -2 * z
    return -2.0 * z.dot(grad_z_i);
}

double cmp::kernel::Uniform::eval(const Eigen::VectorXd& z_diff) const {
    // Check if all elements are in [-1, 1]
    if((z_diff.array() < -1.0).any() || (z_diff.array() > 1.0).any()) {
        return 0.0;
    }
    return 1.0;
}

double cmp::kernel::Uniform::normalizationConstant(const size_t &dim) const {
    return 1.0 / std::pow(2.0, dim);
}

double cmp::kernel::Uniform::applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const {
    // The gradient of the uniform kernel is zero everywhere except at the boundaries, which we ignore here
    return 0.0;
}