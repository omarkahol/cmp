#include "scaler.h"
#include "statistics.h"


Eigen::VectorXd cmp::scaler::StandardScaler::transform(const Eigen::Ref<const Eigen::VectorXd> &x) const {
    return lltDecomposition_.matrixL().solve(x - mean_);
}

Eigen::VectorXd cmp::scaler::StandardScaler::inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &x) const {
    return lltDecomposition_.matrixL() * x + mean_;
}

void cmp::scaler::StandardScaler::fit(const Eigen::Ref<const Eigen::MatrixXd> &data) {

    // Get dimensions
    size_t nSamples = data.rows();
    size_t nFeatures = data.cols();

    Eigen::VectorXd mean = cmp::statistics::mean(data);
    Eigen::MatrixXd scale = cmp::statistics::covariance(data);

    // Check if the matrix is well defined
    for(size_t i = 0; i < nFeatures; i++) {

        // Check if the diagonal is zero, if so we raise an error
        if(scale(i, i) < TOL) {
            throw std::invalid_argument("Covariance matrix is not positive definite");
        }
    }


    lltDecomposition_ = scale.llt();
    mean_ = mean;
}

Eigen::MatrixXd cmp::scaler::StandardScaler::fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    fit(data);
    Eigen::MatrixXd data_t = data;
    for(size_t i = 0; i < data.rows(); i++) {
        data_t.row(i) = transform(data.row(i));
    }
    return data_t;
}

void cmp::scaler::PCA::eigenDecomposition() {
    // Check the number of components
    if(nComponents_ <= 0 || nComponents_ > eigenSolver_.eigenvalues().size()) {

        Eigen::VectorXd eigenvalues = eigenSolver_.eigenvalues();
        eigenvalues.array() = eigenvalues.array().abs().sqrt();

        Eigen::MatrixXd eigenvectors = eigenSolver_.eigenvectors();
        sqrtCov_ = eigenvectors * eigenvalues.asDiagonal();

        // Compute the inverse
        eigenvalues.array() = 1.0 / eigenvalues.array();
        sqrtCovInv_ = eigenvalues.asDiagonal() * eigenvectors.transpose();

    } else {

        // Take the first n_components
        Eigen::VectorXd eigenvalues = eigenSolver_.eigenvalues().tail(nComponents_);
        eigenvalues.array() = eigenvalues.array().abs().sqrt();

        // Take the first n_components eigenvectors
        Eigen::MatrixXd eigenvectors = eigenSolver_.eigenvectors().rightCols(nComponents_);
        sqrtCov_ = eigenvectors * eigenvalues.asDiagonal();

        // Compute the inverse
        eigenvalues.array() = 1.0 / eigenvalues.array();
        sqrtCovInv_ = eigenvalues.asDiagonal() * eigenvectors.transpose();
    }
}

Eigen::VectorXd cmp::scaler::PCA::transform(const Eigen::Ref<const Eigen::VectorXd> &data) const {

    return sqrtCovInv_ * (data - mean_);
}

Eigen::VectorXd cmp::scaler::PCA::inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const {
    return sqrtCov_ * data + mean_;
}

void cmp::scaler::PCA::fit(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    // Get dimensions
    size_t nSamples = data.rows();
    size_t nFeatures = data.cols();

    Eigen::VectorXd mean = cmp::statistics::mean(data);
    Eigen::MatrixXd scale = cmp::statistics::covariance(data);

    // Check if the matrix is well defined
    for(size_t i = 0; i < nFeatures; i++) {
        // Check if the diagonal is zero, if so we raise an error
        if(scale(i, i) < TOL) {
            throw std::invalid_argument("Covariance matrix is not positive definite");
        }
    }

    // Perform the PCA
    eigenSolver_.compute(scale);
    eigenDecomposition();
    mean_ = mean;
}

Eigen::MatrixXd cmp::scaler::PCA::fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    fit(data);
    // Determine the output dimensions based on nComponents_
    size_t output_cols = (nComponents_ > 0 && nComponents_ < data.cols()) ? nComponents_ : data.cols();
    Eigen::MatrixXd data_t(data.rows(), output_cols);
    for(size_t i = 0; i < data.rows(); i++) {
        data_t.row(i) = transform(data.row(i));
    }
    return data_t;
}

void cmp::scaler::PCA::resize(size_t nComponents) {
    nComponents_ = nComponents;
    eigenDecomposition();
}


Eigen::VectorXd cmp::scaler::EllipticScaler::transform(const Eigen::Ref<const Eigen::VectorXd> &data) const {
    Eigen::VectorXd transformed = data - mean_;
    for(size_t i = 0; i < transformed.size(); i++) {
        transformed(i) /= std_[i];
    }
    return transformed;
}

Eigen::VectorXd cmp::scaler::EllipticScaler::inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const {
    Eigen::VectorXd transformed = data;
    for(size_t i = 0; i < transformed.size(); i++) {
        transformed(i) *= std_[i];
    }
    return transformed + mean_;
}

void cmp::scaler::EllipticScaler::fit(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    mean_ = data.colwise().mean();
    std_.resize(data.cols());

    for(size_t i = 0; i < std_.size(); i++) {
        double variance = 0.0;
        for(size_t j = 0; j < data.rows(); j++) {
            variance += std::pow(data(j, i) - mean_(i), 2);
        }
        std_[i] = std::sqrt(variance / double(data.rows() - 1));
    }
}

Eigen::MatrixXd cmp::scaler::EllipticScaler::fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    fit(data);
    Eigen::MatrixXd data_t(data.rows(), data.cols());
    for(size_t i = 0; i < data.rows(); i++) {
        data_t.row(i) = transform(data.row(i));
    }
    return data_t;
}

Eigen::VectorXd cmp::scaler::MinMaxScaler::transform(const Eigen::Ref<const Eigen::VectorXd> &data) const {
    // Scale each feature to [min_, max_] using the computed data_min_ and data_max_
    Eigen::VectorXd scaled = data;
    for(size_t i = 0; i < data.size(); i++) {
        if(data_max_(i) - data_min_(i) > TOL) {
            scaled(i) = (data(i) - data_min_(i)) / (data_max_(i) - data_min_(i)); // Scale to [0, 1]
            scaled(i) = scaled(i) * (max_(i) - min_(i)) + min_(i); // Scale to [min_, max_]
        } else {
            scaled(i) = min_(i); // If all values are the same, set to min_
        }
    }
    return scaled;
}

Eigen::VectorXd cmp::scaler::MinMaxScaler::inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const {
    // Inverse scale each feature from [min_, max_] back to original range using data_min_ and data_max_
    Eigen::VectorXd original = data;
    for(size_t i = 0; i < data.size(); i++) {
        if(max_(i) - min_(i) > TOL) {
            original(i) = (data(i) - min_(i)) / (max_(i) - min_(i)); // Scale to [0, 1]
            original(i) = original(i) * (data_max_(i) - data_min_(i)) + data_min_(i); // Scale back to original range
        } else {
            original(i) = data_min_(i); // If min_ == max_, set to data_min_
        }
    }
    return original;
}

void cmp::scaler::MinMaxScaler::fit(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    // Get dimensions
    size_t nSamples = data.rows();
    size_t nFeatures = data.cols();

    // Initialize data_min_ and data_max_
    data_min_ = Eigen::VectorXd::Constant(nFeatures, std::numeric_limits<double>::max());
    data_max_ = Eigen::VectorXd::Constant(nFeatures, std::numeric_limits<double>::lowest());

    // Compute the min and max for each feature
    for(size_t i = 0; i < nSamples; i++) {
        for(size_t j = 0; j < nFeatures; j++) {
            if(data(i, j) < data_min_(j)) {
                data_min_(j) = data(i, j);
            }
            if(data(i, j) > data_max_(j)) {
                data_max_(j) = data(i, j);
            }
        }
    }
}

Eigen::MatrixXd cmp::scaler::MinMaxScaler::fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) {
    fit(data);
    Eigen::MatrixXd data_t(data.rows(), data.cols());
    for(size_t i = 0; i < data.rows(); i++) {
        data_t.row(i) = transform(data.row(i));
    }
    return data_t;
}

// Intercept and scale are simply the values such that:
// X_scaled = (X - intercept) * scale

Eigen::VectorXd cmp::scaler::MinMaxScaler::getIntercept() const {
    return min_;
}

Eigen::MatrixXd cmp::scaler::MinMaxScaler::getScale() const {
    return (max_ - min_).asDiagonal();
}
