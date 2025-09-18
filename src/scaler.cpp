#include "scaler.h"

double cmp::scaler::StandardScaler::transform(const double &x) const
{
    return (x - mean_)/std_;
}

double cmp::scaler::StandardScaler::inverseTransfrom(const double &x) const
{
    return x*std_ + mean_;
}

std::vector<double> cmp::scaler::StandardScaler::fit_transform(const std::vector<double> &data)
{
    fit(data);
    std::vector<double> data_t(data.size());
    for (size_t i=0; i<data.size(); i++){
        data_t[i] = transform(data[i]);
    }
    return data_t;
}

void cmp::scaler::StandardScaler::fit(const std::vector<double> &data)
{
    mean_, std_ = 0.0, 0.0;
    for (auto &data_i : data){
        mean_ += data_i;
        std_ += data_i*data_i;
    }
    mean_ = mean_/double(data.size());
    std_ = std::sqrt((std_ - mean_*mean_/double(data.size()))/double(data.size()-1.0));

    if (std_ == 0.0){
        std_ = 1.0;
    }
}

Eigen::VectorXd cmp::scaler::StandardVectorScaler::transform(const Eigen::VectorXd &x) const 
{
    return lltDecomposition_.matrixL().solve(x - mean_);
}

Eigen::VectorXd cmp::scaler::StandardVectorScaler::inverseTransfrom(const Eigen::VectorXd &x) const 
{
    return lltDecomposition_.matrixL()*x + mean_;
}

void cmp::scaler::StandardVectorScaler::fit(const std::vector<Eigen::VectorXd> &data)
{
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(data[0].size());
    Eigen::MatrixXd scale = Eigen::MatrixXd::Zero(data[0].size(),data[0].size());

    for (auto &data_i : data){
        mean += data_i;
        scale += data_i*data_i.transpose();
    }
    mean /= double(data.size());
    scale -= data.size()*mean*mean.transpose();
    scale /= double(data.size()-1);

    // Check if the matrix is well defined
    for (size_t i=0; i<scale.rows(); i++){

        // Check if the diagonal is zero, if so set it to 1
        if (scale(i,i) < TOL){
            scale(i,i) = 1.0;
        }

        // Check if the matrix is symmetric
        for (size_t j=i+1; j<scale.cols(); j++){
            if (scale(i,j) != scale(j,i)){
                scale(j,i) = scale(i,j);
            }
        }
    }


    lltDecomposition_ = scale.llt();
    mean_ = mean;
}

std::vector<Eigen::VectorXd> cmp::scaler::StandardVectorScaler::fit_transform(const std::vector<Eigen::VectorXd> &data)
{
    fit(data);
    std::vector<Eigen::VectorXd> data_t(data.size());
    for (size_t i=0; i<data.size(); i++){
        data_t[i] = transform(data[i]);
    }
    return data_t;
}

void cmp::scaler::PCA::eigenDecomposition()
{
    // Check the number of components
    if (nComponents_ > eigenSolver_.eigenvalues().size() || nComponents_ < 0){
        
        Eigen::VectorXd eigenvalues = eigenSolver_.eigenvalues();
        eigenvalues.array() = eigenvalues.array().abs().sqrt();

        Eigen::MatrixXd eigenvectors = eigenSolver_.eigenvectors();
        sqrtCov_ = eigenvectors*eigenvalues.asDiagonal();

        // Compute the inverse
        eigenvalues.array() = 1.0/eigenvalues.array();
        sqrtCovInv_ = eigenvalues.asDiagonal()*eigenvectors.transpose();

    } else {
        
        // Take the first n_components
        Eigen::VectorXd eigenvalues = eigenSolver_.eigenvalues().tail(nComponents_);
        eigenvalues.array() = eigenvalues.array().abs().sqrt();

        // Take the first n_components eigenvectors
        Eigen::MatrixXd eigenvectors = eigenSolver_.eigenvectors().rightCols(nComponents_);
        sqrtCov_ = eigenvectors*eigenvalues.asDiagonal();

        // Compute the inverse
        eigenvalues.array() = 1.0/eigenvalues.array();
        sqrtCovInv_ = eigenvalues.asDiagonal()*eigenvectors.transpose();
    }
}

Eigen::VectorXd cmp::scaler::PCA::transform(const Eigen::VectorXd &data) const
{

    return sqrtCovInv_*(data - mean_);
}

Eigen::VectorXd cmp::scaler::PCA::inverseTransfrom(const Eigen::VectorXd &data) const
{
    return sqrtCov_*data + mean_;
}

void cmp::scaler::PCA::fit(const std::vector<Eigen::VectorXd> &data)
{
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(data[0].size());
    Eigen::MatrixXd scale = Eigen::MatrixXd::Zero(data[0].size(),data[0].size());

    for (auto &data_i : data){
        mean += data_i;
        scale += data_i*data_i.transpose();
    }
    mean /= double(data.size());
    scale -= data.size()*mean*mean.transpose();
    scale /= double(data.size()-1);

    // Check if the matrix is well defined
    for (size_t i=0; i<scale.rows(); i++){

        // Check if the diagonal is zero, if so set it to 1
        if (scale(i,i) < TOL){
            scale(i,i) = 1.0;
        }

        // Check if the matrix is symmetric
        for (size_t j=i+1; j<scale.cols(); j++){
            if (scale(i,j) != scale(j,i)){
                scale(j,i) = scale(i,j);
            }
        }
    }

    // Perform the PCA
    eigenSolver_.compute(scale);
    eigenDecomposition();
    mean_ = mean;
}

std::vector<Eigen::VectorXd> cmp::scaler::PCA::fit_transform(const std::vector<Eigen::VectorXd> &data)
{
    fit(data);
    std::vector<Eigen::VectorXd> data_t(data.size());
    for (size_t i=0; i<data.size(); i++){
        data_t[i] = transform(data[i]);
    }
    return data_t;
}

void cmp::scaler::PCA::resize(size_t nComponents)
{
    nComponents_ = nComponents;
    eigenDecomposition();
}


Eigen::VectorXd cmp::scaler::EllipticScaler::transform(const Eigen::VectorXd &data) const {
    Eigen::VectorXd transformed = data - mean_;
    for (size_t i = 0; i < transformed.size(); i++) {
        transformed(i) /= std_[i];
    }
    return transformed;
}

Eigen::VectorXd cmp::scaler::EllipticScaler::inverseTransfrom(const Eigen::VectorXd &data) const {
    Eigen::VectorXd transformed = data;
    for (size_t i = 0; i < transformed.size(); i++) {
        transformed(i) *= std_[i];
    }
    return transformed + mean_;
}

void cmp::scaler::EllipticScaler::fit(const std::vector<Eigen::VectorXd> &data) {
    mean_ = Eigen::VectorXd::Zero(data[0].size());
    std_.resize(data[0].size());

    for (auto &data_i : data) {
        mean_ += data_i;
    }
    mean_ /= double(data.size());

    for (size_t i = 0; i < std_.size(); i++) {
        std_[i] = 0.0;
        for (auto &data_i : data) {
            std_[i] += (data_i(i) - mean_(i)) * (data_i(i) - mean_(i));
        }
        std_[i] = std::sqrt(std_[i] / double(data.size()-1));
    }
}

std::vector<Eigen::VectorXd> cmp::scaler::EllipticScaler::fit_transform(const std::vector<Eigen::VectorXd> &data) {
    fit(data);
    std::vector<Eigen::VectorXd> data_t(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        data_t[i] = transform(data[i]);
    }
    return data_t;
}
