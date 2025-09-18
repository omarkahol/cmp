#include <multi_gp.h>

using namespace cmp::gp;
using namespace cmp;


void MultiOutputGaussianProcess::fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const method &method, const nlopt::algorithm &alg, const double &tol_rel)
{
    // Fit the GPs
    for (auto &gp : gps_) {
        gp.fit(x0,lb,ub,method,alg,tol_rel);
    }
}

cmp::distribution::MultivariateNormalDistribution MultiOutputGaussianProcess::predictiveDistribution(const Eigen::VectorXd &x) const
{
    Eigen::VectorXd mean(nGPs_);
    Eigen::MatrixXd var(nGPs_, nGPs_);
    for (size_t i=0; i<nGPs_; i++) {
        mean(i) = gps_[i].predict(x);
        var(i,i) = gps_[i].predictVariance(x);
    }
    return cmp::distribution::MultivariateNormalDistribution(mean,var.ldlt());
    
    
}

Eigen::VectorXd cmp::gp::MultiOutputGaussianProcess::predict(const Eigen::VectorXd &x) const
{
    Eigen::VectorXd mean(nGPs_);
    for (size_t i=0; i<nGPs_; i++) {
        mean(i) = gps_[i].predict(x);
    }
    return mean;
}

Eigen::MatrixXd cmp::gp::MultiOutputGaussianProcess::predictVariance(const Eigen::VectorXd &x, const int &i) const
{
    Eigen::MatrixXd var(nGPs_, nGPs_);
    for (size_t j=0; j<nGPs_; j++) {
        var(j,j) = gps_[j].predictVariance(x);
    }
    return var;
}
