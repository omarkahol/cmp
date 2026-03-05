#include <multi_gp.h>

cmp::gp::MultiOutputGaussianProcess::MultiOutputGaussianProcess()  {
    set(1, cmp::covariance::Constant::make(0), cmp::mean::Zero::make(), Eigen::VectorXd::Zero(1), 1e-8);
}

cmp::gp::MultiOutputGaussianProcess::MultiOutputGaussianProcess(
    const size_t& nGps, const std::shared_ptr<covariance::Covariance>& kernel,
    const std::shared_ptr<mean::Mean>& mean, const Eigen::Ref<const Eigen::VectorXd>& params,
    double nugget) {
    set(nGps, kernel, mean, params, nugget);
}

void cmp::gp::MultiOutputGaussianProcess::set(const size_t& nGps,
                                              const std::shared_ptr<covariance::Covariance>& kernel,
                                              const std::shared_ptr<mean::Mean>& mean,
                                              const Eigen::Ref<const Eigen::VectorXd>& params,
                                              double nugget)  {

    assert(nGps > 0 && "The number of Gaussian Processes must be greater than zero");

    gps_.clear();
    for(size_t i = 0; i < nGps; i++) {
        gps_.emplace_back(kernel, mean, params, nugget);
    }
}

void cmp::gp::MultiOutputGaussianProcess::condition(const Eigen::Ref<const Eigen::MatrixXd>& xObs,
                                                    const Eigen::Ref<const Eigen::MatrixXd>& yObs,
                                                    bool copyData) {
    assert(yObs.cols() == gps_.size() && "The number of output dimensions must match the number of Gaussian Processes");

    // Check whether it owns the data or not
    if(copyData) {

        // Make a hard copy of teh data
        xObs_ = xObs;
        yObs_ = yObs;

        // Set internal pointer to the storage (create new Ref objects on heap)
        pXObs_.emplace(xObs_);
        pYObs_.emplace(yObs_);
    } else {

        // Just point to the external data
        pXObs_.emplace(xObs);
        pYObs_.emplace(yObs);
    }

    // Condition each GP (which will not copy the data again)
    for(size_t i = 0; i < gps_.size(); i++) {
        gps_[i].condition(pXObs_.value(), pYObs_.value().col(i), false);
    }
}

void cmp::gp::MultiOutputGaussianProcess::fit(const Eigen::Ref<Eigen::MatrixXd>& xObs,
                                              const Eigen::Ref<Eigen::MatrixXd>& yObs,
                                              const Eigen::Ref<const Eigen::VectorXd>& lb,
                                              const Eigen::Ref<const Eigen::VectorXd>& ub,
                                              const method& method, const nlopt::algorithm& alg,
                                              const double& tol_rel,
                                              bool copyData,
                                              const std::shared_ptr<cmp::prior::Prior>& prior) {
    assert(xObs.rows() == yObs.rows() && "The number of observation points must match the number of observation values");
    assert(yObs.cols() == gps_.size() && "The number of output dimensions must match the number of Gaussian Processes");

    // First we condition the GP with the data
    condition(xObs, yObs, copyData);

    // We then fit each GP individually (without copying the data again)
    for(size_t i = 0; i < gps_.size(); i++) {
        gps_[i].fit(pXObs_.value(), pYObs_.value().col(i), lb, ub, method, alg, tol_rel, true, prior);
    }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> cmp::gp::MultiOutputGaussianProcess::predict(const Eigen::Ref<const Eigen::VectorXd> &x, const type &type) const {
    Eigen::VectorXd mean(gps_.size());
    Eigen::MatrixXd var = Eigen::MatrixXd::Zero(gps_.size(), gps_.size());
    for(size_t i = 0; i < gps_.size(); i++) {
        auto [m, v] = gps_[i].predict(x, type);
        mean(i) = m;
        var(i, i) = v;
    }
    return std::make_pair(mean, var);
}

Eigen::VectorXd cmp::gp::MultiOutputGaussianProcess::predictMean(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const type &t) const {
    Eigen::VectorXd mean(gps_.size());
    for(size_t i = 0; i < gps_.size(); i++) {
        mean(i) = gps_[i].predictMean(x, t);
    }
    return mean;
}
