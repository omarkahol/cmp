#include <kde.h>

cmp::density::KDE::KDE() {
    set(kernel::IsotropicBandwidth::make(1.0, 1), kernel::Gaussian::make());
}

cmp::density::KDE::KDE(std::shared_ptr<kernel::Bandwidth> bandwidth, std::shared_ptr<kernel::Kernel> kernel) {
    set(bandwidth, kernel);
}

cmp::density::KDE::KDE(const KDE& other) {
    set(other.bandwidth_, other.kernel_);
    condition(other.pData_.value(), other.isOwning_);
}

cmp::density::KDE::KDE(KDE&& other) noexcept {
    set(std::move(other.bandwidth_), std::move(other.kernel_));
    condition(std::move(other.pData_.value()), other.isOwning_);
}

cmp::density::KDE& cmp::density::KDE::operator=(const KDE& other) {
    if(this != &other) {
        set(other.bandwidth_, other.kernel_);
        condition(other.pData_.value(), other.isOwning_);
    }
    return *this;
}

cmp::density::KDE& cmp::density::KDE::operator=(KDE&& other) noexcept {
    if(this != &other) {
        set(std::move(other.bandwidth_), std::move(other.kernel_));
        condition(std::move(other.pData_.value()), other.isOwning_);
    }
    return *this;
}

void cmp::density::KDE::set(const std::shared_ptr<kernel::Bandwidth> &bandwidth, const std::shared_ptr<kernel::Kernel> &kernel) {

    bandwidth_ = bandwidth;
    kernel_ = kernel;
}

void cmp::density::KDE::condition(const Eigen::Ref<const Eigen::MatrixXd> &data, bool copyData) {
    isOwning_ = copyData;

    if(copyData) {
        data_ = data;
        dim_ = data.cols();
        nPoints_ = data.rows();
        pData_.emplace(data_);
    } else {
        dim_ = data.cols();
        nPoints_ = data.rows();
        pData_.emplace(data);
    }
}

// KDE
double cmp::density::KDE::eval(const Eigen::VectorXd& x) const {

    double result = 0.0;
    for(size_t i = 0; i < nPoints_; ++i) {
        Eigen::VectorXd z = bandwidth_->apply(pData_->row(i).transpose(), x);
        result += kernel_->eval(z);
    }

    double norm_const = kernel_->normalizationConstant(dim_);
    double determinant = bandwidth_->determinant();
    return result * norm_const * determinant / static_cast<double>(nPoints_);
}

Eigen::MatrixXd cmp::density::bandwidthSelectionRule(const KDE::BandWidthSelectionMethod& method, Eigen::Ref<Eigen::MatrixXd> xObs) {

    if(method == KDE::BandWidthSelectionMethod::SCOTT) {

        // Check if we have observations set
        if(xObs.size() == 0) {
            std::cerr << "No observations set for KDE. Cannot estimate bandwidth using Scott's rule." << std::endl;
            return Eigen::MatrixXd::Identity(1, 1);
        }

        // Compute the STD of the data
        Eigen::MatrixXd cov = cmp::statistics::covariance(xObs);

        return cov * std::pow(static_cast<double>(xObs.cols()), -1.0 / (xObs.rows() + 4.0));

    } else if(method == KDE::BandWidthSelectionMethod::SILVERMAN) {
        if(xObs.rows() == 0) {
            std::cerr << "No observations set for KDE. Cannot estimate bandwidth using Silverman's rule." << std::endl;
            return Eigen::MatrixXd::Identity(1, 1);
        }

        // IQR
        Eigen::VectorXd iqr = cmp::statistics::interQuantileRange(xObs, 0.25, 0.75);

        // Covariance and std dev
        Eigen::MatrixXd cov = cmp::statistics::covariance(xObs);
        Eigen::VectorXd sigma = cov.diagonal().array().sqrt();

        // Use min(sigma, IQR/1.349)
        for(int i = 0; i < sigma.size(); ++i) {
            if(iqr(i) > 0) {
                sigma(i) = std::min(sigma(i), iqr(i) / 1.349);
            }
        }

        int n = xObs.rows();   // samples
        int d = xObs.cols();   // dimension

        double factor = std::pow(n, -1.0 / (d + 4.0));
        return (0.9 * factor) * sigma.asDiagonal();
    }
}

void cmp::density::bandwidthOptimizationCrossValidation(
    const cmp::statistics::KFold& kf,
    Eigen::Ref<Eigen::MatrixXd> data,
    std::shared_ptr<cmp::kernel::Kernel> kernel,
    std::shared_ptr<cmp::kernel::Bandwidth> bandwidth,
    const double& min,
    const double& max,
    nlopt::algorithm alg,
    const double& tol) {
    const size_t n = data.rows();

    auto objective = [&](Eigen::Ref<const Eigen::VectorXd> params,
    Eigen::Ref<Eigen::VectorXd> grad) -> double {
        bandwidth->setFromVector(params);

        double logLikelihood = 0.0;
        Eigen::VectorXd computed_grad = Eigen::VectorXd::Zero(params.size());
        const double eps = 1e-12;

        for(auto [train_indices, test_indices] : kf) {
            Eigen::MatrixXd train_data = cmp::slice(data, train_indices);
            Eigen::MatrixXd test_data  = cmp::slice(data, test_indices);

            const double train_count = static_cast<double>(train_data.rows());
            const double norm_const  = kernel->normalizationConstant(bandwidth->size());
            const double determinant = bandwidth->determinant();
            const double scale       = norm_const * determinant / std::max(1.0, train_count);

            // process each test point
            for(Eigen::Index i = 0; i < test_data.rows(); ++i) {
                Eigen::VectorXd xi = test_data.row(i).transpose();

                // cache values for this test point
                std::vector<double> kvals(train_data.rows());
                std::vector<Eigen::VectorXd> zvals(train_data.rows());

                double raw_sum_i = 0.0;

                for(Eigen::Index j = 0; j < train_data.rows(); ++j) {
                    Eigen::VectorXd xj = train_data.row(j).transpose();

                    // cache z and kernel value
                    zvals[j] = bandwidth->apply(xi, xj);
                    double kVal = kernel->eval(zvals[j]);
                    kvals[j] = kVal;
                    raw_sum_i += kVal;
                }

                double point_likelihood = scale * raw_sum_i;
                logLikelihood += std::log(point_likelihood + eps);

                // gradient accumulation
                for(Eigen::Index p = 0; p < params.size(); ++p) {
                    double dsum = 0.0;

                    for(Eigen::Index j = 0; j < train_data.rows(); ++j) {
                        Eigen::VectorXd grad_z_p =
                            bandwidth->gradientOfApply(xi, train_data.row(j).transpose(), p);

                        // dK/dθ via chain rule
                        double dK = kernel->applyToGradient(zvals[j], grad_z_p);
                        dsum += dK;
                    }

                    // d log p_i / dθ = (∑ dK_j / ∑ K_j) + d log det / dθ
                    double dlog_raw = dsum / (raw_sum_i + eps);
                    double dlog_scale = bandwidth->gradientOfLogDeterminant(p);
                    computed_grad(p) += dlog_raw + dlog_scale;
                }
            }
        }

        // Check if gradient computation is needed
        if(grad.size() != 0) {
            grad = computed_grad;
        }
        return logLikelihood;  // maximise directly
    };



    Eigen::VectorXd initial_params = bandwidth->getParams();
    Eigen::VectorXd lower_bounds = Eigen::VectorXd::Constant(initial_params.size(), min);
    Eigen::VectorXd upper_bounds = Eigen::VectorXd::Constant(initial_params.size(), max);

    cmp::ObjectiveFunctor obj_functor(objective);
    cmp::nlopt_max(obj_functor, initial_params, lower_bounds, upper_bounds, alg, tol);
}
