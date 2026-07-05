#include "gp.h"

using namespace cmp::gp;

void GaussianProcess::compute(const Eigen::Ref<const Eigen::VectorXd> &par) {

    // Store the parameters and scale them
    par_ = par;

    // Compute the covariance matrix
    Eigen::MatrixXd cov = covariance(par);

    // Compute the decomposition
    covDecomposition_.compute(cov);

    // Store the residuals
    residual_ = residual(par_);

    // Compute the solution of [K]x=r for fast prediction
    alpha_ = covDecomposition_.solve(residual_);

    // Compute the inverse of the covariance matrix and store its diagonal
    diagCovInverse_ = covDecomposition_.solve(Eigen::MatrixXd::Identity(pXObs_->rows(), pXObs_->rows())).diagonal();
}

cmp::gp::GaussianProcess::GaussianProcess() {
    set(cmp::covariance::Constant::make(0), cmp::mean::Zero::make(), Eigen::VectorXd::Zero(1), 1e-8);
}

cmp::gp::GaussianProcess::GaussianProcess(const std::shared_ptr<covariance::Covariance>& kernel, const std::shared_ptr<mean::Mean>& mean, Eigen::Ref<const Eigen::VectorXd> params, double nugget) {
    set(kernel, mean, params, nugget);
}

cmp::gp::GaussianProcess::GaussianProcess(const GaussianProcess& other) {

    // Copy all members
    par_ = other.par_;
    pKernel_ = other.pKernel_;
    pMean_ = other.pMean_;
    nugget_ = other.nugget_;

    // Copy the scaler and normalization flag
    normalizeY_ = other.normalizeY_;
    yScaler_ = other.yScaler_;

    // Check if it is conditioned
    if(other.pXObs_.has_value()) {
        condition(other.pXObs_.value(), other.pYObs_.value(), other.xObs_.has_value());
    } else {

        // We do not condition
    }
}

GaussianProcess& cmp::gp::GaussianProcess::operator=(const GaussianProcess& other) {
    if(this != &other) {

        // Copy all members
        par_ = other.par_;
        pKernel_ = other.pKernel_;
        pMean_ = other.pMean_;
        nugget_ = other.nugget_;

        // Check if it is conditioned
        if(other.pXObs_.has_value()) {
            condition(other.pXObs_.value(), other.pYObs_.value(), other.xObs_.has_value());
        } else {
            // We do not condition
        }
    }
    return *this;
}

cmp::gp::GaussianProcess::GaussianProcess(GaussianProcess&& other) noexcept {

    // Move all members
    par_ = std::move(other.par_);
    pKernel_ = std::move(other.pKernel_);
    pMean_ = std::move(other.pMean_);
    nugget_ = other.nugget_;

    // Check if it is conditioned
    if(other.pXObs_.has_value()) {
        condition(other.pXObs_.value(), other.pYObs_.value(), other.xObs_.has_value());
    } else {
        // We do not condition
    }
}

GaussianProcess& cmp::gp::GaussianProcess::operator=(GaussianProcess&& other) noexcept {
    if(this != &other) {

        // Move all members
        par_ = std::move(other.par_);
        pKernel_ = std::move(other.pKernel_);
        pMean_ = std::move(other.pMean_);
        nugget_ = other.nugget_;

        // Check if it is conditioned
        if(other.pXObs_.has_value()) {
            condition(other.pXObs_.value(), other.pYObs_.value(), other.xObs_.has_value());
        } else {
            // We do not condition
        }
    }
    return *this;
}

void cmp::gp::GaussianProcess::set(const std::shared_ptr<covariance::Covariance>& kernel, const std::shared_ptr<mean::Mean>& mean, Eigen::Ref<const Eigen::VectorXd> params, double nugget) {

    // Verify the inputs
    assert(kernel != nullptr && "The Kernel function must be set.");
    assert(mean != nullptr && "The mean function must be set.");
    assert(nugget >= 0 && "The nugget must be non-negative.");

    // Set kernel and mean
    pKernel_ = kernel;
    pMean_ = mean;

    // Set hyperparameters
    par_ = params;

    nugget_ = nugget;

    // Reset the observations
    pXObs_.reset();
    pYObs_.reset();
    xObs_.reset();
    yObs_.reset();

    // Reset the computed quantities
    covDecomposition_ = Eigen::LDLT<Eigen::MatrixXd>();
    alpha_ = Eigen::VectorXd();
    diagCovInverse_ = Eigen::VectorXd();
    residual_ = Eigen::VectorXd();

}

void cmp::gp::GaussianProcess::condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd>& yObs, bool copyData, bool normalizeY) {

    assert(yObs.rows() == xObs.rows() && "X and Y must have the same length.");

    if(normalizeY && !copyData) {
        throw std::invalid_argument("Cannot normalize Y when copyData is false. Set copyData to true to normalize Y.");
    }

    // Set the normalization flag
    normalizeY_ = normalizeY;

    // Check whether it owns the data or not
    if(copyData) {

        if(normalizeY) {
            // Normalize the data
            Eigen::VectorXd yNorm = yScaler_.fit_transform(yObs);

            // Make a hard copy of the data
            xObs_ = xObs;
            yObs_ = yNorm;

            // Set internal pointer to the storage (create new Ref objects on heap)
            pXObs_.emplace(xObs_.value());
            pYObs_.emplace(yObs_.value());
        } else {
            // Make a hard copy of the data
            xObs_ = xObs;
            yObs_ = yObs;

            // Set internal pointer to the storage (create new Ref objects on heap)
            pXObs_.emplace(xObs_.value());
            pYObs_.emplace(yObs_.value());
        }
    } else {

        // Just point to the external data
        pXObs_.emplace(xObs);
        pYObs_.emplace(yObs);
    }

    // Now we compute the quantities needed for prediction
    compute(par_);
}

/*
* FUNCTIONS FOR THE KERNEL
*/
Eigen::MatrixXd GaussianProcess::covariance(Eigen::Ref<const Eigen::VectorXd> par) const {

    // Computation of the kernel matrix
    size_t nObs = pXObs_->rows();
    Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(nObs, nObs);

    #pragma omp parallel for if(nObs > 100) default(none) shared(pXObs_, par, kernel_mat, nObs)
    for(size_t i = 0; i < nObs; i++) {
        for(size_t j = i; j < nObs; j++) {
            kernel_mat(i, j) = pKernel_->eval(pXObs_->row(i).transpose(), pXObs_->row(j).transpose(), par);

            // Kernel matrix is symmetric
            if(i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat + nugget_ * Eigen::MatrixXd::Identity(nObs, nObs);
}

Eigen::MatrixXd GaussianProcess::covarianceGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &n) const {

    // Initialize the kernel derivative
    size_t nObs = pXObs_->rows();
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(nObs, nObs);

    // Fill the matrix that contains the kernel derivative
    #pragma omp parallel for if(nObs > 100) default(none) shared(pXObs_, par, k_der, n, nObs)
    for(size_t i = 0; i < nObs; i++) {
        for(size_t j = i; j < nObs; j++) {

            k_der(i, j) = pKernel_->evalGradient(pXObs_->row(i).transpose(), pXObs_->row(j).transpose(), par, n);

            // The matrix is symmetric
            if(i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

Eigen::MatrixXd GaussianProcess::covarianceHessian(Eigen::Ref<const Eigen::VectorXd> par, const size_t &l, const size_t &k) const {

    // Initialize the kernel hessian
    size_t nObs = pXObs_->rows();
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(nObs, nObs);

    // Fill the matrix that contains the kernel hessian
    #pragma omp parallel for if(nObs > 100) default(none) shared(pXObs_, par, k_der, l, k, nObs)
    for(size_t i = 0; i < nObs; i++) {
        for(size_t j = i; j < nObs; j++) {

            k_der(i, j) = pKernel_->evalHessian(pXObs_->row(i), pXObs_->row(j), par, l, k);

            // The matrix is symmetric
            if(i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}


/*
* FUNCTIONS FOR THE MEAN
*/
Eigen::VectorXd GaussianProcess::priorMean(Eigen::Ref<const Eigen::VectorXd> par) const {
    Eigen::VectorXd y(pXObs_->rows());

    for(size_t i = 0; i < pXObs_->rows(); i++) {
        y(i) = pMean_->eval(pXObs_->row(i).transpose(), par);
    }

    return y;
}

Eigen::VectorXd GaussianProcess::priorMeanGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &i) const {
    Eigen::VectorXd y(pXObs_->rows());
    for(size_t j = 0; j < pXObs_->rows(); j++) {
        y(j) = pMean_->evalGradient(pXObs_->row(j).transpose(), par, i);
    }

    return y;
}

Eigen::VectorXd GaussianProcess::residual(Eigen::Ref<const Eigen::VectorXd> par) const {

    Eigen::VectorXd res(pXObs_->rows());
    for(size_t i = 0; i < pXObs_->rows(); i++) {
        res(i) = pYObs_.value()(i) - pMean_->eval(pXObs_->row(i).transpose(), par);
    }
    return res;
}

/**
 * * FIT THE GP
 */

void GaussianProcess::fit(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, const method &method, const nlopt::algorithm &alg, const double &tol_rel, bool copyData, bool normalizeY, const std::shared_ptr<cmp::prior::Prior> &prior, const std::vector<bool> &logScale) {

    // First we condition the GP on the observations
    condition(xObs, yObs, copyData, normalizeY);

    // Set the optimization function
    switch(method) {
    case MLE: {
        // Create the function and the functor
        auto obj = [&](const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad) {
            return objectiveFunction(x, grad, prior);
        };
        // Optimize
        cmp::ObjectiveFunctor fun(obj);

        if(!logScale.empty()) {
            fun.setLogScale(logScale);
        }

        cmp::nlopt_max(fun, par_, lb, ub, alg, tol_rel);
        break;
    }
    case LOO: {
        // Create the function and the functor
        auto obj = [&](const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad) {
            return objectiveFunctionLOO(x, grad, prior);
        };
        // Optimize
        cmp::ObjectiveFunctor fun(obj);
        if(!logScale.empty()) {
            fun.setLogScale(logScale);
        }
        cmp::nlopt_max(fun, par_, lb, ub, alg, tol_rel);
        break;
    }
    case LOO_MSE: {
        // Create the function and the functor
        auto obj = [&](const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad) {
            return objectiveFunctionLOOMSE(x, grad, prior);
        };
        // Optimize
        cmp::ObjectiveFunctor fun(obj);
        if(!logScale.empty()) {
            fun.setLogScale(logScale);
        }
        cmp::nlopt_max(fun, par_, lb, ub, alg, tol_rel);
        break;
    }
    default:
        throw std::runtime_error("Unknown optimization method");
    }

    // Now we can compute the covariance matrix and its decomposition with the optimized parameters
    compute(par_);
}


/**
 * * MAKING PREDICTIONS
 */

std::pair<double, double> GaussianProcess::predict(const Eigen::Ref<const Eigen::VectorXd> &x, type predictionType) const {

    // If prior prediction or GP not conditioned, return the prior prediction
    if(predictionType == type::PRIOR || !pXObs_.has_value()) {
        double prior_mean = pMean_->eval(x, par_);
        double prior_var = pKernel_->eval(x, x, par_) + nugget_;

        if(normalizeY_) {
            prior_mean = yScaler_.inverseTransform(prior_mean * cmp::ScalarOne)(0);
            prior_var *= yScaler_.getScale()(0, 0) * yScaler_.getScale()(0, 0);
        }

        return std::make_pair(prior_mean, prior_var);
    } else {
        // Compute the dot product (see Rasmussen and Williams)
        double k_star_dot_alpha = 0.0;
        for(size_t i = 0; i < pXObs_->rows(); i++) {
            k_star_dot_alpha += pKernel_->eval(x, pXObs_->row(i), par_) * alpha_(i);
        }

        // Compute the kernel matrix
        Eigen::VectorXd k_star = Eigen::VectorXd::Zero(pXObs_->rows());
        for(size_t i = 0; i < pXObs_->rows(); i++) {
            k_star(i) = pKernel_->eval(x, pXObs_->row(i), par_);
        }

        // Compute the variance
        double var = pKernel_->eval(x, x, par_) - k_star.dot(covDecomposition_.solve(k_star)) + nugget_;
        double mean = pMean_->eval(x, par_) + k_star_dot_alpha;

        if(normalizeY_) {
            mean = yScaler_.inverseTransform(mean * cmp::ScalarOne)(0);
            var *= yScaler_.getScale()(0, 0) * yScaler_.getScale()(0, 0);
        }

        return std::make_pair(mean, var);
    }
}

double GaussianProcess::predictMean(const Eigen::Ref<const Eigen::VectorXd> &x, type predictionType) const {

    if(predictionType == type::PRIOR || !pXObs_.has_value()) {

        double mean = pMean_->eval(x, par_);

        if(normalizeY_) {
            mean = yScaler_.inverseTransform(mean * cmp::ScalarOne)(0);
        }

        return mean;
    } else {
        double k_star_dot_alpha = 0.0;
        for(size_t i = 0; i < pXObs_->rows(); i++) {
            k_star_dot_alpha += pKernel_->eval(x, pXObs_->row(i), par_) * alpha_(i);
        }

        double mean = pMean_->eval(x, par_) + k_star_dot_alpha;

        if(normalizeY_) {
            mean = yScaler_.inverseTransform(mean * cmp::ScalarOne)(0);
        }

        return mean;
    }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> cmp::gp::GaussianProcess::predictMultiple(const Eigen::Ref<const Eigen::MatrixXd> &x_pts, type predictionType) const {

    if(predictionType == type::PRIOR || !pXObs_.has_value()) {
        // Sizes of the matrices
        size_t n_pts = x_pts.rows();

        // Initialize variables
        Eigen::MatrixXd k_xx(n_pts, n_pts);
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_pts);

        #pragma omp parallel for if(n_pts > 100) default(none) shared(x_pts, par_, mean, k_xx, n_pts)
        for(size_t i = 0; i < n_pts; i++) {

            // Evaluate the prior mean
            mean(i) = pMean_->eval(x_pts.row(i), par_);

            // Fill k_xy
            for(size_t j = i; j < n_pts; j++) {
                k_xx(i, j) = pKernel_->eval(x_pts.row(i), x_pts.row(j), par_);

                // Tensor must be symmetric
                if(i != j) {
                    k_xx(j, i) = k_xx(i, j);
                }
            }

            k_xx(i, i) += nugget_;
        }

        if(normalizeY_) {
            for(size_t i = 0; i < n_pts; i++) {
                mean(i) = yScaler_.inverseTransform(mean(i) * cmp::ScalarOne)(0);
                k_xx.row(i) *= yScaler_.getScale()(0, 0) * yScaler_.getScale()(0, 0);
            }
        }

        return std::make_pair(mean, k_xx);
    } else {
        size_t n_pts = x_pts.rows();

        // Initialize variables
        Eigen::MatrixXd k_star(pXObs_->rows(), n_pts);
        Eigen::MatrixXd k_xx(n_pts, n_pts);
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_pts);

        for(size_t i = 0; i < n_pts; i++) {

            // Evaluate the prior mean
            mean(i) = pMean_->eval(x_pts.row(i), par_);

            // Fill k_star and compute the mean
            for(size_t j = 0; j < pXObs_->rows(); j++) {
                k_star(j, i) = pKernel_->eval(pXObs_->row(j), x_pts.row(i), par_);
                mean(i) += k_star(j, i) * alpha_(j);
            }

            // Fill k_xy
            for(size_t j = i; j < n_pts; j++) {
                k_xx(i, j) = pKernel_->eval(x_pts.row(i), x_pts.row(j), par_);

                // Tensor must be symmetric
                if(i != j) {
                    k_xx(j, i) = k_xx(i, j);
                }
            }

        }

        Eigen::MatrixXd cov = k_xx - k_star.transpose() * covDecomposition_.solve(k_star) + nugget_ * Eigen::MatrixXd::Identity(n_pts, n_pts);

        if(normalizeY_) {
            for(size_t i = 0; i < n_pts; i++) {
                mean(i) = yScaler_.inverseTransform(mean(i) * cmp::ScalarOne)(0);
            }

            cov *= yScaler_.getScale()(0, 0) * yScaler_.getScale()(0, 0);
        }

        return std::make_pair(mean, cov);
    }
}

Eigen::VectorXd cmp::gp::GaussianProcess::predictMeanMultiple(const Eigen::Ref<const Eigen::MatrixXd>& x_pts, type predictionType) const {

    if(predictionType == type::PRIOR || !pXObs_.has_value()) {
        // Sizes of the matrices
        size_t n_pts = x_pts.rows();

        // Initialize variables
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_pts);

        for(size_t i = 0; i < n_pts; i++) {

            // Evaluate the prior mean
            mean(i) = pMean_->eval(x_pts.row(i), par_);

            if(normalizeY_) {
                mean(i) = yScaler_.inverseTransform(mean(i) * cmp::ScalarOne)(0);
            }
        }

        return mean;
    }  else {
        size_t n_pts = x_pts.rows();

        // Initialize variables
        Eigen::MatrixXd k_star(pXObs_->rows(), n_pts);
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_pts);

        for(size_t i = 0; i < n_pts; i++) {

            // Evaluate the prior mean
            mean(i) = pMean_->eval(x_pts.row(i), par_);

            // Fill k_star and compute the mean
            for(size_t j = 0; j < pXObs_->rows(); j++) {
                k_star(j, i) = pKernel_->eval(pXObs_->row(j), x_pts.row(i), par_);
                mean(i) += k_star(j, i) * alpha_(j);
            }

            // Inverse transform once per target point
            if(normalizeY_) {
                mean(i) = yScaler_.inverseTransform(mean(i) * cmp::ScalarOne)(0);
            }
        }

        return mean;
    }
}

std::pair<double, double> cmp::gp::GaussianProcess::predictLOO(const size_t& i) const {

    if(!pXObs_.has_value()) {
        throw std::runtime_error("The GP must be conditioned to make LOO predictions");
    } else if(i >= pXObs_->rows()) {
        throw std::out_of_range("Index out of range");
    }

    double loo_mean = pYObs_.value()(i) - alpha_(i) / diagCovInverse_(i);
    double loo_var = 1.0 / diagCovInverse_(i);

    if(normalizeY_) {
        loo_mean = yScaler_.inverseTransform(loo_mean * cmp::ScalarOne)(0);
        loo_var *= yScaler_.getScale()(0, 0) * yScaler_.getScale()(0, 0);
    }

    return std::pair<double, double>(loo_mean, loo_var);
}
double GaussianProcess::logLikelihood() const {
    if(!pXObs_.has_value()) {
        throw std::runtime_error("The GP must be conditioned to compute the log-likelihood");
    }
    return cmp::distribution::MultivariateNormalDistribution::logPDF(residual_, covDecomposition_);
}
double GaussianProcess::logLikelihoodLOO(const size_t &i) const {
    if(!pXObs_.has_value()) {
        throw std::runtime_error("The GP must be conditioned to compute the LOO log-likelihood");
    } else if(i >= pXObs_->rows()) {
        throw std::out_of_range("Index out of range");
    }

    auto [mean, var] = predictLOO(i);

    double yTrue = pYObs_.value()(i);
    if(normalizeY_) {
        yTrue = yScaler_.inverseTransform(yTrue * cmp::ScalarOne)(0);
    }
    return cmp::distribution::NormalDistribution::logPDF(yTrue - mean, std::sqrt(var));
}


Eigen::MatrixXd GaussianProcess::expectedVarianceImprovement(
    const Eigen::Ref<const Eigen::MatrixXd> &x_pts,
    const Eigen::Ref<const Eigen::MatrixXd> &x_pending,
    double nu) const {

    // Sizes of the matrices
    size_t n_pts = x_pts.rows();
    size_t n_obs = pXObs_->rows();
    size_t n_pending = x_pending.rows();

    // Initialize kernel blocks for candidate points
    Eigen::MatrixXd Kop(n_obs, n_pts);
    Eigen::MatrixXd Kpp(n_pts, n_pts);

    // Initialize kernel blocks for pending batch points
    Eigen::MatrixXd K_op_pending(n_obs, n_pending);
    Eigen::MatrixXd K_pending_pts(n_pending, n_pts);
    Eigen::MatrixXd K_pending_pending(n_pending, n_pending);

    // Initialize output matrix
    Eigen::MatrixXd variance_reduction_matrix(n_pts, n_pts);

    // Fill Kop = K(X_obs, X_pts)
    for(size_t i = 0; i < n_pts; i++) {
        for(size_t j = 0; j < n_obs; j++) {
            Kop(j, i) = pKernel_->eval(pXObs_->row(j), x_pts.row(i), par_);
        }
    }

    // Fill Kpp = K(X_pts, X_pts)
    for(size_t i = 0; i < n_pts; i++) {
        for(size_t j = i; j < n_pts; j++) {
            Kpp(i, j) = pKernel_->eval(x_pts.row(i), x_pts.row(j), par_);
            if(i != j) {
                Kpp(j, i) = Kpp(i, j);
            }
        }
    }

    // Fill K_op_pending = K(X_obs, X_pending)
    for(size_t i = 0; i < n_pending; i++) {
        for(size_t j = 0; j < n_obs; j++) {
            K_op_pending(j, i) = pKernel_->eval(pXObs_->row(j), x_pending.row(i), par_);
        }
    }

    // Fill K_pending_pts = K(X_pending, X_pts)
    for(size_t i = 0; i < n_pts; i++) {
        for(size_t j = 0; j < n_pending; j++) {
            K_pending_pts(j, i) = pKernel_->eval(x_pending.row(j), x_pts.row(i), par_);
        }
    }

    // Fill K_pending_pending = K(X_pending, X_pending)
    for(size_t i = 0; i < n_pending; i++) {
        for(size_t j = i; j < n_pending; j++) {
            K_pending_pending(i, j) = pKernel_->eval(x_pending.row(i), x_pending.row(j), par_);
            if(i != j) {
                K_pending_pending(j, i) = K_pending_pending(i, j);
            }
        }
    }

    // Solve K_oo^{-1} * K_op once for all candidate points
    Eigen::MatrixXd Ksol = covDecomposition_.solve(Kop);

    // A = K_po K_oo^{-1} K_op
    Eigen::MatrixXd A = Kop.transpose() * Ksol;

    // Rho matrix for all candidate points (Prior to pending points)
    Eigen::MatrixXd rho = Kpp - A;

    // Solve K_oo^{-1} * K_op_pending for the pending points
    Eigen::MatrixXd Ksol_pending = covDecomposition_.solve(K_op_pending);

    // Compute posterior covariances involving the pending points
    Eigen::MatrixXd Sigma_pending_pts = K_pending_pts - (K_op_pending.transpose() * Ksol);
    Eigen::MatrixXd Sigma_pending_pending = K_pending_pending - (K_op_pending.transpose() * Ksol_pending);

    // Add observation noise nu to the diagonal of the pending covariance
    Sigma_pending_pending.diagonal().array() += nu;

    // Decompose the pending covariance matrix once
    auto pending_decomp = Sigma_pending_pending.ldlt();

    // Pre-calculate the influence of the batch: beta = (Sigma_pending_pending + nu*I)^{-1} * Sigma_pending_pts
    Eigen::MatrixXd beta = pending_decomp.solve(Sigma_pending_pts);

    // Update rho to rho_tilde: This is the covariance AFTER observing x_pending
    Eigen::MatrixXd rho_tilde = rho - (Sigma_pending_pts.transpose() * beta);

    // Select a point and compute the variance reduction assuming that it is observed
    for(size_t sel = 0; sel < n_pts; sel++) {

        // Compute sigma_n using the updated covariance
        double sigma_n = std::abs(rho_tilde(sel, sel)) + nu;

        // The variance reduction is computed here (vectorized for speed)
        variance_reduction_matrix.row(sel) = rho_tilde.col(sel).array().square() / sigma_n;
    }

    return variance_reduction_matrix;
}


double GaussianProcess::objectiveFunction(const Eigen::Ref<const Eigen::VectorXd> &par, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &lp) {

    // --- MATHEMATICAL FORMULATION ---
    // The objective is to maximize the Marginal Log-Likelihood (MLL):
    // log p(y | X, theta) = -1/2 * r^T * K_y^-1 * r - 1/2 * log|K_y| - N/2 * log(2pi)
    // where r = y - m(X) represents the residual vector.
    
    // We compute the LDLT decomposition: K_y = L * D * L^T.
    // This allows stable computation of the quadratic term and the log-determinant (sum of log of D's diagonal).
    Eigen::LDLT<Eigen::MatrixXd> cov_ldlt = cmp::ldltDecomposition(covariance(par));
    Eigen::VectorXd res = residual(par);

    // Compute the log probability density under the multivariate normal distribution.
    double ll = cmp::distribution::MultivariateNormalDistribution::logPDF(res, cov_ldlt);

    // --- ANALYTICAL GRADIENT DERIVATION ---
    // If the gradient vector is non-empty, we compute the partial derivatives:
    // d(MLL)/d(theta_n) = 1/2 * tr((alpha * alpha^T - K_y^-1) * d(K_y)/d(theta_n)) + (d(m(X))/d(theta_n))^T * alpha
    // where alpha = K_y^-1 * r.
    if(grad.size() != 0) {

        // Compute derivatives with respect to each hyperparameter theta_n
        for(int n = 0; n < par.size(); n++) {
            auto cov_gradient = covarianceGradient(par, n);
            auto mean_gradient = priorMeanGradient(par, n);
            
            // Evaluates the derivative of the MVN log-likelihood analytically
            grad[n] = cmp::distribution::MultivariateNormalDistribution::dLogPDF(res, cov_ldlt, cov_gradient, mean_gradient);

            // Add the prior gradient to incorporate prior distributions (MAP estimation)
            // d(Log-Posterior)/d(theta_n) = d(Log-Likelihood)/d(theta_n) + d(Log-Prior)/d(theta_n)
            grad[n] += lp->evalGradient(par, n);
        }
    }

    // Return the log-posterior: log p(theta | y, X) = log p(y | X, theta) + log p(theta)
    return ll + lp->eval(par);
}
double GaussianProcess::objectiveFunctionLOO(const Eigen::Ref<const Eigen::VectorXd> &par, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &lp) {

    // --- MATHEMATICAL FORMULATION (LEAVE-ONE-OUT CROSS-VALIDATION) ---
    // Leave-One-Out cross-validation evaluates the model's predictive capability by training on N-1 points
    // and predicting on the remaining point. Doing this N times is O(N^4).
    // Instead, we use Dubrule's fast LOO formulas (derived from matrix block inversion):
    // LOO residual: r_i = alpha_i / (K^-1)_ii
    // LOO variance: sigma_i^2 = 1 / (K^-1)_ii
    // where alpha = K^-1 * res, and res = y - m(X).
    // The total LOO log-likelihood is:
    // log p_LOO = sum_{i=1}^N [ -1/2 * log(2pi * sigma_i^2) - 1/2 * r_i^2 / sigma_i^2 ]

    const double log2pi = std::log(2.0 * M_PI);
    const double eps = 1e-12; // Numerical safety floor to prevent division by zero

    // Compute covariance matrix and its LDLT decomposition
    Eigen::MatrixXd K = covariance(par);
    Eigen::LDLT<Eigen::MatrixXd> ldlt = cmp::ldltDecomposition(K);
    const Eigen::Index n = K.rows();

    // Compute residuals and solve K * alpha = res
    Eigen::VectorXd res = residual(par);
    Eigen::VectorXd alpha = ldlt.solve(res);

    // Explicitly compute K^-1 to get its diagonal elements (K^-1)_ii
    Eigen::MatrixXd Kinv = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    Eigen::VectorXd v = Kinv.diagonal();

    // Floor the diagonal elements to avoid numerical singularites
    Eigen::VectorXd v_safe = v.array().max(eps);
    Eigen::VectorXd sigma2 = (1.0 / v_safe.array()).matrix();   // LOO predictive variances
    Eigen::VectorXd r = (alpha.array() / v_safe.array()).matrix(); // LOO residuals

    // Evaluate the pointwise LOO log probability densities
    Eigen::VectorXd loo_ll = (-0.5 * (log2pi + sigma2.array().log()) - 0.5 * (r.array().square() / sigma2.array())).matrix();
    double total_loo_ll = loo_ll.sum();

    // --- ANALYTICAL GRADIENT DERIVATION FOR LOO ---
    // If the gradient is requested, we compute derivatives:
    // d(log p_LOO)/d(theta_j) = sum_{i=1}^N [ -1/(2 * sigma_i^2) * d(sigma_i^2)/d(theta_j) - r_i/sigma_i^2 * d(r_i)/d(theta_j) + r_i^2/(2 * sigma_i^4) * d(sigma_i^2)/d(theta_j) ]
    // which simplifies using the derivatives of alpha and the diagonal of K^-1.
    if(grad.size() != 0) {
        const int p = static_cast<int>(par.size());
        grad.setZero();

        for(int j = 0; j < p; ++j) {
            Eigen::MatrixXd dK = covarianceGradient(par, j);   // dK = ∂K/∂θ_j
            Eigen::VectorXd dres = -priorMeanGradient(par, j);   // dres = ∂res/∂θ_j

            // helpers:
            // u = K^{-1} (dK * alpha)  --> used because ∂α has -K^{-1} dK α
            Eigen::VectorXd u = Kinv * (dK * alpha);

            // K^{-1} * dres  --> added term from mean/residual derivative
            Eigen::VectorXd Kinv_dres = Kinv * dres;

            // M = K^{-1} dK K^{-1}  (used for ∂v_i = -M_{ii})
            Eigen::MatrixXd M = Kinv * dK * Kinv;

            double g_j = 0.0;
            for(Eigen::Index i = 0; i < n; ++i) {
                double vi = v[i];
                double vi_safe = std::max(vi, eps); // use safe denom when dividing
                double si2 = 1.0 / vi_safe;         // sigma^2
                double ri = alpha[i] / vi_safe;     // r_i (matches r above)

                // derivative of alpha_i: ∂α_i = -u_i + (K^{-1} dres)_i
                double d_alpha_i = -u[i] + Kinv_dres[i];

                // derivative of v_i: ∂v_i = -M_{ii}
                double d_vi = -M(i, i);

                // derivative of r_i and sigma2
                double d_ri = (d_alpha_i * vi - alpha[i] * d_vi) / (vi * vi);
                double d_sigma2 = -d_vi / (vi * vi);

                // derivative of per-point log-likelihood:
                // ∂ℓ_i = -1/2 * ( (1/σ2) ∂σ2 + (2 r / σ2) ∂r - (r^2 / σ2^2) ∂σ2 )
                double term = -0.5 * ((1.0 / si2) * d_sigma2 + (2.0 * ri / si2) * d_ri - (ri * ri / (si2 * si2)) * d_sigma2);
                g_j += term;
            }
            grad[j] = g_j + lp->evalGradient(par, j);
        }
    }

    // NOTE: if your optimizer *minimizes* (typical), return -total_loo_ll and set grad *= -1.
    return total_loo_ll + lp->eval(par);
}

double GaussianProcess::objectiveFunctionLOOMSE(const Eigen::Ref<const Eigen::VectorXd> &par, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &lp) {

    const double eps = 1e-12;

    // Covariance, factorization, sizes
    Eigen::MatrixXd K = covariance(par);
    Eigen::LDLT<Eigen::MatrixXd> ldlt = cmp::ldltDecomposition(K);
    const Eigen::Index n = K.rows();

    // residuals and alpha = K^{-1} res
    Eigen::VectorXd res = residual(par);
    Eigen::VectorXd alpha = ldlt.solve(res);

    // Full inverse for diagonal terms and derivatives
    Eigen::MatrixXd Kinv = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    Eigen::VectorXd v = Kinv.diagonal();
    Eigen::VectorXd v_safe = v.array().max(eps);

    // LOO residuals and variances:
    // y_i - mu_{-i}(x_i) = alpha_i / v_i, sigma_{-i}^2 = 1 / v_i
    Eigen::VectorXd loo_residual = (alpha.array() / v_safe.array()).matrix();
    Eigen::VectorXd loo_var = (1.0 / v_safe.array()).matrix();

    // MSE-like criterion requested by user: sum_i (y_i - mu_{-i}(x_i))^2 + sigma_{-i}^2
    double loo_mse = loo_residual.squaredNorm() + loo_var.sum();

    if(grad.size() != 0) {
        const int p = static_cast<int>(par.size());
        grad.setZero();

        for(int j = 0; j < p; ++j) {
            Eigen::MatrixXd dK = covarianceGradient(par, j);
            Eigen::VectorXd dres = -priorMeanGradient(par, j);

            // d alpha = -K^{-1} dK alpha + K^{-1} dres
            Eigen::VectorXd d_alpha = -(Kinv * (dK * alpha)) + (Kinv * dres);

            // d v_i = -[K^{-1} dK K^{-1}]_{ii}
            Eigen::MatrixXd M = Kinv * dK * Kinv;

            double d_loo_mse = 0.0;
            for(Eigen::Index i = 0; i < n; ++i) {
                const double vi = v_safe[i];
                const double ri = alpha[i] / vi;
                const double d_vi = -M(i, i);
                const double d_ri = (d_alpha[i] * vi - alpha[i] * d_vi) / (vi * vi);
                const double d_sigma2_i = -d_vi / (vi * vi);

                d_loo_mse += 2.0 * ri * d_ri + d_sigma2_i;
            }

            // We maximize, so negate the criterion gradient and add prior gradient.
            grad[j] = -d_loo_mse + lp->evalGradient(par, j);
        }
    }

    // We maximize, so negate the criterion value and add log-prior.
    return -loo_mse + lp->eval(par);
}
