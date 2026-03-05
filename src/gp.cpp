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

void cmp::gp::GaussianProcess::condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd>& yObs, bool copyData) {

    assert(yObs.rows() == xObs.rows() && "X and Y must have the same length.");

    // Check whether it owns the data or not
    if(copyData) {

        // Make a hard copy of teh data
        xObs_ = xObs;
        yObs_ = yObs;

        // Set internal pointer to the storage (create new Ref objects on heap)
        pXObs_.emplace(xObs_.value());
        pYObs_.emplace(yObs_.value());
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
    Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(pXObs_->rows(), pXObs_->rows());
    for(size_t i = 0; i < pXObs_->rows(); i++) {
        for(size_t j = i; j < pXObs_->rows(); j++) {
            kernel_mat(i, j) = pKernel_->eval(pXObs_->row(i).transpose(), pXObs_->row(j).transpose(), par);

            // Kernel matrix is symmetric
            if(i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat + nugget_ * Eigen::MatrixXd::Identity(pXObs_->rows(), pXObs_->rows());
}

Eigen::MatrixXd GaussianProcess::covarianceGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &n) const {

    // Initialize the kernel derivative
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(pXObs_->rows(), pXObs_->rows());

    // Fill the matrix that contains the kernel derivative
    for(size_t i = 0; i < pXObs_->rows(); i++) {
        for(size_t j = i; j < pXObs_->rows(); j++) {

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
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(pXObs_->rows(), pXObs_->rows());

    // Fill the matrix that contains the kernel hessian
    for(size_t i = 0; i < pXObs_->rows(); i++) {
        for(size_t j = i; j < pXObs_->rows(); j++) {

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

void GaussianProcess::fit(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, const method &method, const nlopt::algorithm &alg, const double &tol_rel, bool copyData, const std::shared_ptr<cmp::prior::Prior> &prior) {

    // First we condition the GP on the observations
    condition(xObs, yObs, copyData);

    // Set the optimization function
    switch(method) {
    case MLE: {
        // Create the function and the functor
        auto obj = [&](const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad) {
            return objectiveFunction(x, grad, prior);
        };
        // Optimize
        cmp::ObjectiveFunctor fun(obj);
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

        return std::make_pair(mean, var);
    }
}

double GaussianProcess::predictMean(const Eigen::Ref<const Eigen::VectorXd> &x, type predictionType) const {

    if(predictionType == type::PRIOR || !pXObs_.has_value()) {
        return pMean_->eval(x, par_);
    } else {
        double k_star_dot_alpha = 0.0;
        for(size_t i = 0; i < pXObs_->rows(); i++) {
            k_star_dot_alpha += pKernel_->eval(x, pXObs_->row(i), par_) * alpha_(i);
        }
        return pMean_->eval(x, par_) + k_star_dot_alpha;
    }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> cmp::gp::GaussianProcess::predictMultiple(const Eigen::Ref<const Eigen::MatrixXd> &x_pts, type predictionType) const {

    if(predictionType == type::PRIOR || !pXObs_.has_value()) {
        // Sizes of the matrices
        size_t n_pts = x_pts.rows();

        // Initialize variables
        Eigen::MatrixXd k_xx(n_pts, n_pts);
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_pts);

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

        auto cov = k_xx - k_star.transpose() * covDecomposition_.solve(k_star) + nugget_ * Eigen::MatrixXd::Identity(n_pts, n_pts);

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
        }

        return mean;
    } else {
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

    return std::pair<double, double>(pYObs_.value()(i) - alpha_(i) / diagCovInverse_(i), 1.0 / diagCovInverse_(i));
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
    return cmp::distribution::NormalDistribution::logPDF(pYObs_.value()(i) - mean, std::sqrt(var));
}

Eigen::MatrixXd GaussianProcess::expectedVarianceImprovement(const Eigen::Ref<const Eigen::MatrixXd> &x_pts, double nu) const {

    // Sizes of the matrices
    size_t n_pts = x_pts.rows();
    size_t n_obs = pXObs_->rows();

    // Initialize kernel blocks
    Eigen::MatrixXd Kop(n_obs, n_pts);
    Eigen::MatrixXd Kpp(n_pts, n_pts);
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

    // Solve K_oo^{-1} * K_op once for all candidate points
    Eigen::MatrixXd Ksol = covDecomposition_.solve(Kop);

    // A = K_po K_oo^{-1} K_op
    Eigen::MatrixXd A = Kop.transpose() * Ksol;

    // Rho matrix for all candidate points
    Eigen::MatrixXd rho = Kpp - A;

    // Select a point and compute the variance reduction assuming that it is observed
    for(size_t sel = 0; sel < n_pts; sel++) {

        // Compute sigma_n
        double sigma_n = std::abs(rho(sel, sel)) + nu;

        // The variance reduction is computed here
        for(size_t i = 0; i < n_pts; i++) {
            double rho_i = rho(i, sel);
            variance_reduction_matrix(sel, i) = rho_i * rho_i / sigma_n;
        }
    }

    return variance_reduction_matrix;
}

Eigen::VectorXd GaussianProcess::expectedVarianceImprovement(const Eigen::Ref<const Eigen::MatrixXd> &x_pts,
                                                             const Eigen::Ref<const Eigen::MatrixXd> &new_x_obs,
                                                             double nu,
                                                             double screeningCutoff) const {

    // Define the two kernel evaluation matrices
    size_t n_pts = x_pts.rows();
    size_t n_new_obs = new_x_obs.rows();
    size_t n_obs = pXObs_->rows();

    // No new observations means no variance reduction
    if(n_new_obs == 0) {
        return Eigen::VectorXd::Zero(n_pts);
    }

    Eigen::MatrixXd k_no = Eigen::MatrixXd::Zero(n_new_obs, n_obs);
    Eigen::MatrixXd k_nn = Eigen::MatrixXd::Zero(n_new_obs, n_new_obs);

    // Fill the matrices
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n_new_obs; i++) {
        for(size_t j = 0; j < n_obs; j++) {
            k_no(i, j) = pKernel_->eval(new_x_obs.row(i), pXObs_->row(j), par_);
        }
        for(size_t j = i; j < n_new_obs; j++) {
            k_nn(i, j) = pKernel_->eval(new_x_obs.row(i), new_x_obs.row(j), par_);

            // The matrix is symmetric
            if(i != j) {
                k_nn(j, i) = k_nn(i, j);
            }
        }
    }

    k_nn = k_nn - k_no * covDecomposition_.solve(k_no.transpose()) + nu * Eigen::MatrixXd::Identity(n_new_obs, n_new_obs);
    Eigen::LDLT<Eigen::MatrixXd> k_nn_ldlt(k_nn);

    // Chunked computation for large x_pts to reduce memory footprint and improve cache locality
    constexpr size_t blockSize = 1024;
    Eigen::VectorXd evi = Eigen::VectorXd::Zero(n_pts);

    for(size_t start = 0; start < n_pts; start += blockSize) {
        size_t b = std::min(blockSize, n_pts - start);

        Eigen::MatrixXd k_np = Eigen::MatrixXd::Zero(n_new_obs, b);
        std::vector<size_t> activeCols;
        activeCols.reserve(b);

        #pragma omp parallel for schedule(static)
        for(size_t ib = 0; ib < b; ib++) {
            size_t i = start + ib;
            double maxAbsKnp = 0.0;
            for(size_t j = 0; j < n_new_obs; j++) {
                double kij = pKernel_->eval(new_x_obs.row(j), x_pts.row(i), par_);
                k_np(j, ib) = kij;
                maxAbsKnp = std::max(maxAbsKnp, std::abs(kij));
            }
            if(maxAbsKnp > screeningCutoff) {
                #pragma omp critical
                activeCols.push_back(ib);
            }
        }

        if(activeCols.empty()) {
            continue;
        }

        Eigen::MatrixXd k_op_active(n_obs, activeCols.size());
        Eigen::MatrixXd k_np_active(n_new_obs, activeCols.size());

        #pragma omp parallel for schedule(static)
        for(size_t ia = 0; ia < activeCols.size(); ia++) {
            size_t ib = activeCols[ia];
            size_t i = start + ib;
            k_np_active.col(ia) = k_np.col(ib);
            for(size_t j = 0; j < n_obs; j++) {
                k_op_active(j, ia) = pKernel_->eval(pXObs_->row(j), x_pts.row(i), par_);
            }
        }

        Eigen::MatrixXd k_op_sol = covDecomposition_.solve(k_op_active);
        Eigen::MatrixXd rho = k_np_active - k_no * k_op_sol;
        Eigen::MatrixXd rho_sol = k_nn_ldlt.solve(rho);

        // Compute block EVI with vectorized column-wise dot products
        Eigen::VectorXd evi_active = (rho.array() * rho_sol.array()).colwise().sum().transpose();
        for(size_t ia = 0; ia < activeCols.size(); ia++) {
            evi(start + activeCols[ia]) = evi_active(ia);
        }
    }

    return evi;
}

Eigen::VectorXd GaussianProcess::expectedVarianceImprovement(const Eigen::Ref<const Eigen::VectorXd> &x,
                                                             const Eigen::Ref<const Eigen::MatrixXd> &x_pts,
                                                             const Eigen::Ref<const Eigen::MatrixXd> &selected_pts,
                                                             double nu,
                                                             double screeningCutoff) const {
    // Number of already selected points
    size_t n_sel = selected_pts.rows();
    size_t n_obs = pXObs_->rows();
    size_t n_pts = x_pts.rows();

    // Build X_aug = [selected_pts; x]
    Eigen::MatrixXd X_aug(n_sel + 1, x.size());
    if(n_sel > 0) X_aug.topRows(n_sel) = selected_pts;
    X_aug.row(n_sel) = x.transpose();

    // K_no: (n_sel+1) x n_obs
    Eigen::MatrixXd K_no(n_sel + 1, n_obs);
    for(size_t i = 0; i < n_sel + 1; ++i)
        for(size_t j = 0; j < n_obs; ++j)
            K_no(i, j) = pKernel_->eval(X_aug.row(i), pXObs_->row(j), par_);

    // K_nn: (n_sel+1) x (n_sel+1)
    Eigen::MatrixXd K_nn(n_sel + 1, n_sel + 1);
    for(size_t i = 0; i < n_sel + 1; ++i)
        for(size_t j = i; j < n_sel + 1; ++j) {
            K_nn(i, j) = pKernel_->eval(X_aug.row(i), X_aug.row(j), par_);
            if(i != j) K_nn(j, i) = K_nn(i, j);
        }

    // Compute Schur complement: C = K_nn - K_no * S * K_no^T + nu * I
    Eigen::MatrixXd C = K_nn - K_no * covDecomposition_.solve(K_no.transpose());
    C.diagonal().array() += nu;
    Eigen::LDLT<Eigen::MatrixXd> C_ldlt(C);

    // Chunked computation for large x_pts
    constexpr size_t blockSize = 1024;
    Eigen::VectorXd evi = Eigen::VectorXd::Zero(n_pts);

    for(size_t start = 0; start < n_pts; start += blockSize) {
        size_t b = std::min(blockSize, n_pts - start);

        Eigen::MatrixXd K_np(n_sel + 1, b);
        std::vector<size_t> activeCols;
        activeCols.reserve(b);

        #pragma omp parallel for schedule(static)
        for(size_t ib = 0; ib < b; ++ib) {
            size_t globalCol = start + ib;
            double maxAbsKnp = 0.0;
            for(size_t i = 0; i < n_sel + 1; ++i) {
                double kij = pKernel_->eval(X_aug.row(i), x_pts.row(globalCol), par_);
                K_np(i, ib) = kij;
                maxAbsKnp = std::max(maxAbsKnp, std::abs(kij));
            }
            if(maxAbsKnp > screeningCutoff) {
                #pragma omp critical
                activeCols.push_back(ib);
            }
        }

        if(activeCols.empty()) {
            continue;
        }

        Eigen::MatrixXd K_op_active(n_obs, activeCols.size());
        Eigen::MatrixXd K_np_active(n_sel + 1, activeCols.size());

        #pragma omp parallel for schedule(static)
        for(size_t ia = 0; ia < activeCols.size(); ++ia) {
            size_t ib = activeCols[ia];
            size_t globalCol = start + ib;
            K_np_active.col(ia) = K_np.col(ib);
            for(size_t i = 0; i < n_obs; ++i) {
                K_op_active(i, ia) = pKernel_->eval(pXObs_->row(i), x_pts.row(globalCol), par_);
            }
        }

        // Compute r = K_np - K_no * K_oo^{-1} * K_op
        Eigen::MatrixXd r = K_np_active - K_no * covDecomposition_.solve(K_op_active);

        // Solve C * y = r for each column of r
        Eigen::MatrixXd Cinv_r = C_ldlt.solve(r);

        // EVI at each x_pts: sum over rows of r .* Cinv_r
        Eigen::VectorXd evi_active = (r.array() * Cinv_r.array()).colwise().sum().transpose();
        for(size_t ia = 0; ia < activeCols.size(); ++ia) {
            evi(start + activeCols[ia]) = evi_active(ia);
        }
    }

    return evi;
}

double GaussianProcess::objectiveFunction(const Eigen::Ref<const Eigen::VectorXd> &par, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &lp) {


    // Compute the LDLT decomposition of the covariance matrix and the residuals
    Eigen::LDLT<Eigen::MatrixXd> cov_ldlt = cmp::ldltDecomposition(covariance(par));
    Eigen::VectorXd res = residual(par);

    // Evaluate the log-likelihood and the log-prior
    double ll = cmp::distribution::MultivariateNormalDistribution::logPDF(res, cov_ldlt);

    // Update the gradient (note that the gradient is optional)
    if(grad.size() != 0) {

        // Update each component of the gradient
        for(int n = 0; n < par.size(); n++) {
            auto cov_gradient = covarianceGradient(par, n);
            auto mean_gradient = priorMeanGradient(par, n);
            grad[n] = cmp::distribution::MultivariateNormalDistribution::dLogPDF(res, cov_ldlt, cov_gradient, mean_gradient);

            // Add the prior gradient
            grad[n] += lp->evalGradient(par, n);
        }
    }

    // Return the log-likelihood + log-prior
    return ll + lp->eval(par);
}
double GaussianProcess::objectiveFunctionLOO(const Eigen::Ref<const Eigen::VectorXd> &par, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &lp) {

    const double log2pi = std::log(2.0 * M_PI);
    const double eps = 1e-12; // numerical floor

    // Covariance, factorization, sizes
    Eigen::MatrixXd K = covariance(par);
    Eigen::LDLT<Eigen::MatrixXd> ldlt = cmp::ldltDecomposition(K);
    const Eigen::Index n = K.rows();

    // residuals and alpha = K^{-1} res
    Eigen::VectorXd res = residual(par);                // res = y - mu(par)
    Eigen::VectorXd alpha = ldlt.solve(res);

    // full inverse (needed for diag and M = K^{-1} dK K^{-1})
    Eigen::MatrixXd Kinv = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    Eigen::VectorXd v = Kinv.diagonal();               // v_i = (K^{-1})_ii

    // safety floor for computations (don't change v itself if you want exact derivatives;
    // we use a floored copy for divisions only to avoid blowups)
    Eigen::VectorXd v_safe = v.array().max(eps);
    Eigen::VectorXd sigma2 = (1.0 / v_safe.array()).matrix();   // loo variances
    Eigen::VectorXd r = (alpha.array() / v_safe.array()).matrix(); // loo residuals

    // per-point loo log-likelihood and total
    Eigen::VectorXd loo_ll = (-0.5 * (log2pi + sigma2.array().log()) - 0.5 * (r.array().square() / sigma2.array())).matrix();
    double total_loo_ll = loo_ll.sum();

    // Gradient (if requested)
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
