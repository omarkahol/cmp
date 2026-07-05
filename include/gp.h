#ifndef GP_H
#define GP_H


#include <distribution.h>
#include <scaler.h>
#include <covariance.h>
#include <mean++.h>
#include <prior++.h>
#include <optimization.h>
#include <memory>

/**
 * @addtogroup surrogate
 * @{
 */
namespace cmp::gp {

/**
 * @brief Optimization method for GP hyperparameters.
 */
enum method {
    MLE,     ///< Maximum Likelihood Estimation (maximizing marginal likelihood)
    LOO,     ///< Leave-One-Out cross-validation predictive probability optimization
    LOO_MSE  ///< Leave-One-Out Mean Squared Error minimization
};

/**
 * @brief GP evaluation type.
 */
enum type {
    PRIOR,     ///< Prior GP response (before conditioning on training data)
    POSTERIOR  ///< Posterior GP response (conditioned on training data)
};

/**
 * @class GaussianProcess
 * @brief This class implements a Gaussian Process (GP) regression model for non-parametric Bayesian regression.
 *
 * @details
 * ### Mathematical Foundations
 * A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.
 * It is completely specified by its mean function \f$m(\mathbf{x})\f$ and covariance (kernel) function \f$k(\mathbf{x}, \mathbf{x}')\f$:
 * \f[ f(\mathbf{x}) \sim \mathcal{GP}\left(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')\right) \f]
 *
 * Given a training dataset \f$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N\f$ where \f$\mathbf{X} \in \mathbb{R}^{N \times D}\f$ and \f$\mathbf{y} \in \mathbb{R}^N\f$,
 * the joint distribution of the training outputs and the predictive function value \f$f(\mathbf{x}_*)\f$ at a test point \f$\mathbf{x}_*\f$ is:
 * \f[
 * \begin{bmatrix} \mathbf{y} \\ f(\mathbf{x}_*) \end{bmatrix} \sim \mathcal{N}\left(
 * \begin{bmatrix} \mathbf{m}(\mathbf{X}) \\ m(\mathbf{x}_*) \end{bmatrix},
 * \begin{bmatrix} \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I} & \mathbf{k}(\mathbf{X}, \mathbf{x}_*) \\ \mathbf{k}(\mathbf{x}_*, \mathbf{X}) & k(\mathbf{x}_*, \mathbf{x}_*) \end{bmatrix}
 * \right)
 * \f]
 * where \f$\sigma_n^2\f$ is the nugget (representing the observation noise variance).
 *
 * The conditional posterior distribution at the test point \f$\mathbf{x}_*\f$ is given by:
 * \f[ f(\mathbf{x}_*) | \mathbf{X}, \mathbf{y}, \mathbf{x}_* \sim \mathcal{N}(\mu_*, \sigma_*^2) \f]
 * where the posterior mean \f$\mu_*\f$ and variance \f$\sigma_*^2\f$ are calculated as:
 * \f[ \mu_* = m(\mathbf{x}_*) + \mathbf{k}(\mathbf{x}_*, \mathbf{X}) \left[ \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I} \right]^{-1} (\mathbf{y} - \mathbf{m}(\mathbf{X})) \f]
 * \f[ \sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}(\mathbf{x}_*, \mathbf{X}) \left[ \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I} \right]^{-1} \mathbf{k}(\mathbf{X}, \mathbf{x}_*) \f]
 *
 * ### Implementation Algorithms
 * 1. **LDLT Decomposition**: Directly inverting \f$\mathbf{K}_y = \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I}\f$ is numerically unstable and computationally expensive (\f$O(N^3)\f$).
 *    We construct \f$\mathbf{K}_y\f$ and compute its LDLT decomposition:
 *    \f[ \mathbf{K}_y = \mathbf{L} \mathbf{D} \mathbf{L}^T \f]
 * 2. **Backsubstitution**: We precompute the weight vector \f$\boldsymbol{\alpha}\f$ by solving:
 *    \f[ \mathbf{L} \mathbf{D} \mathbf{L}^T \boldsymbol{\alpha} = \mathbf{y} - \mathbf{m}(\mathbf{X}) \f]
 *    The predictive posterior mean is then efficiently evaluated as:
 *    \f[ \mu_* = m(\mathbf{x}_*) + \mathbf{k}(\mathbf{x}_*, \mathbf{X}) \boldsymbol{\alpha} \f]
 * 3. **Hyperparameter Optimization**: The hyperparameters \f$\boldsymbol{\theta}\f$ of the covariance and mean functions are optimized by maximizing the marginal log-likelihood:
 *    \f[ \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2} (\mathbf{y} - \mathbf{m}(\mathbf{X}))^T \mathbf{K}_y^{-1} (\mathbf{y} - \mathbf{m}(\mathbf{X})) - \frac{1}{2} \log |\mathbf{K}_y| - \frac{N}{2} \log (2\pi) \f]
 *    using gradient-free (e.g., Subplex) or gradient-based NLopt algorithms.
 *
 * ### Constraints & Invariants
 * - **Nugget (\f$\sigma_n^2\f$)**: Must be strictly positive (\f$\ge 10^{-12}\f$, default \f$10^{-8}\f$) to ensure the covariance matrix remains strictly positive-definite.
 * - **Training Inputs**: Pre-conditions require that the dataset is non-empty (\f$N \ge 1\f$).
 * - **Hyperparameter bounds**: Must satisfy the user-specified bounds \f$\mathbf{lb} \le \boldsymbol{\theta} \le \mathbf{ub}\f$.
 */
class GaussianProcess {
  private:

    // Hyperparameters and prior mean and covariance functions
    Eigen::VectorXd par_;
    std::shared_ptr<covariance::Covariance> pKernel_;
    std::shared_ptr<mean::Mean> pMean_;
    double nugget_;

    // Observations (the GP can own the data or point to external data, if it owns the data the pointer will point to it)
    std::optional<Eigen::MatrixXd> xObs_;
    std::optional<Eigen::VectorXd> yObs_;
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> pXObs_;
    std::optional<Eigen::Ref<const Eigen::VectorXd>> pYObs_;

    // Internal storage for the covariance matrix, its decomposition and related vectors (for fast query)
    Eigen::LDLT<Eigen::MatrixXd> covDecomposition_;
    Eigen::VectorXd alpha_;
    Eigen::VectorXd diagCovInverse_;
    Eigen::VectorXd residual_;

    // Internal flag to check whether we normalize y or not
    bool normalizeY_ = false;
    cmp::scaler::StandardScaler yScaler_;

// Private members
  private:
    void compute(const Eigen::Ref<const Eigen::VectorXd> &par);

  public:

    // Default constructor and destructor
    GaussianProcess();
    ~GaussianProcess() = default;

    // Constructor with parameters
    GaussianProcess(const std::shared_ptr<covariance::Covariance> &kernel, const std::shared_ptr<mean::Mean> &mean, Eigen::Ref<const Eigen::VectorXd> params, double nugget = 1e-8);

    // Copy constructors and operators
    GaussianProcess(const GaussianProcess &other);
    GaussianProcess &operator=(const GaussianProcess &other);
    GaussianProcess(GaussianProcess &&other) noexcept;
    GaussianProcess &operator=(GaussianProcess &&other) noexcept;

    /**
     * A GP is defined by a Kernel function, a mean function, a nugget and its hyperparameters.
     * @param kernel The Kernel function
     * @param mean The mean function
     * @param params The vector of hyperparameters
     * @param nugget The nugget value (default is 1e-8)
     *
     * @note This function is used to set the Kernel and mean functions of the GP.
     */
    void set(const std::shared_ptr<covariance::Covariance> &kernel, const std::shared_ptr<mean::Mean> &mean, Eigen::Ref<const Eigen::VectorXd> params, double nugget = 1e-8);

    /**
     * @brief Condition the GP on a set of observations (allow predictive posterior computation)
     *
     * @param xObs The observation points
     * @param yObs The observation values
     * @param copyData  Decide whether or not to make a deep copy of the data
     * @param normalizeY Whether to normalize the observation values
     *
     * @note if copyData is false, and normalizeY is true the function will throw an exception, since it cannot normalize the data without owning it.
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, bool copyData = true, bool normalizeY = false);

    /**
     * @brief Fit the Gaussian Process to the observations
     *
     * @param xObs the observation points
     * @param yObs the observation values
     * @param lb the lower bound for the hyperparameters
     * @param ub the upper bound for the hyperparameters
     * @param method the method to be used for the optimization
     * @param alg the algorithm to be used for the optimization
     * @param tol_rel the relative tolerance for the optimization
     * @param copyData whether to copy the observation data internally (default is true)
     * @param normalizeY whether to normalize the observation values (default is false)
     * @param prior the prior distribution for the hyperparameters (default is uniform prior)
     * @param logScale a vector indicating which hyperparameters should be optimized in log scale (default is all false)
     *
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, const method &method, const nlopt::algorithm &alg, const double &tol_rel, bool copyData = true, bool normalizeY = false, const std::shared_ptr<cmp::prior::Prior> &prior = cmp::prior::Uniform::make(), const std::vector<bool> &logScale = {});

    /**
     * @brief Get the hyperparameters of the Gaussian Process
     * @return the value of the hyperparameters
     */
    Eigen::VectorXd getParameters() const {
        return par_;
    }

    /**
     * @brief Gets the shared pointer to the GP prior mean function.
     * @return Shared pointer to the Mean object.
     */
    std::shared_ptr<cmp::mean::Mean> getMean() const {
        return this->pMean_;
    }

    /**
     * @brief Gets the shared pointer to the GP covariance kernel.
     * @return Shared pointer to the Covariance object.
     */
    std::shared_ptr<cmp::covariance::Covariance> getKernel() const {
        return this->pKernel_;
    }

    /**
     * @brief Gets the observation noise variance (nugget).
     * @return Nugget value.
     */
    double getNugget() const {
        return this->nugget_;
    }

    /**
     * * FUNCTIONS FOR THE KERNEL
     */

    /**
     * @brief Evaluates the covariance matrix of the Kernel
     * @param par The vector of hyperparameters
     * @return a matrix containing the evaluation of the Kernel on the observation points
     */
    Eigen::MatrixXd covariance(Eigen::Ref<const Eigen::VectorXd> par) const;

    /**
     * Evaluate the i-th component of the gradient of the covariance matrix.
     * @param par The vector of hyperparameters
     * @param i The component of the gradient required
     */
    Eigen::MatrixXd covarianceGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &i) const;

    /**
     * Evaluate the ij component of the hessian of the covariance matrix.
     * @param par The vector of hyperparameters
     * @param i row of the hessian matrix
     * @param j colum of the hessian matrix
     */
    Eigen::MatrixXd covarianceHessian(Eigen::Ref<const Eigen::VectorXd> par, const size_t &i, const size_t &j) const;

    const Eigen::LDLT<Eigen::MatrixXd> &getCovDecomposition() const {
        return covDecomposition_;
    }

    const Eigen::VectorXd &getAlpha() const {
        return alpha_;
    }

    const Eigen::VectorXd &getDiagCovInverse() const {
        return diagCovInverse_;
    }

    const Eigen::VectorXd &getResidualVector() const {
        return residual_;
    }

    const Eigen::Ref<const Eigen::MatrixXd> &getXObs() const {
        return pXObs_.value();
    }

    const Eigen::Ref<const Eigen::VectorXd> &getYObs() const {
        return pYObs_.value();
    }

    /**
     * @brief Returns the number of training observations.
     * @return Number of training data rows.
     */
    size_t nObs() const {
        if(!pXObs_.has_value()) {
            return 0;
        }
        return static_cast<size_t>(pXObs_->rows());
    }
    /**
     * * FUNCTIONS FOR THE MEAN
     */

    /**
     * @brief Evaluates the mean on the observations.
     * @param par The vector of hyperparameters
     * @return The mean of the points.
     */
    Eigen::VectorXd priorMean(Eigen::Ref<const Eigen::VectorXd> par) const;

    /**
     * @brief Evaluate the gradient of the mean function
     * @param par The vector of hyperparameters
     * @param i the index of the hyperparameter
     * @return a vector containing the computation of the gradient
     */
    Eigen::VectorXd priorMeanGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &i) const;

    /**
     * @brief Evaluate the difference between the observation and the mean function
     * @param par The vector of hyperparameters
     * @return a vector containing the computation of the residual
     */
    Eigen::VectorXd residual(Eigen::Ref<const Eigen::VectorXd> par) const;

    /**
     * FUNCTIONS FOR THE PREDICTIVE DISTRIBUTION
     */


    /**
     * @brief Compute the predictive mean and variance at a new point
     *
     * @param x The new prediction point
     * @param predictionType The type of prediction (prior or posterior)
     * @return A pair containing the mean and variance of the prediction
     */
    std::pair<double, double> predict(const Eigen::Ref<const Eigen::VectorXd> &x, type predictionType = type::POSTERIOR) const;

    /**
     * @brief Compute the predictive mean at a new point
     * @note The predictive mean can be computed in O(n) time for the posterior and O(1) time for the prior. Hence it is better to use this function if only the mean is required.
     *
     * @param x The new prediction point
     * @param predictionType The type of prediction (prior or posterior)
     * @return The predictive mean
     */
    double predictMean(const Eigen::Ref<const Eigen::VectorXd> &x, type predictionType = type::POSTERIOR) const;

    /**
     * @brief Compute the predictive variance at a new set of points
     * @param x_pts The new prediction points
     * @param predictionType The type of prediction (prior or posterior)
     * @return A pair containing the mean and variance of the predictions
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predictMultiple(const Eigen::Ref<const Eigen::MatrixXd> &x_pts, type predictionType = type::POSTERIOR) const;

    /**
     * @brief Compute the predictive mean at a new set of points
     * @param x_pts The new prediction points
     * @param predictionType The type of prediction (prior or posterior)
     * @return The predictive mean
     * @note The predictive mean can be computed in O(n*m) time for the posterior and O(m) time for the prior, where n is the number of observations and m is the number of prediction points. Hence it is better to use this function if only the mean is required.
     */
    Eigen::VectorXd predictMeanMultiple(const Eigen::Ref<const Eigen::MatrixXd> &x_pts, type predictionType = type::POSTERIOR) const;


    /**
     * @brief Compute the leave-one-out predictive mean and variance at the i-th observation point
     * @param i The index of the observation point
     * @return A pair containing the mean and variance of the leave-one-out prediction
     * @throws std::out_of_range if i is out of range [0, nObs_) and runtime error if the GP is not conditioned
     */
    std::pair<double, double> predictLOO(const size_t &i) const;

    /**
     * * LOG LIKELIHOOD FUNCTIONS
     */

    /**
     * @brief Compute the log-likelihood of the observations given the hyperparameters
     * @return The value of the log-likelihood
     * @throws runtime_error if the GP is not conditioned
     */
    double logLikelihood() const;

    /**
     * @brief Compute the leave-one-out log-likelihood of the observations given the hyperparameters
     * @return The value of the leave-one-out log-likelihood
     * @throws runtime_error if the GP is not conditioned or out_of_range if i is out of range [0, nObs_)
     */
    double logLikelihoodLOO(const size_t &i) const;

    /**
     * * FUNCTIONS FOR THE EXPECTED VARIANCE IMPROVEMENT
     */

    Eigen::MatrixXd expectedVarianceImprovement(const Eigen::Ref<const Eigen::MatrixXd> &x_pts,
                                                const Eigen::Ref<const Eigen::MatrixXd> &x_pending,
                                                double nu) const;

    /**
     * * * OBJECTIVE FUNCTIONS FOR THE OPTIMIZATION
     */

    double objectiveFunction(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &prior);

    double objectiveFunctionLOO(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &prior);

    double objectiveFunctionLOOMSE(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> grad, const std::shared_ptr<cmp::prior::Prior> &prior);
};
}
/** @} */

#endif // MACRO
