#ifndef MULTI_GP_H
#define MULTI_GP_H

#include "gp.h"

/**
 * @addtogroup surrogate
 * @{
 */
namespace cmp::gp {
/**
 * @brief Manages a collection of independent Gaussian Processes for multi-output regression.
 * 
 * @details Mathematical Formulation
 * Models a vector-valued function \f$\mathbf{f}: \mathbb{R}^D \to \mathbb{R}^M\f$ using \f$M\f$ independent Gaussian Processes:
 * \f[
 * f_j(\mathbf{x}) \sim \mathcal{GP}\left(m_j(\mathbf{x}), k_j(\mathbf{x}, \mathbf{x}')\right), \quad j = 0, \dots, M-1
 * \f]
 * Often used in combination with PCA dimension reduction, where the high-dimensional response vector \f$\mathbf{y} \in \mathbb{R}^Q\f$ is projected onto \f$M\f$ principal components, and each component is modeled by a separate GP.
 * 
 * @details Implementation Algorithm
 * 1. `condition()` partitions the multi-column target matrix \f$\mathbf{Y}\f$ to condition each internal `GaussianProcess` object.
 * 2. `fit()` sequentially optimizes hyperparameters for each GP in the collection using the specified MLE/LOO objectives.
 * 3. `predict()` queries predictions from each internal GP to reconstruct the multivariate output mean and variance vectors.
 */
class MultiOutputGaussianProcess {

  private:

    // Base components
    std::vector<GaussianProcess> gps_;

    // Observations (the GP can own the data or point to external data, if it owns the data the pointer will point to it)
    Eigen::MatrixXd xObs_;
    Eigen::MatrixXd yObs_;
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> pXObs_;
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> pYObs_;



  public:

    // Constructors and destructor
    MultiOutputGaussianProcess();
    MultiOutputGaussianProcess(const size_t &nGps, const std::shared_ptr<covariance::Covariance> &kernel, const std::shared_ptr<mean::Mean> &mean, const Eigen::Ref<const Eigen::VectorXd> &params, double nugget = 1e-8);
    ~MultiOutputGaussianProcess() = default;

    // Copy and move constructors and assignment operators
    MultiOutputGaussianProcess(const MultiOutputGaussianProcess &other) = default;
    MultiOutputGaussianProcess &operator=(const MultiOutputGaussianProcess &other) = default;
    MultiOutputGaussianProcess(MultiOutputGaussianProcess &&other) noexcept = default;
    MultiOutputGaussianProcess &operator=(MultiOutputGaussianProcess &&other) noexcept = default;


    /**
     * @brief Set the number of Gaussian Processes and their parameters
     * @param nGps the number of Gaussian Processes
     * @param kernel the covariance function
     * @param mean the mean function
     * @param params the hyperparameters
     * @param nugget the nugget value
     */
    void set(const size_t &nGps, const std::shared_ptr<covariance::Covariance> &kernel, const std::shared_ptr<mean::Mean> &mean, const Eigen::Ref<const Eigen::VectorXd> &params, double nugget = 1e-8);

    /**
     * @brief Set the values of the observations
     *
     * @param xObs the observation points
     * @param yObs the observation values (each column corresponds to a different output dimension)
     * @param copyData whether to copy the observation data internally (default is true)
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::MatrixXd> &yObs, bool copyData = true);


    /**
     * @brief Fit the Gaussian Process to the observations
     * @param xObs the observation points
     * @param yObs the observation values
     * @param lb the lower bounds for the hyperparameters
     * @param ub the upper bounds for the hyperparameters
     * @param method the fitting method (default is MLE)
     * @param alg the optimization algorithm (default is nlopt::LN_SBPLX)
     * @param tol_rel the relative tolerance for the optimization (default is 1e-3)
     * @param copyData whether to copy the observation data internally (default is true)
     * @param prior the prior distribution for the hyperparameters (default is uniform prior)
     * @param logScale a vector indicating which hyperparameters should be optimized in log scale (default is all false)
     */
    void fit(const Eigen::Ref<Eigen::MatrixXd> &xObs, const Eigen::Ref<Eigen::MatrixXd> &yObs, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub,  const method &method = MLE, const nlopt::algorithm &alg = nlopt::LN_SBPLX, const double &tol_rel = 1e-3, bool copyData = true, const std::shared_ptr<cmp::prior::Prior> &prior = cmp::prior::Uniform::make(), const std::vector<bool> &logScale = {});

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::Ref<const Eigen::VectorXd> &x, const type &t = type::POSTERIOR) const;

    GaussianProcess &operator[](const int &i) {
        return gps_.at(i);
    }

    /**
     * @brief Returns the number of output dimensions (the number of GP models).
     * @return Size of the GP collection.
     */
    size_t size() const {
        return gps_.size();
    }

    /**
     * @brief Predicts the output mean vector at the test point x.
     * @param x The test point.
     * @param t Evaluation type (PRIOR or POSTERIOR).
     * @return Predictive mean vector.
     */
    Eigen::VectorXd predictMean(const Eigen::Ref<const Eigen::VectorXd> &x, const type &t = type::POSTERIOR) const;
};


}
/** @} */

#endif