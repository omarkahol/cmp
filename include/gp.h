#ifndef GP_H
#define GP_H


#include <distribution.h>
#include <scaler.h>
#include <covariance.h>
#include <mean++.h>
#include <prior++.h>
#include <optimization.h>
#include <memory>

namespace cmp::gp {

enum method {
    MLE,
    LOO,
    LOO_MSE,
};

enum type {
    PRIOR,
    POSTERIOR
};

/**
 * @brief This class implements a Gaussian process, providing algorithms for training and prediction.
 *
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
     * A GP is defined by a Kernel function, a mean function, a prior function, a nugget and its hyperparameters.
     * @param kernel The Kernel function
     * @param mean The mean function
     * @param prior The prior function
     * @param params The vector of hyperparameters
     *
     * @note This function is used to set the Kernel, mean and prior functions of the GP.
     */
    void set(const std::shared_ptr<covariance::Covariance> &kernel, const std::shared_ptr<mean::Mean> &mean, Eigen::Ref<const Eigen::VectorXd> params, double nugget = 1e-8);

    /**
     * @brief Condition the GP on a set of observations (allow predictive posterior computation)
     *
     * @param xObs The observation points
     * @param yObs The observation values
     * @param copyData  Decide whether or not to make a deep copy of the data
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, bool copyData);

    /**
     * @brief Fit the Gaussian Process to the observations
     *
     * @param xObs the observation points
     * @param yObs the observation values
     * @param lowerBound the lower bound for the hyperparameters
     * @param upperBound the upper bound for the hyperparameters
     * @param method the method to be used for the optimization
     * @param alg the algorithm to be used for the optimization
     * @param tol_rel the relative tolerance for the optimization
     * @param copyData whether to copy the observation data internally (default is true)
     * @param prior the prior distribution for the hyperparameters (default is uniform prior)
     *
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, const method &method, const nlopt::algorithm &alg, const double &tol_rel, bool copyData = true, const std::shared_ptr<cmp::prior::Prior> &prior = cmp::prior::Uniform::make());

    /**
     * @brief Get the hyperparameters of the Gaussian Process
     * @return the value of the hyperparameters
     */
    Eigen::VectorXd getParameters() const {
        return par_;
    }

    std::shared_ptr<cmp::mean::Mean> getMean() const {
        return this->pMean_;
    }

    std::shared_ptr<cmp::covariance::Covariance> getKernel() const {
        return this->pKernel_;
    }

    double getNugget() const {
        return this->nugget_;
    }

    /**
     * * FUNCTIONS FOR THE KERNEL
     */

    /**
     * @brief Evaluates the covariance matrix of the Kernel
     *
     * @return a matrix containing the evaluation of the Kernel on the observation points
     */
    Eigen::MatrixXd covariance(Eigen::Ref<const Eigen::VectorXd> par) const;

    /**
     * Evaluate the i-th component of the gradient of the covariance matrix.
     * @param i The component of the gradient required
     */
    Eigen::MatrixXd covarianceGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &i) const;

    /**
     * Evaluate the ij component of the hessian of the covariance matrix.
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
     *
     * This function calculates the mean of a set of points given by `x_pts` using the parameters `par`.
     *
     * @return The mean of the points.
     */
    Eigen::VectorXd priorMean(Eigen::Ref<const Eigen::VectorXd> par) const;

    /**
     * @brief Evaluate the gradient of the mean function
     *
     * @param i the index of the hyperparameter
     * @return a vector containing the computation of the gradient
     */
    Eigen::VectorXd priorMeanGradient(Eigen::Ref<const Eigen::VectorXd> par, const int &i) const;

    /**
     * @brief Evaluate the difference between the observation and the mean function
     *
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

#endif // MACRO
