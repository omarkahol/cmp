#ifndef GP_H
#define GP_H

#include "cmp_defines.h"
#include <distribution.h>
#include <utils.h>
#include <scaler.h>
#include <kernel++.h>
#include <mean++.h>
#include <prior++.h>

namespace cmp::gp
{

    enum method
    {
        MLE,
        MAP,
        MLOO,
        MLOOP
    };

    /**
     * @brief This class implements a Gaussian process, providing algorithms for training and prediction.
     *
     */
    class GaussianProcess
    {
    private:
        size_t nObs_;
        Eigen::VectorXd par_;
        Eigen::LDLT<Eigen::MatrixXd> covDecomposition_;
        Eigen::VectorXd alpha_;
        Eigen::VectorXd diagCovInverse_;
        Eigen::VectorXd residual_;

        // Observations
        std::vector<Eigen::VectorXd> XObs_;  ///> The observation points
        std::vector<double> YObs_;           ///> The observation values

        // Kernel, mean and prior functions
        std::shared_ptr<kernel::Kernel> pKernel_; ///> The Kernel function
        std::shared_ptr<mean::Mean> pMean_;       ///> The mean function
        std::shared_ptr<prior::Prior> pPrior_;    ///> The prior function

        // The nuggget
        double nugget_ = 1e-8;

    public:
        GaussianProcess() = default;

        /**
         * @brief Set the values of the observations
         *
         * @param x_obs the observation points
         * @param y_obs the observation values
         */
        void setObservations(const std::vector<Eigen::VectorXd> &xObs, const std::vector<double> &yObs)
        {
            XObs_ = xObs;
            YObs_ = yObs;
            nObs_ = xObs.size();
        }

        void removeObservation(const size_t &i)
        {
            XObs_.erase(XObs_.begin() + i);
            YObs_.erase(YObs_.begin() + i);
            nObs_ = XObs_.size();
        }

        void addObservation(const Eigen::VectorXd &x, const double &y)
        {
            XObs_.push_back(x);
            YObs_.push_back(y);
            nObs_ = XObs_.size();
        }

        std::vector<Eigen::VectorXd> &xObs()
        {
            return XObs_;
        }

        std::shared_ptr<kernel::Kernel> getKernel() const {
            return pKernel_;
        }

        std::shared_ptr<mean::Mean> getMean() const {
            return pMean_;
        }

        std::shared_ptr<prior::Prior> getPrior() const {
            return pPrior_;
        }

        double getNugget() const {
            return nugget_;
        }

        /**
         * Set the GP
         * @param Kernel The Kernel function
         * @param mean The mean function
         * @param prior The prior function
         *
         * @note This function is used to set the Kernel, mean and prior functions of the GP.
         */
        void set(const std::shared_ptr<kernel::Kernel> &Kernel, const std::shared_ptr<mean::Mean> &mean, const std::shared_ptr<prior::Prior> &prior, double nugget = 1e-8)
        {
            pKernel_ = Kernel;
            pMean_ = mean;
            pPrior_ = prior;

            nugget_ = nugget;
        }

        size_t size() const
        {
            return nObs_;
        }

        /**
         * @brief Set the hyperparameters of the Gaussian Process.
         * @note This function computes also the inverse of the covariance matrix.
         *
         * @param par the value of the hyperparameters
         */
        void setParameters(const Eigen::VectorXd &par);

        /**
         * @brief Fit the Gaussian Process to the observations
         *
         * @param parGuess the initial guess for the hyperparameters
         * @param lowerBound the lower bound for the hyperparameters
         * @param upperBound the upper bound for the hyperparameters
         * @param method the method to be used for the optimization (MLE, MAP, MLOO, MLOOP, default is MLE)
         * @param alg the algorithm to be used for the optimization (default is nlopt::LN_SBPLX)
         * @param tol_rel the relative tolerance for the optimization (default is 1e-3)
         *
         */
        void fit(const Eigen::VectorXd &parGuess, const Eigen::VectorXd &lowerBound, const Eigen::VectorXd &upperBound, const method &method = MLE, const nlopt::algorithm &alg = nlopt::LN_SBPLX, const double &tol_rel = 1e-3);

        /**
         * @brief Get the hyperparameters of the Gaussian Process
         *
         * @return the value of the hyperparameters
         */
        Eigen::VectorXd getParameters() const
        {
            return par_;
        }

        /**
         * FUNCTIONS FOR THE KERNEL
         */

        /**
         * @brief Evaluates the covariance matrix of the Kernel
         *
         * @return a matrix containing the evaluation of the Kernel on the observation points
         */
        Eigen::MatrixXd covariance(Eigen::VectorXd par) const;

        /**
         * Evaluate the i-th component of the gradient of the covariance matrix.
         * @param i The component of the gradient required
         */
        Eigen::MatrixXd covarianceGradient(Eigen::VectorXd par, const int &i) const;

        /**
         * Evaluate the ij component of the hessian of the covariance matrix.
         * @param i row of the hessian matrix
         * @param j colum of the hessian matrix
         */
        Eigen::MatrixXd covarianceHessian(Eigen::VectorXd par, const int &i, const int &j) const;

        const Eigen::LDLT<Eigen::MatrixXd> &getCovDecomposition() const
        {
            return covDecomposition_;
        }
        /**
         * FUNCTIONS FOR THE MEAN
         */

        /**
         * @brief Evaluates the mean on the observations.
         *
         * This function calculates the mean of a set of points given by `x_pts` using the parameters `par`.
         *
         * @return The mean of the points.
         */
        Eigen::VectorXd priorMean(Eigen::VectorXd par) const;

        /**
         * @brief Evaluate the gradient of the mean function
         *
         * @param i the index of the hyperparameter
         * @return a vector containing the computation of the gradient
         */
        Eigen::VectorXd priorMeanGradient(Eigen::VectorXd par, const int &i) const;

        /**
         * @brief Evaluate the difference between the observation and the mean function
         *
         * @return a vector containing the computation of the residual
         */
        Eigen::VectorXd residual(Eigen::VectorXd par) const;

        /*
         * FUNCTIONS FOR THE LOG PRIOR
         */

        /**
         * @brief Evaluates the log prior function
         *
         * @return the value of the log prior.
         */
        double logPrior(Eigen::VectorXd par) const
        {
            return pPrior_->eval(par);
        }

        /**
         * This function evaluates the gradient of the log prior function.
         *
         * @param i The index of the hyperparameter
         *
         * @return The value of the gradient of the log prior function
         */
        double logPriorGradient(Eigen::VectorXd par, const int &i) const
        {
            return pPrior_->evalGradient(par, i);
        }

        /**
         * This function evaluates the hessian of the log prior function.
         *
         * @param i The index of the hyperparameter
         * @param j The index of the hyperparameter
         *
         * @return The value of the hessian of the log prior function
         */
        double logPriorHessian(Eigen::VectorXd par, const int &i, const int &j) const
        {
            return pPrior_->evalHessian(par, i, j);
        }

        /**
         * FUNCTIONS FOR THE PREDICTIVE DISTRIBUTION
         */

        /**
         * @brief Compute the predictive distribution at a new point
         *
         * @param x The new prediction point
         * @return The prediction
         */
        distribution::NormalDistribution predictiveDistribution(const Eigen::VectorXd &x) const;

        /**
         * @brief Compute the predictive distribution at a set of new prediction points.
         *
         * @param x_pts The new prediction points.
         * @return The prediction
         */
        distribution::MultivariateNormalDistribution predictiveDistribution(const std::vector<Eigen::VectorXd> &x_pts) const;

        distribution::NormalDistribution predictiveDistributionLooCV(const size_t &i) const;

        double predict(const Eigen::VectorXd &x) const;

        double predictLooCV(const size_t &i) const;

        double predictVariance(const Eigen::VectorXd &x) const;

        double predictiveCovariance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const;

        double predictVarianceLooCV(const size_t &i) const;

        double logLikelihood() const;

        double logLikelihoodLooCV(const size_t &i) const;

        // Compute the predictive mean and covariance at a set of new prediction points.
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> predictiveMeanAndCovariance(const std::vector<Eigen::VectorXd> &x_pts) const;

        /**
         * @brief Compute the variance reduction matrix after observing a set of points.
         *
         * @param x_pts The points where the variance reduction is computed
         * @return Eigen::MatrixXd The variance reduction matrix
         */
        Eigen::MatrixXd expectedVarianceImprovement(const std::vector<Eigen::VectorXd> &x_pts, double nu = 1e-6) const;

        Eigen::VectorXd expectedVarianceImprovement(const std::vector<Eigen::VectorXd> &x_pts, const std::vector<Eigen::VectorXd> &new_x_obs, double nu = 1e-6) const;
    };

    double opt_fun_gp_mle(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);

    double opt_fun_gp_map(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);

    double opt_fun_gp_mloo(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);

    double opt_fun_gp_mloop(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);
}

#endif // MACRO
