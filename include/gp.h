#ifndef GP_H
#define GP_H

#include "cmp_defines.h"
#include <distribution.h>
#include <utils.h>
#include <scaler.h>

namespace cmp {

    enum method {MLE, MAP, MLOO, MLOOP};

    /**
     * @brief This class implements a Gaussian process, providing algorithms for training and prediction.
     * 
     */
    class gp {

        private:
            
            size_t m_size;
            Eigen::VectorXd m_par;
            Eigen::LDLT<Eigen::MatrixXd> m_cov_ldlt;
            Eigen::VectorXd m_alpha;

            scalar_scaler *m_y_obs{nullptr};
            vector_scaler *m_x_obs{nullptr};
            
            kernel_t m_kernel;                                                                                                                  ///> The kernel function
            model_t m_mean;                                                                                                                     ///> The mean function
            prior_t m_log_prior;                                                                                                                ///> The prior function
            std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> m_mean_grad;                                         ///> Function that returns the i-th component of the mean gradient
            std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> m_kernel_grad;              ///> Function that returns the i-th component of the kernel gradient
            std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i, int j)> m_kernel_hess;       ///> Function that returns the ij-th component of the kernel hessian
            std::function<double(Eigen::VectorXd const &, int i)> m_log_prior_grad;                                                             ///> Function that returns the i-th component of the log-prior gradient
            std::function<double(Eigen::VectorXd const &, int i, int j)> m_log_prior_hess;                                                      ///> Function that returns the ij-th component of the log-prior hessian

        public:

            gp()=default;


            /**
             * @brief Set the values of the observations
             * 
             * @param x_obs the observation points
             * @param y_obs the observation values
             */
            void set_observations(cmp::vector_scaler *x_obs, cmp::scalar_scaler *y_obs){
                m_x_obs = x_obs;
                m_y_obs = y_obs;
                m_size = m_y_obs->get_size();
            }

            const scalar_scaler *get_y_obs() const {
                return m_y_obs;
            }

            const vector_scaler *get_x_obs() const {
                return m_x_obs;
            }

            size_t get_size() const{
                return m_size;
            }

            /**
             * @brief Set the hyperparameters of the Gaussian Process.
             * @note This function computes also the inverse of the covariance matrix.
             * 
             * @param par the value of the hyperparameters
             */
            void set_params(const Eigen::VectorXd &par);

            /**
             * @brief Fit the Gaussian Process to the observations
             * 
             * @param x0 the initial guess for the hyperparameters
             * @param lb the lower bound for the hyperparameters
             * @param ub the upper bound for the hyperparameters
             * @param method the method to be used for the optimization (MLE, MAP, MLOO, MLOOP, default is MLE)
             * @param alg the algorithm to be used for the optimization (default is nlopt::LN_SBPLX)
             * @param tol_rel the relative tolerance for the optimization (default is 1e-3)
             * 
             */
            void fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,  const cmp::method &method = cmp::MLE, const nlopt::algorithm &alg = nlopt::LN_SBPLX, const double &tol_rel = 1e-3);


            /**
             * @brief Get the hyperparameters of the Gaussian Process
             * 
             * @return the value of the hyperparameters
             */
            Eigen::VectorXd get_params() const{
                return m_par;
            }

            /**
             * FUNCTIONS FOR THE KERNEL
             */
            
            /**
             * @brief Set the kernel function
             * 
             * @param kernel the function for the kernel.
             */
            void set_kernel(kernel_t kernel){
                m_kernel=kernel;
            };

            /**
            Set the derivative of the kernel with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods and the CMP method.
            This function must evaluate the derivative of the kernel at x, y with respect to hyperparameter number i.
            */
            void set_kernel_grad(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> kernel_grad) { m_kernel_grad = kernel_grad; }

            /**
            Set the hessian of the kernel with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the kernel at x, y with respect to hyperparameter number i and j.
            */
            void set_kernel_hess(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i, int j)> kernel_hess) { m_kernel_hess = kernel_hess; }

            /**
             * @brief Evaluates the covariance matrix of the kernel 
             * 
             * @return a matrix containing the evaluation of the kernel on the observation points 
             */
            Eigen::MatrixXd covariance(Eigen::VectorXd par) const;

            /**
             * Evaluate the i-th component of the gradient of the covariance matrix.
             * @param i The component of the gradient required
            */
            Eigen::MatrixXd covariance_grad(Eigen::VectorXd par,const int &i) const;

            /**
             * Evaluate the ij component of the hessian of the covariance matrix.
             * @param i row of the hessian matrix
             * @param j colum of the hessian matrix
            */
            Eigen::MatrixXd covariance_hess(Eigen::VectorXd par,const int &i, const int &j) const;

            const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt() const {
                return m_cov_ldlt;
            }
            /** 
             * FUNCTIONS FOR THE MEAN
            */

            /**
             * @brief Set the mean function
             * 
             * @param mean the function for the mean.
             */
            void set_mean(model_t mean){
                m_mean=mean;
            };

            /**
             * @brief Sets the mean gradient function.
             *
             * This function sets the mean gradient function to be used in the Gaussian Process.
             * The mean gradient function is a user-defined function that calculates the mean gradient
             * for a given input vector, target vector, and index.
             *
             * @param mean_gradient The mean gradient function to be set.
             */
            void set_mean_grad(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> mean_grad) { m_mean_grad = mean_grad; }
            
            /**
             * @brief Evaluates the mean on the observations.
             *
             * This function calculates the mean of a set of points given by `x_pts` using the parameters `par`.
             *
             * @return The mean of the points.
             */
            Eigen::VectorXd mean(Eigen::VectorXd par) const;

            /**
             * @brief Evaluate the gradient of the mean function
             * 
             * @param i the index of the hyperparameter
             * @return a vector containing the computation of the gradient
             */
            Eigen::VectorXd mean_grad(Eigen::VectorXd par,const int &i) const;

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
             * @brief Set the prior function
             * 
             * @param prior the function for the prior
             */
            void set_log_prior(prior_t log_prior){
                m_log_prior=log_prior;
            };

            /** 
            Set the derivative of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods.
            This function must evaluate the derivative of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_log_prior_grad(std::function<double(Eigen::VectorXd const &, int i)> log_prior_grad) { m_log_prior_grad = log_prior_grad; }

            /** 
            Set the hessian of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_log_prior_hess(std::function<double(Eigen::VectorXd const &, int i, int j)> log_prior_hess) { m_log_prior_hess = log_prior_hess; }

            /**
             * @brief Evaluates the log prior function
             * 
             * @return the value of the log prior. 
             */
            double log_prior(Eigen::VectorXd par) const{
                return m_log_prior(par);
            }

            /**
             * This function evaluates the gradient of the log prior function.
             * 
             * @param i The index of the hyperparameter
             * 
             * @return The value of the gradient of the log prior function
             */
            double log_prior_grad(Eigen::VectorXd par,const int &i) const{
                return m_log_prior_grad(par,i);
            }

            /**
             * This function evaluates the hessian of the log prior function.
             * 
             * @param i The index of the hyperparameter
             * @param j The index of the hyperparameter
             * 
             * @return The value of the hessian of the log prior function
             */
            double log_prior_hess(Eigen::VectorXd par, const int &i, const int&j) const{
                return m_log_prior_hess(par,i,j);
            }


            /**
             * FUNCTIONS FOR THE PREDICTIVE DISTRIBUTION
             */
            
            /**
             * @brief Compute the prediction variance of the GP at a new prediction point. 
             * 
             * @param x The new prediction point.
             * @return The variance of the prediction
             */
            double predictive_var(Eigen::VectorXd x) const;

            /**
             * @brief Compute the prediction mean of the GP at a new prediction point. 
             * 
             * @param x The value of the new prediction point.
             * @return The mean of the prediction.
             */
            double predictive_mean(Eigen::VectorXd x) const;

            /**
            Draw n random samples of the model error evaluated at x_pts.
            It will draw a sample from the normal distribution N(gp_mean, gp_cov)

            decompose COV = V^T D V
            Where V are the eigenvectors and D the eigenvalues

            then draw z from N(0,I) and compute 
            samples = gp_mean + V*sqrt(D)*z ~ N(gp_mean, gp_cov)

            @param x_pts Points where to evaluate the sample
            @param rng Random number generator
            @param n Number of samples

            @return The evaluation of the random sample in vector format 
            */
            std::vector<Eigen::VectorXd> sample(std::vector<Eigen::VectorXd> x_pts, std::default_random_engine &rng, const size_t &n) const;

            /**
             * @brief Compute the variance reduction matrix after observing a set of points.
             * 
             * @param x_pts The points where the variance reduction is computed 
             * @return Eigen::MatrixXd The variance reduction matrix
             */
            Eigen::MatrixXd expected_variance_improvement(std::vector<Eigen::VectorXd> x_pts, double nu=1e-6) const;

            Eigen::VectorXd expected_variance_improvement(std::vector<Eigen::VectorXd> x_pts, const std::vector<Eigen::VectorXd> &new_x_obs, double nu=1e-6) const;
    };

            double opt_fun_gp_mle(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);

            double opt_fun_gp_map(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);
    
            double opt_fun_gp_mloo(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);

            double opt_fun_gp_mloop(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);
}

#endif // MACRO
