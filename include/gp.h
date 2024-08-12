#ifndef GP_H
#define GP_H

#include "cmp_defines.h"
#include <distribution.h>

namespace cmp {

    /**
     * @brief This class implements a Gaussian process, providing algorithms for training and prediction.
     * 
     */
    class gp {

        protected:
            kernel_t m_kernel;
            model_t m_mean;
            prior_t m_logprior;

            std::vector<Eigen::VectorXd> m_x_obs;
            std::vector<double> m_y_obs;

            Eigen::VectorXd m_lb_par;
            Eigen::VectorXd m_ub_par;

            std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> m_mean_gradient;                             ///> Function that returns the i-th component of the mean gradient
            std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> m_kernel_gradient;         ///> Function that returns the i-th component of the kernel gradient
            std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i, int j)> m_kernel_hessian;   ///> Function that returns the ij-th component of the kernel hessian
            std::function<double(Eigen::VectorXd const &, int i)> m_logprior_gradient;                                           ///> Function that returns the i-th component of the log-prior gradient
            std::function<double(Eigen::VectorXd const &, int i, int j)> m_logprior_hessian;                                     ///> Function that returns the ij-th component of the log-prior hessian

        public:

            gp()=default;

            /**
             * @brief Set the kernel function
             * 
             * @param kernel the function for the kernel.
             */
            void set_kernel(kernel_t kernel){m_kernel=kernel;};

            /**
             * @brief Set the mean function
             * 
             * @param mean the function for the mean.
             */
            void set_mean(model_t mean){m_mean=mean;};

            /**
             * @brief Set the prior function
             * 
             * @param prior the function for the prior
             */
            void set_logprior(prior_t logprior){m_logprior=logprior;};

            /**
             * @brief Set the observations
             * 
             * @param x_obs 
             * @param y_obs 
             */
            void set_obs(const std::vector<Eigen::VectorXd> &x_obs, const std::vector<double> &y_obs);

            /**
             * @brief Set the bounds of the gp-parameters (required for optimization)
             * 
             * @param lb_par The lower bound of the gp-parameters.
             * @param ub_par The upper bound of the gp-parameters.
             */
            void set_par_bounds(const Eigen::VectorXd &lb_par, const Eigen::VectorXd &ub_par) {
                m_lb_par = lb_par;
                m_ub_par = ub_par;
            }

            /**
             * @brief Evaluates the covariance matrix of the kernel 
             * 
             * @param par the value of the parameters of the Gaussian Process
             * @return a matrix containing the evaluation of the kernel on the observation points 
             */
            Eigen::MatrixXd covariance(const Eigen::VectorXd &par) const;

            /**
             * @brief Evaluate the difference between the observation and the mean function
             * 
             * @param par the value of the parameters of the Gaussian Process
             * @return a vector containing the computation of the residual
             */
            Eigen::VectorXd residual(const Eigen::VectorXd &par) const;

            /**
            Evaluate the log-likelihood 

            @param res the residual vector: y_obs-model(x_obs,par)
            @param llt the cholesky decomposition of the global error matrix
            @return the log likelihood of the Gaussian Process
            */
            double loglikelihood(Eigen::VectorXd const &res, Eigen::LDLT<Eigen::MatrixXd> const &llt) const;

            /**
             * @brief Evaluates the logprior function
             * 
             * @param par The value of the gp parameters
             * @return the value of the logprior. 
             */
            double logprior(Eigen::VectorXd const &par) const{
                return m_logprior(par);
            }

            double logprior_gradient(Eigen::VectorXd const &par, const int &i) const{
                return m_logprior_gradient(par,i);
            }

            double logprior_hessian(Eigen::VectorXd const &par, const int &i, const int&j) const{
                return m_logprior_hessian(par,i,j);
            }

            
            /**
             * @brief Optimize the parameters of the vector function.
             *
             * This function optimizes the parameters of the vector function using the specified algorithm.
             *
             * @param x0 The initial guess for the parameters.
             * @param ftol_rel The relative tolerance for convergence.
             * @param alg The optimization algorithm to use (default is nlopt::LN_SBPLX).
             * @return The optimized parameters.
             */
            Eigen::VectorXd par_opt(Eigen::VectorXd x0, double ftol_rel, nlopt::algorithm alg = nlopt::LN_SBPLX) const;

            /**
             * \brief Optimize the parameters of the model using leave-one-out cross-validation.
             *
             * This function optimizes the parameters of the model using leave-one-out cross-validation.
             * It takes an initial guess `x0` for the parameters and a relative tolerance `ftol_rel` for convergence.
             *
             * \param x0 The initial guess for the parameters.
             * \param ftol_rel The relative tolerance for convergence.
             * \return The optimized parameters.
             */
            Eigen::VectorXd par_opt_loo(Eigen::VectorXd x0, double ftol_rel) const;

            /**
             * @brief Compute the prediction mean and variance of the GP at a new prediction points. 
             * 
             * @param x The new prediction point.
             * @param par The value of the GP parameters to be used.
             * @param ldlt The Cholesky decomposition of the covariance matrix
             * @param res The residual vector (y_obs - mu(x_obs))
             * @return A vector containing the variance of the prediction at each required point. 
             */
            double prediction_variance(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const Eigen::LDLT<Eigen::MatrixXd> &ldlt) const;

            /**
             * @brief Compute the prediction mean of the GP at a new prediction point. 
             * 
             * @param x The value of the new prediction point.
             * @param par The value of the GP parameters to be used.
             * @param alpha the solution of [K*x = r] where K is the covariance matrix and r is the residual
             * @return The mean of the prediction 
             */
            double prediction_mean(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const Eigen::VectorXd &alpha) const;

            Eigen::MatrixXd predict(const std::vector<Eigen::VectorXd> &x_pts, const Eigen::VectorXd &par, const Eigen::LDLT<Eigen::MatrixXd> &ldlt, const Eigen::VectorXd &res) const;

            /**
            Draw a random sample of the model error evaluated at x_pts.
            It will draw a sample from the normal distribution N(gp_mean, gp_cov)

            decompose COV = V^T D V
            Where V are the eigenvectors and D the eigenvalues

            then draw z from N(0,I) and compute 
            samples = gp_mean + V*sqrt(D)*z ~ N(gp_mean, gp_cov)

            @param x_pts Points where to evaluate the sample
            @param ldlt The Cholesky decomposition of the covariance matrix
            @param par Values of the GP parameters 
            @param rng Random number generator

            @return The evaluation of the random sample in vector format 
            */
            Eigen::VectorXd draw_sample(const std::vector<Eigen::VectorXd> &x_pts, Eigen::VectorXd const &par, const Eigen::LDLT<Eigen::MatrixXd> &ldlt, const Eigen::VectorXd &res, std::default_random_engine &rng) const;


            
            /**
             * @brief Sets the mean gradient function.
             *
             * This function sets the mean gradient function to be used in the Gaussian Process.
             * The mean gradient function is a user-defined function that calculates the mean gradient
             * for a given input vector, target vector, and index.
             *
             * @param mean_gradient The mean gradient function to be set.
             */
            void set_mean_gradient(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> mean_gradient) { m_mean_gradient = mean_gradient; }


            /**
            Set the derivative of the kernel with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods and the CMP method.
            This function must evaluate the derivative of the kernel at x, y with respect to hyperparameter number i.
            */
            void set_kernel_gradient(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> kernel_gradient) { m_kernel_gradient = kernel_gradient; }

            /**
            Set the hessian of the kernel with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the kernel at x, y with respect to hyperparameter number i and j.
            */
            void set_kernel_hessian(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i, int j)> kernel_hessian) { m_kernel_hessian = kernel_hessian; }
            
            /** 
            Set the derivative of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods.
            This function must evaluate the derivative of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_logprior_gradient(std::function<double(Eigen::VectorXd const &, int i)> logprior_gradient) { m_logprior_gradient = logprior_gradient; }

            /** 
            Set the hessian of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_logprior_hessian(std::function<double(Eigen::VectorXd const &, int i, int j)> logprior_hessian) { m_logprior_hessian = logprior_hessian; }


            /**
             * Evaluate the i-th component of the gradient of the covariance matrix.
             * @param par The value of the gp parameters
             * @param i The component of the gradient required
            */
            Eigen::MatrixXd covariance_gradient(const Eigen::VectorXd &par, const int &i) const;

            /**
             * Evaluate the ij component of the hessian of the covariance matrix.
             * @param par The value of the gp parameters
             * @param i row of the hessian matrix
             * @param j colum of the hessian matrix
            */
            Eigen::MatrixXd covariance_hessian(const Eigen::VectorXd &par, const int &i, const int &j) const;

            /**
             * @brief Compute the variance reduction matrix after observing a set of points.
             * 
             * @param x_pts The points where the variance reduction is computed 
             * @param ldlt The Cholesky decomposition of the covariance matrix
             * @param par The value of the GP parameters
             * @return Eigen::MatrixXd The variance reduction matrix
             */
            Eigen::MatrixXd compute_variance_reduction(const std::vector<Eigen::VectorXd> &x_pts, const Eigen::LDLT<Eigen::MatrixXd> &ldlt, const Eigen::VectorXd &par) const;


            Eigen::VectorXd get_lb_par(){return m_lb_par;}
            Eigen::VectorXd get_ub_par(){return m_ub_par;}

            /**
             * @brief Evaluates the mean on the observations.
             *
             * This function calculates the mean of a set of points given by `x_pts` using the parameters `par`.
             *
             * @param par The parameters used for the calculation.
             * @return The mean of the points.
             */
            Eigen::VectorXd evaluate_mean(const Eigen::VectorXd &par) const;

            
            /**
             * @brief Evaluates the mean gradient of the function at the observations.
             *
             * This function calculates the mean gradient of the function at the observations
             * using the provided input points and parameters.
             *
             * @param par The parameters used for evaluation.
             * @param i The component of teh gradient.
             * @return The mean gradient of the function at the specified point.
             */
            Eigen::VectorXd evaluate_mean_gradient(const Eigen::VectorXd &par, int i) const;

            double kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXd par){return m_kernel(x,y,par);}
            double kernel_gradient(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXd par, const int &i){return m_kernel_gradient(x,y,par,i);}
            double kernel_hessian(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXd par, const int &i, const int& j){return m_kernel_hessian(x,y,par,i,j);}


            double logprior(const Eigen::VectorXd &par){return m_logprior(par);}
            double logprior_gradient(const Eigen::VectorXd &par, const int& i){return m_logprior_gradient(par,i);}
            double logprior_hessian(const Eigen::VectorXd &par, const int& i, const int &j){return m_logprior_hessian(par,i,j);}

    };


    /**
    Function to be optimized for the Gaussian Process.
    Optimizes the value of the hyperparameters

    @param x current value of the hyperaparameters
    @param grad the gradient (not used)
    @param data_bit this is a pointer to a pair containing a pointer to a residual vector and the class density_opt
    
    @return computes \f$ \log p(\psi) +  \log p(\mathcal{D} | \psi) \f$

    @note Must be used with a non gradient-based algorithm
    */
    double opt_fun_gp(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);

    double opt_fun_gp_grad(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);
    
    double opt_fun_gp_loo(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);
}

#endif // MACRO
