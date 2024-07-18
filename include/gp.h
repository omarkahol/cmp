#ifndef GP_H
#define GP_H

#include "cmp_defines.h"

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

            std::vector<vector_t> m_x_obs;
            std::vector<double> m_y_obs;

            vector_t m_lb_par;
            vector_t m_ub_par;

            std::function<double(vector_t const &, vector_t const &, int i)> m_mean_gradient;                             ///> Function that returns the i-th component of the mean gradient
            std::function<double(vector_t const &, vector_t const &, vector_t const &, int i)> m_kernel_gradient;         ///> Function that returns the i-th component of the kernel gradient
            std::function<double(vector_t const &, vector_t const &, vector_t const &, int i, int j)> m_kernel_hessian;   ///> Function that returns the ij-th component of the kernel hessian
            std::function<double(vector_t const &, int i)> m_logprior_gradient;                                           ///> Function that returns the i-th component of the log-prior gradient
            std::function<double(vector_t const &, int i, int j)> m_logprior_hessian;                                     ///> Function that returns the ij-th component of the log-prior hessian
        
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
            void set_obs(const std::vector<vector_t> &x_obs, const std::vector<double> &y_obs);

            /**
             * @brief Set the bounds of the gp-parameters (required for optimization)
             * 
             * @param lb_par The lower bound of the gp-parameters.
             * @param ub_par The upper bound of the gp-parameters.
             */
            void set_par_bounds(const vector_t &lb_par, const vector_t &ub_par) {
                m_lb_par = lb_par;
                m_ub_par = ub_par;
            }

            /**
             * @brief Evaluates the covariance matrix of the kernel 
             * 
             * @param par the value of the parameters of the Gaussian Process
             * @return a matrix containing the evaluation of the kernel on the observation points 
             */
            matrix_t covariance(const vector_t &par) const;

            /**
             * @brief Evaluate the difference between the observation and the mean function
             * 
             * @param par the value of the parameters of the Gaussian Process
             * @return a vector containing the computation of the residual
             */
            vector_t residual(const vector_t &par) const;

            /**
            Evaluate the log-likelihood 

            @param res the residual vector: y_obs-model(x_obs,par)
            @param ldlt the cholesky decomposition of the global error matrix
            @return the log likelihood of the Gaussian Process
            */
            double loglikelihood(vector_t const &res, Eigen::LDLT<matrix_t> const &ldlt) const;

            /**
             * @brief Evaluates the logprior function
             * 
             * @param par The value of the gp parameters
             * @return the value of the logprior. 
             */
            double logprior(vector_t const &par) const{
                return m_logprior(par);
            }

            double logprior_gradient(vector_t const &par, const int &i) const{
                return m_logprior_gradient(par,i);
            }

            double logprior_hessian(vector_t const &par, const int &i, const int&j) const{
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
            vector_t par_opt(vector_t x0, double ftol_rel, nlopt::algorithm alg = nlopt::LN_SBPLX) const;

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
            vector_t par_opt_loo(vector_t x0, double ftol_rel) const;

            /**
             * @brief Compute the prediction mean and variance of the GP at some new prediction points (overloaded version)
             * 
             * @param x_pts The prediction points
             * @param par The value of the GP parameters
             * @return a matrix containing as first colum the mean and as second colum the variance
             */
            matrix_t predict(const std::vector<vector_t> &x_pts, const vector_t &par) const;


            /**
             * @brief Compute the prediction mean and variance of the GP at some new prediction points. 
             * 
             * @param x_pts The value of the required prediction points.
             * @param par The value of the GP parameters to be used.
             * @param ldlt The Cholesky decomposition of the covariance matrix
             * @param res The residual vector (y_obs - mu(x_obs))
             * @return A vector containing the variance of the prediction at each required point. 
             */
            matrix_t predict(const std::vector<vector_t> &x_pts, const vector_t &par, const Eigen::LDLT<matrix_t> &ldlt, const vector_t &res) const;

            /**
             * @brief Compute the prediction mean of the GP at a new prediction point. 
             * 
             * @param x The value of the new prediction point.
             * @param par The value of the GP parameters to be used.
             * @param alpha the solution of [K*x = r] where K is the covariance matrix and r is the residual
             * @return The mean of the prediction 
             */
            double prediction_mean(const vector_t &x, const vector_t &par, const vector_t &alpha) const;

            /**
            Draw a random sample of the model error evaluated at x_pts.
            It will draw a sample from the normal distribution N(gp_mean, gp_cov)

            decompose COV = V^T D V
            Where V are the eigenvectors and D the eigenvalues

            then draw z from N(0,I) and compute 
            samples = gp_mean + V*sqrt(D)*z ~ N(gp_mean, gp_cov)

            @param x_pts Points where to evaluate the sample
            @param par Values of the GP parameters 
            @param rng Random number generator

            @return The evaluation of the random sample in vector format 
            */
            vector_t draw_sample(const std::vector<vector_t> &x_pts, vector_t const &par, std::default_random_engine &rng) const;


            
            /**
             * @brief Sets the mean gradient function.
             *
             * This function sets the mean gradient function to be used in the Gaussian Process.
             * The mean gradient function is a user-defined function that calculates the mean gradient
             * for a given input vector, target vector, and index.
             *
             * @param mean_gradient The mean gradient function to be set.
             */
            void set_mean_gradient(std::function<double(vector_t const &, vector_t const &, int i)> mean_gradient) { m_mean_gradient = mean_gradient; }


            /**
            Set the derivative of the kernel with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods and the CMP method.
            This function must evaluate the derivative of the kernel at x, y with respect to hyperparameter number i.
            */
            void set_kernel_gradient(std::function<double(vector_t const &, vector_t const &, vector_t const &, int i)> kernel_gradient) { m_kernel_gradient = kernel_gradient; }

            /**
            Set the hessian of the kernel with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the kernel at x, y with respect to hyperparameter number i and j.
            */
            void set_kernel_hessian(std::function<double(vector_t const &, vector_t const &, vector_t const &, int i, int j)> kernel_hessian) { m_kernel_hessian = kernel_hessian; }
            
            /** 
            Set the derivative of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods.
            This function must evaluate the derivative of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_logprior_gradient(std::function<double(vector_t const &, int i)> logprior_gradient) { m_logprior_gradient = logprior_gradient; }

            /** 
            Set the hessian of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_logprior_hessian(std::function<double(vector_t const &, int i, int j)> logprior_hessian) { m_logprior_hessian = logprior_hessian; }


            /**
             * Evaluate the i-th component of the gradient of the covariance matrix.
             * @param par The value of the gp parameters
             * @param i The component of the gradient required
            */
            matrix_t covariance_gradient(const vector_t &par, const int &i) const;

            /**
             * Evaluate the ij component of the hessian of the covariance matrix.
             * @param par The value of the gp parameters
             * @param i row of the hessian matrix
             * @param j colum of the hessian matrix
            */
            matrix_t covariance_hessian(const vector_t &par, const int &i, const int &j) const;

            matrix_t reduced_covariance_matrix(const std::vector<vector_t> x_pts, const vector_t & par);

            double loglikelihood_gradient(const vector_t &res, const Eigen::LDLT<matrix_t> &ldlt, const vector_t &par, const int &n) const;

            vector_t get_lb_par(){return m_lb_par;}
            vector_t get_ub_par(){return m_ub_par;}

            /**
             * @brief Evaluates the mean of a set of points.
             *
             * This function calculates the mean of a set of points given by `x_pts` using the parameters `par`.
             *
             * @param x_pts The vector of points to calculate the mean for.
             * @param par The parameters used for the calculation.
             * @return The mean of the points.
             */
            vector_t evaluate_mean(const std::vector<vector_t> &x_pts, const vector_t &par) const;

            
            /**
             * @brief Evaluates the mean gradient of the function at a given point.
             *
             * This function calculates the mean gradient of the function at a specific point
             * using the provided input points and parameters.
             *
             * @param x_pts The input points used for evaluation.
             * @param par The parameters used for evaluation.
             * @param i The component of teh gradient.
             * @return The mean gradient of the function at the specified point.
             */
            vector_t evaluate_mean_gradient(const std::vector<vector_t> &x_pts, const vector_t &par, int i) const;

            double kernel(const vector_t &x, const vector_t &y, const vector_t par){return m_kernel(x,y,par);}
            double kernel_gradient(const vector_t &x, const vector_t &y, const vector_t par, const int &i){return m_kernel_gradient(x,y,par,i);}
            double kernel_hessian(const vector_t &x, const vector_t &y, const vector_t par, const int &i, const int& j){return m_kernel_hessian(x,y,par,i,j);}


            double logprior(const vector_t &par){return m_logprior(par);}
            double logprior_gradient(const vector_t &par, const int& i){return m_logprior_gradient(par,i);}
            double logprior_hessian(const vector_t &par, const int& i, const int &j){return m_logprior_hessian(par,i,j);}

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
