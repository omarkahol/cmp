#ifndef DENSITY_H
#define DENSITY_H

#include "grid.h"
#include "cmp_defines.h"
#include "gp.h"

namespace cmp {

    class density {

        // Member data
        protected:
            
            grid *m_grid;                                                       ///> Pointer to the grid on the parameter space 
            gp *m_model_error;                                                  ///> Pointer to the model error Gaussian Process
            model_t m_model;                                                    ///> model
            
            prior_t m_log_prior_par;                                            ///> log of the prior density of the parameters

            vector_t m_lb_par;          ///> lower bounds of the parameters
            vector_t m_ub_par;          ///> upper bounds of the parameters
            
            std::vector<vector_t> m_x_obs;    ///> observations' locations 
            std::vector<double> m_y_obs;                 ///> observations' values

            std::vector<vector_t> m_par_samples;        ///> samples of the parameters from MCMC
            std::vector<vector_t> m_hpar_samples;       ///> samples of the hyperparameters from MCMC

        public:
            
            density(grid *g);
            density(){

            };

        public: 

            
            /**
            Set the value of the experimental observations
            @param x_obs a vector containing the x locations (which is representated as a vector_t)
            @param y_obs a vector containing the value of the observations
            */
            void set_obs(const std::vector<vector_t> &x_obs, const std::vector<double> &y_obs);

            /**
            Set the model function. 
            @param model this function is defined as: \n 
                        vector<vector_t> \f$\boldsymbol{x_{obs}}\f$, vector_t \f$\theta \rightarrow \f$ vector_t \f$ m(\boldsymbol{x_{obs}}, \theta)\f$
            */
            void set_model(model_t model) { m_model = model; }


            void set_model_error(gp *model_error) {m_model_error=model_error;}
            
            /**
            Set the parameters' log-prior function 
            @param log_prior_par this function is defined as \n 
                                vector_t \f$\theta \rightarrow \f$ double \log p( \f$\theta) \f$

            */
            void set_log_prior_par(prior_t log_prior_par) { m_log_prior_par = log_prior_par; };

            /**
            set the samples of the parameters computed with MCMC
            @param s vector of the samples of the parameters
            */
            void set_par_samples(std::vector<vector_t> const &s) { m_par_samples = s; }

            /**
            set the samples of the hyper-parameters computed with MCMC
            @param s vector of the samples of the hyperparameters corresponding to the parameters
            */
            void set_hpar_samples(std::vector<vector_t> const &s) { m_hpar_samples = s; }

            /**
             * Compute the residual vector using the observations provided. 
             * @param par Value of the model parameters
             * @return A vector containing the residuals
            */
            vector_t residuals(vector_t const &par) const;

            /**
            Evaluate the log-likelihood (overloaded version)
            @param res the residual vector: y_obs-model(x_obs,par)
            @param ldlt the cholesky decomposition of the global error matrix
            @return the log likelihood \f$ p(\mathcal{D} | \theta, \psi) \f$
            */
            double loglikelihood(vector_t const &res, Eigen::LDLT<matrix_t> const &ldlt) const;

            /**
            Evaluate the log-likelihood
            @param par current value of the parameters
            @param hpar current value of the hyper-parameters
            @return the log likelihood \f$ p(\mathcal{D} | \theta, \psi) \f$
            */
            double loglikelihood(vector_t const &par, vector_t const &hpar) const;

            /**
            Compute the model hyperparameters using the Kennedy O'Hagan method.
            @param hpar_guess an initial guess
            @param int_const the log of an integration constant. Should be in the order of magnitude of \f$ \log \text{max} p(\theta,\psi) \ p(\mathcal{D} | \theta, \psi) \f$
                and is used to avoid numerical errors
            @param ftol_rel the relative tolerance
            @return the KOH oprimization vector_t \f$ \psi_{\text{KOH}} = \text{argmax} \;\; \int_{\Theta} p(\mathcal{D} | \psi, \theta) p(\psi, \theta) \ \text{d}\theta \f$
            */
            vector_t hpar_KOH(vector_t const &hpar_guess, double int_const ,double ftol_rel) const;

            /** 
            Get the uniform grid on the model parameters
            @return A const pointer to the grid address
            */
            const std::vector<vector_t> *get_grid() const { return &m_grid->m_grid;}
            
            /**
            Get the x locations of the observations
            @return a const pointer to x_obs
            */
            const std::vector<vector_t> *get_x_obs() const { return &m_x_obs;}

            /**
            Get the value of the observations
            @return a const pointer to y_obs
            */
            const std::vector<double> *get_y_obs() const { return &m_y_obs;}

            
            /**
            Evaluate the model
            @param x_pts a matrix containing the value of the required prediction locations
            @param par the value of the parameters
            @return a vector_t containing the model evaluation, \f$ m(\boldsymbol{x_{obs}},\theta) \f$
            */
            vector_t evaluate_model(const std::vector<vector_t> &x_pts, vector_t const &par) const;

            /**
            Evaluate the log prior of the parameters
            @param hpar the value of the parameters
            @return a vector_t containing the log prior, \f$ \log p(\theta) \f$
            */
            double logprior_par(vector_t const &par) const { return m_log_prior_par(par); }

            /**
            Returns the prediction of the calibrated model (without accounting for model error correction)
            @param x_pts prediction points
            @param confidence confidence interval (between 0 and 1, usually 0.95)
            @return a matrix containing \n 
            1) the mean \n 
            2) the lower bound given the confidence interval \n 
            3) the upper bound given the confidence interval \n 
            */
            matrix_t pred_calibrated_model(const std::vector<vector_t> &x_pts, double confidence) const;
            
            /**
            Returns the prediction of the calibrated model (accounting for model error correction)
            Returns the prediction of the calibrated model (without accounting for model error correction)
            @param x_pts prediction points
            @return a matrix containing the mean and variance

            @note \f$ y_\text{pred} | \theta \f$ is a normal distribution \f$ N(\mu_\theta, \sigma_\theta) \f$ under the Gaussian Process predictive equations \n 
                    Using the law of total expectation: \n 
                    \f$ E(y_\text{pred}) = E(E(y_\text{pred} | \theta)) = E(\mu_\theta) \f$ for which we contruct a monte Carlo estimator \n \n 
                    Using the law of total variance: \n 
                    \f$ \text{var}(y_\text{pred}) = E(\text{var}(y_pred | par)) + \text{var}(E(y_\text{pred} | \theta)) = E(\sigma_\theta) + \text{var}(\mu_\theta) \f$ for which we contruct a Monte Carlo estimator
            */
            matrix_t pred_corrected_model(const std::vector<vector_t> &x_pts) const;

            /**
            Draw a random sample of the corrected model evaluated at x_pts.
            Corrected model = Calibrated model + model error
            */
            vector_t draw_corrected_model_sample(const std::vector<vector_t> &x_pts, std::default_random_engine &rng) const;

            /**
            CMP Calculation of the hyperparameters (uses a gradient free method).
            @param par the current value of the parameters
            @param x0 guess for the hyperparameters
            @param ftol_rel tolerance

            @return the MAP value of the hyperparameters vector_t \f$ \psi_\text{FMP} = \text{argmax} \;\; p(\mathcal{D} | \psi, \theta) p(\psi, \theta) \f$
            */
            vector_t hpar_opt(vector_t const &par, vector_t x0, double ftol_rel) const;

            /**
            FMP Calculation of the hyperparameters (uses a gradient free method). Note, the gradients must be available
            
            @param par the current value of the parameters
            @param x0 guess for the hyperparameters
            @param ftol_rel tolerance

            @return the MAP value of the hyperparameters vector_t \f$ \psi_\text{FMP} = \text{argmax} \;\; p(\mathcal{D} | \psi, \theta) p(\psi, \theta) \f$
            */                                                 
            vector_t hpar_opt_grad(vector_t const &par, vector_t x0, double ftol_rel) const;

            /**
             * @brief gradient of the log-likelihood function wrt the hyper-parameters.
             * Evaluate the i-th component of the gradient of the likelihood with respect to the hyper-parameters.
             * 
             * @param hpar value of the hyper-parameter
             * @param cov_inv cholesky decomposition solver of the covariance matrix
             * @param res residual 
             * @param i component of the gradient
             * 
             * @return the required component of the likelihood gradient.
             * 
             * @note Uses the formula from Rasmussen.
            */
            double loglikelihood_gradient(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res, const int &i) const;

            /**
             * @brief hessian of the log-likelihood function wrt the hyper-parameters.
             * Evaluate the ij component of the hessian of the likelihood with respect to the hyper-parameters.
             * 
             * @param hpar value of the hyper-parameter
             * @param cov_inv cholesky decomposition solver of the covariance matrix
             * @param res residual 
             * @param i row of the hessian matrix
             * @param j colum of the hessian matrix
             * 
             * @return the required component of the hessian.
             * 
             * @note Uses the formula from Rasmussen.
            */
            double loglikelihood_hessian(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res, const int &i, const int &j) const;

            /**
             * @brief computes the logarithm of the correction factor for the CMP method.
             * The correction factor for the CMP method is defined, in log units, as \f$ -\frac{1}{2} \mid H \mid\f$ where H is the hessian of the posterior.
             *
             * @param hpar value of the hyper-parameter
             * @param cov_inv cholesky decomposition solver of the covariance matrix
             * @param res residual 
             * 
             * @return The CMP correction factor in log units (already negated).
            */
            double log_cmp_correction(const vector_t &hpar, const Eigen::LDLT<matrix_t> &cov_inv, const vector_t &res);

            gp *model_error() const{return m_model_error;}
    };

    /**
    Function to be optimized in the Complete Maximum a Posteriori method.

    @param x current value of the hyperaparameters
    @param grad the gradient (not used)
    @param data_bit this is a pointer to a pair containg a pointer to a residual vector and the class density_opt
    
    @return computes \f$ \log p(\psi) +  \log p(\mathcal{D} | \theta, \psi) \f$

    @note must be used with a non gradient-based algorithm
    */
    double opt_fun_cmp(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);


    /**
    Function to be optimized in the Complete Maximum a Posteriori method.

    @param x current value of the hyperaparameters
    @param grad the gradient at the current value which must be updated
    @param data_bit this is a pointer to a pair containg a pointer to a residual vector and the class density_opt
    
    @return computes \f$ \log p(\psi) +  \log p(\mathcal{D} | \theta, \psi) \f$
    */
    double opt_fun_cmp_grad(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);


    /**
    Function to be optimized in the Kennedy O'Hagan method.

    @param x current value of the hyperaparameters
    @param grad the gradient (not used)
    @param data this is a pointer to a pair containg a pointer to the class density_opt and an integration constant
    
    @return computes an approximation of \f$ \log \int_{\Theta} p(\mathcal{D} | \psi, \theta) p(\psi, \theta) \ \text{d}\theta \f$ \n

    @note must be used with a non gradient-based algorithm  
    @note this function does not compute the actual integral as it would be too expensive and useless. To avoid it, it sums 
    the value of the \f$ p(\mathcal{D} | \psi, \theta) p(\psi, \theta) \f$ at the grid points. For uniform grid, maximizing this value is equivalent to maximizing the integral.
    @note the optimization can fail if the product becomes too large so the integral is actually divided by a large constant.
    */
    double opt_fun_KOH(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);
}

#endif