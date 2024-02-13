#ifndef DENSITY_H
#define DENSITY_H

#include "doe.h"
#include "cmp_defines.h"

namespace cmp {

    typedef std::function<vector_t(std::vector<vector_t> const &, vector_t const &)> model_t;
    typedef std::function<double(vector_t const &, vector_t const &, vector_t const &)> kernel_t;
    typedef std::function<double(vector_t const &)> prior_t;

    class density {

        
        public:
            density();
            density(const doe &g);
            density(density const &d);

        
        public: 
            /**
            Set the value of the experimental observations
            @param x_obs a vector containing the x locations (which is representated as a vector_t)
            @param y_obs a vector containing the value of the observations
            */
            void set_obs(std::vector<vector_t> const &x_obs, vector_t const &y_obs);

            /**
            Set the model function. 
            @param model this function is defined as: \n 
                        vector<vector_t> \f$\boldsymbol{x_{obs}}\f$, vector_t \f$\theta \rightarrow \f$ vector_t \f$ m(\boldsymbol{x_{obs}}, \theta)\f$
            */
            void set_model(model_t model) { m_model = model; }
            
            /**
            Set the error kernel function.
            @param err_kernel this function is defined as: \n 
                            vector_t \f$ x \f$, vector_t \f$ x' \f$, vector_t \f$ x \rightarrow \f$  double \f$ k_\psi (x,x') \f$

            */
            void set_err_kernel(kernel_t err_kernel) { m_err_kernel = err_kernel; };
            
            /**
            Set the error mean function.
            @param error_mean this function is defined as \n 
                              vector<vector_t> \f$\boldsymbol{x_{obs}}\f$, vector_t \f$\psi \rightarrow \f$ vector_t \f$ \mu_\psi(\boldsymbol{x_{obs}})\f$
            */
            void set_err_mean(model_t err_mean) { m_err_mean = err_mean; }
            
            /**
            Set the parameters' log-prior function 
            @param log_prior_par this function is defined as \n 
                                vector_t \f$\theta \rightarrow \f$ double \log p( \f$\theta) \f$

            */
            void set_log_prior_par(prior_t log_prior_par) { m_log_prior_par = log_prior_par; };
            
            /**
            Set the hyperparameters' log-prior function 
            @param log_prior_hpar this function is defined as \n 
                                vector_t \f$\psi \rightarrow \f$ double \log p( \f$\psi) \f$

            */
            void set_log_prior_hpar(prior_t log_prior_hpar) { m_log_prior_hpar = log_prior_hpar; };
            
            /**
            Set the bounds of the hyperparameters
            @param lb_hpar the lower bound 
            @param ub_hpar the upper bound
            */
            void set_hpar_bounds(vector_t const &lb_hpar, vector_t const &ub_hpar) {
                m_dim_hpar = lb_hpar.size();
                m_lb_hpar = lb_hpar;
                m_ub_hpar = ub_hpar;
            };


        // Utilities
        public :
            
            /**
            set a new doe
            */
            void set_new_doe(doe const &g);


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


        public :
            /**
            Evaluate the error kernel at the desired points
            
            
            @param x_pts the points
            @param hpar the value of the hyperparamters

            @return The error covariance matrix 
            */
            matrix_t covariance(vector_t const &hpar) const;


            /**
             * Compute the residual vector using the observations provided. 
             * 
             * @param par Value of the model parameters
             * 
             * @return A vector containing the residuals
            */
            vector_t residuals(vector_t const &par) const;

            /**
            Evaluate the loglikelihood (overloaded version)

            @param res the residual vector: y_obs-model(x_obs,par)
            @param ldlt the cholesky decomposition of the global error matrix

            @return the log likelihood \f$ p(\mathcal{D} | \theta, \psi) \f$
            */
            double loglikelihood(vector_t const &res, Eigen::LDLT<matrix_t> const &ldlt) const;

            /**
            Evaluate the loglikelihood 

            @param par current value of the parameters
            @param hpar current value of the hyper-parameters

            @return the log likelihood \f$ p(\mathcal{D} | \theta, \psi) \f$

            */
            double loglikelihood(vector_t const &par, vector_t const &hpar) const;

            /*
            Test if the model parameters are in their bounds
            */
            bool in_bounds_par(vector_t const &pars) const;

            /**
            Test if the model hyperpameters are in their bounds
            */
            bool in_bounds_hpar(vector_t const &hpars) const;

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
            const std::vector<vector_t> *get_grid() const { return &m_grid.m_grid;}
            
            /**
            Get the x locations of the observations

            @return a const pointer to x_obs
            */
            const std::vector<vector_t> *get_x_obs() const { return &m_x_obs;}

            /**
            Get the value of the observations

            @return a const pointer to y_obs
            */
            const vector_t *get_y_obs() const { return &m_y_obs;}

            /** 
            Get the bounds of the hyperparameters

            @return a std::pair containing the lower and upper bounds
            */
            std::pair<vector_t, vector_t> get_hpar_bounds() const { return std::make_pair(m_lb_hpar, m_ub_hpar); }
            
            /**
            Evaluate the model

            @param x_pts a vector<vector_t> containing the value of the required prediction locations
            @param par the value of the parameters

            @return a vector_t containing the model evaluation, \f$ m(\boldsymbol{x_{obs}},\theta) \f$
            */
            vector_t evaluate_model(std::vector<vector_t> const &x_pts, vector_t const &par) const { return m_model(x_pts, par); }

            /**
            Evaluate the gp model error mean function

            @param x_pts a vector<vector_t> containing the value of the required locations
            @param hpar the value of the hyperparameters

            @return a vector_t containing the mean function evaluation, \f$ \mu_\psi(\boldsymbol{x_{obs}}) \f$
            */
            vector_t error_mean(std::vector<vector_t> const &x_pts, vector_t const &hpar) const { return m_err_mean(x_pts, hpar); }
            
            /**
            Evaluate the log prior of the hyperparameters

            @param hpar the value of the hyperparameters

            @return a vector_t containing the log prior, \f$ \log p(\psi) \f$
            */
            double logprior_hpar(vector_t const &hpar) const { return m_log_prior_hpar(hpar); }

            /**
            Evaluate the log prior of the parameters

            @param hpar the value of the parameters

            @return a vector_t containing the log prior, \f$ \log p(\theta) \f$
            */
            double logprior_par(vector_t const &par) const { return m_log_prior_par(par); }
            

        public:

            /**
            Returns the prediction of the calibrated model (without accounting for model error correction)
            
            @param x_pts prediction points
            @param confidence confidence interval (between 0 and 1, usually 0.95)
            
            @return a vector containing \n 
            1) the mean \n 
            2) the lower bound given the confidence interval \n 
            3) the upper bound given the confidence interval \n 
            */
            std::vector<vector_t> pred_calibrated_model(std::vector<vector_t> const &x_pts, double confidence) const;
            
            /**
            Returns the prediction of the calibrated model (accounting for model error correction)
            
            Returns the prediction of the calibrated model (without accounting for model error correction)
            
            @param x_pts prediction points
            
            @return a vector containing \n 
            1) the mean \n 
            2) the lower bound given the confidence interval \n 
            3) the upper bound given the confidence interval \n 

            @note \f$ y_\text{pred} | \theta \f$ is a normal distribution \f$ N(\mu_\theta, \sigma_\theta) \f$ under the Gaussian Process predictive equations \n 
                    Using the law of total expectation: \n 
                    \f$ E(y_\text{pred}) = E(E(y_\text{pred} | \theta)) = E(\mu_\theta) \f$ for which we contruct a monte Carlo estimator \n \n 
                    Using the law of total variance: \n 
                    \f$ \text{var}(y_\text{pred}) = E(\text{var}(y_pred | par)) + \text{var}(E(y_\text{pred} | \theta)) = E(\sigma_\theta) + \text{var}(\mu_\theta) \f$ for which we contruct a Monte Carlo estimator
            */
            std::vector<vector_t> pred_corrected_model(std::vector<vector_t> const &x_pts) const;

            /**
            Return the error model kernel mean conditioned on the value of the observations (with assciated errors)
            */
            vector_t gp_cond_mean(std::vector<vector_t> const &x_pts, vector_t const &par, vector_t const &hpar) const;
            
            /**
            Return the error model kernel-covariance conditioned on the value of the observations (with assciated errors)
            */
            matrix_t gp_cond_var(std::vector<vector_t> const &x_pts, vector_t const &par, vector_t const &hpar) const;
            
            /**
            Draw a random sample of the model error evaluated at x_pts.
            It will draw a sample from the normal distribution N(gp_mean, gp_cov)
            Because the covariance matrix at the prediction points is not guaranteed to be positive definite
            the eigendecomposition algorithm has to be used

            decompose COV = V^T D V
            Where V are the eigenvectors and D the eigenvalues

            then draw z from N(0,I) and compute 
            samples = gp_mean + V*sqrt(D)*z ~ N(gp_mean, gp_cov)
            */
            vector_t draw_gp_sample(std::vector<vector_t> const &x_pts, vector_t const &par, vector_t const &hpar, std::default_random_engine &rng) const;

            /**
            Draw a random sample of the corrected model evaluated at x_pts.
            Corrected model = Calibrated model + model error
            */
            vector_t draw_corrected_model_sample(std::vector<vector_t> const &x_pts, std::default_random_engine &rng) const;

        protected:
            
            doe m_grid;                          ///> grid on the parameter space 

            model_t m_model;                                                    ///> model 
            kernel_t m_err_kernel;                                              ///> kernel function of the model error gp
            model_t m_err_mean;                                                 ///> model error mean function
            prior_t m_log_prior_par;                                            ///> log of the prior density of the parameters
            prior_t m_log_prior_hpar;                                           ///> log of the prior density of the hyperparameters
            std::vector<kernel_t> m_err_kernel_derivatives;                     ///> derivatives of the kernel at the observation points

            vector_t m_lb_hpar;         ///> lower bounds of the hyperparameters
            vector_t m_ub_hpar;         ///> upper bounds of the hyperparameters
            vector_t m_lb_par;          ///> lower bounds of the parameters
            vector_t m_ub_par;          ///> upper bounds of the parameters
            int m_dim_hpar;             ///> number of hyperparameters
            int m_dim_par;              ///> number of parameters

            std::vector<vector_t> m_x_obs;    ///> observations' locations 
            vector_t m_y_obs;                 ///> observations' values

            std::vector<vector_t> m_par_samples;        ///> samples of the parameters from MCMC
            std::vector<vector_t> m_hpar_samples;       ///> samples of the hyperparameters from MCMC


            double beta{1.0};  ///> Beta, for the annealed sampling technique
    };


    class density_opt : public density {
        public:
            /** 
            Initialize using a density class 
            */
            density_opt(density const &d);

            /**
            Set the derivative of the kernel with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods and the CMP method.
            This function must evaluate the derivative of the kernel at x, y with respect to hyperparameter number i.
            */
            void set_err_kernel_gradient(std::function<double(vector_t const &, vector_t const &, vector_t const &, int i)> err_kernel_gradient) { m_err_kernel_gradient = err_kernel_gradient; }

            /**
            Set the hessian of the kernel with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the kernel at x, y with respect to hyperparameter number i and j.
            */
            void set_err_kernel_hessian(std::function<double(vector_t const &, vector_t const &, vector_t const &, int i, int j)> err_kernel_hessian) { m_err_kernel_hessian = err_kernel_hessian; }
            
            /** 
            Set the derivative of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary to use gradient based-optimization methods.
            This function must evaluate the derivative of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_logprior_hpar_gradient(std::function<double(vector_t const &, int i)> logprior_hpar_gradient) { m_logprior_hpar_gradient = logprior_hpar_gradient; }

            /** 
            Set the hessian of the hyperparameters prior with respect to the hyperparameters.
            This function is necessary for the CMP method.
            This function must evaluate the hessian of the log of the hyperparameter prior with respect to hyperparameter number i.
            */
            void set_logprior_hpar_hessian(std::function<double(vector_t const &, int i, int j)> logprior_hpar_hessian) { m_logprior_hpar_hessian = logprior_hpar_hessian; }

            /**
            FMP Calculation of the hyperparameters (uses a gradient free method).
            
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
             * Set the annealing parameter
            */
            void set_beta(double beta); 

            /**
             * Get the annealing constant
            */
            double get_beta() const; 

            /**
             * Evaluate the i-th component of the gradient of the covariance matrix.
             * @param hpar The value of hyper-parameters
             * @param i The component of the gradient required
            */
            matrix_t covariance_gradient(const vector_t &hpar, const int &i) const;

            /**
             * Evaluate the ij component of the hessian of the covariance matrix.
             * @param hpar The value of hyper-parameters
             * @param i row of the hessian matrix
             * @param j colum of the hessian matrix
            */
            matrix_t covariance_hessian(const vector_t &hpar, const int &i, const int &j) const;

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
             * @brief gradient of the prior function wrt the hyper-parameters.
             * Evaluate the i-th component of the gradient of the prior with respect to the hyper-parameters.
             * 
             * @param hpar value of the hyper-parameter
             * @param i component of the gradient
             * 
             * @return The required component of the prior gradient.
            */
            double logprior_hpar_gradient(const vector_t & hpar, const int &i) const{return m_logprior_hpar_gradient(hpar,i);};

            /**
             * @brief hessian of the prior function wrt the hyper-parameters.
             * Evaluate the ij component of the hessian of the prior with respect to the hyper-parameters.
             * 
             * @param hpar value of the hyper-parameter
             * @param i row of the hessian matrix
             * @param j colum of the hessian matrix
             * 
             * @return The required component of the prior hessian.
            */
            double logprior_hpar_hessian(const vector_t & hpar, const int &i, const int &j) const{return m_logprior_hpar_hessian(hpar,i,j);};

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

        protected:
            
            std::function<double(vector_t const &, vector_t const &, vector_t const &, int i)> m_err_kernel_gradient; 
            std::function<double(vector_t const &, vector_t const &, vector_t const &, int i, int j)> m_err_kernel_hessian; 
            std::function<double(vector_t const &, int i)> m_logprior_hpar_gradient;
            std::function<double(vector_t const &, int i, int j)> m_logprior_hpar_hessian;
            double m_beta{1.0};          ///> annealing constant beta                                   
    };

    /**
    Optimize a function using a non-gradient based method.
    @param opt_fun function to optimize in the type of double x, void *data \f$ \rightarrow \f$ double \f$ f(x) \f$ 
    @param data_ptr additional data
    @param x0 initial guess
    @param lb lower bounds
    @param ub upper bounds
    @param ftol_rel realtive tolerance
    @param algorithm the algorithm
    
    @return the value of the function at the maximum \n 

    Suggested algorithms : \n
        1. nlopt::LN_SBPLX for a non-gradient based method \n 
        2. nlopt::LD_TNEWTON_PRECOND_RESTART for a grandient based method \n 

    @note If a gradient based method is used, you should define the gradients
    */
    double opt_routine(nlopt::vfunc opt_fun, void *data_ptr, vector_t &x0, const vector_t &lb, const vector_t &ub, double ftol_rel, nlopt::algorithm alg);

    /**
    Function to be optimized in the Full Maximum a Posteriori method.

    @param x current value of the hyperaparameters
    @param grad the gradient (not used)
    @param data_bit this is a pointer to a pair containg a pointer to a residual vector and the class density_opt
    
    @return computes \f$ \log p(\psi) +  \log p(\mathcal{D} | \theta, \psi) \f$

    @note must be used with a non gradient-based algorithm
    */
    double opt_fun_fmp(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);


    /**
    Function to be optimized in the Full Maximum a Posteriori method.

    @param x current value of the hyperaparameters
    @param grad the gradient at the current value which must be updated
    @param data_bit this is a pointer to a pair containg a pointer to a residual vector and the class density_opt
    
    @return computes \f$ \log p(\psi) +  \log p(\mathcal{D} | \theta, \psi) \f$
    */
    double opt_fun_fmp_grad(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);


    /** 
     * Not currently defined
    */
    double opt_fun_par(const std::vector<double> &x, std::vector<double> &grad, void *data_bit);


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