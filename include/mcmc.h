/**
 * Sampling using the metropolis algorithm
*/

#ifndef MCMC_H
#define MCMC_H

#include "cmp_defines.h"

namespace cmp {

    /**
     * A class for generating samples from a Markov Chain.
     * The code supports the separation between parameters, which are sampled, and
     * hyperparameters, which are computed using the get_hpar function. 
     * For this reason, it can be adapted to work with Fully Bayesian or Modular approaches.
    */
    class mcmc_chain {

        
        protected:
            matrix_t m_cov_prop; ///<proposal covariance matrix
            vector_t m_par; ///<current parameter value
            vector_t m_hpar; ///<current hyperparameter value
            double m_score; ///<current value of the score
            score_t m_compute_score; ///<log - likelihood function
            get_hpar_t m_get_hpar; ///< compute the hyperparameters
            std::default_random_engine m_rng; ///<random number generator
            matrix_t m_lt; ///<lower triangular decomposition of the proposal covariance
            
            size_t m_dim_par;
            size_t m_dim_hpar;

            in_bounds_t m_in_bounds;

            size_t m_steps{0};
            size_t m_accepts{0};

            vector_t m_mean;
            matrix_t m_cov;
        
        
        protected:
            std::normal_distribution<double> m_dist_n{0,1};
            std::uniform_real_distribution<double> m_dist_u{0,1};

        //constructors
        public:
        /**
         * Deafult constructor. 
         * To correctly initialize the chain one must supply:
         * @param cov_prop A proposal covariance matrix that will be used to propse samples.
         * @param par An initial value for the sampled parameters.
         * @param hpar An initial value for the computed hyperparameters.
         * @param score A function to evaluate the score. 
         *                         Defined as vector_t parameter \f$\theta\f$ , vector_t hyperparameter \f$\psi\f$ \f$\rightarrow\f$ double \f$ \log p (\theta | \psi(\theta), \mathcal{D}) \f$
         * @param get_hpar A function that computes the chosen value of the hyperparameters given the proposed value of the parameters and the previous value of the hyperparameters (used as an initial guess).
         * Defined as vector_t parameter \f$\theta\f$ , vector_t hyperparameter \f$\psi_{-1}\f$ \f$\rightarrow\f$ vector_t \f$\psi(\theta)\f$
         * @param in_bounds A function that checks if the proposed parameter is in it's bounds. Defined as vector_t parameter \f$\theta\f$ \f$\rightarrow\f$ bool in_bounds.
         * @param rng A random number generator.
        */
            mcmc_chain(matrix_t cov_prop, vector_t par, vector_t hpar, score_t score, get_hpar_t get_hpar, in_bounds_t in_bounds,std::default_random_engine rng):
                m_cov_prop(cov_prop), m_par(par), m_hpar(hpar), m_compute_score(score), m_get_hpar(get_hpar), m_rng(rng), m_in_bounds(in_bounds){
                    
                    //compute the cholesky decomposition
                    m_lt = cov_prop.llt().matrixL();

                    m_dim_par = par.size();
                    m_dim_hpar = hpar.size();

                    m_mean = vector_t(par.size());
                    m_cov = matrix_t(par.size(), par.size());

                    for(int i=0; i<par.size(); i++) {
                        m_mean(i)=0.0;
                        for(int j=0; j < par.size(); j++) {
                            m_cov(i,j)=0.0;
                        }
                    }

                    m_score = score(par,hpar);
                    
                    spdlog::info("Setting up mcmc chain with initial value \n{0}\n and covariance \n{1}", par,cov_prop);
                };
            
            mcmc_chain(const mcmc_chain & other) = default;
            
            mcmc_chain &operator=(const mcmc_chain &other) = default;

            mcmc_chain() = default;
        
        // Functions
        public:
            
            /**
             * @brief Perform a single chain step. 
             * @note does not update the mean and the covariance function, to do so you must call update()
             **/ 
            void step();

            /**
             * @brief Perform a single chain step and updates the mean and covariance matrix.
             **/ 
            void step_update();

            /**
             * @brief update the mean vector and covariance matrix.
             * The method simply sums the current sample and its cross product to the mean and covariance matrices
             * @note The actual mean and covariances is computed when adap_cov() or get_mean_cov() are called
            */
            void update();

            /**
             * Perform and adaptation of the current covariance matrix using the data covariance matrix. \n 
             * \f$ C_{\text{prop}} = \frac{2.38^2}{d} C_{\text{curr}} \f$ \n 
             * The current covariance matrix is stored and updated automatically at every step.
            */
            void adapt_cov();

            /**
             * Reset the value of the mean and of the covaraince matrix.
             * Reset the number of steps and number of accepted steps.
             * Does @b not reset the value of the proposal covariance matrix.
            */
            void reset();

            /**
             * Return the current parameter
            */
            vector_t get_par() const;

            /**
             * Return the current hyperparameter
            */
            vector_t get_hpar() const;

            /**
             * Return the dimension of the chain
            */
            size_t get_dim() const;

            /**
             * Return the number of steps performed
            */
            size_t get_steps() const;

            /**
             * Evaluate the current mean vector and covariance matrix of the samples.
            */
            std::pair<vector_t, matrix_t> get_mean_cov() const;

            /**
             * Log the total number of steps, acceptance ratio, proposal covariance and data mean and covariance.
            */
            void info() const;
            
    };

    /**
     * @brief Compute the lagged self correlation between samples
     * @param samples samples on which to compute the self correlation
     * @param lag the lag, k, (between 0 and n-1 where n is the samples size)
     * 
     * @return the lagged self correlation \f$ \frac{1}{n} \sum_{i=1}^{n-k} \theta_i \theta_{i+k}\f$
     * 
    */
    vector_t self_correlation_lag(const std::vector<vector_t> &samples, int lag);

    /**
     * @brief computes the mean vector and covariance matrix of some MCMC samples
     * @param samples samples 
     * @return a pair containing the mean and the covariance of the samples
    */
    std::pair<vector_t, matrix_t> mean_cov(const std::vector<vector_t> & samples);

    /**
     * @brief computes the correlation length and the effective sample size of some samples
     * @param samples the samples from the mcmc sampling
     * @return a pair containing the correlations lengths and the effective sample size
    */
    std::pair<vector_t, double> single_chain_diagnosis(std::vector<vector_t> samples);


    /**
     * @brief compute the r_hat statistics of multiple chains. Chains have converged if it is less than 1.1
     * 
     * @param chains a vector containing multiple chains sampling the same distribution.
     * @return the r_hat statistics. 
     */
    double r_hat(const std::vector<mcmc_chain> & chains);








    







}

#endif 