/**
 * Sampling using the metropolis algorithm
*/

#ifndef MCMC_H
#define MCMC_H

#include "cmp_defines.h"

namespace cmp {

    class mcmc {

        protected:

            std::default_random_engine m_rng;    // Random number generator
            matrix_t m_lt;                       // Lower triangular decomposition of the proposal covariance
            
            vector_t m_par;                      //  Current parameter value
            double m_score{0.0};                 //  Current value of the score
            
            size_t m_dim;                        // Dimension of the chain

            size_t m_steps{0};                   // Steps done
            size_t m_accepts{0};                 // Accepted candidates
            size_t m_updates{0};                 // Number of times the mean has been updated

            vector_t m_mean;                     // Sample-mean vector
            matrix_t m_cov;                      // Sample-covariance matrix
        
        
        protected:
            std::normal_distribution<double> m_dist_n{0,1};
            std::uniform_real_distribution<double> m_dist_u{0,1};

        //constructors
        public:
            
            /**
             * @brief Construct a new mcmc chain object 
             * 
             * @param dim the size of the chain (number of parameters)
             * @param rng the random number generator
             */
            mcmc(size_t dim, std::default_random_engine rng);

            mcmc() = default;
        
        // Functions
        public:

            /**
             * @brief Initialize the chain
             * 
             * @param cov_prop The proposal covariance matrix
             * @param par The proposal value of the parameters
             * @param score The initial score (default is zero)
             */
            void seed(matrix_t cov_prop, vector_t par, double score = 0.0);
            
            /**
             * @brief Propose a candidate using a normal distribution using as mean the previous mean and as covariance the proposal covariance
             * 
             * @return vector_t, The value of the proposed candidate
             */
            vector_t propose();

            /**
             * @brief Perform a single MCMC step (without updating the mean and cov)
             * 
             * @param get_score The score function
             */
            void step(const score_t &get_score);

            /**
             * @brief Perform a single MCMC step and update the mean and covariance
             * 
             * @param get_score The score function
             */
            void step_update(const score_t &get_score);

            /**
             * @brief Accept or reject a candidate
             * 
             * @param par The candidate
             * @param score The score of the candidate
             * @return true if the candidate is accepted
             */
            bool accept(const vector_t &par, double score);

            /**
             * @brief Updates the value of the mean and covariance matrix.
             * 
             */
            void update();

            /**
             * Perform and adaptation of the current covariance matrix using the data covariance matrix. \n 
             * \f$ C_{\text{prop}} = \frac{2.38^2}{d} C_{\text{curr}} \f$ \n 
             * The current covariance matrix is stored and updated automatically at every step.
            */
            void adapt_cov();

            /**
             * Reset the value of the mean and of the covariance matrix.
             * Reset the number of steps and number of accepted steps.
             * Does not reset the value of the proposal covariance matrix.
            */
            void reset();

            /**
             * Return the current parameter
            */
            vector_t get_par() const;

            /**
             * @brief Get the current score
             * 
             * @return the score
             */
            double get_score() const {
                return m_score;
            }

            /**
             * Return the dimension of the chain
            */
            size_t get_dim() const;

            /**
             * Return the number of steps performed
            */
            size_t get_steps() const;

            /**
             * Return the number of steps performed
            */
            double get_acceptance_ratio() const {
                return m_accepts/static_cast<double>(m_steps);
            }

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
    double r_hat(const std::vector<mcmc> & chains);








    







}

#endif 