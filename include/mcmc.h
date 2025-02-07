/**
 * Sampling using the metropolis algorithm
*/

#ifndef MCMC_H
#define MCMC_H

#include "cmp_defines.h"
#include <distribution.h>

namespace cmp {

    class mcmc {

        protected:
            
            double m_score{0.0};                 //  Current value of the score
            
            size_t m_dim;                        // Dimension of the chain

            size_t m_steps{0};                   // Steps done
            size_t m_accepts{0};                 // Accepted candidates

            Eigen::VectorXd m_mean;                     // Sample-mean vector
            Eigen::MatrixXd m_cov;                      // Sample-covariance matrix
        
        
        protected:
            // Proposal distribution
            cmp::proposal_distribution *m_proposal;
            std::uniform_real_distribution<double> m_dist_u{0,1};
            std::default_random_engine m_rng;

        //constructors
        public:
            
            /**
             * @brief Construct a new mcmc chain object 
             * 
             * @param rng the random number generator
             */
            mcmc(cmp::proposal_distribution *proposal, std::default_random_engine &rng, const double &score = -std::numeric_limits<double>::infinity());

            mcmc() = default;
        
        // Functions
        public:

            /**
             * @brief Increase the number of steps
             * 
             */
            void increase_steps() {
                m_steps++;
            }

            /**
             * @brief Perform a single MCMC step (without updating the mean and cov)
             * 
             * @param get_score The score function
             */
            void step(const score_t &get_score, bool DR_STEP = false, double gamma = 0.2);

            /**
             * @brief Accept or reject a candidate
             * 
             * @param par The candidate
             * @param score The score of the candidate
             */
            bool accept(const Eigen::VectorXd &par, double score);

            /**
             * @brief Updates the value of the mean and covariance matrix.
             * 
             */
            void update();

            /**
             * @brief Get a covariance matrix adapted to the samples
             * @note The covariance matrix is adapted to the samples by computing their covariance and multiplying it by a factor of 2.38^2/d
             * 
             * @return Eigen::LLT<Eigen::MatrixXd> 
             */
            Eigen::LDLT<Eigen::MatrixXd> get_adapted_cov();

            /**
             * Reset the value of the mean and of the covariance matrix.
             * Reset the number of steps and number of accepted steps.
             * Does not reset the value of the proposal covariance matrix.
            */
            void reset();

            /**
             * Return the current parameter
            */
            Eigen::VectorXd get_par() const;

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
            double get_acceptance_ratio() const;

            /**
             * Return the mean of the data
            */
            Eigen::VectorXd get_mean() const;

            /**
             * Return the covariance of the data
            */
            Eigen::MatrixXd get_cov() const;

            /**
             * Log the total number of steps, acceptance ratio, data mean and covariance.
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
    Eigen::VectorXd self_correlation_lag(const std::vector<Eigen::VectorXd> &samples, int lag);

    /**
     * @brief computes the mean vector and covariance matrix of some MCMC samples
     * @param samples samples 
     * @return a pair containing the mean and the covariance of the samples
    */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> mean_cov(const std::vector<Eigen::VectorXd> & samples);

    /**
     * @brief computes the correlation length and the effective sample size of some samples
     * @param samples the samples from the mcmc sampling
     * @return a pair containing the correlations lengths and the effective sample size
    */
    std::pair<Eigen::VectorXd, double> single_chain_diagnosis(std::vector<Eigen::VectorXd> samples);


    /**
     * @brief compute the r_hat statistics of multiple chains. Chains have converged if it is less than 1.1
     * 
     * @param chains a vector containing multiple chains sampling the same distribution.
     * @return the r_hat statistics. 
     */
    double r_hat(const std::vector<mcmc> & chains);








    







}

#endif 