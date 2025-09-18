/**
 * Sampling using the metropolis algorithm
*/

#ifndef MCMC_H
#define MCMC_H

#include "cmp_defines.h"
#include <distribution.h>

namespace cmp::mcmc {

    class MarkovChain {

        protected:
            
            double score_{0.0};                 //  Current value of the score
            
            size_t dim_;                        // Dimension of the chain

            size_t nSteps_{0};                   // Steps done
            size_t nAccepts_{0};                 // Accepted candidates

            Eigen::VectorXd mean_;                     // Sample-mean vector
            Eigen::MatrixXd cov_;                      // Sample-covariance matrix
        
        
        protected:
            // Proposal distribution
            cmp::distribution::ProposalDistribution *proposal_;
            std::uniform_real_distribution<double> distU_{0,1};
            std::default_random_engine rng_;

        //constructors
        public:
            
            /**
             * @brief Construct a new mcmc chain object 
             * 
             * @param rng the random number generator
             */
            MarkovChain(cmp::distribution::ProposalDistribution *proposal, std::default_random_engine &rng, const double &score = -std::numeric_limits<double>::infinity());

            MarkovChain() = default;
        
        // Functions
        public:

            /**
             * @brief Increase the number of steps
             * 
             */
            void increaseSteps() {
                nSteps_++;
            }

            /**
             * @brief Perform a single MCMC step (without updating the mean and cov)
             * 
             * @param getScore The score function
             */
            void step(const score_t &getScore, bool DR_STEP = false, double gamma = 0.2);

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
            Eigen::LDLT<Eigen::MatrixXd> getAdaptedCovariance();

            /**
             * Reset the value of the mean and of the covariance matrix.
             * Reset the number of steps and number of accepted steps.
             * Does not reset the value of the proposal covariance matrix.
            */
            void reset();

            /**
             * Return the current parameter
            */
            Eigen::VectorXd getCurrent() const;

            /**
             * @brief Get the current score
             * 
             * @return the score
             */
            double getScore() const {
                return score_;
            }

            /**
             * Return the dimension of the chain
            */
            size_t getDim() const;

            /**
             * Return the number of steps performed
            */
            size_t getSteps() const;

            /**
             * Return the number of steps performed
            */
            double getAcceptanceRatio() const;

            /**
             * Return the mean of the data
            */
            Eigen::VectorXd getMean() const;

            /**
             * Return the covariance of the data
            */
            Eigen::MatrixXd getCovariance() const;

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
    Eigen::VectorXd selfCorrelation(const std::vector<Eigen::VectorXd> &samples, int lag);

    /**
     * @brief computes the mean vector and covariance matrix of some MCMC samples
     * @param samples samples 
     * @return a pair containing the mean and the covariance of the samples
    */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> samplesStatistics(const std::vector<Eigen::VectorXd> & samples);

    /**
     * @brief computes the correlation length and the effective sample size of some samples
     * @param samples the samples from the mcmc sampling
     * @return a pair containing the correlations lengths and the effective sample size
    */
    std::pair<Eigen::VectorXd, double> singleChainDiagnosis(std::vector<Eigen::VectorXd> samples);


    /**
     * @brief compute the r_hat statistics of multiple chains. Chains have converged if it is less than 1.1
     * 
     * @param chains a vector containing multiple chains sampling the same distribution.
     * @return the r_hat statistics. 
     */
    double multiChainDiagnosis(const std::vector<MarkovChain> & chains);








    







}

#endif 