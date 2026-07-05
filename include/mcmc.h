#ifndef MCMC_H
#define MCMC_H

#include "cmp_defines.h"
#include "distribution.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <random>
#include <Eigen/Dense>

/**
 * @addtogroup sampling
 * @{
 */
namespace cmp::mcmc {

// ==============================================================================
// HELPER FUNCTIONS
// ==============================================================================

// Numerically safe log(1 - α)
inline double log1m(double a) {
    return (a < 1.0) ? std::log1p(-a) : -std::numeric_limits<double>::infinity();
}

// ==============================================================================
// MARKOV CHAIN CLASS TEMPLATE
// ==============================================================================


/**
 * @class MarkovChain
 * @brief Represents a single Markov Chain utilizing Delayed Rejection Adaptive Metropolis (DRAM) for sampling.
 *
 * @tparam ProposalType The class type of the proposal distribution used to generate candidate states.
 *
 * @details
 * ### Mathematical Foundations & DRAM Algorithm
 * This class implements **Delayed Rejection Adaptive Metropolis (DRAM)**, combining adaptive covariance tuning with multi-stage candidate proposals:
 * 
 * 1. **Stage 1 (Adaptive Metropolis)**:
 *    A candidate state \f$\boldsymbol{\theta}^*\f$ is proposed from a Gaussian distribution centered at the current state \f$\boldsymbol{\theta}^{(t)}\f$
 *    with scaling covariance \f$\boldsymbol{\Sigma}_t\f$:
 *    \f[ \boldsymbol{\theta}^* \sim \mathcal{N}\left(\boldsymbol{\theta}^{(t)}, \gamma_1 \boldsymbol{\Sigma}_t\right) \f]
 *    The Stage 1 acceptance probability is:
 *    \f[ \alpha_1(\boldsymbol{\theta}^{(t)}, \boldsymbol{\theta}^*) = \min\left(1, \frac{\pi(\boldsymbol{\theta}^*)}{\pi(\boldsymbol{\theta}^{(t)})}\right) \f]
 *    where \f$\pi(\boldsymbol{\theta})\f$ is the target density.
 *
 * 2. **Stage 2 (Delayed Rejection)**:
 *    If the Stage 1 proposal is rejected, a Stage 2 candidate \f$\boldsymbol{\theta}^{**}\f$ is drawn from a narrower proposal distribution:
 *    \f[ \boldsymbol{\theta}^{**} \sim \mathcal{N}\left(\boldsymbol{\theta}^{(t)}, \gamma_2 \boldsymbol{\Sigma}_t\right) \f]
 *    with \f$\gamma_2 < \gamma_1\f$. To preserve **detailed balance**, the Stage 2 acceptance probability must account for the first-stage rejection:
 *    \f[ \alpha_2(\boldsymbol{\theta}^{(t)}, \boldsymbol{\theta}^*, \boldsymbol{\theta}^{**}) = \min\left(1, \frac{\pi(\boldsymbol{\theta}^{**}) q_1(\boldsymbol{\theta}^* | \boldsymbol{\theta}^{**}) [1 - \alpha_1(\boldsymbol{\theta}^{**}, \boldsymbol{\theta}^*)]}{\pi(\boldsymbol{\theta}^{(t)}) q_1(\boldsymbol{\theta}^* | \boldsymbol{\theta}^{(t)}) [1 - \alpha_1(\boldsymbol{\theta}^{(t)}, \boldsymbol{\theta}^*)]}\right) \f]
 *    where \f$q_1(\mathbf{x} | \mathbf{y})\f$ is the probability density of proposing \f$\mathbf{x}\f$ given current state \f$\mathbf{y}\f$ under the Stage-1 covariance.
 *
 * 3. **Adaptive Covariance Scaling**:
 *    The proposal covariance \f$\boldsymbol{\Sigma}_t\f$ is updated recursively using the sample covariance of the history of the chain:
 *    \f[ \boldsymbol{\Sigma}_t = s_d \mathrm{Cov}\left(\boldsymbol{\theta}^{(1)}, \dots, \boldsymbol{\theta}^{(t)}\right) + s_d \epsilon \mathbf{I}_d \f]
 *    where \f$s_d = \frac{2.4^2}{d}\f$ is the optimal scaling factor for \f$d\f$-dimensional Gaussian targets, and \f$\epsilon > 0\f$ is a regularization term.
 *
 * ### Implementation Algorithms
 * - **Triangular Backsubstitution**: The multivariate proposals evaluate Mahalanobis distances \f$(\mathbf{x}-\mathbf{y})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\mathbf{y})\f$ using LDLT factorization.
 * - **Robbins-Monro Tuning**: Scale factor \f$s_d\f$ is tuned dynamically to match the target acceptance rate:
 *   \f[ \log s_{t+1} = \log s_t + \eta_t (\alpha_{1} - \alpha_{\text{target}}) \f]
 *
 * ### Constraints & Invariants
 * - **Detailed Balance**: The multi-stage rejection ratio calculation must use log-space subtraction to prevent numerical underflow.
 * - **Positive Definiteness**: Regularization nugget \f$\epsilon\f$ must be added to the diagonal of the covariance matrix to guarantee invertibility.
 */
template <typename ProposalType>
class MarkovChain {
  protected:
    double score_{0.0};                  // Current value of the score
    size_t dim_;                         // Dimension of the chain
    size_t nSteps_{0};                   // Steps done
    size_t nAccepts_{0};                 // Accepted candidates

    Eigen::VectorXd mean_;               // Sample-mean vector
    Eigen::MatrixXd cov_;                // Sample-covariance matrix

    ProposalType proposal_;              // Proposal distribution (stored by value)

    std::uniform_real_distribution<double> distU_{0.0, 1.0};
    std::default_random_engine rng_;

    double scale_;                       // Scaling factor
    double targetAcceptanceRatio_;       // Target acceptance ratio for adaptive MCMC

  public:
    /**
     * @brief Construct a new mcmc chain object
     *
     * @param proposal The proposal distribution (copied by value)
     * @param rng the random number generator
     * @param targetAcceptanceRatio the target acceptance ratio
     */
    MarkovChain(const ProposalType &proposal,
                std::default_random_engine &rng,
                const double &targetAcceptanceRatio = 0.234)
        : rng_(rng),
          proposal_(proposal),
          score_(-std::numeric_limits<double>::infinity()),
          dim_(proposal.get().size()),
          targetAcceptanceRatio_(targetAcceptanceRatio),
          mean_(Eigen::VectorXd::Zero(dim_)),
          cov_(Eigen::MatrixXd::Zero(dim_, dim_)),
          scale_(5.6644 / static_cast<double>(dim_)) // 2.38^2 = 5.6644
    {}

    MarkovChain() = default;
    ~MarkovChain() = default;

    /**
     * @brief Increase the number of steps
     */
    void increaseSteps() {
        nSteps_++;
    }

    /**
     * @brief Accept or reject a candidate
     *
     * @param par The candidate
     * @param score The score of the candidate
     */
    bool accept(const Eigen::Ref<const Eigen::VectorXd> &par, double score) {
        if((score - score_) > std::log(distU_(rng_))) {
            proposal_.set(par);
            score_ = score;
            nAccepts_++;
            return true;
        }
        return false;
    }

    /**
     * @brief Perform a single MCMC step
     */
    void step(const score_t &getScore, const bool &delayedRejection = false, const double& gamma = 0.1) {
        increaseSteps();

        // ========================================================================
        // STAGE 1: Standard Proposal
        // ========================================================================
        Eigen::VectorXd cand_prev = proposal_.sample(rng_, 1.0);
        double score_prev = getScore(cand_prev);

        // Calculate Stage 1 continuous acceptance probability (alpha_1_fwd)
        double log_alpha_1_fwd = std::min(0.0, score_prev - score_);
        double alpha_1_fwd = std::exp(log_alpha_1_fwd);

        bool accepted = accept(cand_prev, score_prev);

        // Robbins-Monro Global Scale Update (Tuning to First-Stage efficiency)
        double rm_gamma = 1.0 / std::sqrt(static_cast<double>(nSteps_));
        scale_ *= std::exp(rm_gamma * (alpha_1_fwd - targetAcceptanceRatio_));

        // ========================================================================
        // STAGE 2: Delayed Rejection (DRAM Detailed Balance)
        // ========================================================================
        if(!accepted && delayedRejection) {

            // Use the first gamma for the narrower DR backup proposal
            Eigen::VectorXd cand_new = proposal_.sample(rng_, gamma);
            double score_new = getScore(cand_new);

            // 1. The Backward Rejection Probability
            double log_alpha_1_bwd = std::min(0.0, score_prev - score_new);
            double alpha_1_bwd = std::exp(log_alpha_1_bwd);

            double log_rej_fwd = log1m(alpha_1_fwd);
            double log_rej_bwd = log1m(alpha_1_bwd);

            // 2. The Proposal Asymmetry
            Eigen::VectorXd jump_fwd = cand_prev - getCurrent();
            Eigen::VectorXd jump_bwd = cand_prev - cand_new;

            double log_q1_fwd = -0.5 * proposal_.squaredMahalanobis(jump_fwd);
            double log_q1_bwd = -0.5 * proposal_.squaredMahalanobis(jump_bwd);

            // 3. The True DRAM Stage-2 Acceptance Ratio
            double log_ratio_stage2 = (score_new - score_) +         // Pi ratio
                                      (log_q1_bwd - log_q1_fwd) +    // Proposal ratio
                                      (log_rej_bwd - log_rej_fwd);   // Rejection ratio

            double alpha_2 = std::exp(std::min(0.0, log_ratio_stage2));

            // Final Stage-2 Coin Flip
            if(std::log(distU_(rng_)) < std::log(alpha_2)) {
                proposal_.set(cand_new);
                score_ = score_new;
                accepted = true;
            }
        }

        update();
    }

    /**
     * @brief Updates the value of the mean and covariance matrix.
     */
    void update() {
        Eigen::VectorXd par = proposal_.get();

        if(nSteps_ == 1) {
            mean_ = par;
            cov_.setZero();
        } else {
            // Welford's stable 1-pass multi-dimensional algorithm
            Eigen::VectorXd delta_old = par - mean_;
            mean_ += delta_old / static_cast<double>(nSteps_);
            Eigen::VectorXd delta_new = par - mean_;

            cov_ += delta_old * delta_new.transpose();
        }
    }

    /**
     * @brief Get a covariance matrix adapted to the samples
     */
    Eigen::MatrixXd getAdaptedCovariance() const {
        double epsilon = 1e-6; // Regularization
        Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim_, dim_);
        return scale_ * getCovariance() + scale_ * epsilon * identity;
    }

    /**
     * Reset the value of the mean and of the covariance matrix.
     */
    void reset() {
        nSteps_ = 0;
        nAccepts_ = 0;
        mean_.setZero();
        cov_.setZero();
        scale_ = 5.6644 / static_cast<double>(dim_);
    }

    /**
     * @brief Gets the current parameters of the chain.
     * @return Parameter vector.
     */
    Eigen::VectorXd getCurrent() const {
        return proposal_.get();
    }

    /**
     * @brief Gets the current log probability score of the chain.
     * @return Log score.
     */
    double getScore() const {
        return score_;
    }

    /**
     * @brief Gets the dimensionality of the parameter space.
     * @return Parameter dimension.
     */
    size_t getDim() const {
        return dim_;
    }

    /**
     * @brief Gets the total number of steps run in this chain.
     * @return Steps count.
     */
    size_t getSteps() const {
        return nSteps_;
    }

    /**
     * @brief Gets the current acceptance ratio of proposals.
     * @return Acceptance ratio value in [0, 1].
     */
    double getAcceptanceRatio() const {
        return static_cast<double>(nAccepts_) / static_cast<double>(nSteps_);
    }

    /**
     * @brief Gets the running mean of the parameter samples.
     * @return Mean vector.
     */
    Eigen::VectorXd getMean() const {
        return mean_;
    }

    /**
     * @brief Gets the running covariance of the parameter samples.
     * @return Covariance matrix.
     */
    Eigen::MatrixXd getCovariance() const {
        if(nSteps_ <= 1) {
            return Eigen::MatrixXd::Zero(dim_, dim_);
        }
        return cov_ / (static_cast<double>(nSteps_) - 1.0);
    }

    /**
     * Log the total number of steps, acceptance ratio, data mean and covariance.
     */
    void info() const {
        std::cout << "run " << nSteps_ << " steps\n"
                  << "acceptance ratio: " << std::fixed << std::setprecision(3) << getAcceptanceRatio() << "\n"
                  << "Data covariance: \n" << getCovariance() << "\n"
                  << "Data mean: \n" << getMean() << std::endl;
    }
};

// ==============================================================================
// CHAIN DIAGNOSTICS (Also templated)
// ==============================================================================

/**
 * @brief Computes the Gelman-Rubin convergence diagnostic metric (R-hat).
 * 
 * @details Mathematical Formulation
 * For \f$M\f$ chains of length \f$N\f$, it computes within-chain variance \f$W\f$ and between-chain variance \f$B\f$:
 * \f[
 * W = \frac{1}{M} \sum_{m=1}^M s_m^2, \quad B = \frac{N}{M-1} \sum_{m=1}^M (\bar{\theta}_{m} - \bar{\theta}_{\cdot})^2
 * \f]
 * The posterior marginal variance is estimated as:
 * \f[
 * \widehat{\text{Var}}(\theta \mid y) = \frac{N-1}{N} W + \frac{1}{N} B
 * \f]
 * The potential scale reduction factor (PSRF or \f$\hat{R}\f$) is:
 * \f[
 * \hat{R} = \sqrt{\frac{\widehat{\text{Var}}(\theta \mid y)}{W}}
 * \f]
 * An \f$\hat{R} \to 1.0\f$ indicates successful convergence.
 * 
 * @details Implementation Algorithm
 * Computes coordinate-wise mean and variance vectors across all active chains, calculates the within-chain and between-chain variances, and returns the maximum coordinate value of \f$\hat{R}\f$.
 */
template <typename ProposalType>
double multiChainDiagnosis(const std::vector<MarkovChain<ProposalType>> &chains) {
    if(chains.empty()) return 0.0;

    size_t dim_chain = chains[0].getDim();
    size_t num_chains = chains.size();

    Eigen::VectorXd chainwise_mean = Eigen::VectorXd::Zero(dim_chain);
    Eigen::VectorXd chainwise_cov = Eigen::VectorXd::Zero(dim_chain);
    Eigen::VectorXd chainwise_mean_cov = Eigen::VectorXd::Zero(dim_chain);

    for(const auto &chain : chains) {
        Eigen::VectorXd current_mean = chain.getMean();
        Eigen::MatrixXd current_cov = chain.getCovariance();

        chainwise_mean += current_mean;
        chainwise_cov += current_cov.diagonal();
        chainwise_mean_cov += current_mean.cwiseProduct(current_mean);
    }

    chainwise_mean /= static_cast<double>(num_chains);
    chainwise_cov /= static_cast<double>(num_chains);

    chainwise_mean_cov = chainwise_mean_cov / static_cast<double>(num_chains) - chainwise_mean.cwiseProduct(chainwise_mean);

    if(num_chains > 1) {
        chainwise_mean_cov *= static_cast<double>(num_chains) / static_cast<double>(num_chains - 1);
    }

    Eigen::VectorXd var_chain = chainwise_cov + chainwise_mean_cov;

    // Prevent division by zero if variance is perfectly flat
    Eigen::VectorXd r_hat = Eigen::VectorXd::Zero(dim_chain);
    for(size_t i = 0; i < dim_chain; ++i) {
        if(chainwise_cov(i) > 0.0) {
            r_hat(i) = std::sqrt(var_chain(i) / chainwise_cov(i));
        } else {
            r_hat(i) = 1.0;
        }
    }

    return r_hat.maxCoeff();
}


/**
 * @brief Implements an Evolutionary Markov Chain Monte Carlo sampler.
 * 
 * @details Mathematical Formulation
 * Uses differential evolution crossover proposals. For each chain \f$i\f$, candidate states \f$\boldsymbol{\theta}_i^*\f$ are generated using states from two other distinct, randomly selected chains \f$j\f$ and \f$k\f$:
 * \f[
 * \boldsymbol{\theta}_i^* = \boldsymbol{\theta}_i + \gamma (\boldsymbol{\theta}_j - \boldsymbol{\theta}_k) + \mathbf{e}
 * \f]
 * where \f$\gamma\f$ is the scale factor and \f$\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma_e^2 \mathbf{I})\f$ represents small mutation noise.
 * The candidate state is accepted according to the Metropolis-Hastings probability:
 * \f[
 * \alpha = \min\left(1, \exp\left(\log \pi(\boldsymbol{\theta}_i^*) - \log \pi(\boldsymbol{\theta}_i)\right)\right)
 * \f]
 * 
 * @details Implementation Algorithm
 * 1. For each chain \f$i\f$, randomly selects distinct partner chains \f$j\f$ and \f$k\f$.
 * 2. Evaluates mutated crossover states.
 * 3. Applies Metropolis-Hastings rejection step based on the likelihood evaluations.
 */
class EvolutionaryMarkovChain {
  protected:
    size_t nChains_{0};
    size_t dim_{0};
    std::vector<Eigen::VectorXd> chainSamples_;          // Current samples of each chain
    std::vector<double> chainScores_;                    // Current scores of each chain

    // Random distribution to pick a random chain
    std::uniform_int_distribution<size_t> distChain_;

    // Random distribution to pick a random number
    std::uniform_real_distribution<double> distU_{0, 1};
    std::normal_distribution<double> distN_{0, 1};

    double nugget_{1e-6};                               // Nugget to add to the covariance matrix for numerical stability

  public:
    EvolutionaryMarkovChain() = default;


    /**
     * @brief Constructs an EvolutionaryMarkovChain sampler.
     * @param initialSamples Initial parameter states for each chain.
     * @param initialScores Initial log probability scores for each chain.
     * @param nugget Standard deviation scaling for crossover mutation noise.
     */
    EvolutionaryMarkovChain(std::vector<Eigen::VectorXd> initialSamples, std::vector<double> initialScores, double nugget = 1e-6);

    /**
     * @brief Performs one crossover and mutation step for all chains.
     * @param getScore The log posterior probability evaluator.
     * @param rng Random number engine.
     * @param gamma Scale factor for difference vector in differential evolution.
     */
    void step(const score_t &getScore, std::default_random_engine &rng, double gamma = 0.2);

    /**
     * @brief Gets the current states of all chains.
     * @return Vector of chain samples.
     */
    std::vector<Eigen::VectorXd> getCurrent() const {
        return chainSamples_;
    }

    /**
     * @brief Gets the current log probability scores of all chains.
     * @return Vector of chain scores.
     */
    std::vector<double> getScores() const {
        return chainScores_;
    }
};

















}

/** @} */

#endif