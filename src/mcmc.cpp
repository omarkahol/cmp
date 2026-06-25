#include "mcmc.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// Numerically safe log(1 - α)
double log1m(double a) {
    return (a < 1.0) ? std::log1p(-a) : -std::numeric_limits<double>::infinity();
}

// Optimized Constructor using Initializer Lists
cmp::mcmc::MarkovChain::MarkovChain(cmp::distribution::ProposalDistribution *proposal,
                                    std::default_random_engine &rng,
                                    const double &targetAcceptanceRatio)
    : rng_(rng),
      proposal_(proposal),
      score_(-std::numeric_limits<double>::infinity()),
      dim_(proposal->get().size()),
      targetAcceptanceRatio_(targetAcceptanceRatio),
      mean_(Eigen::VectorXd::Zero(dim_)),
      cov_(Eigen::MatrixXd::Zero(dim_, dim_)),
      scale_(5.6644 / static_cast<double>(dim_)) // 2.38^2 = 5.6644
{}

bool cmp::mcmc::MarkovChain::accept(const Eigen::Ref<const Eigen::VectorXd> &par, double score) {
    if((score - score_) > std::log(distU_(rng_))) {
        proposal_->set(par);
        score_ = score;
        nAccepts_++;
        return true;
    }
    return false;
}

void cmp::mcmc::MarkovChain::step(const score_t &getScore, const bool &delayedRejection, const double& gamma) {
    increaseSteps();

    // ========================================================================
    // STAGE 1: Standard Proposal
    // ========================================================================
    Eigen::VectorXd cand_prev = proposal_->sample(rng_, 1.0);
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
        Eigen::VectorXd cand_new = proposal_->sample(rng_, gamma);
        double score_new = getScore(cand_new);

        // 1. The Backward Rejection Probability
        // What is the probability we WOULD HAVE accepted cand_prev if we started at cand_new?
        double log_alpha_1_bwd = std::min(0.0, score_prev - score_new);
        double alpha_1_bwd = std::exp(log_alpha_1_bwd);

        double log_rej_fwd = log1m(alpha_1_fwd);
        double log_rej_bwd = log1m(alpha_1_bwd);

        // 2. The Proposal Asymmetry
        // Requires the squared Mahalanobis distance using the unscaled (gamma=1.0) covariance
        Eigen::VectorXd jump_fwd = cand_prev - getCurrent();
        Eigen::VectorXd jump_bwd = cand_prev - cand_new;

        double log_q1_fwd = -0.5 * proposal_->squaredMahalanobis(jump_fwd);
        double log_q1_bwd = -0.5 * proposal_->squaredMahalanobis(jump_bwd);

        // 3. The True DRAM Stage-2 Acceptance Ratio
        double log_ratio_stage2 = (score_new - score_) +         // Pi ratio
                                  (log_q1_bwd - log_q1_fwd) +    // Proposal ratio
                                  (log_rej_bwd - log_rej_fwd);   // Rejection ratio

        double alpha_2 = std::exp(std::min(0.0, log_ratio_stage2));

        // Final Stage-2 Coin Flip
        if(std::log(distU_(rng_)) < std::log(alpha_2)) {
            proposal_->set(cand_new);
            score_ = score_new;
            // Deliberately NOT incrementing nAccepts_ here to keep scale_ adaptation pure
            accepted = true;
        }
    }

    // O(1) Constant-Time Covariance Tracking
    update();
}

void cmp::mcmc::MarkovChain::update() {
    Eigen::VectorXd par = proposal_->get();

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

Eigen::MatrixXd cmp::mcmc::MarkovChain::getCovariance() const {
    if(nSteps_ <= 1) {
        return Eigen::MatrixXd::Zero(dim_, dim_);
    }
    // Fixed: Applied Bessel's correction (- 1.0) for unbiased sample covariance
    return cov_ / (static_cast<double>(nSteps_) - 1.0);
}

Eigen::MatrixXd cmp::mcmc::MarkovChain::getAdaptedCovariance() {
    double epsilon = 1e-6; // Regularization
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim_, dim_);

    return scale_ * getCovariance() + scale_ * epsilon * identity;
}

void cmp::mcmc::MarkovChain::reset() {
    nSteps_ = 0;
    nAccepts_ = 0;
    mean_.setZero();
    cov_.setZero();
    scale_ = 5.6644 / static_cast<double>(dim_);
}

Eigen::VectorXd cmp::mcmc::MarkovChain::getCurrent() const {
    return proposal_->get();
}
size_t cmp::mcmc::MarkovChain::getDim() const {
    return dim_;
}
Eigen::VectorXd cmp::mcmc::MarkovChain::getMean() const {
    return mean_;
}
size_t cmp::mcmc::MarkovChain::getSteps() const {
    return nSteps_;
}

double cmp::mcmc::MarkovChain::getAcceptanceRatio() const {
    return static_cast<double>(nAccepts_) / static_cast<double>(nSteps_);
}

void cmp::mcmc::MarkovChain::info() const {
    std::cout << "run " << nSteps_ << " steps\n"
              << "acceptance ratio: " << std::fixed << std::setprecision(3) << getAcceptanceRatio() << "\n"
              << "Data covariance: \n" << getCovariance() << "\n"
              << "Data mean: \n" << getMean() << std::endl;
}

double cmp::mcmc::multiChainDiagnosis(const std::vector<mcmc::MarkovChain> &chains) {
    size_t dim_chain = chains[0].getDim();
    size_t num_chains = chains.size();

    // FIXED: Must initialize to Zero to avoid garbage memory!
    Eigen::VectorXd chainwise_mean = Eigen::VectorXd::Zero(dim_chain);
    Eigen::VectorXd chainwise_cov = Eigen::VectorXd::Zero(dim_chain);
    Eigen::VectorXd chainwise_mean_cov = Eigen::VectorXd::Zero(dim_chain);

    for(const mcmc::MarkovChain &chain : chains) {
        Eigen::VectorXd current_mean = chain.getMean();
        Eigen::MatrixXd current_cov = chain.getCovariance();

        chainwise_mean += current_mean;
        chainwise_cov += current_cov.diagonal();
        chainwise_mean_cov += current_mean.cwiseProduct(current_mean);
    }

    chainwise_mean /= num_chains;
    chainwise_cov /= num_chains;

    chainwise_mean_cov = chainwise_mean_cov / num_chains - chainwise_mean.cwiseProduct(chainwise_mean);
    chainwise_mean_cov *= static_cast<double>(num_chains) / (num_chains - 1);

    Eigen::VectorXd var_chain = chainwise_cov + chainwise_mean_cov;
    Eigen::VectorXd r_hat = (var_chain.cwiseQuotient(chainwise_cov)).cwiseSqrt();

    return r_hat.maxCoeff();
}

// Constructor Optimized with Initializer Lists
cmp::mcmc::EvolutionaryMarkovChain::EvolutionaryMarkovChain(std::vector<Eigen::VectorXd> initialSamples,
                                                            std::vector<double> initialScores,
                                                            double nugget)
    : chainSamples_(std::move(initialSamples)),
      chainScores_(std::move(initialScores)),
      nugget_(nugget),
      nChains_(chainSamples_.size()),
      dim_(chainSamples_[0].size()),
      distChain_(0, nChains_ - 1)
{}

void cmp::mcmc::EvolutionaryMarkovChain::step(const score_t &getScore, std::default_random_engine &rng, double gamma) {
    for(size_t i = 0; i < nChains_; ++i) {
        size_t r1 = distChain_(rng);
        size_t r2 = distChain_(rng);

        while(r1 == i) r1 = distChain_(rng);
        while(r2 == i || r2 == r1) r2 = distChain_(rng);

        Eigen::VectorXd cand_prop = chainSamples_[i] + gamma * (chainSamples_[r1] - chainSamples_[r2]) + nugget_ * distN_(rng) * Eigen::VectorXd::Ones(dim_);
        double score_cand = getScore(cand_prop);

        if(std::log(distU_(rng)) < score_cand - chainScores_[i]) {
            chainSamples_[i] = cand_prop;
            chainScores_[i] = score_cand;
        }
    }
}