#include "mcmc.h"
#include <algorithm>

// Numerically safe log(1 - α)
double log1m(double a) {
    return (a < 1.0) ? std::log1p(-a) : -std::numeric_limits<double>::infinity();
};

cmp::mcmc::MarkovChain::MarkovChain(cmp::distribution::ProposalDistribution *proposal, std::default_random_engine &rng, const double &score): rng_(rng) {

    // Initialize the parameter vector
    proposal_ = proposal;
    score_ = score;
    dim_ = proposal_->get().size();

    // Initialize mean and variance
    mean_ = Eigen::VectorXd::Zero(dim_);
    cov_ = Eigen::MatrixXd::Zero(dim_, dim_);
}

bool cmp::mcmc::MarkovChain::accept(const Eigen::Ref<const Eigen::VectorXd> &par, double score) {

    // Increase the number of accepts
    if((score - score_) > log(distU_(rng_))) {
        proposal_->set(par);
        score_ = score;
        nAccepts_++;
        return true;
    }
    return false;
}

void cmp::mcmc::MarkovChain::step(const score_t &getScore,
                                  const std::vector<double> &gammas) {
    increaseSteps();

    // --- Stage 0: initial proposal ---
    Eigen::VectorXd cand_prev = proposal_->sample(rng_, 1.0);
    double score_prev = getScore(cand_prev);

    bool accepted = accept(cand_prev, score_prev);

    // --- Delayed Rejection stages ---
    if(!accepted && !gammas.empty()) {

        std::vector<Eigen::VectorXd> cands = {cand_prev};
        std::vector<double> scores = {score_prev};
        std::vector<double> alphas = {};

        for(size_t k = 0; k < gammas.size(); ++k) {

            // New candidate from narrower proposal
            Eigen::VectorXd cand_new = proposal_->sample(rng_, gammas[k]);
            double score_new = getScore(cand_new);

            // Stage-0 acceptance probability (for numerator and denominator)
            double alpha_num = 1.0, alpha_den = 1.0;

            double log_alpha_num = 0.0, log_alpha_den = 0.0;
            for(auto a : alphas) {
                log_alpha_num += log1m(a);
                log_alpha_den += log1m(a);
            }
            double log_ratio = score_new - score_ + (log_alpha_num - log_alpha_den);

            // Adjust for multi-stage rejection probability chain
            log_ratio += std::log(alpha_num / alpha_den);

            // Symmetric proposal → jump PDFs cancel out
            double alpha_k = std::min(1.0, std::exp(log_ratio));

            if(std::log(distU_(rng_)) < std::log(alpha_k)) {
                proposal_->set(cand_new);
                score_ = score_new;
                nAccepts_++;
                accepted = true;
                break;
            }

            // If rejected, prepare for next stage
            cands.push_back(cand_new);
            scores.push_back(score_new);
            alphas.push_back(alpha_k);
        }
    }

    update();
}


void cmp::mcmc::MarkovChain::update() {
    // Get the current parameter
    Eigen::VectorXd par = proposal_->get();
    // Update estimates of mean and covariance
    mean_ += par;
    cov_ += par * par.transpose();
}

Eigen::MatrixXd cmp::mcmc::MarkovChain::getAdaptedCovariance() {
    return ((pow(2.38, 2) / static_cast<double>(dim_)) * getCovariance());
}

void cmp::mcmc::MarkovChain::reset() {

    // Reset the steps and accepts
    nSteps_ = 0;
    nAccepts_ = 0;

    // Reset the mean and the covariance
    mean_ = Eigen::VectorXd::Zero(dim_);
    cov_ = Eigen::MatrixXd::Zero(dim_, dim_);
}

Eigen::VectorXd cmp::mcmc::MarkovChain::getCurrent() const {
    return proposal_->get();
}

size_t cmp::mcmc::MarkovChain::getDim() const {
    return dim_;
}

Eigen::VectorXd cmp::mcmc::MarkovChain::getMean() const {
    return mean_ / static_cast<double>(nSteps_);
}

Eigen::MatrixXd cmp::mcmc::MarkovChain::getCovariance() const {
    Eigen::VectorXd mean = getMean();
    return cov_ / static_cast<double>(nSteps_) - mean * mean.transpose();
}

size_t cmp::mcmc::MarkovChain::getSteps() const {
    return nSteps_;
}

double cmp::mcmc::MarkovChain::getAcceptanceRatio() const {
    return nAccepts_ / static_cast<double>(nSteps_);
}

void cmp::mcmc::MarkovChain::info() const {

    auto mean = getMean();
    auto cov = getCovariance();

    std::cout << "run " << nSteps_ << " steps" << std::endl;
    std::cout << "acceptance ratio: " << std::fixed << std::setprecision(3) << getAcceptanceRatio() << std::endl;
    std::cout << "Data covariance: \n" << cov << std::endl;
    std::cout << "Data mean: \n" << mean << std::endl;

}

double cmp::mcmc::multiChainDiagnosis(const std::vector<mcmc::MarkovChain> &chains) {

    // We have the intra chain mean and intra chain variance
    // We the compute the inter-chain mean which is the chainwise_mean of the mean

    size_t dim_chain = chains[0].getDim();
    size_t num_chains = chains.size();

    Eigen::VectorXd chainwise_mean(dim_chain);
    Eigen::VectorXd chainwise_cov(dim_chain);
    Eigen::VectorXd chainwise_mean_cov(dim_chain);

    for(const mcmc::MarkovChain &chain : chains) {
        auto data = std::make_pair(chain.getMean(), chain.getCovariance());

        // Compute the mean of the mean
        chainwise_mean += data.first;

        // Compute the mean of the variance
        chainwise_cov += data.second.diagonal();

        // compute the varaince of the means
        chainwise_mean_cov += data.first.cwiseProduct(data.first);

    }
    // divide by the number of chains
    chainwise_mean /= num_chains;
    chainwise_cov /= num_chains;

    // get the varaince from the second moment
    chainwise_mean_cov = chainwise_mean_cov / num_chains - chainwise_mean.cwiseProduct(chainwise_mean);

    // rescale
    chainwise_mean_cov *= num_chains / (num_chains - 1);

    Eigen::VectorXd var_chain = chainwise_cov + chainwise_mean_cov;
    Eigen::VectorXd r_hat = (var_chain.cwiseQuotient(chainwise_cov)).cwiseSqrt();
    return r_hat.maxCoeff();
}

cmp::mcmc::EvolutionaryMarkovChain::EvolutionaryMarkovChain(std::vector<Eigen::VectorXd> initialSamples, std::vector<double> initialScores, double nugget) {
    chainSamples_ = std::move(initialSamples);
    chainScores_ = std::move(initialScores);
    nugget_ = nugget;

    nChains_ = chainSamples_.size();
    dim_ = chainSamples_[0].size();

    distChain_ = std::uniform_int_distribution<size_t>(0, nChains_ - 1);
}

void cmp::mcmc::EvolutionaryMarkovChain::step(const score_t &getScore, std::default_random_engine &rng, double gamma) {
    // For each chain
    for(size_t i = 0; i < nChains_; i++) {

        // Pick two random chains
        size_t r1 = distChain_(rng);
        size_t r2 = distChain_(rng);
        while(r1 == i) r1 = distChain_(rng);
        while(r2 == i || r2 == r1) r2 = distChain_(rng);

        // Propose a candidate
        Eigen::VectorXd cand_prop = chainSamples_[i] + gamma * (chainSamples_[r1] - chainSamples_[r2]) + nugget_ * distN_(rng) * Eigen::VectorXd::Ones(dim_);

        // Compute the score
        double score_cand = getScore(cand_prop);

        // Accept the candidate, with uniform probability
        if(log(distU_(rng)) < score_cand - chainScores_[i]) {
            chainSamples_[i] = cand_prop;
            chainScores_[i] = score_cand;
        }
    }
}