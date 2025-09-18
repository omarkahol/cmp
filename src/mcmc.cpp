#include "mcmc.h"
#include <algorithm>

using namespace cmp;

mcmc::MarkovChain::MarkovChain(cmp::distribution::ProposalDistribution *proposal, std::default_random_engine &rng, const double &score): rng_(rng) {
    
    // Initialize the parameter vector
    proposal_ = proposal;
    score_ = score;
    dim_ = proposal_->get().size();

    // Initialize mean and variance
    mean_ = Eigen::VectorXd::Zero(dim_);
    cov_ = Eigen::MatrixXd::Zero(dim_,dim_);
}

bool mcmc::MarkovChain::accept(const Eigen::VectorXd &par, double score) {

    // Increase the number of accepts
    if ((score-score_) > log(distU_(rng_))) {
        proposal_->set(par);
        score_ = score;
        nAccepts_++;
        return true;
    }
    return false;
}

void mcmc::MarkovChain::step(const score_t &getScore, bool DRAM_STEP, double gamma)
{
    // Increase the number of steps
    increaseSteps();

    // Propose a candidate
    Eigen::VectorXd cand_prop = proposal_->sample(rng_, 1.0);

    // Compute the score
    double score_prop = getScore(cand_prop);

    // Decide whether to accept the candidate
    bool accepted = accept(cand_prop, score_prop);

    // Perform a DRAM step
    if ((!accepted) && DRAM_STEP) {

        // Generate a new porposal from a narrower distribution
        Eigen::VectorXd cand_prop_2 = proposal_->sample(rng_, gamma);

        // Compute the score
        double score_prop_2 = getScore(cand_prop_2);

        // Compute acceptance probabilities
        double alfa_qs_qsm1 = std::min(1.0,std::exp(score_prop - score_));
        double alfa_qs_qs2 = std::min(1.0,std::exp(score_prop - score_prop_2));

        // Compute difference between the two jumping distributions
        double j_qs_qs2 = proposal_->logJumpPDF(cand_prop-cand_prop_2);
        double j_qs_qsm1 = proposal_->logPDF(cand_prop);

        double log_j_diff = j_qs_qs2 - j_qs_qsm1;
        double log_score_diff = score_prop_2 - score_;
        double log_alfa_diff = std::log(1.0-alfa_qs_qs2) - std::log(1-alfa_qs_qsm1);

        // Evaluate the acceptance
        if ((log_alfa_diff + log_j_diff + log_score_diff)>distU_(rng_)) {
            proposal_->set(cand_prop_2);
            score_ = score_prop_2;
            nAccepts_++;
        }
    }

    // Update
    update();
}

void mcmc::MarkovChain::update()
{
    // Get the current parameter
    Eigen::VectorXd par = proposal_->get();
    // Update estimates of mean and covariance
    mean_ += par;
    cov_ += par*par.transpose();
}

Eigen::LDLT<Eigen::MatrixXd> mcmc::MarkovChain::getAdaptedCovariance() {
    return ((pow(2.38,2)/static_cast<double>(dim_)) * getCovariance()).ldlt();

}

void mcmc::MarkovChain::reset()
{

    // Reset the steps and accepts
    nSteps_ = 0;
    nAccepts_ = 0;

    // Reset the mean and the covariance
    mean_ = Eigen::VectorXd::Zero(dim_);
    cov_ = Eigen::MatrixXd::Zero(dim_,dim_);
}

Eigen::VectorXd mcmc::MarkovChain::getCurrent() const {
    return proposal_->get();
}

size_t cmp::mcmc::MarkovChain::getDim() const {
    return dim_;
}

Eigen::VectorXd mcmc::MarkovChain::getMean() const {
    return mean_/static_cast<double>(nSteps_);
}

Eigen::MatrixXd mcmc::MarkovChain::getCovariance() const {
    Eigen::VectorXd mean = getMean();
    return cov_/static_cast<double>(nSteps_) - mean*mean.transpose();
}

size_t cmp::mcmc::MarkovChain::getSteps() const {
    return nSteps_;
}

double cmp::mcmc::MarkovChain::getAcceptanceRatio() const
{
    return nAccepts_/static_cast<double>(nSteps_);
}

void mcmc::MarkovChain::info() const{

    auto mean = getMean();
    auto cov = getCovariance();
    
    std::cout << "run " << nSteps_ << " steps" << std::endl;
    std::cout << "acceptance ratio: " << std::fixed << std::setprecision(3) << getAcceptanceRatio() << std::endl;
    std::cout << "Data covariance: \n" << cov << std::endl;
    std::cout << "Data mean: \n" << mean << std::endl;
    
}

Eigen::VectorXd cmp::mcmc::selfCorrelation(const std::vector<Eigen::VectorXd> &samples, int lag) {
    
    Eigen::VectorXd self_corr(samples[0].size());

    for (int i=0; i<samples.size() - lag; i++) {
        self_corr += samples[i].cwiseProduct(samples[i+lag]);
    }

    return self_corr/(samples.size() - lag);
}

std::pair<Eigen::VectorXd, double> cmp::mcmc::singleChainDiagnosis(std::vector<Eigen::VectorXd> samples) {

    auto mean_cov_pair = cmp::mcmc::samplesStatistics(samples);
    const Eigen::VectorXd &mean = mean_cov_pair.first;
    const Eigen::VectorXd &var = mean_cov_pair.second.diagonal();

    Eigen::VectorXd corr_length=Eigen::VectorXd::Zero(samples[0].size());
    Eigen::VectorXd self_corr_prev(samples[0].size());

    for(int lag=0; lag<samples.size(); lag++) {
        Eigen::VectorXd self_corr(samples[0].size());
        for (int i=0; i<samples.size() - lag; i++) {
            self_corr += (samples[i]-mean).cwiseProduct(samples[i+lag]-mean);
        }
        self_corr /= (samples.size()-lag);

        if ((self_corr_prev + self_corr).minCoeff() < 0 && (lag-1)%2==0) {
            spdlog::warn("Self correlation estimate at lag {0:d} has become unreliable... Integration stopped.",lag);
            break;
        } else {
            corr_length += self_corr;
            self_corr_prev = self_corr;
        }
    }

    corr_length = corr_length.cwiseQuotient(var);
    double ess = static_cast<double>(samples.size()) / (-1+2*corr_length.maxCoeff());

    return std::make_pair(corr_length, ess);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> cmp::mcmc::samplesStatistics(const std::vector<Eigen::VectorXd> &samples) {
    Eigen::VectorXd mean(samples[0].size());
    Eigen::MatrixXd cov(samples[0].size(), samples[0].size());

    for (const Eigen::VectorXd &sample : samples) {
        mean += sample;
        cov += sample*sample.transpose();
    }

    mean /= samples.size();
    cov = cov/samples.size() - mean*mean.transpose();

    return std::make_pair(mean,cov);
}

double cmp::mcmc::multiChainDiagnosis(const std::vector<mcmc::MarkovChain> &chains) {

    // We have the intra chain mean and intra chain variance
    // We the compute the inter-chain mean which is the chainwise_mean of the mean

    size_t dim_chain = chains[0].getDim();
    size_t num_chains = chains.size();

    Eigen::VectorXd chainwise_mean(dim_chain);
    Eigen::VectorXd chainwise_cov(dim_chain);
    Eigen::VectorXd chainwise_mean_cov(dim_chain);

    for (const mcmc::MarkovChain &chain : chains) {
        auto data = std::make_pair(chain.getMean(),chain.getCovariance());

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
    chainwise_mean_cov = chainwise_mean_cov/num_chains - chainwise_mean.cwiseProduct(chainwise_mean);

    // rescale
    chainwise_mean_cov *= num_chains/(num_chains-1);

    double steps = static_cast<double>(chains[0].getSteps());

    Eigen::VectorXd var_chain = chainwise_cov + chainwise_mean_cov;
    Eigen::VectorXd r_hat = (var_chain.cwiseQuotient(chainwise_cov)).cwiseSqrt();
    return r_hat.maxCoeff();
}
