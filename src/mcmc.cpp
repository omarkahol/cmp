#include "mcmc.h"
#include <algorithm>

using namespace cmp;

mcmc::mcmc(size_t dim): m_dim(dim) {
    
    // Initialize the parameter vector
    m_par = vector_t::Zero(dim);
    m_score = 0.0;

    // Initialize mean and variance
    m_mean = vector_t::Zero(dim);
    m_cov = matrix_t::Zero(dim,dim);
}

void mcmc::seed(vector_t par, double score) {
    m_par = par;
    m_score = score;
}

void cmp::mcmc::init(cmp::multivariate_distribution *proposal) {
    m_proposal = proposal;
}

bool mcmc::accept(const vector_t &par, double score) {

    // Increase the number of accepts
    if ((score-m_score) > log(m_dist_u(m_rng))) {
        m_par = par;
        m_score = score;
        m_accepts++;
        return true;
    }
    return false;
}

void mcmc::step(const score_t &get_score, bool DRAM_STEP, double gamma)
{
    // Increase the number of steps
    increase_steps();

    // Propose a candidate
    vector_t cand_prop = m_proposal->jump(m_par, 1.0);

    // Compute the score
    double score_prop = get_score(cand_prop);

    // Decide whether to accept the candidate
    bool accepted = accept(cand_prop, score_prop);

    // Perform a DRAM step
    if ((!accepted) && DRAM_STEP) {

        // Generate a new porposal from a narrower distribution
        vector_t cand_prop_2 = m_proposal->jump(m_par, gamma);

        // Compute the score
        double score_prop_2 = get_score(cand_prop_2);

        // Compute acceptance probabilities
        double alfa_qs_qsm1 = std::min(1.0,std::exp(score_prop - m_score));
        double alfa_qs_qs2 = std::min(1.0,std::exp(score_prop - score_prop_2));

        // Compute difference between the two jumping distributions
        double j_qs_qs2 = m_proposal->log_jump_prob(cand_prop-cand_prop_2, 1.0);
        double j_qs_qsm1 = m_proposal->log_jump_prob(cand_prop-m_par, 1.0);

        double log_j_diff = j_qs_qs2 - j_qs_qsm1;
        double log_score_diff = score_prop_2 - m_score;
        double log_alfa_diff = std::log(1.0-alfa_qs_qs2) - std::log(1-alfa_qs_qsm1);

        // Evaluate the acceptance
        if ((log_alfa_diff + log_j_diff + log_score_diff)>m_dist_u(m_rng)) {
            m_par = cand_prop_2;
            m_score = score_prop_2;
            m_accepts++;
        }
    }

    // Update
    update();
}

void mcmc::update()
{
    // Update estimates of mean and covariance
    m_mean += m_par;
    m_cov += m_par*m_par.transpose();
}

Eigen::LDLT<Eigen::MatrixXd> mcmc::get_adapted_cov() {
    return ((pow(2.38,2)/static_cast<double>(m_dim)) * get_cov()).ldlt();

}

void mcmc::reset()
{

    // Reset the steps and accepts
    m_steps = 0;
    m_accepts = 0;

    // Reset the mean and the covariance
    m_mean = vector_t::Zero(m_dim);
    m_cov = matrix_t::Zero(m_dim,m_dim);
}

vector_t mcmc::get_par() const {
    return m_par;
}

size_t cmp::mcmc::get_dim() const {
    return m_dim;
}

vector_t mcmc::get_mean() const {
    return m_mean/static_cast<double>(m_steps);
}

matrix_t mcmc::get_cov() const {
    vector_t mean = get_mean();
    return m_cov/static_cast<double>(m_steps) - mean*mean.transpose();
}

size_t cmp::mcmc::get_steps() const {
    return m_steps;
}

double cmp::mcmc::get_acceptance_ratio() const
{
    return m_accepts/static_cast<double>(m_steps);
}

void mcmc::info() const{

    auto mean = get_mean();
    auto cov = get_cov();
    
    spdlog::info("run {0:d} steps", m_steps);
    spdlog::info("acceptance ratio: {:.3f}", get_acceptance_ratio());
    spdlog::info("Data covariance: \n{}",cov);
    spdlog::info("Data mean: \n{}",mean);
    
}

vector_t cmp::self_correlation_lag(const std::vector<vector_t> &samples, int lag) {
    
    vector_t self_corr(samples[0].size());

    for (int i=0; i<samples.size() - lag; i++) {
        self_corr += samples[i].cwiseProduct(samples[i+lag]);
    }

    return self_corr/(samples.size() - lag);
}

std::pair<vector_t, double> cmp::single_chain_diagnosis(std::vector<vector_t> samples) {

    auto mean_cov_pair = mean_cov(samples);
    const vector_t &mean = mean_cov_pair.first;
    const vector_t &var = mean_cov_pair.second.diagonal();

    vector_t corr_length=vector_t::Zero(samples[0].size());
    vector_t self_corr_prev(samples[0].size());

    for(int lag=0; lag<samples.size(); lag++) {
        vector_t self_corr(samples[0].size());
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

std::pair<vector_t, matrix_t> cmp::mean_cov(const std::vector<vector_t> &samples) {
    vector_t mean(samples[0].size());
    matrix_t cov(samples[0].size(), samples[0].size());

    for (const vector_t &sample : samples) {
        mean += sample;
        cov += sample*sample.transpose();
    }

    mean /= samples.size();
    cov = cov/samples.size() - mean*mean.transpose();

    return std::make_pair(mean,cov);
}

double cmp::r_hat(const std::vector<mcmc> &chains) {

    // We have the intra chain mean and intra chain variance
    // We the compute the inter-chain mean which is the chainwise_mean of the mean

    size_t dim_chain = chains[0].get_dim();
    size_t num_chains = chains.size();

    vector_t chainwise_mean(dim_chain);
    vector_t chainwise_cov(dim_chain);
    vector_t chainwise_mean_cov(dim_chain);

    for (const mcmc &chain : chains) {
        auto data = std::make_pair(chain.get_mean(),chain.get_cov());

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

    double steps = static_cast<double>(chains[0].get_steps());

    vector_t var_chain = chainwise_cov + chainwise_mean_cov;
    vector_t r_hat = (var_chain.cwiseQuotient(chainwise_cov)).cwiseSqrt();
    return r_hat.maxCoeff();
}
