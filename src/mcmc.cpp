#include "mcmc.h"
#include <algorithm>

using namespace cmp;

mcmc::mcmc(size_t dim, std::default_random_engine rng): m_rng(rng), m_dim(dim) {

    // Initialize proposed steps
    m_lt = matrix_t::Identity(dim,dim).llt();
    m_par = vector_t::Zero(dim);

    // Initialize mean and variance
    m_mean = vector_t::Zero(dim);
    m_cov = matrix_t::Zero(dim,dim);
}

void mcmc::seed(matrix_t cov_prop, vector_t par, double score) {

    // Extract the covariance matrix
    m_lt = cov_prop.llt();
    m_par = par;
    m_score=score;
}

vector_t mcmc::propose() {
    
    // Sample a standard normal distribution 
    vector_t random_variate(m_dim);
    for (size_t i=0; i<m_dim; i++) {
        random_variate(i) = m_dist_n(m_rng);
    }

    // Return the proposed parameter
    return m_par + m_lt.matrixL()*random_variate;
}

bool mcmc::accept(const vector_t &par, double score) {

    // Increase the number of steps
    m_steps++;

    // Decide whether to accept the sample
    bool accept = score -  m_score > log(m_dist_u(m_rng));

    // Update chain parameters
    if (accept) {
        m_par = par;
        m_score = score;
        m_accepts++;
    }

    return accept;
}

void mcmc::step(const score_t &get_score, bool DRAM_STEP, double gamma) {
    
    // Propose a candidate
    vector_t cand_prop = propose();

    // Compute the score
    double score_prop = get_score(cand_prop);

    // Decide whether to accept the candidate 
    bool accepted = accept(cand_prop,score_prop);


    // Perform a DRAM step
    if (!accepted && DRAM_STEP) {

        // Generate a new porposal from a narrower distribution
        vector_t random_variate(m_dim);
        for (size_t i=0; i<m_dim; i++) {
            random_variate(i) = m_dist_n(m_rng);
        }
        vector_t cand_prop_2 = m_par + (m_lt.matrixL()*random_variate)*gamma;

        // Compute the score
        double score_prop_2 = get_score(cand_prop_2);

        // Compute acceptance probabilities
        double alfa_qs_qsm1 = std::min(1.0,score_prop - m_score);
        double alfa_qs_qs2 = std::min(1.0,score_prop - score_prop_2);

        // Compute difference between the two jumping distributions
        double j_qs_qs2 = m_lt.solve(cand_prop - cand_prop_2).squaredNorm();
        double j_qs_qsm1 = m_lt.solve(cand_prop - m_par).squaredNorm();

        double log_j_diff = 0.5*(j_qs_qsm1 - j_qs_qs2);
        double log_score_diff = score_prop_2 - m_score;
        double log_alfa_diff = log(1-alfa_qs_qs2)-log(1-alfa_qs_qsm1);

        // Generate a random number
        double u = m_dist_u(m_rng);

        if (log(u) < log_j_diff + log_score_diff + log_alfa_diff) {
            m_par = cand_prop_2;
            m_score = score_prop_2;
            m_accepts++;
        }
    }

    // Update
    update();
}

void mcmc::update() {
    
    // Update estimates of mean and covariance
    m_mean = (double(m_steps)*m_mean + m_par)/double(m_steps+1);
    m_cov = (double(m_steps)*m_cov + m_par*m_par.transpose())/double(m_steps+1);

    // Note that to get the true covariance we still need to subtract the mean squared
}

void mcmc::adapt_cov(double epsilon) {

    // Extract the mean vector and the covariance matrix
    auto cov = get_cov();

    // Rescale the proposal matrix
    matrix_t cov_prop = (pow(2.38,2)/static_cast<double>(m_dim)) * cov + epsilon*matrix_t::Identity(m_dim,m_dim);
    
    // Compute the cholesky decomposition
    m_lt = cov_prop.llt();
}

void mcmc::reset() {

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
    return m_mean;
}

matrix_t mcmc::get_cov() const {
    return m_cov - m_mean*m_mean.transpose();
}

size_t cmp::mcmc::get_steps() const {
    return m_steps;
}

void mcmc::info() const{

    auto mean = get_mean();
    auto cov = get_cov();
    
    spdlog::info("run {0:d} steps", m_steps);
    spdlog::info("acceptance ratio: {:.3f}", get_acceptance_ratio());
    spdlog::info("Proposal covariance: \n{}",m_lt.matrixLLT());
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
