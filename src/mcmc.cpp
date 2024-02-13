#include "mcmc.h"

using namespace cmp;

void mcmc_chain::step() {

    m_steps++;
    
    // Sample a standard normal distribution 
    vector_t random_variate(m_dim_par);
    for (size_t i=0; i<m_dim_par; i++) {
        random_variate(i) = m_dist_n(m_rng);
    }

    // Propose a candidate parameter
    vector_t par_prop = m_par + m_lt*random_variate;

    //check if the parameters are in their respective bounds
    if(m_in_bounds(par_prop)) {
        
        // Get the value of the hyperparameter - use the current value of the hpar as guess
        vector_t hpar_prop = m_get_hpar(par_prop, m_hpar);

        // Accept with predefined probability
        double score_prop = m_compute_score(par_prop,hpar_prop);
        bool accept = score_prop -  m_score > log(m_dist_u(m_rng));
        
        if (accept) {
            m_par = par_prop;
            m_hpar = hpar_prop;
            m_score = score_prop;
            m_accepts++;
        }
    }

}

void cmp::mcmc_chain::update() {
    
    // Update estimates of mean and covariance
    m_mean += m_par;
    m_cov += m_par*m_par.transpose();
}

void mcmc_chain::step_update() {
    // perform a step
    step();

    // update
    update();
}

void mcmc_chain::adapt_cov() {

    auto mean_cov = get_mean_cov();
    m_cov_prop = (pow(2.38,2)/static_cast<double>(m_dim_par)) * mean_cov.second;
    
    //compute the cholesky decomposition
    m_lt = m_cov_prop.llt().matrixL();
}

void mcmc_chain::reset() {
    m_steps = 0;
    m_accepts = 0;

    m_mean = m_mean*0;
    m_cov = m_cov*0;
}

vector_t mcmc_chain::get_par() const {
    return m_par;
}

vector_t mcmc_chain::get_hpar() const {
    return m_hpar;
}

size_t cmp::mcmc_chain::get_dim() const {
    return m_dim_par;
}

std::pair<vector_t, matrix_t> mcmc_chain::get_mean_cov() const
{
    vector_t mean = m_mean / static_cast<double>(m_steps);
    matrix_t cov = m_cov / static_cast<double>(m_steps - 1) - mean*mean.transpose();
    return std::make_pair(mean, cov);
}

size_t cmp::mcmc_chain::get_steps() const {
    return m_steps;
}

void mcmc_chain::info() const{

    auto data = get_mean_cov();
    
    spdlog::info("run {0:d} steps", m_steps);
    spdlog::info("acceptance ratio: {:.3f}", static_cast<double>(m_accepts)/ static_cast<double> (m_steps));
    spdlog::info("Proposal covariance: \n{}",m_cov_prop);
    spdlog::info("Data covariance: \n{}",data.second);
    spdlog::info("Data mean: \n{}",data.first);
    
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

std::pair<vector_t, matrix_t> cmp::mean_cov(const std::vector<vector_t> &samples)
{
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

double cmp::r_hat(const std::vector<mcmc_chain> &chains) {

    // We have the intra chain mean and intra chain variance
    // We the compute the inter-chain mean which is the chainwise_mean of the mean

    size_t dim_chain = chains[0].get_dim();
    size_t num_chains = chains.size();

    vector_t chainwise_mean(dim_chain);
    vector_t chainwise_cov(dim_chain);
    vector_t chainwise_mean_cov(dim_chain);

    for (const mcmc_chain &chain : chains) {
        auto data = chain.get_mean_cov();

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
