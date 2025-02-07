#include "mcmc.h"
#include <algorithm>

using namespace cmp;

mcmc::mcmc(cmp::proposal_distribution *proposal, std::default_random_engine &rng, const double &score): m_rng(rng) {
    
    // Initialize the parameter vector
    m_proposal = proposal;
    m_score = score;
    m_dim = m_proposal->get().size();

    // Initialize mean and variance
    m_mean = Eigen::VectorXd::Zero(m_dim);
    m_cov = Eigen::MatrixXd::Zero(m_dim,m_dim);
}

bool mcmc::accept(const Eigen::VectorXd &par, double score) {

    // Increase the number of accepts
    if ((score-m_score) > log(m_dist_u(m_rng))) {
        m_proposal->set(par);
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
    Eigen::VectorXd cand_prop = m_proposal->sample(m_rng, 1.0);

    // Compute the score
    double score_prop = get_score(cand_prop);

    // Decide whether to accept the candidate
    bool accepted = accept(cand_prop, score_prop);

    // Perform a DRAM step
    if ((!accepted) && DRAM_STEP) {

        // Generate a new porposal from a narrower distribution
        Eigen::VectorXd cand_prop_2 = m_proposal->sample(m_rng, gamma);

        // Compute the score
        double score_prop_2 = get_score(cand_prop_2);

        // Compute acceptance probabilities
        double alfa_qs_qsm1 = std::min(1.0,std::exp(score_prop - m_score));
        double alfa_qs_qs2 = std::min(1.0,std::exp(score_prop - score_prop_2));

        // Compute difference between the two jumping distributions
        double j_qs_qs2 = m_proposal->log_jump_pdf(cand_prop-cand_prop_2);
        double j_qs_qsm1 = m_proposal->log_pdf(cand_prop);

        double log_j_diff = j_qs_qs2 - j_qs_qsm1;
        double log_score_diff = score_prop_2 - m_score;
        double log_alfa_diff = std::log(1.0-alfa_qs_qs2) - std::log(1-alfa_qs_qsm1);

        // Evaluate the acceptance
        if ((log_alfa_diff + log_j_diff + log_score_diff)>m_dist_u(m_rng)) {
            m_proposal->set(cand_prop_2);
            m_score = score_prop_2;
            m_accepts++;
        }
    }

    // Update
    update();
}

void mcmc::update()
{
    // Get the current parameter
    Eigen::VectorXd par = m_proposal->get();
    // Update estimates of mean and covariance
    m_mean += par;
    m_cov += par*par.transpose();
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
    m_mean = Eigen::VectorXd::Zero(m_dim);
    m_cov = Eigen::MatrixXd::Zero(m_dim,m_dim);
}

Eigen::VectorXd mcmc::get_par() const {
    return m_proposal->get();
}

size_t cmp::mcmc::get_dim() const {
    return m_dim;
}

Eigen::VectorXd mcmc::get_mean() const {
    return m_mean/static_cast<double>(m_steps);
}

Eigen::MatrixXd mcmc::get_cov() const {
    Eigen::VectorXd mean = get_mean();
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
    
    std::cout << "run " << m_steps << " steps" << std::endl;
    std::cout << "acceptance ratio: " << std::fixed << std::setprecision(3) << get_acceptance_ratio() << std::endl;
    std::cout << "Data covariance: \n" << cov << std::endl;
    std::cout << "Data mean: \n" << mean << std::endl;
    
}

Eigen::VectorXd cmp::self_correlation_lag(const std::vector<Eigen::VectorXd> &samples, int lag) {
    
    Eigen::VectorXd self_corr(samples[0].size());

    for (int i=0; i<samples.size() - lag; i++) {
        self_corr += samples[i].cwiseProduct(samples[i+lag]);
    }

    return self_corr/(samples.size() - lag);
}

std::pair<Eigen::VectorXd, double> cmp::single_chain_diagnosis(std::vector<Eigen::VectorXd> samples) {

    auto mean_cov_pair = mean_cov(samples);
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

std::pair<Eigen::VectorXd, Eigen::MatrixXd> cmp::mean_cov(const std::vector<Eigen::VectorXd> &samples) {
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

double cmp::r_hat(const std::vector<mcmc> &chains) {

    // We have the intra chain mean and intra chain variance
    // We the compute the inter-chain mean which is the chainwise_mean of the mean

    size_t dim_chain = chains[0].get_dim();
    size_t num_chains = chains.size();

    Eigen::VectorXd chainwise_mean(dim_chain);
    Eigen::VectorXd chainwise_cov(dim_chain);
    Eigen::VectorXd chainwise_mean_cov(dim_chain);

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

    Eigen::VectorXd var_chain = chainwise_cov + chainwise_mean_cov;
    Eigen::VectorXd r_hat = (var_chain.cwiseQuotient(chainwise_cov)).cwiseSqrt();
    return r_hat.maxCoeff();
}
