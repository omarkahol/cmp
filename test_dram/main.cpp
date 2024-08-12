#include <cmp_defines.h>
#include <gp.h>
#include <kernel.h>
#include <mcmc.h>
#include <distribution.h>
#include <omp.h>
#include <utils.h>

using namespace cmp;

const double gamma = 16;
const double eps = 1e-3;

double score(const vector_t &x) {
    return -gamma*pow(x.squaredNorm()-1,2);
}

int main() {

    std::default_random_engine rng(140998);
    std::normal_distribution dist_n(0.,1.);

    size_t n_burn = 10'000;
    size_t n_samples = 100'000;
    std::vector<cmp::vector_t> data(n_samples);

    // Create a chain
    cmp::multivariate_t_distribution proposal(vector_t::Zero(1),(2*matrix_t::Identity(1,1)).ldlt(),1,rng);
    mcmc chain(1);
    chain.init(&proposal);
    chain.seed(vector_t::Zero(1),score(vector_t::Zero(1)));

    // Run the chain
    for (size_t i=0; i<n_burn; i++) {
        chain.step(score,true,0.1);

        if (i%1000==0 && i>0) {
            proposal.set_cov_ldlt(chain.get_adapted_cov());
        }
    }

    chain.info();

    for (size_t i=0; i<n_samples; i++) {
        chain.step(score,true,0.1);
        data[i] = chain.get_par();
    }

    chain.info();

    std::ofstream file("chain.txt");
    write_vector(data,file);
    file.close();

    


    return 0; 
}