#include <cmp_defines.h>
#include <gp.h>
#include <kernel.h>
#include <pdf.h>
#include <io.h>
#include <mcmc.h>
#include <omp.h>

using namespace cmp;

const double gamma = 16;
const double eps = 1e-3;

double score(const double &x) {
    return -gamma*pow(x*x-1,2);
}

int main() {

    std::default_random_engine rng(140998);
    std::normal_distribution dist_n(0.,1.);

    size_t n_chains = 10;
    size_t n_samples = 100000;

    matrix_t data = matrix_t::Zero(n_samples,n_chains);

    // Create a random starting point for each chain
    for(size_t i=0; i<n_chains; i++) {
        data(0,i) = dist_n(rng);
    }

    // Create the chains
    std::vector<mcmc> chains(n_chains);
    std::vector<std::default_random_engine> rngs(n_chains);
    std::vector<std::uniform_int_distribution<int>> dists(n_chains);
    for(size_t i=0; i<n_chains; i++) {
        chains[i] = mcmc(1,rng);
        chains[i].seed(matrix_t::Identity(1,1),vector_t::Zero(1),score(data(0,i)));

        rngs[i] = std::default_random_engine(140998+(i+1)*1000);
        dists[i] = std::uniform_int_distribution<int>(0,n_chains-1);
    }

    // Run the chains
    double gamma_chain = 1.0;
    for(size_t i=1; i<n_samples; i++) {
        
        #pragma omp parallel
        for(size_t j=0; j<n_chains; j++) {

            // Pick two random chains
            size_t chain1 = dists[j](rngs[j]);
            size_t chain2 = dists[j](rngs[j]);
            
            // Propose a candidate by looking at the previous step

            if (i%10!=0) {
                gamma_chain = 2.38/sqrt(2);
            } else {
                gamma_chain = 1.0;
            }
            double cand_prop = data(i-1,j) + gamma_chain*(data(i-1,chain1)-data(i-1,chain2)) + eps*dist_n(rng);
            vector_t cand(1);
            cand << cand_prop;

            // Compute the score
            double score_cand = score(cand_prop);

            // Accept the candidate
            if(chains[j].accept(cand,score_cand)) {
                data(i,j) = cand_prop;
            } else {
                data(i,j) = data(i-1,j);
            }
            chains[j].update();
        }
    }

    std::ofstream file("chain.txt");
    write_vector(matrix_to_vvxd(data),file);
    file.close();

    chains[0].info();

    


    return 0; 
}