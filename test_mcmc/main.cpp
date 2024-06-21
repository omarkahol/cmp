#include <cmp_defines.h>
#include <gp.h>
#include <kernel.h>
#include <pdf.h>
#include <io.h>
#include <mcmc.h>

using namespace cmp;

double score(double x, double m) {
    return -0.5*m*x*x;
}

int main() {

    std::default_random_engine rng(140998);
    std::normal_distribution dist_n(0.,1.);

    mcmc chain(1,rng);

    rng.seed(151200);

    chain.seed(matrix_t::Ones(1,1),vector_t::Zero(1),0);

    int n_burn = 1'000'000;
    for(int i=0; i<n_burn; i++) {

        vector_t par = chain.propose();
        double s = score(par(0),pow(dist_n(rng),2));
        chain.accept(par,s);
        chain.update();

    }
    chain.adapt_cov();
    chain.info();
    


    return 0; 
}