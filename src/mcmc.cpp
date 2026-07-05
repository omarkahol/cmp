#include "mcmc.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>


// Constructor Optimized with Initializer Lists
cmp::mcmc::EvolutionaryMarkovChain::EvolutionaryMarkovChain(std::vector<Eigen::VectorXd> initialSamples,
                                                            std::vector<double> initialScores,
                                                            double nugget)
{
    std::cout << "[Constructor] initialSamples.size() = " << initialSamples.size() << std::endl;
    if (!initialSamples.empty()) {
        std::cout << "[Constructor] initialSamples[0].size() = " << initialSamples[0].size() << std::endl;
    }
    chainSamples_ = std::move(initialSamples);
    chainScores_ = std::move(initialScores);
    nugget_ = nugget;
    nChains_ = chainSamples_.size();
    if (!chainSamples_.empty()) {
        dim_ = chainSamples_[0].size();
    } else {
        dim_ = 0;
    }
    distChain_ = std::uniform_int_distribution<size_t>(0, nChains_ - 1);
    std::cout << "[Constructor] nChains_ = " << nChains_ << ", dim_ = " << dim_ << std::endl;
}

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