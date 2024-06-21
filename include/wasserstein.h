#ifndef WASSERSTEIN_H
#define WASSERSTEIN_H

#include <cmp_defines.h>

namespace cmp {

    /**
     * @brief Computes the Wasserstein distance two distributions given a set of independent samples.
     * 
     * @param samples_1 The first set of samples.
     * @param samples_2 The second set of samples.
     * @param p The order of the distance.
     * @return W_p^p, the p-th power of the Wasserstein distance between the samples
     */
    double wasserstein_1d(std::vector<double> &samples_1, std::vector<double> &samples_2, const double &p);

    /**
     * @brief Generate a uniform sample from the surface of a hypersphere (maybe move to utils.h)
     * 
     * @param dim The dimension of the hyper-sphere (>=1)
     * @param dist_n A normal distribution
     * @param rng A prng
     * @return vector_t, The sample from the hypersphere.
     */
    vector_t hypersphere_sample(int dim, std::normal_distribution<double> &dist_n, std::default_random_engine &rng);

    /**
     * @brief Computes the sliced-Wasserstein distance of order p between some n-dimensional points. 
     * 
     * @param samples_1 The first set of samples.
     * @param samples_2 The second set of samples.
     * @param p The order of the distance.
     * @param slices The number of slices
     * @param dist_n A normal distribution 
     * @param rng A prng 
     * @return SW_p^p, the p-th power of the sliced Wasserstein distance between the samples
     */
    double sliced_wasserstein(const std::vector<vector_t> &samples_1, const std::vector<vector_t> &samples_2, const double &p, const int &slices, std::normal_distribution<double> &dist_n, std::default_random_engine &rng);

}

#endif