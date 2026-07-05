#include "wasserstein.h"
#include "distribution.h" // Assuming this contains your UniformSphereDistribution
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace cmp {

double wasserstein1D(Eigen::VectorXd &s1, Eigen::VectorXd &s2, double p) {
    size_t N = s1.size();
    size_t M = s2.size();

    // 1. Sort the arrays in-place using direct data pointers
    std::sort(s1.data(), s1.data() + N);
    std::sort(s2.data(), s2.data() + M);

    double dist = 0.0;
    size_t i = 0, j = 0;

    // 2. Initial probability mass for the current elements
    double mass_1 = 1.0 / static_cast<double>(N);
    double mass_2 = 1.0 / static_cast<double>(M);

    // 3. Greedily match probability mass
    while(i < N && j < M) {
        double mass_to_move = std::min(mass_1, mass_2);
        double diff = std::abs(s1[i] - s2[j]);

        // Heavily optimize the most common powers
        if(p == 1.0) {
            dist += mass_to_move * diff;
        } else if(p == 2.0) {
            dist += mass_to_move * diff * diff;
        } else {
            dist += mass_to_move * std::pow(diff, p);
        }

        mass_1 -= mass_to_move;
        mass_2 -= mass_to_move;

        // Advance pointers if mass is exhausted (using epsilon for float safety)
        if(mass_1 <= 1e-15) {
            i++;
            if(i < N) mass_1 = 1.0 / static_cast<double>(N);
        }
        if(mass_2 <= 1e-15) {
            j++;
            if(j < M) mass_2 = 1.0 / static_cast<double>(M);
        }
    }

    return dist;
}


double slicedWassersteinDistance(const Eigen::MatrixXd &samples_1, const Eigen::MatrixXd &samples_2, double p, int slices, size_t seed) {

    size_t dim = samples_1.cols();
    double V_n = 2.0 * std::pow(M_PI, dim / 2.0) / std::tgamma(dim / 2.0);
    double total_dist = 0.0;

    // OpenMP parallel region
    #pragma omp parallel
    {
        // Thread-local PRNG: Seed uniquely using the base_rng and thread ID
        std::default_random_engine local_rng(seed + omp_get_thread_num() * 1729); // 1729 is a random prime to help decorrelate seeds

        // Thread-local distribution
        cmp::distribution::UniformSphereDistribution sphere(dim);

        // Thread-local workspaces for projections
        Eigen::VectorXd proj_1(samples_1.rows());
        Eigen::VectorXd proj_2(samples_2.rows());

        // Distribute the slices across available CPU threads
        #pragma omp for reduction(+:total_dist)
        for(int i = 0; i < slices; i++) {

            // 1. Pick a sample on the hyper-sphere
            Eigen::VectorXd theta = sphere.sample(local_rng);

            // 2. Project all samples onto the 1D line using Eigen's SIMD-optimized BLAS.
            // Using .noalias() prevents Eigen from creating a temporary matrix during evaluation.
            proj_1.noalias() = samples_1 * theta;
            proj_2.noalias() = samples_2 * theta;

            // 3. Compute the 1D Wasserstein distance (modifies proj_1 and proj_2 in-place)
            total_dist += wasserstein1D(proj_1, proj_2, p);
        }
    }

    // Average the accumulated distances over the sphere volume and number of slices
    return total_dist / (V_n * static_cast<double>(slices));
}

} // namespace cmp