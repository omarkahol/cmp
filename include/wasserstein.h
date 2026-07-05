#ifndef WASSERSTEIN_H
#define WASSERSTEIN_H

#include <cmp_defines.h>

#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>

namespace cmp {

/**
 * @brief Computes the 1D Wasserstein distance between two distributions using the greedy
 * mass-transport algorithm. Handles unequal sample sizes.
 * * @warning This function sorts the input vectors in-place to avoid memory allocation overhead.
 *
 * @param s1 The first set of 1D projected samples. (Modified in-place)
 * @param s2 The second set of 1D projected samples. (Modified in-place)
 * @param p The order of the distance.
 * @return W_p^p, the p-th power of the Wasserstein distance.
 */
double wasserstein1D(Eigen::VectorXd &s1, Eigen::VectorXd &s2, double p);

/**
 * @brief Computes the sliced-Wasserstein distance of order p between two n-dimensional datasets.
 *
 * @param samples_1 The first set of samples (N x D matrix).
 * @param samples_2 The second set of samples (M x D matrix).
 * @param p The order of the distance.
 * @param slices The number of slices to compute.
 * @param seed The seed for the random number generator (passed to thread-local PRNGs).
 * @return SW_p^p, the p-th power of the sliced Wasserstein distance.
 */
double slicedWassersteinDistance(const Eigen::MatrixXd &samples_1, const Eigen::MatrixXd &samples_2, double p, int slices, size_t seed = 42);

} // namespace cmp

#endif