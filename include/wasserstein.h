#ifndef WASSERSTEIN_H
#define WASSERSTEIN_H

#include <cmp_defines.h>

#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>

/**
 * @addtogroup core
 * @{
 */
namespace cmp {

/**
 * @brief Computes the 1D Wasserstein distance between two empirical distributions.
 * 
 * @details Mathematical Formulation
 * The 1D Wasserstein distance of order \f$p \ge 1\f$ between two empirical measures \f$\mu = \frac{1}{N}\sum \delta_{x_i}\f$ and \f$\nu = \frac{1}{M}\sum \delta_{y_j}\f$ is defined as the \f$L^p\f$ distance between their quantile functions \f$F^{-1}(t)\f$ and \f$G^{-1}(t)\f$:
 * \f[
 * W_p^p(\mu, \nu) = \int_0^1 |F^{-1}(t) - G^{-1}(t)|^p dt
 * \f]
 * When \f$N = M\f$, this reduces to:
 * \f[
 * W_p^p(\mu, \nu) = \frac{1}{N} \sum_{i=1}^N |x_{(i)} - y_{(i)}|^p
 * \f]
 * where \f$x_{(i)}\f$ and \f$y_{(i)}\f$ are sorted samples. For unequal sample sizes, the integral is solved by integrating the differences of the step functions over partitions of \f$[0, 1]\f$.
 * 
 * @details Implementation Algorithm
 * 1. Sorts `s1` and `s2` in-place to form the empirical quantiles.
 * 2. Iterates through the joint partition of \f$[0, 1]\f$ defined by the cumulative weight steps \f$i/N\f$ and \f$j/M\f$.
 * 3. Integrates the constant difference raised to power \f$p\f$ over each interval.
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
 * @details Mathematical Formulation
 * The Sliced Wasserstein distance projects multivariate distributions \f$\mu, \nu\f$ in \f$\mathbb{R}^D\f$ onto 1D lines along random directions \f$\boldsymbol{\theta} \in \mathbb{S}^{D-1}\f$ (the unit sphere):
 * \f[
 * SW_p^p(\mu, \nu) = \int_{\mathbb{S}^{D-1}} W_p^p(\boldsymbol{\theta}^* \mu, \boldsymbol{\theta}^* \nu) d\boldsymbol{\theta} \approx \frac{1}{L} \sum_{l=1}^L W_p^p(\boldsymbol{\theta}_l^T \mathbf{X}_1, \boldsymbol{\theta}_l^T \mathbf{X}_2)
 * \f]
 * where \f$\boldsymbol{\theta}_l\f$ are random projection directions drawn uniformly from the unit sphere.
 * 
 * @details Implementation Algorithm
 * 1. Generates \f$L\f$ random directions by sampling from a standard multivariate Gaussian and normalizing to unit norm.
 * 2. Projects the multivariate samples \f$\mathbf{X}_1\f$ and \f$\mathbf{X}_2\f$ onto each direction.
 * 3. Computes the 1D Wasserstein distance for each projection and returns the arithmetic mean.
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

/** @} */

#endif