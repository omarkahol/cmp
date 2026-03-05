/**
 * @file sobol.h
 * @brief Header file for Sobol sensitivity analysis using the Saltelli sampling scheme.
 * @author Omar Kahol
 * @date September 2025
 */

#ifndef CMP_SOBOL_H
#define CMP_SOBOL_H

#include <cmp_defines.h>
#include <statistics.h>
#include <grid.h>

namespace cmp::sobol {

struct SobolResults {
    Eigen::VectorXd firstOrder; // First order indices
    Eigen::VectorXd totalOrder; // Total order indices

    // Stores the interaction indices for second order if computed
    // First element is interaction between 0 and 1, second between 0 and 2, ..., (d-1) and d, etc.
    Eigen::VectorXd secondOrder; // Second order indices (if computed)

    double getFirstOrder(size_t i) const {
        if(i >= firstOrder.size()) throw std::out_of_range("Index out of range");
        return firstOrder(i);
    }
    double getTotalOrder(size_t i) const {
        if(i >= totalOrder.size()) throw std::out_of_range("Index out of range");
        return totalOrder(i);
    }
    double getSecondOrder(size_t i, size_t j) const {
        if(secondOrder.size() == 0) throw std::logic_error("Second order Sobol indices were not computed.");
        if(i >= firstOrder.size() || j >= firstOrder.size()) throw std::out_of_range("Index out of range");
        if(i == j) throw std::invalid_argument("Indices must be different for second order Sobol index.");
        if(i > j) std::swap(i, j);
        size_t index = i * (2 * firstOrder.size() - i - 1) / 2 + (j - i - 1);
        return secondOrder(index);
    }
};

class SobolSaltelli {
  private:
    size_t nObs_ = 0;
    size_t nTotalObs_ = 0;
    size_t dim_ = 0;
    bool secondOrder_ = false;

  public:
    SobolSaltelli() = default;

    SobolSaltelli(size_t nObs, size_t dim, bool secondOrder = false)
        : nObs_(nObs), dim_(dim), secondOrder_(secondOrder) {
        nTotalObs_ = (2 + dim_ + (secondOrder_ ? dim_ * (dim_ - 1) / 2 : 0)) * nObs_;
    }

    size_t nTotalObs() const {
        return nTotalObs_;
    }
    size_t dim() const {
        return dim_;
    }
    size_t nObs() const {
        return nObs_;
    }

    /**
     * \brief Constructs the Sobol evaluation grid using the Saltelli sampling scheme.
     * \param lowerBound The lower bounds for each dimension.
     * \param upperBound The upper bounds for each dimension.
     * \param nObs The number of observations (size of A and B matrices).
     * \param gridType A shared pointer to a UnitGrid object that defines the grid type (e.g., random, Latin Hypercube).
     * \param secondOrder Whether to compute second-order indices.
     * \return An Eigen::MatrixXd containing the evaluation grid with size [(2 + dim + dim*(dim-1)/2) * nObs, dim] if secondOrder is true, otherwise [(2 + dim) * nObs, dim].
     */
    Eigen::MatrixXd evaluationGrid(size_t nObs, std::shared_ptr<cmp::grid::Grid> gridType, bool secondOrder = false);

    /**
     * \brief Slices the output vector Y from the Sobol evaluation grid to keep only the indices specified in idx.
     * \param Y The output vector from the Sobol evaluation grid.
     * \param idx The indices of the observations to keep.
     * \param nObs The number of observations (size of A and B matrices).
     * \param dim The dimensionality of the input space.
     * \param secondOrder Whether second-order indices were computed.
     * \return A sliced output vector containing only the specified indices.
     */
    Eigen::VectorXd sliceSaltelliOutput(const Eigen::VectorXd &Y, const Eigen::VectorXs &idx, size_t nObs, size_t dim, bool secondOrder);

    /**
     * \brief Computes Sobol indices from the output vector Y obtained by evaluating the Sobol evaluation grid.
     * \param Y The output vector from the Sobol evaluation grid.
     * \return A SobolResults struct containing the first-order (firstOrder), total-order (T), and if computed, second-order (secondOrder) Sobol indices.
     */
    SobolResults compute(const Eigen::VectorXd &Y);

    /**
     * \brief Computes Sobol indices with bootstrap resampling to estimate confidence intervals.
     * \param Y The output vector from the Sobol evaluation grid.
     * \param nBootstrap Number of bootstrap samples to use.
     * \param randomSeed Seed for the random number generator (default is 42).
     * \return A pair of SobolResults which is just a storage for 3 Eigen vectors firstOrder, T, and secondOrder (if second order is computed). The first element contains the mean estimates, and the second element contains the standard deviations from the bootstrap samples.
     */
    std::pair<SobolResults, SobolResults> computeWithBootstrap(const Eigen::VectorXd &Y, size_t nBootstrap, size_t randomSeed = 42);

};
}
#endif // CMP_SOBOL_H