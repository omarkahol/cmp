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

/**
 * @class SobolSaltelli
 * @brief Performs Global Sensitivity Analysis (GSA) using the Saltelli formulation of Sobol indices.
 *
 * @details
 * ### Mathematical Foundations
 * Sobol sensitivity analysis decomposes the total variance \f$V(Y)\f$ of a model \f$Y = f(X_1, \dots, X_d)\f$
 * with independent inputs into contributions from individual inputs and their interactions:
 * \f[ V(Y) = \sum_{i=1}^d V_i + \sum_{1 \le i < j \le d} V_{ij} + \dots + V_{1\dots d} \f]
 *
 * Where:
 * - **First-Order Index (Main Effect)**: Measures the fractional variance contribution of input \f$X_i\f$ alone:
 *   \f[ S_i = \frac{V_i}{V(Y)} = \frac{\mathrm{Var}_{X_i}\left(\mathbb{E}_{\mathbf{X}_{\sim i}}[Y | X_i]\right)}{\mathrm{Var}(Y)} \f]
 * - **Second-Order Index (Interaction Effect)**: Measures the interaction effect of inputs \f$X_i\f$ and \f$X_j\f$:
 *   \f[ S_{ij} = \frac{V_{ij}}{V(Y)} = \frac{\mathrm{Var}_{X_i, X_j}\left(\mathbb{E}_{\mathbf{X}_{\sim i,j}}[Y | X_i, X_j]\right) - V_i - V_j}{\mathrm{Var}(Y)} \f]
 * - **Total-Order Index**: Measures the total effect of \f$X_i\f$ including all its interactions:
 *   \f[ S_{Ti} = 1 - \frac{\mathrm{Var}_{\mathbf{X}_{\sim i}}\left(\mathbb{E}_{X_i}[Y | \mathbf{X}_{\sim i}]\right)}{\mathrm{Var}(Y)} \f]
 *
 * ### Implementation Algorithms (Saltelli 2010 Scheme)
 * 1. **Grid Generation**: We draw two independent sample matrices \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ of size \f$N \times d\f$.
 *    We construct \f$d\f$ hybrid matrices \f$\mathbf{A}_B^i\f$ (where column \f$i\f$ is from \f$\mathbf{B}\f$, rest from \f$\mathbf{A}\f$)
 *    and \f$d(d-1)/2\f$ hybrid matrices \f$\mathbf{A}_B^{ij}\f$ (where columns \f$i\f$ and \f$j\f$ are from \f$\mathbf{B}\f$, rest from \f$\mathbf{A}\f$).
 * 2. **Evaluation**: Evaluate the model outputs: \f$\mathbf{y}_A\f$, \f$\mathbf{y}_B\f$, \f$\mathbf{y}_{AB}^i\f$, and \f$\mathbf{y}_{AB}^{ij}\f$.
 * 3. **Estimator Formulations**:
 *    - First-Order:
 *      \f[ S_i = \frac{\frac{1}{N} \sum_{k=1}^N y_B^{(k)} (y_{AB}^{(i)(k)} - y_A^{(k)})}{\mathrm{Var}(Y)} \f]
 *    - Total-Order:
 *      \f[ S_{Ti} = \frac{\frac{1}{2N} \sum_{k=1}^N (y_A^{(k)} - y_{AB}^{(i)(k)})^2}{\mathrm{Var}(Y)} \f]
 *
 * ### Constraints & Invariants
 * - **Sample Independence**: The input matrices \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ must be statistically independent.
 * - **Computational Complexity**: The number of model evaluations required is \f$N(d + 2)\f$ for first and total order,
 *   and \f$N(2d + 2 + d(d-1)/2)\f$ if second-order interaction indices are requested.
 */
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