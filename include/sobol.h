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
#include <stdexcept>
#include <utility>
#include <memory>

/**
 * @addtogroup sensitivity
 * @{
 */
namespace cmp::sobol {

/**
 * @brief Struct to hold calculated Sobol sensitivity indices.
 * * @details Stores the resultant vectors for main effects, total effects,
 * and optional interaction effects calculated from a variance-based
 * sensitivity analysis.
 */
struct SobolResults {
    /** @brief Vector storing the first-order (main effect) Sobol indices. */
    Eigen::VectorXd firstOrder;

    /** @brief Vector storing the total-order Sobol indices. */
    Eigen::VectorXd totalOrder;

    /** @brief Vector storing the second-order (interaction effect) Sobol indices (if computed). */
    Eigen::VectorXd secondOrder;

    /**
     * @brief Gets the first-order Sobol index for input parameter \f$ i \f$.
     * @param i Index of the input parameter.
     * @return First-order Sobol index value \f$ S_i \f$.
     * @throws std::out_of_range If the requested index is out of bounds.
     */
    double getFirstOrder(size_t i) const {
        if(i >= firstOrder.size()) throw std::out_of_range("Index out of range");
        return firstOrder(i);
    }

    /**
     * @brief Gets the total-order Sobol index for input parameter \f$ i \f$.
     * @param i Index of the input parameter.
     * @return Total-order Sobol index value \f$ S_{Ti} \f$.
     * @throws std::out_of_range If the requested index is out of bounds.
     */
    double getTotalOrder(size_t i) const {
        if(i >= totalOrder.size()) throw std::out_of_range("Index out of range");
        return totalOrder(i);
    }

    /**
     * @brief Gets the second-order Sobol index for interaction between parameters \f$ i \f$ and \f$ j \f$.
     * * @details Computes the 1D flat index for the upper triangular portion (excluding diagonal) mapping \f$(i, j)\f$.
     * * @param i Index of the first parameter.
     * @param j Index of the second parameter.
     * @return Second-order Sobol index value \f$ S_{ij} \f$.
     * @throws std::logic_error If second-order indices were not computed.
     * @throws std::out_of_range If indices exceed parameter bounds.
     * @throws std::invalid_argument If \f$ i = j \f$.
     */
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
 * \f[ S_i = \frac{V_i}{V(Y)} = \frac{\mathrm{Var}_{X_i}\left(\mathbb{E}_{\mathbf{X}_{\sim i}}[Y | X_i]\right)}{\mathrm{Var}(Y)} \f]
 * - **Second-Order Index (Interaction Effect)**: Measures the interaction effect of inputs \f$X_i\f$ and \f$X_j\f$:
 * \f[ S_{ij} = \frac{V_{ij}}{V(Y)} = \frac{\mathrm{Var}_{X_i, X_j}\left(\mathbb{E}_{\mathbf{X}_{\sim i,j}}[Y | X_i, X_j]\right) - V_i - V_j}{\mathrm{Var}(Y)} \f]
 * - **Total-Order Index**: Measures the total effect of \f$X_i\f$ including all its interactions:
 * \f[ S_{Ti} = 1 - \frac{\mathrm{Var}_{\mathbf{X}_{\sim i}}\left(\mathbb{E}_{X_i}[Y | \mathbf{X}_{\sim i}]\right)}{\mathrm{Var}(Y)} \f]
 *
 * ### Implementation Algorithms (Saltelli 2010 Scheme)
 * 1. **Grid Generation**: We draw two independent sample matrices \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ of size \f$N \times d\f$.
 * We construct \f$d\f$ hybrid matrices \f$\mathbf{A}_B^i\f$ (where column \f$i\f$ is from \f$\mathbf{B}\f$, rest from \f$\mathbf{A}\f$)
 * and \f$d(d-1)/2\f$ hybrid matrices \f$\mathbf{A}_B^{ij}\f$ (where columns \f$i\f$ and \f$j\f$ are from \f$\mathbf{B}\f$, rest from \f$\mathbf{A}\f$).
 * 2. **Evaluation**: Evaluate the model outputs: \f$\mathbf{y}_A\f$, \f$\mathbf{y}_B\f$, \f$\mathbf{y}_{AB}^i\f$, and \f$\mathbf{y}_{AB}^{ij}\f$.
 * 3. **Estimator Formulations**:
 * - First-Order:
 * \f[ S_i = \frac{\frac{1}{N} \sum_{k=1}^N y_B^{(k)} (y_{AB}^{(i)(k)} - y_A^{(k)})}{\mathrm{Var}(Y)} \f]
 * - Total-Order:
 * \f[ S_{Ti} = \frac{\frac{1}{2N} \sum_{k=1}^N (y_A^{(k)} - y_{AB}^{(i)(k)})^2}{\mathrm{Var}(Y)} \f]
 *
 * ### Constraints & Invariants
 * - **Sample Independence**: The input matrices \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ must be statistically independent.
 * - **Computational Complexity**: The number of model evaluations required is \f$N(d + 2)\f$ for first and total order,
 * and \f$N(2d + 2 + d(d-1)/2)\f$ if second-order interaction indices are requested.
 */
class SobolSaltelli {
  private:
    /** @brief The number of base samples \f$ N \f$. */
    size_t nObs_ = 0;

    /** @brief The total number of evaluations required across all matrices. */
    size_t nTotalObs_ = 0;

    /** @brief The dimensionality \f$ d \f$ of the input parameter space. */
    size_t dim_ = 0;

    /** @brief Flag indicating whether second-order indices should be computed. */
    bool secondOrder_ = false;

  public:
    /** @brief Default constructor. */
    SobolSaltelli() = default;

    /**
     * @brief Constructs a Sobol-Saltelli analyzer with the given configuration.
     * @param nObs The number of base observations \f$ N \f$ (size of matrices A and B).
     * @param dim The dimensionality \f$ d \f$ of the input space.
     * @param secondOrder Boolean flag to enable computation of second-order interaction indices.
     */
    SobolSaltelli(size_t nObs, size_t dim, bool secondOrder = false)
        : nObs_(nObs), dim_(dim), secondOrder_(secondOrder) {
        nTotalObs_ = (2 + dim_ + (secondOrder_ ? dim_ * (dim_ - 1) / 2 : 0)) * nObs_;
    }

    /**
     * @brief Gets the total number of evaluations required based on \f$ N \f$, \f$ d \f$, and analysis order.
     * @return The total evaluation count.
     */
    size_t nTotalObs() const {
        return nTotalObs_;
    }

    /**
     * @brief Gets the number of dimension parameters \f$ d \f$.
     * @return The dimensionality of the input space.
     */
    size_t dim() const {
        return dim_;
    }

    /**
     * @brief Gets the number of base samples \f$ N \f$.
     * @return The number of base samples.
     */
    size_t nObs() const {
        return nObs_;
    }

    /**
     * @brief Constructs the Sobol evaluation grid using the Saltelli sampling scheme.
     * @param nObs The number of observations \f$ N \f$ (size of A and B matrices).
     * @param gridType A shared pointer to a Grid object that defines the sampling strategy (e.g., Random, Latin Hypercube).
     * @param secondOrder Whether to construct hybrid grids for second-order indices.
     * @return An Eigen::MatrixXd containing the evaluation grid with size \f$[(2 + d + d(d-1)/2) \times N, d]\f$ if secondOrder is true, otherwise \f$[(2 + d) \times N, d]\f$.
     */
    Eigen::MatrixXd evaluationGrid(size_t nObs, std::shared_ptr<cmp::grid::Grid> gridType, bool secondOrder = false);

    /**
     * @brief Slices the output vector Y from the Sobol evaluation grid to keep only valid or specified indices.
     * * @details Useful for handling simulations that failed or returned invalid data. It safely trims the outputs
     * while preserving the structured blocks of the Saltelli design matrix needed for index estimation.
     * * @param Y The raw output vector from the Sobol evaluation grid.
     * @param idx The indices of the valid observations to keep.
     * @param nObs The number of base observations \f$ N \f$.
     * @param dim The dimensionality of the input space \f$ d \f$.
     * @param secondOrder Whether second-order evaluations were included in Y.
     * @return A sliced output vector containing only the specified valid outputs.
     */
    Eigen::VectorXd sliceSaltelliOutput(const Eigen::VectorXd &Y, const Eigen::VectorXs &idx, size_t nObs, size_t dim, bool secondOrder);

    /**
     * @brief Computes Sobol indices from the output vector Y obtained by evaluating the Saltelli grid.
     * @param Y The output vector evaluating the full Saltelli grid sequence.
     * @return A SobolResults struct containing the estimated first-order, total-order, and optionally second-order indices.
     */
    SobolResults compute(const Eigen::VectorXd &Y);

    /**
     * @brief Computes Sobol indices with bootstrap resampling to estimate statistical confidence intervals.
     * * @details Resamples the \f$ N \f$ base block sequences with replacement to construct an empirical
     * distribution of the sensitivity indices.
     * * @param Y The output vector evaluating the full Saltelli grid sequence.
     * @param nBootstrap Number of bootstrap resampling iterations.
     * @param randomSeed Seed for the internal random number generator (default is 42).
     * @return A std::pair of SobolResults. The `first` element contains the mean parameter estimates,
     * and the `second` element contains the standard deviations (standard error) of the bootstrap samples.
     */
    std::pair<SobolResults, SobolResults> computeWithBootstrap(const Eigen::VectorXd &Y, size_t nBootstrap, size_t randomSeed = 42);

};
}
/** @} */

#endif // CMP_SOBOL_H