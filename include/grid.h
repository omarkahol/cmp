#ifndef GRID_H
#define GRID_H

#include "cmp_defines.h"
#include <boost/math/special_functions/prime.hpp>
#include <boost/random/sobol.hpp>
#include <boost/random/niederreiter_base2.hpp>
#include <boost/random/faure.hpp>
#include <poly.h>

namespace cmp::grid {

/**
 * @brief Abstract base class for numerical integration grids and space-filling designs.
 * 
 * @details Mathematical Formulation
 * Defines a mapping that generates a point set \f$\mathcal{P} = \{\mathbf{x}_i\}_{i=1}^N\f$ in a hyperrectangle \f$\prod_{d=1}^D [a_d, b_d]\f$ to represent the design of experiments or coordinate nodes.
 * 
 * @details Implementation Algorithm
 * Provides pure virtual interfaces `construct()` to construct a matrix of coordinates, and `dim()` to query the dimensionality.
 */
class Grid {
  public:
    virtual Eigen::MatrixXd construct(const size_t &nPoints) = 0;
    virtual size_t dim() const = 0;
    virtual ~Grid() = default;
};

/**
 * @brief Sobol low-discrepancy sequence grid generator.
 * 
 * @details Mathematical Formulation
 * Generates points using the quasi-random Sobol sequence to minimize star discrepancy:
 * \f[
 * D_N^*(\mathcal{P}) = O\left( \frac{\log^D N}{N} \right)
 * \f]
 * ensuring faster convergence of multi-dimensional integrals compared to standard Monte Carlo.
 * 
 * @details Implementation Algorithm
 * 1. Invokes Boost's Sobol generator.
 * 2. Multiplies generated uniform fractions by dimension bounds \f$[a_d, b_d]\f$.
 * 3. Shuffles output coordinates to break spatial correlations in short prefixes.
 */
class SobolGrid : public Grid {
  private:
    boost::random::sobol gen_;   ///< Boost Sobol sequence generator.
    size_t dimension_;           ///< Number of dimensions.

    Eigen::VectorXd lowerBound_; ///< Lower bounds vector.
    Eigen::VectorXd upperBound_; ///< Upper bounds vector.
  public:
    SobolGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound) : gen_(lowerBound.size()), dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound) {}

    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd points(nPoints, dimension_);

        std::vector<boost::uint_least64_t> seq(dimension_);

        for(size_t i = 0; i < nPoints; ++i) {
            gen_.generate(seq.begin(), seq.end());

            for(size_t d = 0; d < dimension_; ++d) {
                points(i, d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * static_cast<double>(seq[d]) / static_cast<double>(gen_.max());
            }
        }

        // Make a list of indices and shuffle them
        Eigen::VectorXs indices(nPoints);
        std::iota(indices.data(), indices.data() + indices.size(), 0);
        std::shuffle(indices.data(), indices.data() + indices.size(), gen_);

        points = cmp::slice(points, indices);

        gen_.discard(nPoints * dimension_); // Advance the generator state

        return points;
    }

    // Pure lazy evaluation
    Eigen::VectorXd operator()() {
        Eigen::VectorXd point(dimension_);
        std::vector<boost::uint_least64_t> seq(dimension_);

        gen_.generate(seq.begin(), seq.end());

        for(size_t d = 0; d < dimension_; ++d) {
            point(d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * static_cast<double>(seq[d]) / static_cast<double>(gen_.max());
        }
        return point;
    }

    size_t dim() const override {
        return dimension_;
    }
};

/**
 * @brief Owen-scrambled Sobol low-discrepancy sequence grid generator.
 * 
 * @details Mathematical Formulation
 * Owen scrambling randomized digital nets by applying random permutations on the digital trees. It preserves the low-discrepancy property:
 * \f[
 * D_N^*(\mathcal{P}) = O\left( \frac{\log^D N}{N} \right)
 * \f]
 * while providing unbiased estimates and improving the numerical integration variance rate to \f$o(N^{-1})\f$.
 * 
 * @details Implementation Algorithm
 * 1. Obtains Sobol sequences from the underlying Boost generator.
 * 2. Maps values to 64-bit unsigned integers representing fractional components.
 * 3. Applies a recursive bitwise Owen scrambled hash permutation using the prefix history.
 * 4. Maps the scrambled values back to floating point space and scales them to the boundaries.
 */
class ScrambledSobolGrid : public Grid {
  private:
    boost::random::sobol gen_;         ///< Boost Sobol sequence generator.
    size_t dimension_;                 ///< Number of dimensions.

    Eigen::VectorXd lowerBound_;       ///< Lower bounds vector.
    Eigen::VectorXd upperBound_;       ///< Upper bounds vector.

    // We need a unique seed for each dimension's permutation tree
    std::vector<uint64_t> dim_seeds_;  ///< Seeds used to initialize scrambling for each coordinate dimension.

    // Fast bit-mixer to generate deterministic pseudo-randomness based on bit history
    static inline uint64_t hash_prefix(uint64_t prefix, uint64_t seed) {
        uint64_t v = prefix ^ seed;
        v ^= v >> 30;
        v *= 0xbf58476d1ce4e5b9ULL;
        v ^= v >> 27;
        v *= 0x94d049bb133111ebULL;
        v ^= v >> 31;
        return v;
    }

    // True stateless nested Owen Scrambling
    static inline uint64_t owen_scramble(uint64_t val, uint64_t seed) {
        uint64_t result = 0;
        for(int i = 0; i < 64; ++i) {
            int shift = 63 - i;

            // The permutation of this bit depends ONLY on the preceding bits (the prefix)
            uint64_t prefix = (i == 0) ? 0 : (val >> (shift + 1));

            // Hash the prefix and seed to decide whether to flip the current bit
            uint64_t flip = hash_prefix(prefix, seed) & 1;
            uint64_t bit = ((val >> shift) & 1) ^ flip;

            result = (result << 1) | bit;
        }
        return result;
    }

  public:
    ScrambledSobolGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound,
                       const Eigen::Ref<const Eigen::VectorXd> &upperBound,
                       unsigned int seed = 42)
        : gen_(lowerBound.size()), dimension_(lowerBound.size()),
          lowerBound_(lowerBound), upperBound_(upperBound) {

        // Generate a unique scrambling seed for each dimension
        std::default_random_engine rng(seed);
        dim_seeds_.resize(dimension_);
        for(size_t i = 0; i < dimension_; ++i) {
            dim_seeds_[i] = rng();
        }
    }

    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd points(nPoints, dimension_);
        for(size_t i = 0; i < nPoints; ++i) {
            points.row(i) = operator()();
        }
        return points;
    }

    // Pure lazy evaluation
    Eigen::VectorXd operator()() {
        Eigen::VectorXd point(dimension_);
        std::vector<boost::uint_least64_t> seq(dimension_);

        gen_.generate(seq.begin(), seq.end());
        double max_val = static_cast<double>(gen_.max());

        for(size_t d = 0; d < dimension_; ++d) {
            // 1. Normalize the boost integer to [0, 1)
            double u = static_cast<double>(seq[d]) / max_val;

            // 2. Map into a full 64-bit integer representing the fractional bits
            // (Bit 63 is the 0.5 space cut, bit 62 is 0.25, etc.)
            // Using 18446744073709551615.0 (which is 2^64 - 1)
            uint64_t fractional_bits = static_cast<uint64_t>(u * 18446744073709551615.0);

            // 3. Apply the true Owen scrambling algorithm
            uint64_t scrambled_bits = owen_scramble(fractional_bits, dim_seeds_[d]);

            // 4. Map back to [0, 1] float space
            double scrambled_u = static_cast<double>(scrambled_bits) / 18446744073709551615.0;

            // 5. Scale to final parameter bounds
            point(d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * scrambled_u;
        }
        return point;
    }

    size_t dim() const override {
        return dimension_;
    }
};

/**
 * @brief Standard Monte Carlo random grid generator.
 * 
 * @details Mathematical Formulation
 * Generates independent and identically distributed (i.i.d.) samples drawn uniformly from the design space:
 * \f[
 * \mathbf{x}_i \sim \mathcal{U}\left( \prod_{d=1}^D [a_d, b_d] \right)
 * \f]
 */
class MonteCarloGrid : public Grid {
  private:
    size_t dimension_;                            ///< Number of dimensions.
    std::default_random_engine rng_;              ///< Pseudo-random number generator.
    std::uniform_real_distribution<double> dist_; ///< Uniform distribution range [0, 1].

    Eigen::VectorXd lowerBound_;                  ///< Lower bounds vector.
    Eigen::VectorXd upperBound_;                  ///< Upper bounds vector.
  public:
    /**
     * @brief Constructs a MonteCarloGrid generator.
     * @param lowerBound The lower bounds of the hyperrectangle.
     * @param upperBound The upper bounds of the hyperrectangle.
     * @param seed Seed for the pseudo-random number generator.
     */
    MonteCarloGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, unsigned int seed = 42)
        : dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound), rng_(seed), dist_(0.0, 1.0) {}

    /**
     * @brief Constructs a matrix of random Monte Carlo samples.
     * @param nPoints Number of points to generate.
     */
    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd points(nPoints, dimension_);
        for(size_t i = 0; i < nPoints; ++i) {
            for(size_t j = 0; j < dimension_; ++j) {
                points(i, j) = lowerBound_(j) + (upperBound_(j) - lowerBound_(j)) * dist_(rng_);
            }
        }
        return points;
    }

    /**
     * @brief Evaluates a single random Monte Carlo sample.
     */
    Eigen::VectorXd operator()() {
        Eigen::VectorXd point(dimension_);
        for(size_t j = 0; j < dimension_; ++j) {
            point(j) = lowerBound_(j) + (upperBound_(j) - lowerBound_(j)) * dist_(rng_);
        }
        return point;
    }

    /**
     * @brief Returns the dimension of the domain.
     */
    size_t dim() const override {
        return dimension_;
    }
};

/**
 * @brief Latin Hypercube Sampling (LHS) grid generator.
 * 
 * @details Mathematical Formulation
 * For each dimension \f$d \in \{1,\dots,D\}\f$, LHS divides the domain range \f$[a_d, b_d]\f$ into \f$N\f$ equal-probability bins. The coordinate values for the \f$i\f$-th sample is placed inside the \f$\pi_d(i)\f$-th bin:
 * \f[
 * x_{i, d} = a_d + \frac{b_d - a_d}{N} \left( \pi_d(i) - 1 + u_{i,d} \right)
 * \f]
 * where \f$\pi_d\f$ is a random permutation of the set \f$\{1,\dots,N\}\f$ chosen independently for each dimension, and \f$u_{i,d} \sim \mathcal{U}(0, 1)\f$ is a random jitter.
 * 
 * @details Implementation Algorithm
 * 1. Allocates \f$N \times D\f$ matrix.
 * 2. For each dimension \f$d\f$, generates a vector of indices \f$\{0, \dots, N-1\}\f$ and shuffles it.
 * 3. Draws a random uniform offset \f$u\f$ inside each bin and maps the resulting value to the target hyperrectangle bounds.
 */
class LatinHypercubeGrid : public Grid {
  private:
    size_t dimension_;                 ///< Number of dimensions.
    std::default_random_engine rng_;    ///< Pseudo-random number generator.

    Eigen::VectorXd lowerBound_;       ///< Lower bounds vector.
    Eigen::VectorXd upperBound_;       ///< Upper bounds vector.
  public:
    /**
     * @brief Constructs a LatinHypercubeGrid generator.
     * @param lowerBound The lower bounds of the hyperrectangle.
     * @param upperBound The upper bounds of the hyperrectangle.
     * @param seed Seed for the random number generator.
     */
    LatinHypercubeGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, unsigned int seed = 42)
        : lowerBound_(lowerBound), upperBound_(upperBound), rng_(seed), dimension_(lowerBound.size()) {}

    /**
     * @brief Constructs a Latin Hypercube matrix of samples.
     * @param nPoints Number of points to generate.
     */
    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd grid_points(nPoints, dimension_);

        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for(size_t j = 0; j < dimension_; j++) {
            // bins 0..nPoints-1
            std::vector<size_t> bins(nPoints);
            std::iota(bins.begin(), bins.end(), 0);

            // shuffle the bin order for this dimension
            std::shuffle(bins.begin(), bins.end(), rng_);

            // assign a random location inside each bin
            for(size_t i = 0; i < nPoints; i++) {
                double u = dist(rng_);  // random offset
                grid_points(i, j) = lowerBound_(j) + (upperBound_(j) - lowerBound_(j)) * (bins[i] + u) / double(nPoints);
            }
        }

        return grid_points;
    }

    /**
     * @brief Returns the dimension of the domain.
     */
    size_t dim() const override {
        return dimension_;
    }
};

/**
 * @brief Linearly spaced Cartesian tensor product grid generator.
 * 
 * @details Mathematical Formulation
 * Given a total number of target points \f$N\f$, computes the number of points per dimension:
 * \f[
 * N_d = \lceil N^{1/D} \rceil
 * \f]
 * Constructs a uniform Cartesian grid of size \f$N_d^D\f$.
 */
class LinspacedGrid : public Grid {
  private:
    size_t dimension_;           ///< Number of dimensions.
    Eigen::VectorXd lowerBound_; ///< Lower bounds vector.
    Eigen::VectorXd upperBound_; ///< Upper bounds vector.

    /**
     * @brief Helper to retrieve the multi-dimensional grid index matching a flat index.
     */
    Eigen::VectorXs gridElement(const size_t &index, const size_t &n_pts, const size_t &dim);

  public:
    /**
     * @brief Constructs a LinspacedGrid.
     * @param lowerBound The lower bounds of the hyperrectangle.
     * @param upperBound The upper bounds of the hyperrectangle.
     */
    LinspacedGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound)
        : lowerBound_(lowerBound), upperBound_(upperBound), dimension_(lowerBound.size()) {}

    /**
     * @brief Constructs the linearly spaced grid matrix.
     * @param nPoints Target number of points (actual size will be rounded up to the nearest perfect power).
     */
    Eigen::MatrixXd construct(const size_t &nPoints) override {

        // Compute the number of points per dimension
        size_t n_pts_per_dim = static_cast<size_t>(std::ceil(std::pow(nPoints, 1.0 / dimension_)));
        size_t total_points = static_cast<size_t>(std::pow(n_pts_per_dim, dimension_));

        Eigen::MatrixXd grid_points(total_points, dimension_);
        for(size_t i = 0; i < total_points; i++) {

            //Index will be updated to contain numbers from 0 to n-1
            Eigen::VectorXs index(dimension_);

            //Generate current indices for the parameters
            index = gridElement(i, n_pts_per_dim, dimension_);

            for(size_t j = 0; j < dimension_; j++) {

                // perform a linear mapping to [0,1]
                grid_points(i, j) = lowerBound_(j) + (upperBound_(j) - lowerBound_(j)) * (index[j] + 0.5) / double(n_pts_per_dim);
            }
        }

        return grid_points;
    }

    /**
     * @brief Returns the dimension of the domain.
     */
    size_t dim() const override {
        return dimension_;
    }
};


/**
 * @brief Niederreiter (base 2) low-discrepancy sequence grid generator.
 * 
 * @details Mathematical Formulation
 * Uses the Niederreiter sequence in base 2 to generate quasi-random samples with low discrepancy.
 */
class NiederreiterGrid : public Grid {
  private:
    boost::random::niederreiter_base2 gen_; ///< Niederreiter sequence generator.
    size_t dimension_;                       ///< Number of dimensions.

    Eigen::VectorXd lowerBound_;            ///< Lower bounds vector.
    Eigen::VectorXd upperBound_;            ///< Upper bounds vector.
  public:
    /**
     * @brief Constructs a NiederreiterGrid generator.
     * @param lowerBound Lower bounds of the domain.
     * @param upperBound Upper bounds of the domain.
     */
    NiederreiterGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound)
        : gen_(lowerBound.size()), dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound) {}

    /**
     * @brief Constructs a matrix of Niederreiter points.
     * @param nPoints Number of points to generate.
     */
    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd points(nPoints, dimension_);
        std::vector<boost::uint_least64_t> seq(dimension_);

        for(size_t i = 0; i < nPoints; ++i) {
            gen_.generate(seq.begin(), seq.end());

            for(size_t d = 0; d < dimension_; ++d) {
                points(i, d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * static_cast<double>(seq[d]) / static_cast<double>(gen_.max());
            }
        }

        // Make a list of indices and shuffle them
        Eigen::VectorXs indices(nPoints);
        std::iota(indices.data(), indices.data() + indices.size(), 0);
        std::shuffle(indices.data(), indices.data() + indices.size(), gen_);

        points = cmp::slice(points, indices);

        gen_.discard(nPoints * dimension_); // Advance the generator state

        return points;
    }

    /**
     * @brief Evaluates a single Niederreiter sample.
     */
    Eigen::VectorXd operator()() {
        Eigen::VectorXd point(dimension_);
        std::vector<boost::uint_least64_t> seq(dimension_);

        gen_.generate(seq.begin(), seq.end());

        for(size_t d = 0; d < dimension_; ++d) {
            point(d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * static_cast<double>(seq[d]) / static_cast<double>(gen_.max());
        }
        return point;
    }

    /**
     * @brief Returns the dimension of the domain.
     */
    size_t dim() const override {
        return dimension_;
    }
};

/**
 * @brief Faure low-discrepancy sequence grid generator.
 * 
 * @details Mathematical Formulation
 * Uses the Faure sequence (often used in prime base) to generate quasi-random samples with low discrepancy.
 */
class FaureGrid : public Grid {
  private:
    boost::random::faure gen_;        ///< Faure sequence generator.
    std::default_random_engine rng_;  ///< Random engine for shuffling.
    size_t dimension_;                ///< Number of dimensions.

    Eigen::VectorXd lowerBound_;      ///< Lower bounds vector.
    Eigen::VectorXd upperBound_;      ///< Upper bounds vector.
  public:
    /**
     * @brief Constructs a FaureGrid generator.
     * @param lowerBound Lower bounds of the domain.
     * @param upperBound Upper bounds of the domain.
     * @param seed Seed for the random shuffle generator.
     */
    FaureGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, unsigned int seed = 42)
        : gen_(lowerBound.size()), rng_(seed), dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound) {}

    /**
     * @brief Constructs a matrix of Faure points.
     * @param nPoints Number of points to generate.
     */
    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd points(nPoints, dimension_);
        std::vector<boost::uint_least64_t> seq(dimension_);

        for(size_t i = 0; i < nPoints; ++i) {
            gen_.generate(seq.begin(), seq.end());

            for(size_t d = 0; d < dimension_; ++d) {
                points(i, d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * static_cast<double>(seq[d]) / static_cast<double>(gen_.max());
            }
        }

        // Make a list of indices and shuffle them
        Eigen::VectorXs indices(nPoints);
        std::iota(indices.data(), indices.data() + indices.size(), 0);
        std::shuffle(indices.data(), indices.data() + indices.size(), rng_);

        points = cmp::slice(points, indices);

        gen_.discard(nPoints * dimension_); // Advance the generator state

        return points;
    }

    /**
     * @brief Evaluates a single Faure sample.
     */
    Eigen::VectorXd operator()() {
        Eigen::VectorXd point(dimension_);
        std::vector<boost::uint_least64_t> seq(dimension_);

        gen_.generate(seq.begin(), seq.end());

        for(size_t d = 0; d < dimension_; ++d) {
            point(d) = lowerBound_(d) + (upperBound_(d) - lowerBound_(d)) * static_cast<double>(seq[d]) / static_cast<double>(gen_.max());
        }
        return point;
    }

    /**
     * @brief Returns the dimension of the domain.
     */
    size_t dim() const override {
        return dimension_;
    }
};

}


#endif