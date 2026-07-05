#ifndef GRID_H
#define GRID_H

#include "cmp_defines.h"
#include <boost/math/special_functions/prime.hpp>
#include <boost/random/sobol.hpp>
#include <boost/random/niederreiter_base2.hpp>
#include <boost/random/faure.hpp>
#include <poly.h>

namespace cmp::grid {

class Grid {
  public:
    virtual Eigen::MatrixXd construct(const size_t &nPoints) = 0;
    virtual size_t dim() const = 0;
    virtual ~Grid() = default;
};

class SobolGrid : public Grid {
  private:
    boost::random::sobol gen_;
    size_t dimension_;

    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;
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

class ScrambledSobolGrid : public Grid {
  private:
    boost::random::sobol gen_;
    size_t dimension_;

    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;

    // We need a unique seed for each dimension's permutation tree
    std::vector<uint64_t> dim_seeds_;

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

class MonteCarloGrid : public Grid {
  private:
    size_t dimension_;
    std::default_random_engine rng_;
    std::uniform_real_distribution<double> dist_;

    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;
  public:
    MonteCarloGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, unsigned int seed = 42)
        : dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound), rng_(seed), dist_(0.0, 1.0) {}

    Eigen::MatrixXd construct(const size_t &nPoints) override {
        Eigen::MatrixXd points(nPoints, dimension_);
        for(size_t i = 0; i < nPoints; ++i) {
            for(size_t j = 0; j < dimension_; ++j) {
                points(i, j) = lowerBound_(j) + (upperBound_(j) - lowerBound_(j)) * dist_(rng_);
            }
        }
        return points;
    }

    // Pure lazy evaluation
    Eigen::VectorXd operator()() {
        Eigen::VectorXd point(dimension_);
        for(size_t j = 0; j < dimension_; ++j) {
            point(j) = lowerBound_(j) + (upperBound_(j) - lowerBound_(j)) * dist_(rng_);
        }
        return point;
    }

    size_t dim() const override {
        return dimension_;
    }
};

class LatinHypercubeGrid : public Grid {
  private:
    size_t dimension_;
    std::default_random_engine rng_;

    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;
  public:
    LatinHypercubeGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, unsigned int seed = 42)
        : lowerBound_(lowerBound), upperBound_(upperBound), rng_(seed), dimension_(lowerBound.size()) {}

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


    size_t dim() const override {
        return dimension_;
    }
};

class LinspacedGrid : public Grid {
  private:
    size_t dimension_;
    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;

    Eigen::VectorXs gridElement(const size_t &index, const size_t &n_pts, const size_t &dim);

  public:
    LinspacedGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound)
        : lowerBound_(lowerBound), upperBound_(upperBound), dimension_(lowerBound.size()) {}

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

    size_t dim() const override {
        return dimension_;
    }
};


class NiederreiterGrid : public Grid {
  private:
    boost::random::niederreiter_base2 gen_;
    size_t dimension_;

    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;
  public:
    NiederreiterGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound)
        : gen_(lowerBound.size()), dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound) {}

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

class FaureGrid : public Grid {
  private:
    boost::random::faure gen_;
    std::default_random_engine rng_;
    size_t dimension_;

    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;
  public:
    FaureGrid(const Eigen::Ref<const Eigen::VectorXd> &lowerBound, const Eigen::Ref<const Eigen::VectorXd> &upperBound, unsigned int seed = 42)
        : gen_(lowerBound.size()), rng_(seed), dimension_(lowerBound.size()), lowerBound_(lowerBound), upperBound_(upperBound) {}

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

}


#endif