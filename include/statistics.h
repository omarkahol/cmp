/**
 * @file statistics.h
 * @brief Header file for statistical functions and classes
 * @details This file contains declarations for various statistical functions,
 * resampling techniques (K-Fold, Bootstrap), and spatial statistics utilities
 * used throughout the project.
 */

#ifndef CMP_STATISTICS_H
#define CMP_STATISTICS_H

#include <concepts>
#include <numeric>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <cmp_defines.h>
#include <unsupported/Eigen/FFT>
#include <complex>

/**
 * @addtogroup core
 * @{
 */
namespace cmp::statistics {

/**
 * @class KFold
 * @brief K-Fold cross-validation partition generator.
 * * @details
 * ### Mathematical Formulation
 * Partitions a dataset \f$\mathcal{D}\f$ of size \f$N\f$ into \f$K\f$ mutually exclusive, approximately equal-sized subsets \f$\{F_1, \dots, F_K\}\f$ such that:
 * \f[
 * \bigcup_{k=1}^K F_k = \mathcal{D}, \quad F_i \cap F_j = \emptyset \quad \forall i \neq j
 * \f]
 * During the \f$k\f$-th fold iteration, the model is trained on \f$\mathcal{D} \setminus F_k\f$ and validated on \f$F_k\f$.
 * * ### Implementation Algorithm
 * 1. Computes the base size of each fold: \f$S = \lfloor N/K \rfloor\f$.
 * 2. Assigns indices to splits. The \f$k\f$-th split covers index range \f$[k \cdot S, (k+1) \cdot S)\f$, and the last fold covers any remainder.
 * 3. Returns the respective train and test index vectors.
 */
class KFold {
  private:
    /** @brief Number of folds/splits \f$ K \f$. */
    Eigen::Index nSplits_;

    /** @brief Total number of observations \f$ N \f$ in the dataset. */
    Eigen::Index nObs_;

    /** @brief Flag indicating if the dataset indices should be shuffled before splitting. */
    bool shuffle_;

    /** @brief Seed for the random number generator. */
    unsigned int rngState_;

    /** @brief Random number engine used for shuffling. */
    std::default_random_engine rng_;

    /** @brief Internal array storing the (potentially shuffled) sequence of indices. */
    Eigen::VectorXs indices_;

  public:

    /**
     * @brief Constructor for the KFold partitioner.
     * @param n_splits Number of folds \f$ K \f$. Must be at least 2 and at most n_obs.
     * @param n_obs Total number of observations \f$ N \f$ in the dataset. Must be at least 2.
     * @param shuffle Whether to shuffle the data before splitting into folds.
     * @param random_state Seed for the random number generator (used if shuffle is true).
     * @throws std::invalid_argument if the input parameters violate minimum constraints.
     */
    KFold(Eigen::Index n_splits, Eigen::Index n_obs, bool shuffle = false, unsigned int random_state = 42)
        : nSplits_(n_splits),
          nObs_(n_obs),
          shuffle_(shuffle),
          rngState_(random_state),
          rng_(random_state) {
        if(n_obs < 2) {
            throw std::invalid_argument("Number of observations must be at least 2.");
        }
        if(n_splits < 2 || n_splits > n_obs) {
            throw std::invalid_argument("Number of splits must be between 2 and the number of observations.");
        }

        // Generate indices 0..nObs-1
        indices_.resize(nObs_);
        for(Eigen::Index i = 0; i < nObs_; ++i) {
            indices_(i) = i;
        }

        if(shuffle_) {
            std::shuffle(indices_.data(), indices_.data() + indices_.size(), rng_);
        }
    }

    /**
     * @class KFoldIterator
     * @brief Iterator class to traverse folds for K-Fold cross-validation.
     */
    class KFoldIterator {
      private:
        /** @brief The current active fold index \f$ k \f$. */
        Eigen::Index currentSplit_;

        /** @brief Reference to the parent KFold object to access index arrays and sizes. */
        const KFold &parent_;

      public:
        /**
         * @brief Constructs a KFoldIterator.
         * @param kf Reference to the parent KFold object.
         * @param split Index of the current split/fold.
         */
        KFoldIterator(const KFold& kf, Eigen::Index split)
            : currentSplit_(split), parent_(kf) {}

        /**
         * @brief Checks inequality with another iterator (used for loop termination).
         * @param other The iterator to compare against.
         * @return True if iterators point to different splits, false otherwise.
         */
        bool operator!=(const KFoldIterator& other) const {
            return currentSplit_ != other.currentSplit_;
        }

        /**
         * @brief Dereferences the iterator to compute training and test index sets for the current fold.
         * @return A std::pair containing {train_indices, test_indices} as Eigen vectors.
         */
        std::pair<Eigen::VectorXs, Eigen::VectorXs> operator*() const {
            Eigen::Index fold_size = parent_.nObs_ / parent_.nSplits_;
            Eigen::Index start = currentSplit_ * fold_size;
            Eigen::Index end   = (currentSplit_ == parent_.nSplits_ - 1)
                                 ? parent_.nObs_ : start + fold_size;

            // test set
            Eigen::VectorXs test_indices = parent_.indices_.segment(start, end - start);

            // train set
            Eigen::VectorXs train_indices(parent_.nObs_ - (end - start));
            Eigen::Index train_idx = 0;
            for(Eigen::Index i = 0; i < parent_.nObs_; ++i) {
                if(i < start || i >= end) {
                    train_indices(train_idx++) = parent_.indices_(i);
                }
            }

            return {train_indices, test_indices};
        }

        /**
         * @brief Increments the iterator to point to the next fold.
         * @return Reference to the incremented iterator.
         */
        KFoldIterator& operator++() {
            ++currentSplit_;
            return *this;
        }
    };

    /**
     * @brief Gets the iterator pointing to the first fold.
     * @return A KFoldIterator initialized to split 0.
     */
    KFoldIterator begin() const {
        return KFoldIterator(*this, 0);
    }

    /**
     * @brief Gets the iterator pointing to the end of the folds.
     * @return A KFoldIterator initialized to split K.
     */
    KFoldIterator end()   const {
        return KFoldIterator(*this, nSplits_);
    }

    /**
     * @brief Directly accesses the train/test indices for a specific split without iterating.
     * @param split The specific split index \f$ k \f$ to extract.
     * @return A pair of training and test index vectors.
     * @throws std::out_of_range if the requested split is invalid.
     */
    std::pair<Eigen::VectorXs, Eigen::VectorXs> operator()(Eigen::Index split) const {
        if(split < 0 || split >= nSplits_) {
            throw std::out_of_range("Split index out of range.");
        }
        return *(KFoldIterator(*this, split));
    }

    /**
     * @brief Gets the total number of splits/folds \f$ K \f$.
     * @return The number of folds.
     */
    size_t nFolds() const {
        return this->nSplits_;
    }
};


/**
 * @class Bootstrap
 * @brief Bootstrap resampling index generator.
 * * @details
 * ### Mathematical Formulation
 * Draws a random sample \f$\mathcal{D}^*\f$ of size \f$M\f$ from a dataset \f$\mathcal{D}\f$ of size \f$N\f$.
 * - If with replacement:
 * \f[
 * P(x_i^* = x_j) = \frac{1}{N} \quad \forall i \in \{1,\dots,M\}, \ j \in \{1,\dots,N\}
 * \f]
 * - If without replacement:
 * \f[
 * P(x_i^* = x_j) = \frac{1}{N - i + 1}
 * \f]
 * subject to all selections being distinct.
 * * ### Implementation Algorithm
 * 1. If `withReplacement_` is true, generates \f$M\f$ random integers uniformly distributed in the range \f$[0, N-1]\f$.
 * 2. If false, shuffles the index array using `std::shuffle` and extracts the first \f$M\f$ elements.
 */
class Bootstrap {
  private:
    /** @brief Original number of observations \f$ N \f$. */
    size_t nObs_;

    /** @brief Target number of observations to resample \f$ M \f$. */
    size_t nObsResample_;

    /** @brief Seed for the random number generator. */
    unsigned int rngState_;

    /** @brief Random number engine. */
    std::default_random_engine rng_;

    /** @brief Uniform integer distribution for sampling with replacement. */
    std::uniform_int_distribution<size_t> dist_;

    /** @brief Internal array storing the base indices for sampling. */
    Eigen::VectorXs indices_;

    /** @brief Flag indicating whether to sample with replacement. */
    bool withReplacement_;

  public:
    /**
     * @brief Constructor for the Bootstrap resampler.
     * @param n_obs Total number of original observations \f$ N \f$.
     * @param n_obs_resample Number of samples to draw \f$ M \f$.
     * @param with_replacement Boolean flag indicating whether duplicates are allowed.
     * @param random_state Seed for the random number generator.
     * @throws std::invalid_argument if invalid sizes are provided.
     */
    Bootstrap(size_t n_obs, size_t n_obs_resample, bool with_replacement = true, unsigned int random_state = 42)
        : nObs_(n_obs),
          nObsResample_(n_obs_resample),
          rngState_(random_state),
          rng_(random_state),
          dist_(0, n_obs - 1),
          withReplacement_(with_replacement) {
        if(n_obs < 1) {
            throw std::invalid_argument("Number of observations must be at least 1.");
        }
        if(n_obs_resample < 1) {
            throw std::invalid_argument("Number of resampled observations must be at least 1.");
        }
        if(!withReplacement_ && n_obs_resample > n_obs) {
            throw std::invalid_argument("Number of resampled observations cannot be greater than number of observations when sampling without replacement.");
        }

        // Generate indices 0..nObs-1
        indices_.resize(nObs_);
        for(size_t i = 0; i < nObs_; ++i) {
            indices_(i) = i;
        }
    }

    /**
     * @brief Executes the bootstrap sampling process.
     * @return A vector containing the \f$ M \f$ randomly sampled indices.
     */
    Eigen::VectorXs operator()() {
        Eigen::VectorXs sample_indices(nObsResample_);

        if(withReplacement_) {
            // Sample with replacement
            for(size_t i = 0; i < nObsResample_; ++i) {
                sample_indices(i) = dist_(rng_);
            }
        } else {
            // Sample without replacement
            // Shuffle all indices and take the first nObsResample_
            Eigen::VectorXs shuffled_indices = indices_;  // make a copy
            std::shuffle(shuffled_indices.data(), shuffled_indices.data() + shuffled_indices.size(), rng_);
            sample_indices = shuffled_indices.head(nObsResample_).eval();
        }

        return sample_indices;
    }
};

/**
 * @brief Computes the empirical mean vector of a dataset.
 * * @details
 * ### Mathematical Formulation
 * For a dataset matrix \f$\mathbf{X} \in \mathbb{R}^{N \times D}\f$, the mean vector \f$\boldsymbol{\mu} \in \mathbb{R}^D\f$ is:
 * \f[
 * \boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i
 * \f]
 * @param data A reference to a matrix containing the data (points in rows, features in columns).
 * @return A column vector containing the mean of each feature.
 * @throws std::invalid_argument if the input matrix is empty.
 */
Eigen::VectorXd mean(const Eigen::Ref<const Eigen::MatrixXd>& data);

/**
 * @brief Computes the sample covariance matrix of a dataset.
 * * @details
 * ### Mathematical Formulation
 * Given a dataset matrix \f$\mathbf{X} \in \mathbb{R}^{N \times D}\f$ with mean vector \f$\boldsymbol{\mu}\f$, the unbiased sample covariance matrix \f$\mathbf{\Sigma} \in \mathbb{R}^{D \times D}\f$ is:
 * \f[
 * \mathbf{\Sigma} = \frac{1}{N-1} (\mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T)^T (\mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T)
 * \f]
 * @param data A reference to an Eigen matrix containing the data (points in rows).
 * @return The covariance matrix of the data.
 * @throws std::invalid_argument if the input matrix has fewer than 2 rows.
 */
Eigen::MatrixXd covariance(const Eigen::Ref<const Eigen::MatrixXd>& data);

/**
 * @brief Computes a specific quantile of a 1D vector of data.
 *
 * @param data A reference to an Eigen vector containing the data.
 * @param quantile The target probability quantile \f$ p \in [0, 1] \f$.
 * @return The computed value representing the \f$ p \f$-th quantile.
 * @throws std::invalid_argument if the input vector is empty or the quantile is out of bounds.
 */
double quantile(const Eigen::Ref<const Eigen::VectorXd>& data, double quantile);

/**
 * @brief Computes the interquartile range (IQR) column-wise for a dataset.
 *
 * @details
 * ### Mathematical Formulation
 * For each feature column \f$ j \f$:
 * \f[
 * \text{IQR}_j = Q_{\text{upper}, j} - Q_{\text{lower}, j}
 * \f]
 * where \f$Q\f$ represents the value at the specified quantile probabilities.
 * * @param data A reference to an Eigen matrix containing the data (points in rows).
 * @param lowerQuantile The lower probability bound (e.g., 0.25).
 * @param upperQuantile The upper probability bound (e.g., 0.75).
 * @return A vector containing the IQR for each feature column.
 * @throws std::invalid_argument if the input matrix is empty or the quantiles are invalid.
 */
Eigen::VectorXd interQuantileRange(const Eigen::Ref<const Eigen::MatrixXd>& data, double lowerQuantile, double upperQuantile);

/**
 * @brief Computes the Pearson correlation matrix between two datasets.
 * * @details
 * ### Mathematical Formulation
 * For columns \f$X_i\f$ from `data1` and \f$Y_j\f$ from `data2`, the correlation coefficient is:
 * \f[
 * \rho_{ij} = \frac{\mathrm{cov}(X_i, Y_j)}{\sigma_{X_i} \sigma_{Y_j}}
 * \f]
 * Manages the case where the datasets have different numbers of points by aligning them appropriately if supported by the implementation.
 * * @param data1 First dataset (points in rows).
 * @param data2 Second dataset (points in rows).
 * @return The Pearson correlation matrix \f$ \mathbf{R} \in \mathbb{R}^{D_1 \times D_2} \f$.
 */
Eigen::MatrixXd pearsonCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2);

/**
 * @brief Computes the cross-correlation matrix between two datasets with a specified temporal lag.
 *
 * @details
 * Correlates \f$X(t)\f$ from `data1` with \f$Y(t + \tau)\f$ from `data2` where \f$\tau\f$ is the `lag`.
 * * @param data1 First dataset (time series points in rows).
 * @param data2 Second dataset (time series points in rows).
 * @param lag The discrete time lag \f$ \tau \f$ to apply to the second dataset.
 * @return The lagged correlation matrix.
 */
Eigen::MatrixXd laggedCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2, int lag);

/**
 * @brief Returns the cross-correlation matrices between two datasets for a sequence of lags.
 *
 * @param data1 First dataset (time series points in rows).
 * @param data2 Second dataset (time series points in rows).
 * @param minLag Minimum lag \f$ \tau_{\text{min}} \f$ to consider.
 * @param maxLag Maximum lag \f$ \tau_{\text{max}} \f$ to consider.
 * @return A std::vector of correlation matrices, ordered from minLag to maxLag.
 */
std::vector<Eigen::MatrixXd> laggedCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2, int minLag, int maxLag);

/**
 * @brief Computes the self-correlation (autocorrelation) length and the effective sample size.
 * * @details Useful for MCMC diagnostics to determine the number of independent samples.
 * * @param data the dataset containing the samples (each sample is a row).
 * @return a std::pair containing:
 * 1. An Eigen::VectorXd of correlation lengths for each feature.
 * 2. A double representing the overall effective sample size (ESS).
 */
std::pair<Eigen::VectorXd, double> selfCorrelationLength(const Eigen::Ref<const Eigen::MatrixXd>& data);

/**
 * @brief Computes the complete auto- and cross-correlation functions across all lags efficiently using the Fast Fourier Transform (FFT).
 * * @details
 * ### Mathematical Formulation (Wiener-Khinchin Theorem)
 * The cross-correlation \f$R_{ij}(\tau)\f$ between feature \f$i\f$ and \f$j\f$ can be computed in the frequency domain:
 * \f[
 * R_{ij}(\tau) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{x_i\} \cdot \mathcal{F}\{x_j\}^* \right\}
 * \f]
 * where \f$\mathcal{F}\f$ is the discrete Fourier transform. Data is mean-centered, zero-padded to the next power of 2 to avoid circular convolution artifacts, and properly normalized by sample variances.
 * * @param data A reference to the dataset matrix (time points in rows, features in columns).
 * @return A vector of length `nPoints`. Each element is a `nFeatures x nFeatures` matrix representing the correlation at lag \f$\tau\f$.
 */
inline std::vector<Eigen::MatrixXd> selfCrossCorrelationFFT(
    const Eigen::Ref<const Eigen::MatrixXd>& data) {
    size_t nPoints = data.rows();
    size_t nFeatures = data.cols();
    size_t nFFT = 1;
    while(nFFT < 2 * nPoints) nFFT <<= 1;  // zero-pad to next power of 2

    Eigen::FFT<double> fft;

    // Precompute FFTs
    std::vector<std::vector<std::complex<double>>> X(nFeatures, std::vector<std::complex<double>>(nFFT));
    for(size_t i = 0; i < nFeatures; ++i) {
        Eigen::VectorXd x = data.col(i);
        x.array() -= x.mean();

        std::vector<double> x_pad(nFFT, 0.0);
        for(size_t t = 0; t < nPoints; ++t) x_pad[t] = x[t];

        fft.fwd(X[i], x_pad);
    }

    // Compute cross-correlations for all lags
    std::vector<Eigen::MatrixXd> corr(nPoints, Eigen::MatrixXd::Zero(nFeatures, nFeatures));

    for(size_t i = 0; i < nFeatures; ++i) {
        double var_i = data.col(i).array().square().sum() / nPoints;
        for(size_t j = 0; j < nFeatures; ++j) {
            double var_j = data.col(j).array().square().sum() / nPoints;

            // Multiply FFTs
            std::vector<std::complex<double>> R_fft(nFFT);
            for(size_t k = 0; k < nFFT; ++k)
                R_fft[k] = X[i][k] * std::conj(X[j][k]);

            // Inverse FFT
            std::vector<double> r_pad(nFFT, 0.0);
            fft.inv(r_pad, R_fft);

            // Normalize and store first nPoints lags
            for(size_t lag = 0; lag < nPoints; ++lag)
                corr[lag](i, j) = r_pad[lag] / (nPoints * std::sqrt(var_i * var_j));
        }
    }

    return corr; // vector of size nPoints, each element = nFeatures x nFeatures
};


/**
 * @class PairwiseDistanceStats
 * @brief Computes and stores pairwise spatial distances between high-dimensional data points.
 * * @details
 * Efficiently computes the upper triangle of the distance matrix to save memory,
 * utilizing OpenMP for multi-threaded parallel execution. Provides fast access to
 * mean distances, spatial quantiles, and the spatial correlation integral.
 */
class PairwiseDistanceStats {
  private:
    /** @brief 1D flattened array of the strictly upper triangular distance matrix. */
    std::vector<double> distances;

    /** @brief The globally computed mean distance across all pairs. */
    double mean_val;

    /** @brief Flag indicating if the distances have been computed and the class is ready to query. */
    bool is_computed;

  public:
    /** @brief Default constructor. */
    PairwiseDistanceStats() : mean_val(0.0), is_computed(false) {}

    /**
     * @brief Computes the upper triangle of the pairwise Euclidean distance matrix.
     * * @details
     * ### Mathematical Formulation
     * Evaluates the \f$ L_2 \f$-norm between all unique pairs:
     * \f[
     * d_{ij} = ||\mathbf{x}_i - \mathbf{x}_j||_2 \quad \text{for } 1 \le i < j \le N
     * \f]
     * Utilizes OpenMP `dynamic` scheduling to balance the uneven workload of the triangular nested loops.
     * * @param data Matrix of size (N, D) where N is number of points, D is dimensions.
     * @throws std::invalid_argument if fewer than 2 points are provided.
     */
    void compute(const Eigen::MatrixXd& data) {
        size_t n = data.rows();
        if(n < 2) throw std::invalid_argument("Need at least 2 points to compute pairwise distances.");

        // Number of strictly upper triangular elements: N * (N - 1) / 2
        size_t num_pairs = n * (n - 1) / 2;
        distances.resize(num_pairs);
        mean_val = 0.0;

        // OpenMP parallelization over the outer loop
        #pragma omp parallel
        {
            // Thread-local sum prevents atomic locking overhead during the loop
            double local_sum = 0.0;

            // dynamic scheduling balances the triangular workload distribution
            #pragma omp for schedule(dynamic)
            for(size_t i = 0; i < n - 1; ++i) {
                // Calculate the 1D flat array offset for row i
                size_t offset = i * n - i * (i + 1) / 2 - i - 1;

                for(size_t j = i + 1; j < n; ++j) {
                    // Eigen's .row() subtraction and .norm() naturally utilize SIMD
                    double dist = (data.row(i) - data.row(j)).norm();
                    distances[offset + j] = dist;
                    local_sum += dist;
                }
            }

            // Safely accumulate the thread-local sums into the global mean
            #pragma omp atomic
            mean_val += local_sum;
        }

        mean_val /= static_cast<double>(num_pairs);
        is_computed = true;
    }

    /**
     * @brief Retrieves the mean pairwise distance \f$ \bar{d} \f$.
     * @return The mean distance.
     * @throws std::runtime_error if called before `compute()`.
     */
    double mean() const {
        if(!is_computed) throw std::runtime_error("Distances not yet computed.");
        return mean_val;
    }

    /**
     * @brief Fast quantile retrieval without executing a full sort.
     * * @details Uses `std::nth_element` to achieve \f$ \mathcal{O}(M) \f$ partial sorting performance
     * compared to \f$ \mathcal{O}(M \log M) \f$ for full sorting. Note this mutates the internal
     * distance array's ordering but preserves the exact values.
     * * @param q Target quantile probability \f$ p \in [0, 1] \f$ (e.g., 0.5 for median).
     * @return The distance value representing that quantile.
     * @throws std::runtime_error if called before `compute()`.
     * @throws std::invalid_argument if \f$ p \f$ is outside bounds.
     */
    double quantile(double q) {
        if(!is_computed) throw std::runtime_error("Distances not yet computed.");
        if(q < 0.0 || q > 1.0) throw std::invalid_argument("Quantile must be in [0, 1].");

        size_t idx = static_cast<size_t>(q * (distances.size() - 1));

        // std::nth_element provides an O(M) partial sort, which is significantly
        // faster than an O(M log M) full std::sort.
        std::nth_element(distances.begin(), distances.begin() + idx, distances.end());

        return distances[idx];
    }

    /**
     * @brief Computes the spatial Correlation Integral \f$ C(r) \f$ for a grid of radial thresholds.
     * * @details
     * ### Mathematical Formulation
     * Computes the fraction of pairwise distances strictly less than \f$ r \f$:
     * \f[
     * C(r) = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^N \Theta(r - d_{ij})
     * \f]
     * where \f$ \Theta \f$ is the Heaviside step function.
     * * ### Implementation Algorithm
     * 1. Copies and sorts the requested threshold bounds \f$ r \f$.
     * 2. Utilizes OpenMP parallelization and binary search (`std::upper_bound`) in \f$ \mathcal{O}(\log K) \f$ to rapidly bin every pairwise distance into thread-local histograms.
     * 3. Calculates the cumulative sum globally to construct the integral profile \f$ C(r) \f$.
     * * @param r_thresholds A vector of spatial radii (\f$ r \f$). Does not need to be pre-sorted.
     * @return std::vector<double> The fraction of pairwise distances less than each corresponding radius \f$ r \f$.
     * @throws std::runtime_error if called before `compute()`.
     */
    std::vector<double> correlation_integral(const std::vector<double>& r_thresholds) const {
        if(!is_computed) throw std::runtime_error("Distances not yet computed.");
        if(r_thresholds.empty()) return {};

        size_t k = r_thresholds.size();
        size_t m = distances.size();

        // 1. Create a sorted copy of the thresholds for binary search tracking
        std::vector<std::pair<double, size_t>> sorted_r(k);
        for(size_t i = 0; i < k; ++i) {
            sorted_r[i] = {r_thresholds[i], i};
        }
        std::sort(sorted_r.begin(), sorted_r.end(),
        [](const auto & a, const auto & b) {
            return a.first < b.first;
        });

        // Extract just the sorted values for std::upper_bound
        std::vector<double> r_vals(k);
        for(size_t i = 0; i < k; ++i) r_vals[i] = sorted_r[i].first;

        // Global histogram bins
        std::vector<size_t> global_counts(k + 1, 0);

        // 2. OpenMP parallel histogram construction
        #pragma omp parallel
        {
            // Thread-local histogram prevents atomic contention
            std::vector<size_t> local_counts(k + 1, 0);

            #pragma omp for schedule(static)
            for(size_t i = 0; i < m; ++i) {
                double d = distances[i];

                // Binary search: O(log K) instead of O(K)
                auto it = std::upper_bound(r_vals.begin(), r_vals.end(), d);
                size_t idx = std::distance(r_vals.begin(), it);

                // idx represents the first threshold strictly greater than d
                local_counts[idx]++;
            }

            // Merge local histograms safely
            #pragma omp critical
            {
                for(size_t j = 0; j <= k; ++j) {
                    global_counts[j] += local_counts[j];
                }
            }
        }

        // 3. Prefix Sum (Cumulative integration)
        // If a distance falls in bin j, it is smaller than all thresholds >= j.
        std::vector<double> C_r(k, 0.0);
        size_t cumulative_sum = 0;

        for(size_t j = 0; j < k; ++j) {
            cumulative_sum += global_counts[j];

            // Map the sorted result back to the original input order
            size_t original_idx = sorted_r[j].second;
            C_r[original_idx] = static_cast<double>(cumulative_sum) / static_cast<double>(m);
        }

        return C_r;
    }
};

} // namespace cmp::statistics

/** @} */

#endif // CMP_STATISTICS_H