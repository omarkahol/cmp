/**
 * @file statistics.h
 * @brief Header file for statistical functions and classes
 * This file contains declarations for various statistical functions and classes used in the project.
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

namespace cmp::statistics {

/**
 * @brief K-Fold Cross-Validation class
 */
class KFold {
  private:
    Eigen::Index nSplits_;
    Eigen::Index nObs_;
    bool shuffle_;
    unsigned int rngState_;
    std::default_random_engine rng_;
    Eigen::VectorXs indices_;

  public:

    /**
     * @brief Constructor for KFold class
     * @param n_splits Number of folds. Must be at least 2 and at most n_obs.
     * @param n_obs Total number of observations in the dataset. Must be at least 2.
     * @param shuffle Whether to shuffle the data before splitting into folds.
     * @param random_state Seed for the random number generator (used if shuffle is true).
     * @throws std::invalid_argument if the input parameters are invalid.
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

    class KFoldIterator {
      private:
        Eigen::Index currentSplit_;
        const KFold &parent_;

      public:
        KFoldIterator(const KFold& kf, Eigen::Index split)
            : currentSplit_(split), parent_(kf) {}

        bool operator!=(const KFoldIterator& other) const {
            return currentSplit_ != other.currentSplit_;
        }

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

        KFoldIterator& operator++() {
            ++currentSplit_;
            return *this;
        }
    };

    KFoldIterator begin() const {
        return KFoldIterator(*this, 0);
    }
    KFoldIterator end()   const {
        return KFoldIterator(*this, nSplits_);
    }

    std::pair<Eigen::VectorXs, Eigen::VectorXs> operator()(Eigen::Index split) const {
        if(split < 0 || split >= nSplits_) {
            throw std::out_of_range("Split index out of range.");
        }
        return *(KFoldIterator(*this, split));
    }

    size_t nFolds() const {
        return this->nSplits_;
    }
};


class Bootstrap {
  private:
    size_t nObs_;
    size_t nObsResample_;
    unsigned int rngState_;
    std::default_random_engine rng_;
    std::uniform_int_distribution<size_t> dist_;
    Eigen::VectorXs indices_;
    bool withReplacement_;

  public:
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


/* @brief Computes the mean of a vector of data.
 *
 * @param data A reference to an matrix containing the data (different points in rows).
 * @return The mean of the data.
 * @throws std::invalid_argument if the input vector is empty.
 */
Eigen::VectorXd mean(const Eigen::Ref<const Eigen::MatrixXd>& data);

/**
 * @brief Computes the variance of a vector of data.
 *
 * @param data A reference to an Eigen vector containing the data.
 * @return The variance of the data.
 * @throws std::invalid_argument if the input vector is empty.
 */
Eigen::MatrixXd covariance(const Eigen::Ref<const Eigen::MatrixXd>& data);

/**
 * @brief Computes the median of a vector of data.
 *
 * @param data A reference to an Eigen vector containing the data.
 * @return The median of the data.
 * @throws std::invalid_argument if the input vector is empty.
 */
double quantile(const Eigen::Ref<const Eigen::VectorXd>& data, double quantile);


Eigen::VectorXd interQuantileRange(const Eigen::Ref<const Eigen::MatrixXd>& data, double lowerQuantile, double upperQuantile);

/**
 * @brief Compute the Pearson correlation matrix between two datasets.
 * Manages the case where the datasets have different numbers of points.
 * @param data1 First dataset (points in rows).
 * @param data2 Second dataset (points in rows).
 * @return The Pearson correlation matrix.
 */
Eigen::MatrixXd pearsonCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2);


/**
 * @brief Computes the correlation matrix between two datasets with a specified lag.
 *
 * @param data1 First dataset (points in rows).
 * @param data2 Second dataset (points in rows).
 * @return The correlation matrix.
 */
Eigen::MatrixXd laggedCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2, int lag);


/**
 * @brief Return the correlation matrix between two datasets for all lags in the range [minLag, maxLag].
 *
 * @param data1 First dataset (points in rows).
 * @param data2 Second dataset (points in rows).
 * @param minLag Minimum lag to consider.
 * @param maxLag Maximum lag to consider.
 * @return The correlation matrix.
 */
std::vector<Eigen::MatrixXd> laggedCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2, int minLag, int maxLag);


/**
 * @brief Computes the self-correlation length and the effective sample size of some samples.
 * @param data the dataset containing the samples (each sample is a row)
 * @return a pair containing the correlations lengths and the effective sample size
 */
std::pair<Eigen::VectorXd, double> selfCorrelationLength(const Eigen::Ref<const Eigen::MatrixXd>& data);

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


} // namespace cmp::statistics

#endif // CMP_STATISTICS_H