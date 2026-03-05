#include "statistics.h"
#include <iostream>

Eigen::VectorXd cmp::statistics::mean(const Eigen::Ref<const Eigen::MatrixXd>& data) {
    if(data.size() == 0) {
        throw std::invalid_argument("Data matrix is empty");
    }

    // Points are stored in different rows
    return data.colwise().mean();
}
Eigen::MatrixXd cmp::statistics::covariance(const Eigen::Ref<const Eigen::MatrixXd>& data) {
    if(data.size() == 0) {
        throw std::invalid_argument("Data matrix is empty");
    }

    Eigen::MatrixXd centered = data.rowwise() - mean(data).transpose();
    return (centered.adjoint() * centered) / static_cast<double>(data.rows() - 1);
}

double cmp::statistics::quantile(const Eigen::Ref<const Eigen::VectorXd>& data, double q) {
    if(data.size() == 0) {
        throw std::invalid_argument("Data vector is empty");
    }
    if(q < 0.0 || q > 1.0) {
        throw std::invalid_argument("Quantile must be between 0 and 1");
    }

    std::vector<double> sorted_data(data.data(), data.data() + data.size());
    std::sort(sorted_data.begin(), sorted_data.end());

    if(sorted_data.size() == 1) return sorted_data[0];

    double pos  = (sorted_data.size() - 1) * q;
    auto idx    = static_cast<size_t>(std::floor(pos));
    double frac = pos - idx;

    if(idx + 1 < sorted_data.size()) {
        return sorted_data[idx] * (1.0 - frac) + sorted_data[idx + 1] * frac;
    } else {
        return sorted_data[idx];
    }
}

Eigen::VectorXd cmp::statistics::interQuantileRange(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                                    double lowerQuantile, double upperQuantile) {
    if(data.size() == 0) {
        throw std::invalid_argument("Data matrix is empty");
    }
    if(lowerQuantile < 0.0 || upperQuantile > 1.0 || lowerQuantile > upperQuantile) {
        throw std::invalid_argument("Invalid quantile range");
    }

    Eigen::Index nFeatures = data.cols();
    Eigen::VectorXd iqr(nFeatures);

    for(Eigen::Index j = 0; j < nFeatures; ++j) {
        Eigen::VectorXd col = data.col(j); // copia sicura
        double q_low  = quantile(col, lowerQuantile);
        double q_high = quantile(col, upperQuantile);
        iqr(j) = q_high - q_low;
    }

    return iqr;
}



Eigen::MatrixXd cmp::statistics::pearsonCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2) {

    // Handle the case where datasets have different number of points
    size_t nPoints = std::min(data1.rows(), data2.rows());
    size_t nFeatures = data1.cols();

    assert(nFeatures == data2.cols());

    Eigen::MatrixXd corr(nFeatures, nFeatures);
    for(size_t i = 0; i < nFeatures; i++) {
        for(size_t j = i; j < nFeatures; j++) {
            Eigen::VectorXd x = data1.col(i).head(nPoints);
            Eigen::VectorXd y = data2.col(j).head(nPoints);

            double mean_x = x.mean();
            double mean_y = y.mean();

            double numerator = ((x.array() - mean_x) * (y.array() - mean_y)).sum();
            double denominator = std::sqrt(((x.array() - mean_x).square().sum()) * ((y.array() - mean_y).square().sum()));

            if(denominator == 0) {
                corr(i, j) = 0; // or handle as needed
            } else {
                corr(i, j) = numerator / denominator;
            }

            // Fill the symmetric element
            if(i != j) {
                corr(j, i) = corr(i, j); // Symmetric matrix
            }
        }
    }
    return corr;
}

Eigen::MatrixXd cmp::statistics::laggedCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2, int lag) {

    size_t nPoints = std::min(data1.rows(), data2.rows());
    size_t nFeatures = data1.cols();

    assert(nFeatures == data2.cols());

    Eigen::MatrixXd corr(nFeatures, nFeatures);
    for(size_t i = 0; i < nFeatures; i++) {
        for(size_t j = 0; j < nFeatures; j++) {
            Eigen::VectorXd x, y;
            if(lag >= 0) {
                x = data1.col(i).head(nPoints - lag);
                y = data2.col(j).segment(lag, nPoints - lag);
            } else {
                x = data1.col(i).segment(-lag, nPoints + lag);
                y = data2.col(j).head(nPoints + lag);
            }

            double mean_x = x.mean();
            double mean_y = y.mean();

            double numerator = ((x.array() - mean_x) * (y.array() - mean_y)).sum();
            double denominator = std::sqrt(((x.array() - mean_x).square().sum()) * ((y.array() - mean_y).square().sum()));

            if(denominator == 0) {
                corr(i, j) = 0; // or handle as needed
            } else {
                corr(i, j) = numerator / denominator;
            }
        }
    }
    return corr;
}

std::vector<Eigen::MatrixXd> cmp::statistics::laggedCorrelation(const Eigen::Ref<const Eigen::MatrixXd>& data1, const Eigen::Ref<const Eigen::MatrixXd>& data2, int minLag, int maxLag) {

    if(minLag >= maxLag) {
        throw std::invalid_argument("minLag must be less than maxLag");
    }

    std::vector<Eigen::MatrixXd> correlations;
    for(int lag = minLag; lag <= maxLag; lag++) {
        correlations.push_back(laggedCorrelation(data1, data2, lag));
    }
    return correlations;
}

std::pair<Eigen::VectorXd, double> cmp::statistics::selfCorrelationLength(const Eigen::Ref<const Eigen::MatrixXd>& data) {

    Eigen::VectorXd mean = cmp::statistics::mean(data);
    Eigen::VectorXd var = cmp::statistics::covariance(data).diagonal();

    // Get the dimensions
    size_t nSamples = data.rows();
    size_t nFeatures = data.cols();

    Eigen::VectorXd corr_length = Eigen::VectorXd::Zero(nFeatures);
    Eigen::VectorXd self_corr_prev(nFeatures);

    for(int lag = 0; lag < nSamples; lag++) {

        Eigen::MatrixXd self_corr = cmp::statistics::laggedCorrelation(data, data, lag).diagonal();

        if((self_corr_prev + self_corr).minCoeff() < 0 && (lag - 1) % 2 == 0) {
            break;
        } else {
            corr_length += self_corr;
            self_corr_prev = self_corr;
        }
    }

    corr_length = corr_length.cwiseQuotient(var);
    double ess = static_cast<double>(nSamples) / (-1.0 + 2.0 * corr_length.maxCoeff());

    return std::make_pair(corr_length, ess);
}
