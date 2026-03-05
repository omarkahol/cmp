#ifndef MODEL_CLUSTER_H
#define MODEL_CLUSTER_H

#include <iostream>
#include <limits>
#include <distribution.h>
#include <cmp_defines.h>
#include <grid.h>
#include <classifier.h>
#include <gp.h>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cstring>


namespace cmp {

class ModelCluster {

  private:
    std::default_random_engine rng_;

    Eigen::MatrixXd xObs_;
    Eigen::VectorXd yObs_;

    size_t nClusters_;
    size_t nObs_;
    size_t dimX_;

    // The labels of the points
    Eigen::VectorXs labels_;

    // The local index table
    Eigen::VectorXs localIndexTable_;

    // The GPs for each cluster, along with their centroids and fit status
    std::vector<bool> fit_;
    std::vector<Eigen::VectorXd> centroids_;
    std::vector<cmp::gp::GaussianProcess> gps_;

    // The cluster sizes
    std::vector<size_t> clusterSize_;

    // The Gamma parameter
    double gamma_{1.0};

    // Kernel, mean and nugget for each cluster
    std::shared_ptr<covariance::Covariance> kernel_;
    std::shared_ptr<mean::Mean> mean_;
    Eigen::VectorXd parameters_;
    double nugget_{1e-8};
  public:

    ModelCluster() = default;

    void set(std::shared_ptr<covariance::Covariance> kernel, std::shared_ptr<mean::Mean> mean, Eigen::VectorXd parameters, double nugget, double gamma, unsigned int seed) {
        kernel_ = kernel;
        mean_ = mean;
        nugget_ = nugget;
        parameters_ = parameters;
        gamma_ = gamma;
        rng_.seed(seed);
    }

    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, const Eigen::Ref<const Eigen::VectorXs> &labels) {

        // Initialize the members
        nObs_ = xObs.rows();
        dimX_ = xObs.cols();
        xObs_ = xObs;
        yObs_ = yObs;

        // Count the number of clusters
        nClusters_ = 0;
        for(size_t i = 0; i < nObs_; i++) {
            if(labels(i) + 1 > nClusters_) {
                nClusters_ = labels(i) + 1;
            }
        }

        // Set the GPs
        gps_.clear();
        for(size_t i = 0; i < nClusters_; i++) {
            gps_.emplace_back(kernel_, mean_, parameters_, nugget_);
        }

        // Initialize the containers
        labels_ = labels;
        localIndexTable_ = Eigen::VectorXs::Zero(nObs_);
        centroids_ = std::vector<Eigen::VectorXd>(nClusters_, Eigen::VectorXd::Zero(dimX_));
        // Ensure per-cluster bookkeeping is correctly sized now that nClusters_ is known
        fit_ = std::vector<bool>(nClusters_, false);
        clusterSize_ = std::vector<size_t>(nClusters_, 0);

        // Call the update model function
        updateModel(std::vector<bool>(nClusters_, true));
    }

    std::pair<double, double> predict(const Eigen::VectorXd &xStar, cmp::classifier::Classifier *classifier) const {

        double mu = 0.0;
        double var = 0.0;
        std::vector<double> probs = classifier->predictProbabilities(xStar);
        for(size_t i = 0; i < nClusters_; i++) {
            auto [mu_i, var_i] = gps_[i].predict(xStar);
            mu += probs[i] * mu_i;
            var += probs[i] * probs[i] * var_i;
        }

        return {mu, var};
    }

    size_t nClusters() const {
        return nClusters_;
    }

    void fit(Eigen::Ref<const Eigen::VectorXd> lowerBound, Eigen::Ref<const Eigen::VectorXd> upperBound, cmp::gp::method fitType, nlopt::algorithm algorithm = nlopt::LN_SBPLX, double tol = 1e-3, std::shared_ptr<cmp::prior::Prior> prior = nullptr) {


        for(size_t i = 0; i < nClusters(); i++) {
            if(!fit_[i]) {
                auto index = getIndices(i);
                auto xObsi = cmp::slice(xObs_, index);
                auto yObsi = cmp::slice(yObs_, index);

                gps_[i].fit(xObsi, yObsi, lowerBound, upperBound, fitType, algorithm, tol, true, prior);
                fit_[i] = true;
            }
        }
    }

    size_t nPoints() const {
        return nObs_;
    }

    size_t dim() const {
        return dimX_;
    }

    size_t getMembership(size_t i) const {
        return labels_[i];
    }

    const Eigen::VectorXs &getLabels() const {
        return labels_;
    }

    size_t getClusterSize(size_t i) const {
        return clusterSize_[i];
    }

    cmp::gp::GaussianProcess &operator[](size_t i) {
        return gps_[i];
    }

    const Eigen::VectorXd &centroid(size_t i) const {
        return centroids_[i];
    }

    Eigen::VectorXs getIndices(size_t clusterIndex) const {
        Eigen::VectorXs indices = Eigen::VectorXs::Zero(clusterSize_[clusterIndex]);
        size_t counter = 0;
        for(size_t j = 0; j < nObs_; j++) {
            if(labels_[j] == clusterIndex) {
                indices[counter] = j;
                counter++;
            }
        }
        return indices;
    }

    void updateModel(const std::vector<bool> &affectedClusters) {

        // Set the observations
        for(size_t i = 0; i < nClusters_; i++) {

            // Check if the cluster is affected
            if(!affectedClusters[i]) {
                continue;
            }

            // Centroids
            centroids_[i] = Eigen::VectorXd::Zero(dimX_);

            // Counter for the cluster size
            size_t counter = 0;

            // Set the observations
            for(size_t j = 0; j < nObs_; j++) {
                if(labels_[j] == i) {

                    // Update the centroid
                    centroids_[i] += xObs_.row(j).transpose();

                    // Set the membership
                    localIndexTable_[j] = counter;

                    // Update the counter
                    counter++;
                }
            }

            // Compute the centroid
            centroids_[i] /= double(counter);

            // Set the cluster size
            clusterSize_[i] = counter;

            // GP is not fit
            fit_[i] = false;
        }
    }

    /**
     * This function performs the switch of the points
     * @param newOwners The new owners of the points, the first element is the global index and the second element is the owner
     */
    bool performSwitches(const std::vector<std::pair<size_t, size_t>> &newOwners, size_t minSize) {

        // Affected clusters
        std::vector<bool> affectedClusters(nClusters_, false);

        // Iterate through the points
        for(size_t i = 0; i < newOwners.size(); i++) {

            // Get the global index
            size_t globalIndex = newOwners[i].first;

            // Get the new and old owners
            size_t newOwner = newOwners[i].second;
            size_t oldOwner = labels_[globalIndex];

            // The clusters are affected
            affectedClusters[oldOwner] = true;
            affectedClusters[newOwner] = true;

            // Update the membership
            labels_[globalIndex] = newOwner;
        }

        // Call the update model function
        updateModel(affectedClusters);

        // Now check if we can purge clusters
        size_t numPurges = 0;
        bool finished = false;
        while(!finished) {
            finished = true;
            for(size_t i = 0; i < nClusters_; i++) {
                if(clusterSize_[i] < minSize) {
                    purge(i);
                    numPurges++;
                    finished = false;
                    break;
                }
            }
        }

        return (numPurges > 0) || (newOwners.size() > 0);
    }

    /**
     * This function computes the cross validation error for each cluster
     */
    double computeScore(size_t globalIndex) const {

        // Find the owner of the point
        size_t owner = labels_[globalIndex];
        size_t localIndex = localIndexTable_[globalIndex];

        // Evaluate the error
        auto [mean, var] = gps_[owner].predictLOO(localIndex);
        return cmp::distribution::NormalDistribution::logPDF(yObs_(globalIndex) - mean, std::sqrt(var));
    }

    /**
     * This function computes the score of cluster \p clusterIndex on the point \p globalIndex
     *
     * @param clusterIndex The index of the cluster
     * @param globalIndex The global index of the point
     * @param errorType The type of error to compute
     *
     * @return The error
     */
    double computeScore(size_t clusterIndex, size_t globalIndex) const {

        auto [mean, var] = gps_[clusterIndex].predict(xObs_.row(globalIndex).transpose());
        return cmp::distribution::NormalDistribution::logPDF(yObs_(globalIndex) - mean, std::sqrt(var));

    }

    std::vector<std::pair<size_t, size_t>> switchStep(cmp::classifier::Classifier* classifier, size_t maxAllowedSwitches = 10, double minProb = 0.1, double T = 1.0) {

        // Compute membership probabilities
        auto probabilities = computeProbabilities(T, classifier, minProb);

        // Compute The probability of actually changing the membership for each point
        // This is done by summing the probabilities of all clusters except the current owner
        std::vector<double> switchingProbabilities(nObs_, 0.0);
        for(size_t i = 0; i < nObs_; i++) {
            size_t owner = labels_[i];
            double sumProb = 0.0;
            for(size_t j = 0; j < nClusters_; ++j) {
                if(j != owner) {
                    sumProb += probabilities[i][j];
                }
            }
            switchingProbabilities[i] = sumProb;
        }

        // Precompute a list with the active indices (i.e., those with non-zero switching probability)
        std::vector<size_t> activeIndices;
        activeIndices.reserve(nObs_);
        for(size_t i = 0; i < nObs_; ++i)
            if(switchingProbabilities[i] > 0.0)
                activeIndices.push_back(i);

        std::vector<std::pair<size_t, size_t>> switches;
        switches.reserve(maxAllowedSwitches);

        // Iteratively sample a new cluster and remove selected entries
        for(size_t s = 0; s < maxAllowedSwitches && !activeIndices.empty(); ++s) {

            // Build discrete distribution from current weights
            std::vector<double> weights;
            weights.reserve(activeIndices.size());
            for(auto idx : activeIndices)
                weights.push_back(switchingProbabilities[idx]);

            std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
            size_t sampledPos = dist(rng_);
            size_t globalIndex = activeIndices[sampledPos];

            // Remove selected index from active pool (swap & pop)
            activeIndices[sampledPos] = activeIndices.back();
            activeIndices.pop_back();

            // Sample new cluster for this point
            size_t owner = labels_[globalIndex];
            const auto& probs = probabilities[globalIndex];
            std::discrete_distribution<size_t> clusterDist(probs.begin(), probs.end());
            size_t newOwner = clusterDist(rng_);

            if(newOwner != owner)
                switches.emplace_back(globalIndex, newOwner);
        }

        return switches;
    }



    std::vector<std::pair<size_t, size_t>> deterministicSwitchStep(cmp::classifier::Classifier *cls, const size_t &maxAllowedSwitches, const double &minProb) {

        // Compute the score changes
        std::vector<std::pair<double, std::pair<size_t, size_t>>> allSwitches;
        allSwitches.reserve(nObs_);

        // Precompute classifier probabilities once for all observations
        std::vector<std::vector<double>> classifierProbabilities(nObs_);
        for(size_t i = 0; i < nObs_; i++) {
            classifierProbabilities[i] = cls->predictProbabilities(xObs_.row(i));
        }

        for(size_t i = 0; i < nObs_; i++) {

            // Compute SVM probabilities
            const std::vector<double> &prob = classifierProbabilities[i];

            // Find the owner
            size_t owner = labels_[i];

            // Compute the LOO score
            double looScore = computeScore(i);

            // Starting score is -inf if the current cluster has a probability below the threshold, to ensure we switch it if there is any other cluster with a probability above the threshold
            double startingScore = 0;
            if(prob[owner] < minProb) {
                startingScore = -std::numeric_limits<double>::infinity();
            }

            // Iterate through the clusters
            std::pair<double, size_t> bestCluster{startingScore, owner};
            for(size_t c = 0; c < nClusters_; c++) {
                if(c == owner) {
                    continue;
                } else if(prob[c] <= minProb) {
                    continue;
                } else {
                    // Predict the score
                    double predScore = computeScore(c, i);
                    double deltaScore = (predScore - looScore);

                    // Check if it is better than the owner
                    if(deltaScore > bestCluster.first) {
                        bestCluster.first = deltaScore;
                        bestCluster.second = c;
                    }

                }
            }

            // If it is the owner we skip
            if(bestCluster.second == owner) {
                continue;
            } else {
                // We shall add it in the list
                allSwitches.push_back({bestCluster.first, {i, bestCluster.second}});
            }
        }




        // We sort the previous vector
        std::sort(allSwitches.begin(), allSwitches.end(), [](std::pair<double, std::pair<size_t, size_t>> a, std::pair<double, std::pair<size_t, size_t>> b) {
            return a.first > b.first;
        });

        size_t nSwitches = std::min(maxAllowedSwitches, allSwitches.size());

        std::vector<std::pair<size_t, size_t>> switches(nSwitches);
        for(size_t i = 0; i < nSwitches; i++) {
            switches[i] = allSwitches[i].second;
        }

        return switches;
    }

    /**
     * This function purges a cluster
     *
     * @param clusterIndex The index of the cluster to purge
     *
     */
    void purge(const size_t &clusterIndex) {

        std::vector<bool> affectedClusters(nClusters_, false);
        std::cout << "Purging cluster " << clusterIndex << " with size " << getClusterSize(clusterIndex) << std::endl;

        // Cycle through the observations
        for(size_t i = 0; i < nObs_; i++) {

            // Check if the point is in the cluster to purge
            if(labels_[i] == clusterIndex) {

                // We need to redistribute the point
                std::pair<size_t, double> chosenCluster = std::make_pair(0, std::numeric_limits<double>::infinity());

                for(size_t j = 0; j < nClusters_; j++) {

                    // Skip the current cluster
                    if(j == clusterIndex) {
                        continue;
                    }

                    // Compute the error
                    double centroidDistance = (xObs_.row(i).transpose() - centroids_[j]).squaredNorm();

                    // Check if we have a new minimum
                    if(centroidDistance < chosenCluster.second) {
                        chosenCluster = std::make_pair(j, centroidDistance);
                    }
                }

                // Perform the switch
                labels_[i] = chosenCluster.first;
                affectedClusters[chosenCluster.first] = true;
            }
        }

        // Purge the cluster
        gps_.erase(gps_.begin() + clusterIndex);
        fit_.erase(fit_.begin() + clusterIndex);
        clusterSize_.erase(clusterSize_.begin() + clusterIndex);
        centroids_.erase(centroids_.begin() + clusterIndex);
        affectedClusters.erase(affectedClusters.begin() + clusterIndex);

        nClusters_--;

        // Now we need to decrease the cluster index of the points
        for(size_t i = 0; i < nObs_; i++) {
            if(labels_[i] > clusterIndex) {
                labels_[i]--;
            }
        }

        // Now we need to update the model
        updateModel(affectedClusters);
    }

    std::vector<std::vector<double>> computeProbabilities(const double &T, cmp::classifier::Classifier *classifier, const double &minProb = 0.1) const {

        std::vector<std::vector<double>> switchingProbabilities(nObs_, std::vector<double>(nClusters_, 0.0));

        // Precompute classifier probabilities and leave-one-out scores once per point
        std::vector<std::vector<double>> classifierProbabilities(nObs_);
        std::vector<double> looScores(nObs_, 0.0);
        for(size_t i = 0; i < nObs_; i++) {
            classifierProbabilities[i] = classifier->predictProbabilities(xObs_.row(i));
            looScores[i] = computeScore(i);
        }

        // Iterate through the points
        for(size_t i = 0; i < nObs_; i++) {

            // Find the owner of the point
            size_t owner = labels_[i];
            size_t localIndex = localIndexTable_[i];

            // Predict the SVM probabilities for the point
            const std::vector<double> &svmProbabilities = classifierProbabilities[i];

            // Compute the LOO score for the current point
            double looScore = looScores[i];
            switchingProbabilities[i][owner] = svmProbabilities[owner];

            for(size_t j = 0; j < nClusters_; j++) {
                if(j == owner) {
                    continue;
                } else {

                    if(svmProbabilities[j] <= minProb) {
                        switchingProbabilities[i][j] = 0;
                        continue;
                    }

                    // Compute the predictive error
                    double predictiveScore = computeScore(j, i);

                    switchingProbabilities[i][j] = std::exp((predictiveScore - looScore) / T) * svmProbabilities[j];

                }
            }

            // Compute the sum of the probabilities
            double sum = 0.0;
            for(size_t j = 0; j < nClusters_; j++) {
                sum += switchingProbabilities[i][j];
            }

            // Check if the sum is valid
            if(sum < 1e-12 || std::isnan(sum) || std::isinf(sum)) {

                // We revert back to the SVM probabilities
                for(size_t j = 0; j < nClusters_; j++) {
                    switchingProbabilities[i][j] = svmProbabilities[j];
                }
            } else {
                // Normalize the probabilities
                for(size_t j = 0; j < nClusters_; j++) {
                    switchingProbabilities[i][j] /= sum;
                }
            }

        }

        // Now we return the probabilities
        return switchingProbabilities;
    }

    Eigen::MatrixXd confusionMatrix(std::vector<std::vector<double>> switchingProbability) const {
        Eigen::MatrixXd confusion_num(nClusters_, nClusters_);
        Eigen::MatrixXd confusion_den(nClusters_, nClusters_);
        confusion_num.setZero();
        confusion_den.setZero();

        for(size_t i = 0; i < nObs_; i++) {
            for(size_t j = 0; j < nClusters_; j++) {
                for(size_t k = 0; k < nClusters_; k++) {

                    // Only consider points that are in cluster j
                    if(labels_[i] == j) {
                        confusion_num(j, k) += std::abs(switchingProbability[i][j] - switchingProbability[i][k]);
                        confusion_den(j, k) += switchingProbability[i][j] + switchingProbability[i][k];
                    }
                }
            }
        }

        // Divide the matrices elementwise
        for(size_t i = 0; i < nClusters_; i++) {
            for(size_t j = 0; j < nClusters_; j++) {
                if(confusion_den(i, j) != 0) {
                    confusion_num(i, j) /= confusion_den(i, j);
                }
            }
        }

        // Now we return the confusion matrix
        return confusion_num;
    }

    // Merge two clusters
    void merge(const size_t &clusterIndex1, const size_t &clusterIndex2) {

        // Check if the clusters are the same
        if(clusterIndex1 == clusterIndex2) {
            return;
        }

        std::vector<std::pair<size_t, size_t>> switches;

        // Cycle through the observations
        for(size_t i = 0; i < nObs_; i++) {

            // Check if the point is in the cluster to purge
            if(labels_[i] == clusterIndex2) {
                switches.push_back(std::make_pair(i, clusterIndex1));
            }
        }

        // Perform the switches
        performSwitches(switches, 0);
    }

};

namespace covariance {
class ModelClusterCovariance : public cmp::covariance::Covariance {
  private:
    cmp::ModelCluster *pModelCluster_;
    cmp::classifier::Classifier *pClassifier_;
    mutable std::unordered_map<std::string, std::vector<double>> probabilityCache_;

    std::string makeProbabilityCacheKey(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        const std::size_t n = static_cast<std::size_t>(x.size());
        std::string key(sizeof(std::size_t) + n * sizeof(double), '\0');
        std::memcpy(key.data(), &n, sizeof(std::size_t));
        std::memcpy(key.data() + sizeof(std::size_t), x.data(), n * sizeof(double));
        return key;
    }

    const std::vector<double> &getCachedProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        std::string key = makeProbabilityCacheKey(x);
        auto it = probabilityCache_.find(key);
        if(it == probabilityCache_.end()) {
            auto probs = pClassifier_->predictProbabilities(x);
            it = probabilityCache_.emplace(std::move(key), std::move(probs)).first;
        }
        return it->second;
    }
  public:

    ModelClusterCovariance(cmp::ModelCluster *modelCluster, cmp::classifier::Classifier *classifier) : pModelCluster_(modelCluster), pClassifier_(classifier) {};

    void clearProbabilityCache() const {
        probabilityCache_.clear();
    }

    void precomputeProbabilities(const Eigen::Ref<const Eigen::MatrixXd> &xObs) const {
        probabilityCache_.reserve(probabilityCache_.size() + static_cast<std::size_t>(xObs.rows()));
        for(size_t i = 0; i < xObs.rows(); i++) {
            (void)getCachedProbabilities(xObs.row(i));
        }
    }

    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        const auto &p1 = getCachedProbabilities(x1);
        const auto &p2 = getCachedProbabilities(x2);
        double result = 0.0;
        for(size_t k = 0; k < p1.size(); k++) {
            double cov = (*pModelCluster_)[k].getKernel()->eval(x1, x2, (*pModelCluster_)[k].getParameters());
            result += std::sqrt(p1[k] * p2[k]) * cov;
        }
        return result;
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return 0.0;
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return 0.0;
    };

    static std::shared_ptr<Covariance> make(cmp::ModelCluster *modelCluster, cmp::classifier::Classifier *classifier) {
        return std::make_shared<ModelClusterCovariance>(modelCluster, classifier);
    };
};
} // namespace covariance

namespace mean {
class ModelClusterMean: public cmp::mean::Mean {
  private:
    cmp::ModelCluster *pModelCluster_;
    cmp::classifier::Classifier *pClassifier_;
  public:

    ModelClusterMean(cmp::ModelCluster *modelCluster, cmp::classifier::Classifier *classifier) : pModelCluster_(modelCluster), pClassifier_(classifier) {};
    double eval(const Eigen::VectorXd& x, const Eigen::VectorXd &par) const {
        // Preidict the probabilities
        auto probs = pClassifier_->predictProbabilities(x);
        double mu = 0.0;
        for(size_t i = 0; i < pModelCluster_->nClusters(); i++) {
            mu += probs[i] * (*pModelCluster_)[i].getMean()->eval(x, (*pModelCluster_)[i].getParameters());
        }
        return mu;

    };
    double evalGradient(const Eigen::VectorXd& x, const Eigen::VectorXd &par, const size_t &i) const {
        return 0.0;
    };
    double evalHessian(const Eigen::VectorXd& x, const Eigen::VectorXd &par, const size_t &i, const size_t &j) const {
        return 0.0;
    };

    static std::shared_ptr<Mean> make(cmp::ModelCluster *modelCluster, cmp::classifier::Classifier *classifier) {
        return std::make_shared<ModelClusterMean>(modelCluster, classifier);
    };
};
} // namespace mean

} // namespace cmp

#endif