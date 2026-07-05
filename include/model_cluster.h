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


/**
 * @addtogroup surrogate
 * @{
 */
namespace cmp {

/**
 * @class ModelCluster
 * @brief Manages a clustered set of Gaussian Processes for localized regression.
 *
 * @details Mathematical Formulation
 * Let the dataset be composed of $N$ observations $(x_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$.
 * The data is partitioned into $K$ clusters. Each cluster $k \in \{1, \ldots, K\}$ is modeled
 * by an independent Gaussian Process (GP) with a local covariance kernel $k_k(x, x'; \theta_k)$
 * and a mean function $m_k(x; \theta_k)$.
 * The cluster membership of observation $i$ is denoted by the label $c_i \in \{0, \ldots, K-1\}$.
 *
 * For a query point $x^*$, a classifier predicts the probability of cluster membership $P(C = k \mid x^*)$.
 * The blended prediction is given by:
 * \f[
 * \mu(x^*) = \sum_{k=1}^K P(C = k \mid x^*) \mu_k(x^*)
 * \f]
 * \f[
 * \sigma^2(x^*) = \sum_{k=1}^K P(C = k \mid x^*)^2 \sigma_k^2(x^*)
 * \f]
 * where $\mu_k(x^*)$ and $\sigma_k^2(x^*)$ are the mean and variance predictions of the $k$-th GP.
 *
 * Switching steps evaluate the change in predictive log-likelihood (LOO error for current cluster,
 * or standard predictive log-likelihood for candidate clusters). The stochastic switching probability
 * for point $i$ to cluster $j$ is computed as:
 * \f[
 * p_{ij} \propto P(C = j \mid x_i) \exp\left( \frac{\Delta_{ij}}{T} \right)
 * \f]
 * where $\Delta_{ij}$ represents the log-predictive score difference if point $i$ is moved to cluster $j$,
 * and $T$ is the temperature parameter.
 *
 * @details Implementation Algorithm
 * 1. **Initialization (`condition`)**: Groups the input observations according to their labels,
 *    instantiates the GP models, computes initial cluster centroids, and sets up index lookup tables.
 * 2. **Blended Prediction (`predict`)**: Obtains probabilities from the classifier, queries predictions
 *    from each local GP, and performs the weighted combination.
 * 3. **Stochastic Switching (`switchStep`)**: Precomputes probability distributions for all points
 *    based on likelihood shifts, filters active points, and samples switches up to `maxAllowedSwitches`.
 * 4. **Deterministic Switching (`deterministicSwitchStep`)**: Calculates the log-likelihood gain
 *    for all possible re-assignments, sorts them, and executes the top beneficial moves.
 * 5. **Purging / Merging**: Reassigns points from a dissolved cluster to their nearest centroids,
 *    shrinks the list of active GPs, and shifts all cluster labels to maintain contiguous indices.
 */
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

    /**
     * @brief Configures parameters, kernel, and random seed for the clustered local GP model.
     * @param kernel Shared covariance function.
     * @param mean Shared mean function.
     * @param parameters Covariance hyperparameter vector.
     * @param nugget Standard observation noise.
     * @param gamma Blending scale/regularization factor.
     * @param seed Random seed.
     */
    void set(std::shared_ptr<covariance::Covariance> kernel, std::shared_ptr<mean::Mean> mean, Eigen::VectorXd parameters, double nugget, double gamma, unsigned int seed) {
        kernel_ = kernel;
        mean_ = mean;
        nugget_ = nugget;
        parameters_ = parameters;
        gamma_ = gamma;
        rng_.seed(seed);
    }

    /**
     * @brief Conditions the model on observations and initial cluster labels.
     * @param xObs Observation point matrix.
     * @param yObs Response vector.
     * @param labels Cluster assignments vector.
     */
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

    void fit(Eigen::Ref<const Eigen::VectorXd> lowerBound, Eigen::Ref<const Eigen::VectorXd> upperBound, cmp::gp::method fitType, nlopt::algorithm algorithm = nlopt::LN_SBPLX, double tol = 1e-3, std::shared_ptr<cmp::prior::Prior> prior = nullptr, std::vector<bool> logScale = {}) {


        for(size_t i = 0; i < nClusters(); i++) {
            if(!fit_[i]) {
                auto index = getIndices(i);
                auto xObsi = cmp::slice(xObs_, index);
                auto yObsi = cmp::slice(yObs_, index);

                gps_[i].fit(xObsi, yObsi, lowerBound, upperBound, fitType, algorithm, tol, true, false, prior, logScale);
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
     * @param minSize The minimum size of the clusters
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
     * This function computes the score of the point \p globalIndex on its current cluster
     *
     * @param globalIndex The global index of the point
     * @return The LOO log-predictive contribution of the point under its current cluster assignment.
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

            // Skip expensive delta evaluations if no eligible alternative cluster.
            bool hasEligibleAlternative = false;
            for(size_t c = 0; c < nClusters_; ++c) {
                if(c != owner && prob[c] > minProb) {
                    hasEligibleAlternative = true;
                    break;
                }
            }
            if(!hasEligibleAlternative) {
                continue;
            }

            // Compute remove-from-owner term once.
            const double ownerRemovalDelta = computeScore(i);

            // Strict monotone mode: only accept positive total move deltas.
            double startingScore = 0.0;

            // Iterate through the clusters
            std::pair<double, size_t> bestCluster{startingScore, owner};
            for(size_t c = 0; c < nClusters_; c++) {
                if(c == owner) {
                    continue;
                } else if(prob[c] <= minProb) {
                    continue;
                } else {
                    // Exact move gain = remove-from-owner delta + add-to-candidate delta.
                    double deltaScore = computeScore(c, i) - ownerRemovalDelta;

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

        // Precompute classifier probabilities once per point
        std::vector<std::vector<double>> classifierProbabilities(nObs_);
        for(size_t i = 0; i < nObs_; i++) {
            classifierProbabilities[i] = classifier->predictProbabilities(xObs_.row(i));
        }

        // Iterate through the points
        for(size_t i = 0; i < nObs_; i++) {

            // Find the owner of the point
            size_t owner = labels_[i];

            // Exact cluster-level delta for removing point i from its owner.
            const double ownerScore = computeScore(i);

            // Compute the log-weights for each cluster
            std::vector<double> logWeights(nClusters_, -std::numeric_limits<double>::infinity());

            for(size_t j = 0; j < nClusters_; j++) {


                if(j == owner) {

                    // If j is the owner, we see if it can actually own the point (i.e., if the classifier probability is above the threshold)
                    if(classifierProbabilities[i][j] <= minProb) {
                        logWeights[j] = -std::numeric_limits<double>::infinity();
                    } else {
                        // If j is the owner and can own the point, we compute the delta score and combine it with the classifier probability.
                        logWeights[j] = std::log(classifierProbabilities[i][j]);
                    }

                } else {

                    // If j is not the owner but can't own the point due to low classifier probability, we skip it.
                    if(classifierProbabilities[i][j] <= minProb) {
                        logWeights[j] = -std::numeric_limits<double>::infinity();
                    } else {

                        // If j is not the owner and can own the point, we compute the delta score and combine it with the classifier probability.
                        const double deltaMove = computeScore(j, i) - ownerScore;
                        logWeights[j] = std::log(classifierProbabilities[i][j]) + (deltaMove / T);
                    }
                }
            }

            // Find the maximum log-weight to prevent overflow during exponentiation
            double maxLogWeight = -std::numeric_limits<double>::infinity();
            for(size_t j = 0; j < nClusters_; j++) {
                if(logWeights[j] > maxLogWeight) {
                    maxLogWeight = logWeights[j];
                }
            }

            // Exponentiate safely and normalize
            double sum = 0.0;
            for(size_t j = 0; j < nClusters_; j++) {
                if(logWeights[j] > -std::numeric_limits<double>::infinity()) {
                    // Subtracting maxLogWeight ensures the largest value exponentiated is 0 (exp(0) = 1)
                    switchingProbabilities[i][j] = std::exp(logWeights[j] - maxLogWeight);
                    sum += switchingProbabilities[i][j];
                } else {
                    switchingProbabilities[i][j] = 0.0;
                }
            }

            // Final normalization
            if(sum > 0.0) {
                for(size_t j = 0; j < nClusters_; j++) {
                    switchingProbabilities[i][j] /= sum;
                }
            } else {
                // Extremely rare edge case where all weights are -inf (e.g. invalid probability inputs, we fall back to classifier probabilities)
                for(size_t j = 0; j < nClusters_; j++) {
                    switchingProbabilities[i][j] = classifierProbabilities[i][j];
                }
            }
        }

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

        // Guard invalid indices.
        if(clusterIndex1 >= nClusters_ || clusterIndex2 >= nClusters_) {
            return;
        }

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
        // Use minSize=1 so the emptied source cluster is purged immediately.
        performSwitches(switches, 1);
    }

};

namespace covariance {
/**
 * @class ModelClusterCovariance
 * @brief Blended covariance kernel that interpolates local GP kernels using classifier probabilities.
 *
 * @details Mathematical Formulation
 * The covariance function between two inputs $x$ and $x'$ is computed as:
 * \f[
 * K(x, x') = \sum_{k=1}^K \sqrt{P(C=k \mid x) P(C=k \mid x')} k_k(x, x'; \theta_k)
 * \f]
 * where $P(C=k \mid x)$ represents the probability that input $x$ belongs to cluster $k$
 * (evaluated by the classifier), and $k_k$ is the kernel of the $k$-th cluster's GP.
 *
 * @details Implementation Algorithm
 * 1. **Cache Lookups (`getCachedProbabilities`)**: Generates a binary string key representing
 *    the coordinates of the input vector and queries the cache `probabilityCache_` to avoid redundant classifier evaluations.
 * 2. **Covariance Evaluation (`eval`)**: Fetches membership probabilities for both inputs,
 *    loops over all clusters to evaluate their respective covariance kernels, and sums the scaled products.
 */
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
/**
 * @class ModelClusterMean
 * @brief Blended mean function that interpolates local GP means using classifier probabilities.
 *
 * @details Mathematical Formulation
 * The blended mean function at input $x$ is evaluated as:
 * \f[
 * M(x) = \sum_{k=1}^K P(C=k | x) m_k(x; \theta_k)
 * \f]
 * where $P(C=k | x)$ is the classifier-derived probability that $x$ belongs to cluster $k$,
 * and $m_k$ is the mean function of the $k$-th cluster's GP.
 *
 * @details Implementation Algorithm
 * 1. Queries the classifier for the cluster membership probability vector at input $x$.
 * 2. Evaluates the local mean function for each cluster GP at $x$.
 * 3. Returns the dot product of the probability vector and the local GP mean vector.
 */
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
/** @} */

#endif