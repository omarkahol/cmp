#ifndef MODEL_CLUSTER_H
#define MODEL_CLUSTER_H

#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <distribution.h>
#include <cmp_defines.h>
#include <grid.h>
#include <cluster.h>
#include <classifier.h>
#include <poly.h>
#include <svm.h>
#include <set>


namespace cmp {
class ModelClusterPoly {

  private:
    std::default_random_engine rng_;

    Eigen::MatrixXd xObs_;
    Eigen::VectorXd yObs_;

    size_t nClusters_;
    size_t nObs_;
    size_t dimX_;

    // The labels of the points
    Eigen::VectorXs labels_;

    // The GPs for each cluster, along with their centroids and fit status
    std::vector<bool> fit_;
    std::vector<Eigen::VectorXd> centroids_;
    std::vector<cmp::PolynomialExpansion> *polynomials_;

    // The cluster sizes
    std::vector<size_t> clusterSize_;

    // The Gamma parameter
    double gamma_{1.0};

  public:

    ModelClusterPoly() = default;
    ~ModelClusterPoly() = default;

    void set(std::vector<cmp::PolynomialExpansion> *polynomials, const double &gamma = 0, const double &seed = 42) {
        polynomials_ = polynomials;
        nClusters_ = polynomials->size();
        fit_ = std::vector<bool>(nClusters_, false);
        clusterSize_ = std::vector<size_t>(nClusters_, 0);
        gamma_ = gamma;
        rng_.seed(seed);
    }

    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXd> &yObs, const Eigen::Ref<const Eigen::VectorXs> &labels) {

        // Get only the unique labels
        std::set<int> uniqueLabels;
        for(int i = 0; i < labels.size(); i++) {
            uniqueLabels.insert(labels(i));
        }
        if(uniqueLabels.size() != nClusters_) {
            throw std::runtime_error("The number of unique labels in the labels vector does not match the number of clusters.");
        }
        if(xObs.rows() != yObs.size() || xObs.rows() != labels.size()) {
            throw std::runtime_error("The number of observations in xObs, yObs and labels must be the same.");
        }

        // Remap the labels to be in the range [0, nClusters_-1]
        if(*uniqueLabels.begin() != 0 || *uniqueLabels.rbegin() != nClusters_ - 1) {
            std::cout << "Remapping labels to be in the range [0, " << nClusters_ - 1 << "]" << std::endl;
            Eigen::VectorXs remappedLabels = Eigen::VectorXs::Zero(labels.size());
            std::map<int, int> labelMap;
            int newLabel = 0;
            for(const auto &label : uniqueLabels) {
                labelMap[label] = newLabel++;
            }
            for(int i = 0; i < labels.size(); i++) {
                remappedLabels(i) = labelMap[labels(i)];
            }
            condition(xObs, yObs, remappedLabels);
            return;
        }


        // Initialize the members
        nObs_ = xObs.rows();
        dimX_ = xObs.cols();
        xObs_ = xObs;
        yObs_ = yObs;

        // Initialize the containers
        labels_ = labels;
        for(size_t i = 0; i < nClusters_; i++) {
            centroids_.emplace_back(Eigen::VectorXd::Zero(dimX_));
        }

        // Call the update model function
        updateModel(std::vector<bool>(nClusters_, true));
    }

    size_t nClusters() const {
        return nClusters_;
    }

    void fit() {
        #pragma omp parallel for
        for(size_t i = 0; i < nClusters(); i++) {
            if(!fit_[i]) {
                auto index = getIndices(i);
                auto xObsi = xObs_(index, Eigen::all);
                auto yObsi = yObs_(index);

                (*polynomials_)[i].fit(xObsi, yObsi);
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

    cmp::PolynomialExpansion &operator[](size_t i) {
        return (*polynomials_)[i];
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

            // Count the number of observations in the cluster
            size_t nObsCluster = 0;
            for(size_t j = 0; j < nObs_; j++) {
                if(labels_[j] >= 0 && static_cast<size_t>(labels_[j]) == i) {
                    nObsCluster++;
                }
            }

            // Centroids
            centroids_[i] = Eigen::VectorXd::Zero(dimX_);

            // Counter for the cluster size
            size_t counter = 0;

            // Set the observations
            for(size_t j = 0; j < nObs_; j++) {
                if(labels_[j] >= 0 && static_cast<size_t>(labels_[j]) == i) {

                    // Update the centroid
                    centroids_[i] += xObs_.row(j);

                    // Set the membership
                    labels_[j] = i;

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
    void performSwitches(const std::vector<std::pair<size_t, size_t>> &newOwners) {

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
    }



    bool isFit(const size_t &clusterIndex) const {
        return fit_[clusterIndex];
    }

    void setFit(const size_t &clusterIndex, const bool &fit) {
        fit_[clusterIndex] = fit;
    }

    /**
     * This function computes the error for the point with global index \par globalIndex
     */
    double computeScore(size_t model, size_t globalIndex) const {

        // Make a prediction
        auto [mean, var] = (*polynomials_)[model].predict(xObs_.row(globalIndex));

        // Return the score -(MSE + geometrical regularization)
        return -((yObs_(globalIndex) - mean) * (yObs_(globalIndex) - mean) + gamma_ * (xObs_.row(globalIndex).transpose() - centroids_[model]).squaredNorm());
    }

    std::vector<std::pair<size_t, size_t>> switchStep(cmp::classifier::Classifier *classifier, const double &T = 1.0, const size_t &maxAllowedSwitches = 10, const double &minProb = 0.1) {

        // Compute the probabilities for each point to switch to each cluster
        std::vector<std::vector<double>> probabilities = computeProbabilities(T, classifier, minProb);

        // Compute the probabilities of switching for each point
        std::vector<double> switchingProbabilities(nObs_, 0.0);
        for(size_t i = 0; i < nObs_; i++) {
            // Find the owner of the point
            size_t owner = labels_[i];

            // Compute the switching probabilities
            for(size_t j = 0; j < nClusters_; j++) {
                if(j != owner) {
                    switchingProbabilities[i] += probabilities[i][j];
                }
            }
        }

        // Sample maxAllowedSwitches from the points
        std::vector<size_t> candidateSwitches(maxAllowedSwitches, 0);
        for(size_t i = 0; i < maxAllowedSwitches; i++) {

            // Select a random point according to the switching probabilities
            std::discrete_distribution<size_t> dist(switchingProbabilities.begin(), switchingProbabilities.end());
            candidateSwitches[i] = dist(rng_);

            // Do not allow the same point to be selected multiple times
            switchingProbabilities[candidateSwitches[i]] = 0.0;
        }

        // Prepare a vector to hold the selected switches
        std::vector<std::pair<size_t, size_t>> switches;

        // Iterate through the candidate switches
        for(size_t i = 0; i < candidateSwitches.size(); i++) {
            size_t globalIndex = candidateSwitches[i];

            // Find the owner of the point
            size_t owner = labels_[globalIndex];

            // Prepare a discrete distribution for the switching probabilities
            std::discrete_distribution<size_t> dist(probabilities[globalIndex].begin(), probabilities[globalIndex].end());

            // Sample a new owner
            size_t newOwner = dist(rng_);

            // If the new owner is different from the current one add the switch
            if(newOwner != owner) {
                switches.push_back(std::make_pair(globalIndex, newOwner));
            }
        }

        // Return the switches
        return switches;
    }

    /**
     * This function purges a cluster
     *
     * @param clusterIndex The index of the cluster to purge
     *
     */
    void purgeStep(const size_t &clusterIndex) {

        std::vector<bool> affectedClusters(nClusters_, true);

        std::vector<size_t> globalIndicesOfPointsToRedistribute;

        // Cycle through the observations
        for(size_t i = 0; i < nObs_; i++) {

            // Check if the point is in the cluster to purge
            if(labels_[i] == clusterIndex) {

                // Add the point to the list of points to redistribute
                globalIndicesOfPointsToRedistribute.push_back(i);

                // We need to redistribute the point
                std::pair<size_t, double> chosenCluster = std::make_pair(-1, std::numeric_limits<double>::infinity());

                for(size_t j = 0; j < nClusters_; j++) {

                    // Skip the current cluster
                    if(j == clusterIndex) {
                        continue;
                    }

                    // Compute the error
                    double error = computeScore(j, i);

                    // Check if we have a new minimum
                    if(error < chosenCluster.second) {
                        chosenCluster = std::make_pair(j, error);
                    }
                }

                // Perform the switch
                labels_[i] = chosenCluster.first;
                affectedClusters[chosenCluster.first] = true;
            }
        }

        // Purge the cluster
        polynomials_->erase(polynomials_->begin() + clusterIndex);
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

    double compactnessScore(const size_t & clusterIndex) {
        double avgCompactness = 0.0;

        for(size_t i = 0; i < nObs_; i++) {
            if(labels_[i] == clusterIndex) {
                avgCompactness += (xObs_.row(i).transpose() - centroids_[clusterIndex]).norm();
            }
        }
        if(clusterSize_[clusterIndex] > 0) {
            avgCompactness /= double(clusterSize_[clusterIndex]);
        } else {
            avgCompactness = std::numeric_limits<double>::infinity();
        }
        return avgCompactness;
    }

    std::vector<std::vector<double>> computeProbabilities(const double &T, cmp::classifier::Classifier *classifier, const double &minProb = 0.1) const {

        std::vector<std::vector<double>> switchingProbabilities(nObs_, std::vector<double>(nClusters_, 0.0));

        // Iterate through the points
        for(size_t i = 0; i < nObs_; i++) {

            // Find the owner of the point
            size_t owner = labels_[i];

            // Predict the SVM probabilities for the point
            std::vector<double> svmProbabilities = classifier->predictProbabilities(xObs_.row(i));

            // Compute the score for the current point
            double looScore = computeScore(owner, i);
            switchingProbabilities[i][owner] = svmProbabilities[owner];

            for(size_t j = 0; j < nClusters_; j++) {
                if(j == owner) {
                    continue;
                } else {

                    // Compute the predictive error
                    double predictiveScore = computeScore(j, i);

                    if(svmProbabilities[j] > minProb) {
                        switchingProbabilities[i][j] = std::exp((predictiveScore - looScore) / T) * svmProbabilities[j];
                    } else {
                        switchingProbabilities[i][j] = 0;
                    }

                }
            }

            // Compute the sum of the probabilities
            double sum = 0.0;
            for(size_t j = 0; j < nClusters_; j++) {
                sum += switchingProbabilities[i][j];
            }

            // Normalize again
            for(size_t j = 0; j < nClusters_; j++) {
                switchingProbabilities[i][j] /= sum;
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
    void mergeClusters(const size_t &clusterIndex1, const size_t &clusterIndex2, bool hparGuess = false) {

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
        performSwitches(switches);

        // Purge the cluster
        purgeStep(clusterIndex2);
    }

};

} // namespace cmp

#endif