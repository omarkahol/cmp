#include "cluster.h"

bool cmp::cluster::GeometricCluster::fit(const Eigen::Ref<const Eigen::MatrixXd> &points, size_t n_clusters, std::default_random_engine &rng, size_t max_iter) {

    // Set the number of clusters
    nClusters_ = n_clusters;
    nPoints_ = points.rows();
    dim_ = points.cols();

    // Initialize the centroids
    centroids_ = Eigen::MatrixXd::Zero(n_clusters, dim_);


    // Pick random points as the initial centroids WITHOUT REPLACEMENT
    std::vector<double> weights = std::vector<double>(nPoints_, 1.0);
    for(size_t i = 0; i < n_clusters; i++) {
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        size_t pick = dist(rng);
        centroids_.row(i) = points.row(pick);
        weights[pick] = 0;
    }

    // Initialize the location
    labels_ = Eigen::VectorXs::Zero(nPoints_);

    // Perform the iterations
    for(size_t iter = 0; iter < max_iter; iter++) {

        bool converged = true;

        // Assign each grid point to a cluster
        for(size_t i = 0; i < nPoints_; i++) {
            double min_dist = (points.row(i) - centroids_.row(labels_[i])).norm();
            for(size_t j = 0; j < n_clusters; j++) {

                // Skip this iteration
                if(j == labels_[i]) {
                    continue;
                }

                double dist = (points.row(i) - centroids_.row(j)).norm();
                if(dist < min_dist) {
                    min_dist = dist;
                    if(labels_[i] != j) {
                        labels_[i] = j;
                        converged = false;
                    }
                }
            }
        }

        // Recompute the centroids
        for(size_t j = 0; j < n_clusters; j++) {
            Eigen::VectorXd sum = Eigen::VectorXd::Zero(dim_);
            size_t count = 0;
            for(size_t i = 0; i < nPoints_; i++) {
                if(labels_[i] == j) {
                    sum += points.row(i);
                    count++;
                }
            }
            centroids_.row(j) = sum.transpose() / count;
        }

        if(converged) {
            return true;
        }
    }
    return false;
}

const size_t &cmp::cluster::GeometricCluster::operator[](size_t i) const {
    return labels_[i];
}

Eigen::VectorXd cmp::cluster::GeometricCluster::centroid(size_t i) const {
    return centroids_.row(i);
}

size_t cmp::cluster::GeometricCluster::nClusters() const {
    return nClusters_;
}

size_t cmp::cluster::GeometricCluster::nPoints() const {
    return nPoints_;
}

size_t cmp::cluster::GeometricCluster::dim() const {
    return dim_;
}

/**
 * DirichletProcessMixtureModel implementation
 */


cmp::cluster::DirichletProcessMixtureModel::DirichletProcessMixtureModel(double alpha, const cmp::distribution::NormalInverseWishartDistribution& hyper, unsigned int seed) :
    alpha_(alpha),
    hyper_(hyper),
    nextClusterId_(0),
    rng_(seed) {
    if(alpha_ <= 0) {
        throw std::invalid_argument("Concentration parameter alpha must be positive");
    }
}

void cmp::cluster::DirichletProcessMixtureModel::init(const Eigen::VectorXs& initialLabels) {

    // Clear any existing clusters
    clusters_.clear();
    size_t nextClusterID = 0;

    // Map from label to cluster id
    std::map<size_t, size_t> labelToClusterID;

    // Iterate over initial labels and create clusters
    for(int i = 0; i < initialLabels.size(); ++i) {
        size_t label = initialLabels(i);

        // If this label is new, create a new cluster
        if(!labelToClusterID.count(label)) {

            labelToClusterID[label] = nextClusterID;
            clusters_.emplace(nextClusterID, Cluster(nextClusterID, dim_));

            // Add this point to the new cluster
            addPointToCluster(i, nextClusterID);

            nextClusterID++;
        } else {

            // Existing label, get the corresponding cluster id
            size_t clusterID = labelToClusterID[label];
            addPointToCluster(i, clusterID);
        }
    }
}

void cmp::cluster::DirichletProcessMixtureModel::removePointFromCluster(const size_t &pointIndex, const size_t &clusterID) {

    // Validate point index
    if(pointIndex >= nPoints_) {
        throw std::out_of_range("Point index out of range");
    }

    // Find the cluster and ensure it exists
    auto iterator = clusters_.find(clusterID);
    if(iterator == clusters_.end()) {
        throw std::invalid_argument("Cluster ID not found");
    }

    // Sanity check
    Cluster& c = iterator->second;
    if(c.nPoints_ == 0) {
        throw std::invalid_argument("Cannot remove point from empty cluster");
    }

    // If this was the last point in the cluster, remove the cluster completely
    if(c.nPoints_ == 1) {
        clusters_.erase(iterator);
    } else {

        // Update the cluster statistics
        const Eigen::VectorXd& x = xObs_.row(pointIndex);
        c.nPoints_ -= 1;
        c.sumOfX_ -= x;
        c.sumOfXXT_ -= x * x.transpose();
    }
}

void cmp::cluster::DirichletProcessMixtureModel::addPointToCluster(const size_t &pointIndex, const size_t &clusterID) {

    // Validate point index
    if(pointIndex >= nPoints_) {
        throw std::out_of_range("Point index out of range");
    }

    // Find the cluster, or create it if it doesn't exist
    auto iterator = clusters_.find(clusterID);

    // If not found, create a new cluster
    if(iterator == clusters_.end()) {
        clusters_.emplace(clusterID, Cluster(clusterID, dim_));
        iterator = clusters_.find(clusterID);
    }

    Cluster& c = iterator->second;

    // Update the cluster statistics
    const Eigen::VectorXd& x = xObs_.row(pointIndex);
    c.nPoints_ += 1;
    c.sumOfX_ += x;
    c.sumOfXXT_ += x * x.transpose();
    labels_(pointIndex) = clusterID;
}


double cmp::cluster::DirichletProcessMixtureModel::logNIWPosterior(const Eigen::VectorXd& x, const Cluster& c) const {

    // If the cluster is empty, return the prior predictive
    size_t nPoints = c.nPoints_;
    if(nPoints == 0) {
        return hyper_.logPDF(x);
    }

    // Posterior parameters for Normal-Inverse-Wishart
    double n = static_cast<double>(nPoints);
    Eigen::VectorXd clusterMean = c.sumOfX_ / n;

    // Compute S = sum of squared deviations from sample mean
    // S = Σ(x_i - xbar)(x_i - xbar)^T = sumOfXXT - n * xbar * xbar^T
    Eigen::MatrixXd S = c.sumOfXXT_ - n * (clusterMean * clusterMean.transpose());

    // The NormalInverseWishartDistribution stores the PRIOR PREDICTIVE Student-t
    // parameters: df' = nu - d + 1 and Sigma0 = ((kappa+1)/(kappa*df')) * Psi.
    // Reconstruct the IW scale Psi from the stored predictive covariance Sigma0.
    double kappa0 = hyper_.kappa();
    double df0 = hyper_.nu(); // this is already nu' = nu - d + 1
    Eigen::MatrixXd Sigma0 = hyper_.covariance(); // predictive covariance, not Psi

    // Recover prior IW scale matrix Psi0: Sigma0 = ((kappa0+1)/(kappa0*df0)) * Psi0
    Eigen::MatrixXd Psi0 = Sigma0 * (kappa0 * df0) / (kappa0 + 1.0);

    double kappa_n = kappa0 + n;
    double df = df0 + n; // posterior predictive dof

    // NIW posterior scale matrix update (Lambda_n = Psi_n)
    // Psi_n = Psi0 + S + (kappa0 * n)/(kappa0 + n) * (xbar - mu0)(xbar - mu0)^T
    Eigen::VectorXd diff = clusterMean - hyper_.mean();
    Eigen::MatrixXd Lambda_n = Psi0 + S + ((kappa0 * n) / kappa_n) * (diff * diff.transpose());

    // Posterior mean for the NIW predictive distribution:
    // mu_n = (kappa0 * mu0 + n * xbar) / (kappa0 + n)
    Eigen::VectorXd mu_n = (kappa0 * hyper_.mean() + n * clusterMean) / kappa_n;

    // Posterior predictive covariance: Sigma_n = ((kappa_n + 1)/(kappa_n * df)) * Psi_n
    double scale = (kappa_n + 1.0) / (kappa_n * df);
    Eigen::MatrixXd Sigma = scale * Lambda_n;

    // Return log PDF of multivariate Student-t at x with location mu_n
    return cmp::distribution::MultivariateStudentDistribution::logPDF(x - mu_n, Sigma.ldlt(), df);
}

void cmp::cluster::DirichletProcessMixtureModel::condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::VectorXs& init_labels) {

    // Store the data
    xObs_ = xObs;
    nPoints_ = xObs.rows();
    dim_ = xObs.cols();

    // Initialize labels and clusters
    labels_ = init_labels;
    if(labels_.size() != nPoints_) {
        throw std::invalid_argument("Initial labels size does not match number of data points");
    }

    // Validate labels are within reasonable bounds
    for(int i = 0; i < labels_.size(); ++i) {
        if(labels_(i) < 0) {
            throw std::invalid_argument("Labels must be non-negative");
        }
    }

    init(labels_);

    // Remap the labels to be contiguous starting from 0
    remapLabels();

    // Holder for cluster ids
    clusterIDs_.reserve(clusters_.size());

}

void cmp::cluster::DirichletProcessMixtureModel::step() {

    // Iterate over all points in random order
    std::vector<size_t> ordering(nPoints_);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::shuffle(ordering.begin(), ordering.end(), rng_);

    for(size_t pointIndex : ordering) {

        size_t clusterID = labels_(pointIndex);

        // Remove point from current cluster
        removePointFromCluster(pointIndex, clusterID);

        // Gather candidate clusters (existing)
        clusterIDs_.clear();
        for(const auto& kv : clusters_) clusterIDs_.push_back(kv.first);

        int K = static_cast<int>(clusterIDs_.size());
        std::vector<double> logp;
        logp.reserve(K + 1);

        Eigen::VectorXd x = xObs_.row(static_cast<int>(pointIndex)).transpose();

        // existing clusters
        for(size_t cid : clusterIDs_) {
            const Cluster& c = clusters_.at(cid);
            double log_count = std::log(static_cast<double>(c.nPoints_));
            double lp = log_count + logNIWPosterior(x, c);
            logp.push_back(lp);
        }
        // new cluster
        double lp_new = std::log(alpha_) + hyper_.logPDF(x);
        logp.push_back(lp_new);

        // normalize log probabilities (log-sum-exp), sample
        double maxlog = *std::max_element(logp.begin(), logp.end());
        double sum = 0.0;
        std::vector<double> probs(logp.size());
        for(size_t i = 0; i < logp.size(); ++i) {
            double v = std::exp(logp[i] - maxlog);
            probs[i] = v;
            sum += v;
        }
        for(double &v : probs) v /= sum;

        // sample categorical
        double u = distU_(rng_);
        double cumsum = 0.0;
        int choice = static_cast<int>(probs.size()) - 1; // default new
        for(size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if(u <= cumsum) {
                choice = static_cast<int>(i);
                break;
            }
        }

        if(choice < K) {
            // assign to existing cluster clusterIDs[choice]
            addPointToCluster(pointIndex, clusterIDs_[choice]);
        } else {
            // Create a new cluster and add the point to it
            clusters_.emplace(nextClusterId_, Cluster(nextClusterId_, dim_));
            addPointToCluster(pointIndex, nextClusterId_);
            nextClusterId_++;
        }
    }
}

Eigen::VectorXs cmp::cluster::DirichletProcessMixtureModel::mostProbableLabels() {

    // Iterate over all points in random order
    std::vector<size_t> ordering(nPoints_);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::shuffle(ordering.begin(), ordering.end(), rng_);

    for(size_t pointIndex : ordering) {

        size_t clusterID = labels_(pointIndex);

        // Remove point from current cluster
        removePointFromCluster(pointIndex, clusterID);

        // Gather candidate clusters (existing)
        clusterIDs_.clear();
        for(const auto& kv : clusters_) clusterIDs_.push_back(kv.first);

        int K = static_cast<int>(clusterIDs_.size());
        std::vector<double> logp;
        logp.reserve(K + 1);

        Eigen::VectorXd x = xObs_.row(static_cast<int>(pointIndex)).transpose();

        // existing clusters
        for(size_t cid : clusterIDs_) {
            const Cluster& c = clusters_.at(cid);
            double log_count = std::log(static_cast<double>(c.nPoints_));
            double lp = log_count + logNIWPosterior(x, c);
            logp.push_back(lp);
        }
        // new cluster
        double lp_new = std::log(alpha_) + hyper_.logPDF(x);
        logp.push_back(lp_new);

        // normalize log probabilities (log-sum-exp), sample
        double maxlog = *std::max_element(logp.begin(), logp.end());
        double sum = 0.0;
        std::vector<double> probs(logp.size());
        for(size_t i = 0; i < logp.size(); ++i) {
            double v = std::exp(logp[i] - maxlog);
            probs[i] = v;
            sum += v;
        }
        for(double &v : probs) v /= sum;

        // Pick the most probable
        int choice = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

        if(choice < K) {
            // assign to existing cluster clusterIDs[choice]
            addPointToCluster(pointIndex, clusterIDs_[choice]);
        } else {
            // Create a new cluster and add the point to it
            clusters_.emplace(nextClusterId_, Cluster(nextClusterId_, dim_));
            addPointToCluster(pointIndex, nextClusterId_);
            nextClusterId_++;
        }
    }

    remapLabels();

    return labels_;

}

void cmp::cluster::DirichletProcessMixtureModel::remapLabels() {

    // Map from old cluster ID to new contiguous ID
    std::map<size_t, size_t> oldToNewID;
    size_t newID = 0;

    for(const auto& kv : clusters_) {
        size_t oldID = kv.first;
        oldToNewID[oldID] = newID;
        newID++;
    }

    // Update labels
    for(int i = 0; i < labels_.size(); ++i) {
        size_t oldID = labels_(i);
        if(oldToNewID.count(oldID)) {
            labels_(i) = oldToNewID[oldID];
        } else {
            throw std::runtime_error("Label refers to non-existent cluster");
        }
    }

    // Rebuild clusters_ map with new IDs
    std::map<size_t, Cluster> newClusters;
    for(const auto& kv : clusters_) {
        size_t oldID = kv.first;
        size_t updatedID = oldToNewID[oldID];
        Cluster c = kv.second;
        c.id_ = updatedID;
        newClusters[updatedID] = c;
    }
    clusters_ = std::move(newClusters);
    nextClusterId_ = clusters_.size();
}