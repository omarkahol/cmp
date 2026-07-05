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

cmp::cluster::DirichletProcessMixtureModel::DirichletProcessMixtureModel(double alpha, const cmp::distribution::GammaDistribution &alphaPrior, const cmp::distribution::NormalInverseWishartDistribution& hyper, unsigned int seed) :
    alpha_(alpha),
    alphaPrior_(alphaPrior),
    hyper_(hyper),
    nextClusterId_(0),
    rng_(seed) {
}

void cmp::cluster::DirichletProcessMixtureModel::init(const Eigen::VectorXs& initialLabels) {
    clusters_.clear();
    size_t nextClusterID = 0;
    std::map<size_t, size_t> labelToClusterID;

    for(int i = 0; i < initialLabels.size(); ++i) {
        size_t label = initialLabels(i);

        if(!labelToClusterID.count(label)) {
            labelToClusterID[label] = nextClusterID;
            clusters_.emplace(nextClusterID, Cluster(nextClusterID, dim_));
            addPointToCluster(i, nextClusterID);
            nextClusterID++;
        } else {
            size_t clusterID = labelToClusterID[label];
            addPointToCluster(i, clusterID);
        }
    }
}

void cmp::cluster::DirichletProcessMixtureModel::removePointFromCluster(const size_t &pointIndex, const size_t &clusterID) {
    if(pointIndex >= nPoints_) throw std::out_of_range("Point index out of range");

    auto iterator = clusters_.find(clusterID);
    if(iterator == clusters_.end()) throw std::invalid_argument("Cluster ID not found");

    Cluster& c = iterator->second;
    if(c.nPoints_ == 0) throw std::invalid_argument("Cannot remove point from empty cluster");

    if(c.nPoints_ == 1) {
        clusters_.erase(iterator);
    } else {
        const Eigen::VectorXd& x = xObs_.row(pointIndex);
        c.nPoints_ -= 1;
        c.sumOfX_ -= x;
        // Using noalias() avoids a temporary matrix allocation
        c.sumOfXXT_.noalias() -= x * x.transpose();
    }
}

void cmp::cluster::DirichletProcessMixtureModel::addPointToCluster(const size_t &pointIndex, const size_t &clusterID) {
    if(pointIndex >= nPoints_) throw std::out_of_range("Point index out of range");

    auto iterator = clusters_.find(clusterID);
    if(iterator == clusters_.end()) {
        clusters_.emplace(clusterID, Cluster(clusterID, dim_));
        iterator = clusters_.find(clusterID);
    }

    Cluster& c = iterator->second;
    const Eigen::VectorXd& x = xObs_.row(pointIndex);

    c.nPoints_ += 1;
    c.sumOfX_ += x;
    // Using noalias() avoids a temporary matrix allocation
    c.sumOfXXT_.noalias() += x * x.transpose();
    labels_(pointIndex) = clusterID;
}

double cmp::cluster::DirichletProcessMixtureModel::logNIWPosterior(const Eigen::VectorXd& x, const Cluster& c) const {
    size_t nPoints = c.nPoints_;
    if(nPoints == 0) {
        return hyper_.logPDF(x);
    }

    double n = static_cast<double>(nPoints);
    Eigen::VectorXd clusterMean = c.sumOfX_ / n;

    // S is symmetric. We only compute the lower triangular part to save CPU cycles.
    Eigen::MatrixXd S = c.sumOfXXT_;
    S.selfadjointView<Eigen::Lower>().rankUpdate(clusterMean, -n);

    double kappa0 = hyper_.kappa();
    double df0 = hyper_.nu();
    Eigen::MatrixXd Sigma0 = hyper_.covariance();

    Eigen::MatrixXd Psi0 = Sigma0 * (kappa0 * df0) / (kappa0 + 1.0);
    double kappa_n = kappa0 + n;
    double df = df0 + n;

    Eigen::VectorXd diff = clusterMean - hyper_.mean();

    // Construct Lambda_n efficiently
    Eigen::MatrixXd Lambda_n = Psi0 + S;
    Lambda_n.selfadjointView<Eigen::Lower>().rankUpdate(diff, (kappa0 * n) / kappa_n);

    Eigen::VectorXd mu_n = (kappa0 * hyper_.mean() + n * clusterMean) / kappa_n;

    double scale = (kappa_n + 1.0) / (kappa_n * df);
    // Since Lambda_n was only built on the lower triangle, we extract the self-adjoint view
    Eigen::MatrixXd Sigma = scale * Lambda_n.selfadjointView<Eigen::Lower>();

    return cmp::distribution::MultivariateStudentDistribution::logPDF(x - mu_n, Sigma.ldlt(), df);
}

void cmp::cluster::DirichletProcessMixtureModel::condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::VectorXs& init_labels) {
    xObs_ = xObs;
    nPoints_ = xObs.rows();
    dim_ = xObs.cols();
    labels_ = init_labels;

    if(labels_.size() != nPoints_) throw std::invalid_argument("Initial labels size mismatch");
    for(int i = 0; i < labels_.size(); ++i) {
        if(labels_(i) < 0) throw std::invalid_argument("Labels must be non-negative");
    }

    init(labels_);
    remapLabels();

    // Pre-allocate workspaces to prevent inner-loop memory thrashing
    clusterIDs_workspace_.reserve(nPoints_);
    logp_workspace_.reserve(nPoints_ + 1);
    probs_workspace_.reserve(nPoints_ + 1);
}

void cmp::cluster::DirichletProcessMixtureModel::step() {
    std::vector<size_t> ordering(nPoints_);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::shuffle(ordering.begin(), ordering.end(), rng_);

    // Pre-allocate the vector once. Reassigning inside the loop is allocation-free.
    Eigen::VectorXd x(dim_);

    for(size_t pointIndex : ordering) {
        size_t clusterID = labels_(pointIndex);
        removePointFromCluster(pointIndex, clusterID);

        // Reset workspaces (clears size, retains capacity)
        clusterIDs_workspace_.clear();
        logp_workspace_.clear();
        probs_workspace_.clear();

        for(const auto& kv : clusters_) clusterIDs_workspace_.push_back(kv.first);
        int K = static_cast<int>(clusterIDs_workspace_.size());

        // Assign the row without allocating new memory
        x = xObs_.row(static_cast<int>(pointIndex)).transpose();

        // existing clusters
        for(size_t cid : clusterIDs_workspace_) {
            const Cluster& c = clusters_.at(cid);
            double log_count = std::log(static_cast<double>(c.nPoints_));
            logp_workspace_.push_back(log_count + logNIWPosterior(x, c));
        }

        // new cluster
        logp_workspace_.push_back(std::log(alpha_) + hyper_.logPDF(x));

        // normalize log probabilities (log-sum-exp)
        double maxlog = *std::max_element(logp_workspace_.begin(), logp_workspace_.end());
        double sum = 0.0;

        for(double lp : logp_workspace_) {
            double v = std::exp(lp - maxlog);
            probs_workspace_.push_back(v);
            sum += v;
        }
        for(double &v : probs_workspace_) v /= sum;

        // sample categorical
        double u = distU_(rng_);
        double cumsum = 0.0;
        int choice = K; // default to new cluster

        for(size_t i = 0; i < probs_workspace_.size(); ++i) {
            cumsum += probs_workspace_[i];
            if(u <= cumsum) {
                choice = static_cast<int>(i);
                break;
            }
        }

        if(choice < K) {
            addPointToCluster(pointIndex, clusterIDs_workspace_[choice]);
        } else {
            clusters_.emplace(nextClusterId_, Cluster(nextClusterId_, dim_));
            addPointToCluster(pointIndex, nextClusterId_);
            nextClusterId_++;
        }
    }

    // Update alpha using the auxiliary variable method
    updateAlpha();
}

void cmp::cluster::DirichletProcessMixtureModel::remapLabels() {
    std::map<size_t, size_t> oldToNewID;
    size_t newID = 0;

    for(const auto& kv : clusters_) {
        oldToNewID[kv.first] = newID++;
    }

    for(int i = 0; i < labels_.size(); ++i) {
        size_t oldID = labels_(i);
        if(oldToNewID.count(oldID)) {
            labels_(i) = oldToNewID[oldID];
        } else {
            throw std::runtime_error("Label refers to non-existent cluster");
        }
    }

    std::map<size_t, Cluster> newClusters;
    for(const auto& kv : clusters_) {
        size_t updatedID = oldToNewID[kv.first];
        Cluster c = kv.second;
        c.id_ = updatedID;
        newClusters[updatedID] = c;
    }

    clusters_ = std::move(newClusters);
    nextClusterId_ = clusters_.size();
}