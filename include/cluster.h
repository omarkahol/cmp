#ifndef CLUSTER_H
#define CLUSTER_H

#include <cmp_defines.h>
#include <distribution.h>
#include <grid.h>

namespace cmp::cluster {

/**
 * @brief Implements a k-means clustering algorithm.
 */
class GeometricCluster {
  private:
    Eigen::VectorXs labels_;
    Eigen::MatrixXd centroids_;

    size_t nClusters_;
    size_t nPoints_;
    size_t dim_;

  public:
    GeometricCluster() = default;

    bool fit(const Eigen::Ref<const Eigen::MatrixXd> &points, size_t nClusters, std::default_random_engine &rng, size_t max_iter = 1000);
    const size_t &operator[](size_t i) const;
    Eigen::VectorXd centroid(size_t i) const;

    const Eigen::VectorXs &getLabels() const {
        return labels_;
    }

    size_t nClusters() const;
    size_t nPoints() const;
    size_t dim() const;
};

/**
 * @brief Implements a Dirichlet Process Mixture Model using Gibbs sampling for clustering.
 * The model uses a Normal-Inverse-Wishart prior for the Gaussian components.
 */
class DirichletProcessMixtureModel {

  private:
    struct Cluster {
        size_t id_;                 // Cluster id (not necessarily contiguous)
        size_t nPoints_;            // Number of points in cluster

        // Sufficient statistics
        Eigen::VectorXd sumOfX_;       // Sum of all points in the cluster
        Eigen::MatrixXd sumOfXXT_;     // Sum of all outer products x * x^T for points in the cluster

        // Default constructor
        Cluster() : id_(0), nPoints_(0) {}

        // Constructor with id and dimension
        Cluster(size_t id, size_t dim) : id_(id), nPoints_(0),
            sumOfX_(Eigen::VectorXd::Zero(dim)),
            sumOfXXT_(Eigen::MatrixXd::Zero(dim, dim)) {}
    };

  private:
    Eigen::MatrixXd xObs_;                  // Observed data points (N x D)
    Eigen::VectorXs labels_;                // Current labels
    size_t nPoints_{0};                     // Number of data points
    int dim_{0};                            // Dimensionality of the feature space
    double alpha_{1.0};                     // Concentration parameter for the Dirichlet Process
    std::vector<size_t>  clusterIDs_;       // Current cluster IDs

    // This prior is conjugate to the Gaussian likelihood
    const cmp::distribution::NormalInverseWishartDistribution &hyper_;

    std::map<size_t, Cluster> clusters_; // Map from cluster id to Cluster struct
    size_t nextClusterId_;               // Used to assign unique cluster IDs

    // Random number generator for sampling
    std::default_random_engine rng_;
    std::uniform_real_distribution<double> distU_{0.0, 1.0};

  public:

// public API
    DirichletProcessMixtureModel(
        double alpha,
        const cmp::distribution::NormalInverseWishartDistribution &hyper,
        unsigned int seed = 12345
    );

    void condition(const Eigen::Ref<const Eigen::MatrixXd> &data, const Eigen::VectorXs& init_labels);

    void step();

    Eigen::VectorXs mostProbableLabels();

    void remapLabels();

// get current assignment vector (size N)
    Eigen::VectorXs getLabels() const {
        return labels_;
    }

// get current number of clusters
    size_t nClusters() const {
        return clusters_.size();
    }

  private:

// helpers
    void init(const Eigen::VectorXs& init_labels);
    void removePointFromCluster(const size_t &pointIndex, const size_t &clusterID);
    void addPointToCluster(const size_t &pointIndex, const size_t &clusterID);
    double logNIWPosterior(const Eigen::VectorXd& x, const Cluster& c) const;

};


class DummyCluster {
  private:
    size_t nClusters_;
    std::default_random_engine rng_;
    Eigen::VectorXs labels_;
  public:
    DummyCluster(size_t nClusters, unsigned int seed = 42)
        : nClusters_(nClusters), rng_(seed) {}

    void fit(const Eigen::Ref<const Eigen::MatrixXd> &points) {
        size_t nPoints = points.rows();
        std::uniform_int_distribution<size_t> dist(0, nClusters_ - 1);
        labels_ = Eigen::VectorXs::Zero(nPoints);
        for(size_t i = 0; i < nPoints; i++) {
            labels_(i) = dist(rng_);
        }
    }

    const Eigen::VectorXs &getLabels() const {
        return labels_;
    }
};

} // namespace cmp::cluster


#endif