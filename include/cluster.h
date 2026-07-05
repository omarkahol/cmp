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
 * @class DirichletProcessMixtureModel
 * @brief Implements an infinite Gaussian Mixture Model using a Dirichlet Process Mixture Model (DPMM) and Gibbs sampling.
 *
 * @details
 * ### Mathematical Foundations
 * The Dirichlet Process Mixture Model (DPMM) assumes an infinite number of mixture components, allowing the model
 * to adaptively determine the number of clusters from the data. The generative process for observations \f$\mathbf{x}_i \in \mathbb{R}^d\f$ is:
 * \f[ \mathbf{x}_i | z_i, \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\} \sim \mathcal{N}(\boldsymbol{\mu}_{z_i}, \boldsymbol{\Sigma}_{z_i}) \f]
 * \f[ z_i | \mathbf{w} \sim \mathrm{Categorical}(\mathbf{w}) \f]
 * \f[ \mathbf{w} \sim \mathrm{GEM}(\alpha) \quad \text{(Stick-breaking construction)} \f]
 * \f[ (\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \sim \mathrm{NIW}(\boldsymbol{\mu}_0, \kappa_0, \boldsymbol{\Lambda}_0, \nu_0) \f]
 * where \f$\mathrm{NIW}\f$ is the Normal-Inverse-Wishart conjugate prior, and \f$\alpha > 0\f$ is the concentration parameter.
 *
 * The NIW prior is defined as:
 * \f[ \boldsymbol{\Sigma} \sim \mathrm{Inv-Wishart}(\boldsymbol{\Lambda}_0, \nu_0), \quad \boldsymbol{\mu} | \boldsymbol{\Sigma} \sim \mathcal{N}\left(\boldsymbol{\mu}_0, \frac{1}{\kappa_0} \boldsymbol{\Sigma}\right) \f]
 *
 * ### Implementation Algorithms (Collapsed Gibbs Sampling)
 * The algorithm integrates out the parameters \f$\{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}\f$, sampling only the cluster indicators \f$z_i\f$:
 * 1. **Existing Cluster Probability**: The probability that data point \f$\mathbf{x}_i\f$ belongs to an existing cluster \f$k\f$ is:
 *    \f[ P(z_i = k | \mathbf{z}_{-i}, \mathbf{x}_i) \propto n_k^{-i} \cdot t_{\nu_n}\left(\mathbf{x}_i ; \boldsymbol{\mu}_n, \frac{\kappa_n + 1}{\kappa_n(\nu_n - d + 1)} \boldsymbol{\Lambda}_n\right) \f]
 *    where \f$n_k^{-i}\f$ is the cluster size excluding \f$\mathbf{x}_i\f$, and \f$t_{\nu}\f$ is the multivariate Student-t distribution representing the posterior predictive distribution.
 * 2. **New Cluster Probability**: The probability that \f$\mathbf{x}_i\f$ spawns a new cluster is:
 *    \f[ P(z_i = \text{new} | \mathbf{z}_{-i}, \mathbf{x}_i) \propto \alpha \cdot t_{\nu_0}\left(\mathbf{x}_i ; \boldsymbol{\mu}_0, \frac{\kappa_0 + 1}{\kappa_0(\nu_0 - d + 1)} \boldsymbol{\Lambda}_0\right) \f]
 * 3. **NIW Posterior Updates**: For each cluster \f$k\f$, the sufficient statistics are updated as:
 *    \f[ \kappa_n = \kappa_0 + n_k, \quad \nu_n = \nu_0 + n_k, \quad \boldsymbol{\mu}_n = \frac{\kappa_0 \boldsymbol{\mu}_0 + n_k \bar{\mathbf{x}}}{\kappa_n} \f]
 *    \f[ \boldsymbol{\Lambda}_n = \boldsymbol{\Lambda}_0 + \mathbf{C} + \frac{\kappa_0 n_k}{\kappa_n} (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^T \f]
 *    where \f$\bar{\mathbf{x}}\f$ is the sample mean, and \f$\mathbf{C} = \sum_{j \in \text{cluster}} (\mathbf{x}_j - \bar{\mathbf{x}})(\mathbf{x}_j - \bar{\mathbf{x}})^T\f$.
 *
 * ### Constraints & Invariants
 * - **Degrees of Freedom**: Prior degrees of freedom must satisfy \f$\nu_0 > d - 1\f$ to guarantee prior scale matrix normalization.
 * - **Positive Definiteness**: The prior scale matrix \f$\boldsymbol{\Lambda}_0\f$ must be symmetric positive-definite.
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

    // Workspaces for the Gibbs sampler to avoid heap allocations
    std::vector<double> logp_workspace_;
    std::vector<double> probs_workspace_;
    std::vector<size_t> clusterIDs_workspace_;

    Eigen::MatrixXd xObs_;                  // Observed data points (N x D)
    Eigen::VectorXs labels_;                // Current labels
    size_t nPoints_{0};                     // Number of data points
    int dim_{0};                            // Dimensionality of the feature space
    std::vector<size_t>  clusterIDs_;       // Current cluster IDs

    // This prior is conjugate to the Gaussian likelihood
    cmp::distribution::NormalInverseWishartDistribution hyper_;
    cmp::distribution::GammaDistribution alphaPrior_;
    double alpha_{1.0}; // Concentration parameter for the Dirichlet Process

    std::map<size_t, Cluster> clusters_; // Map from cluster id to Cluster struct
    size_t nextClusterId_;               // Used to assign unique cluster IDs

    // Random number generator for sampling
    std::default_random_engine rng_;
    std::uniform_real_distribution<double> distU_{0.0, 1.0};

  public:

// public API
    DirichletProcessMixtureModel(
        double alpha,
        const cmp::distribution::GammaDistribution &alphaPrior,
        const cmp::distribution::NormalInverseWishartDistribution &hyper,
        unsigned int seed = 12345
    );

    void condition(const Eigen::Ref<const Eigen::MatrixXd> &data, const Eigen::VectorXs& init_labels);

    void step();

    void remapLabels();

// get current assignment vector (size N)
    Eigen::VectorXs getLabels() const {
        return labels_;
    }

// get current number of clusters
    size_t nClusters() const {
        return clusters_.size();
    }

    double getAlpha() const {
        return alpha_;
    }

  private:

// helpers
    void init(const Eigen::VectorXs& init_labels);
    void removePointFromCluster(const size_t &pointIndex, const size_t &clusterID);
    void addPointToCluster(const size_t &pointIndex, const size_t &clusterID);
    double logNIWPosterior(const Eigen::VectorXd& x, const Cluster& c) const;

    void updateAlpha() {
        double K = static_cast<double>(clusters_.size());
        double N = static_cast<double>(nPoints_);

        // Extract prior parameters from your distribution object
        double a = alphaPrior_.getAlpha();
        double b = alphaPrior_.getBeta();

        // 1. Instantiate the Beta distribution and sample eta
        cmp::distribution::BetaDistribution beta_dist(alpha_ + 1.0, N);
        double eta = beta_dist.sample(rng_);

        // 2. Calculate the weights for the Gamma mixture
        double weight1 = (a + K - 1.0) / (N * (b - std::log(eta)));
        double pi_eta = weight1 / (weight1 + 1.0);

        // 3. Sample the new alpha using your custom cmp::distribution classes
        double u = distU_(rng_);
        double updated_beta = b - std::log(eta); // The rate parameter

        if(u < pi_eta) {
            // Gamma(a + K, b - ln(eta))
            cmp::distribution::GammaDistribution new_gamma(a + K, updated_beta);
            alpha_ = new_gamma.sample(rng_);
        } else {
            // Gamma(a + K - 1, b - ln(eta))
            cmp::distribution::GammaDistribution new_gamma(a + K - 1.0, updated_beta);
            alpha_ = new_gamma.sample(rng_);
        }

        // Safety bounds just in case of severe numerical underflow
        if(alpha_ < 1e-5) alpha_ = 1e-5;
    }

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