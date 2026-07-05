#ifndef CLUSTER_H
#define CLUSTER_H

#include <cmp_defines.h>
#include <distribution.h>
#include <grid.h>

/**
 * @addtogroup clustering
 * @{
 */
namespace cmp::cluster {

/**
 * @brief Implements a standard k-means clustering algorithm.
 * 
 * @details Mathematical Formulation
 * K-means partitions \f$N\f$ observations \f$\{\mathbf{x}_1, \dots, \mathbf{x}_N\}\f$ where \f$\mathbf{x}_j \in \mathbb{R}^d\f$ into \f$K\f$ clusters \f$\mathbf{S} = \{S_1, \dots, S_K\}\f$ to minimize the within-cluster sum of squares (WCSS):
 * \f[
 * \arg\min_{\mathbf{S}} \sum_{i=1}^K \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|_2^2
 * \f]
 * where \f$\boldsymbol{\mu}_i\f$ is the centroid of the points in \f$S_i\f$.
 * 
 * @details Implementation Algorithm
 * 1. Initialize centroids \f$\boldsymbol{\mu}_i\f$ by randomly selecting \f$K\f$ distinct data points.
 * 2. Assign each point \f$\mathbf{x}_j\f$ to the cluster with the nearest centroid:
 *    \f[
 *    S_i^{(t)} = \left\{ \mathbf{x}_j : \|\mathbf{x}_j - \boldsymbol{\mu}_i^{(t)}\|_2 \le \|\mathbf{x}_j - \boldsymbol{\mu}_m^{(t)}\|_2 \ \forall m=1,\dots,K \right\}
 *    \f]
 * 3. Update the centroids to be the arithmetic mean of all points in that cluster:
 *    \f[
 *    \boldsymbol{\mu}_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{\mathbf{x}_j \in S_i^{(t)}} \mathbf{x}_j
 *    \f]
 * 4. Iterate steps 2-3 until convergence (centroids no longer change or `max_iter` is reached).
 */
class GeometricCluster {
  private:
    Eigen::VectorXs labels_;    ///< Cluster label assigned to each data point.
    Eigen::MatrixXd centroids_; ///< Centroid coordinates for each cluster.

    size_t nClusters_;          ///< Number of clusters.
    size_t nPoints_;            ///< Number of data points.
    size_t dim_;                ///< Dimensionality of data features.

  public:
    GeometricCluster() = default;

    /**
     * @brief Fits the K-means clustering model on the given dataset.
     * 
     * @param points N x d matrix of points to cluster.
     * @param nClusters The number of clusters K.
     * @param rng Random number generator used to initialize centroids.
     * @param max_iter Maximum number of K-means iterations.
     * @return true if convergence was reached, false if maximum iterations was exceeded.
     */
    bool fit(const Eigen::Ref<const Eigen::MatrixXd> &points, size_t nClusters, std::default_random_engine &rng, size_t max_iter = 1000);

    /**
     * @brief Accesses the cluster label of the i-th point.
     * 
     * @param i Index of the point.
     * @return Cluster label index.
     */
    const size_t &operator[](size_t i) const;

    /**
     * @brief Returns the centroid of the i-th cluster.
     * 
     * @param i Cluster index.
     * @return Vector representation of the cluster centroid.
     */
    Eigen::VectorXd centroid(size_t i) const;

    /**
     * @brief Returns the number of clusters.
     */
    size_t nClusters() const;

    /**
     * @brief Returns the number of data points.
     */
    size_t nPoints() const;

    /**
     * @brief Returns the dimension of the data.
     */
    size_t dim() const;

    /**
     * @brief Returns the cluster assignments vector.
     * @return Reference to the vector of cluster labels.
     */
    const Eigen::VectorXs &getLabels() const {
        return labels_;
    }
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
        size_t id_;                 ///< Unique cluster ID (not necessarily contiguous).
        size_t nPoints_;            ///< Number of points assigned to this cluster.

        // Sufficient statistics
        Eigen::VectorXd sumOfX_;       ///< Sum of all points in the cluster.
        Eigen::MatrixXd sumOfXXT_;     ///< Sum of all outer products x * x^T for points in the cluster.

        // Default constructor
        Cluster() : id_(0), nPoints_(0) {}

        // Constructor with id and dimension
        Cluster(size_t id, size_t dim) : id_(id), nPoints_(0),
            sumOfX_(Eigen::VectorXd::Zero(dim)),
            sumOfXXT_(Eigen::MatrixXd::Zero(dim, dim)) {}
    };

  private:

    // Workspaces for the Gibbs sampler to avoid heap allocations
    std::vector<double> logp_workspace_;        ///< Temporary buffer for cluster log-probabilities.
    std::vector<double> probs_workspace_;       ///< Temporary buffer for cluster probability weights.
    std::vector<size_t> clusterIDs_workspace_;  ///< Temporary buffer for cluster IDs.

    Eigen::MatrixXd xObs_;                      ///< Observed data points matrix (N x D).
    Eigen::VectorXs labels_;                    ///< Current cluster assignment label vector.
    size_t nPoints_{0};                         ///< Number of observation points.
    int dim_{0};                                ///< Dimensionality of the feature space.
    std::vector<size_t>  clusterIDs_;           ///< List of active cluster IDs.

    // This prior is conjugate to the Gaussian likelihood
    cmp::distribution::NormalInverseWishartDistribution hyper_; ///< Hyperprior distribution parameters for clusters.
    cmp::distribution::GammaDistribution alphaPrior_;           ///< Prior distribution parameters for concentration parameter.
    double alpha_{1.0};                                         ///< Concentration parameter alpha for the Dirichlet Process.

    std::map<size_t, Cluster> clusters_;        ///< Map from cluster ID to sufficient statistics cluster structure.
    size_t nextClusterId_;                      ///< Counter to generate unique new cluster IDs.

    // Random number generator for sampling
    std::default_random_engine rng_;            ///< Pseudo-random number generator engine.
    std::uniform_real_distribution<double> distU_{0.0, 1.0}; ///< Uniform real generator for rejection sampler.

  public:

// public API
    /**
     * @brief Constructs a new DirichletProcessMixtureModel object.
     * 
     * @param alpha Initial concentration parameter for the Dirichlet Process.
     * @param alphaPrior Gamma prior distribution parameters for concentration updates.
     * @param hyper Normal-Inverse-Wishart hyperprior parameters for clusters.
     * @param seed Seed value for the random number generator.
     */
    DirichletProcessMixtureModel(
        double alpha,
        const cmp::distribution::GammaDistribution &alphaPrior,
        const cmp::distribution::NormalInverseWishartDistribution &hyper,
        unsigned int seed = 12345
    );

    /**
     * @brief Conditions the DPMM model on the given dataset with initial cluster labels.
     * 
     * @param data N x D matrix of observed points.
     * @param init_labels Initial cluster assignment vector of size N.
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &data, const Eigen::VectorXs& init_labels);

    /**
     * @brief Performs one complete sweep of collapsed Gibbs sampling over all points.
     */
    void step();

    /**
     * @brief Remaps cluster labels to be contiguous integers starting from 0.
     */
    void remapLabels();

    /**
     * @brief Gets the current cluster label assignments vector.
     * 
     * @return Eigen::VectorXs of cluster labels.
     */
    Eigen::VectorXs getLabels() const {
        return labels_;
    }

    /**
     * @brief Gets the current number of active clusters.
     * 
     * @return Number of clusters.
     */
    size_t nClusters() const {
        return clusters_.size();
    }

    /**
     * @brief Gets the current concentration parameter alpha.
     * 
     * @return Value of alpha.
     */
    double getAlpha() const {
        return alpha_;
    }

  private:

    /**
     * @brief Initializes cluster counts, sums, and sufficient statistics.
     * 
     * @param init_labels Initial cluster labels.
     */
    void init(const Eigen::VectorXs& init_labels);

    /**
     * @brief Removes a point from the sufficient statistics of a specified cluster.
     * 
     * @param pointIndex Index of the data point.
     * @param clusterID ID of the cluster.
     */
    void removePointFromCluster(const size_t &pointIndex, const size_t &clusterID);

    /**
     * @brief Adds a point to the sufficient statistics of a specified cluster.
     * 
     * @param pointIndex Index of the data point.
     * @param clusterID ID of the cluster.
     */
    void addPointToCluster(const size_t &pointIndex, const size_t &clusterID);

    /**
     * @brief Computes the log posterior probability of a point under a cluster's NIW predictive distribution.
     * 
     * @param x The point vector.
     * @param c The cluster structure.
     * @return The log posterior probability.
     */
    double logNIWPosterior(const Eigen::VectorXd& x, const Cluster& c) const;

    /**
     * @brief Updates the concentration parameter alpha via auxiliary variable sampling.
     */
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


/**
 * @brief Simple partitioning algorithm that assigns observations to clusters uniformly at random.
 * 
 * @details Mathematical Formulation
 * Each sample index \f$i\f$ is assigned a cluster assignment variable \f$z_i\f$ drawn from a discrete uniform distribution:
 * \f[
 * P(z_i = k) = \frac{1}{K}, \quad k \in \{0, 1, \dots, K-1\}
 * \f]
 * where \f$K\f$ is the number of target clusters.
 * 
 * @details Implementation Algorithm
 * The `fit()` function constructs a `std::uniform_int_distribution` and samples an integer in the range \f$[0, K-1]\f$ for each data point sequentially.
 */
class DummyCluster {
  private:
    size_t nClusters_;               ///< Number of target clusters.
    std::default_random_engine rng_;  ///< Random number generator.
    Eigen::VectorXs labels_;         ///< Vector of randomly assigned labels.
  public:
    DummyCluster(size_t nClusters, unsigned int seed = 42)
        : nClusters_(nClusters), rng_(seed) {}

    /**
     * @brief Randomly assigns each data point to a cluster uniformly at random.
     * 
     * @param points Matrix of points.
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &points) {
        size_t nPoints = points.rows();
        std::uniform_int_distribution<size_t> dist(0, nClusters_ - 1);
        labels_ = Eigen::VectorXs::Zero(nPoints);
        for(size_t i = 0; i < nPoints; i++) {
            labels_(i) = dist(rng_);
        }
    }

    /**
     * @brief Gets the randomly generated cluster labels.
     * 
     * @return Reference to the label vector.
     */
    const Eigen::VectorXs &getLabels() const {
        return labels_;
    }


};

} // namespace cmp::cluster


/** @} */

#endif