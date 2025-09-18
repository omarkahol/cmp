#include "kluster.h"

bool cmp::cluster::GeometricCluster::compute(const std::vector<Eigen::VectorXd> &points, size_t n_clusters, std::default_random_engine &rng, size_t max_iter)
{
    // Set the number of clusters
    nClusters_ = n_clusters;
    nPoints_ = points.size();
    dim_ = points[0].size();

    // Initialize the centroids
    Eigen::VectorXd one = Eigen::VectorXd::Ones(dim_);

    // Pick random points as the initial centroids WITHOUT REPLACEMENT
    std::vector<double> weights = std::vector<double>(nPoints_, 1.0);
    for (size_t i = 0; i < n_clusters; i++) {
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        size_t pick = dist(rng);
        centroids_.push_back(points[pick]);
        weights[pick] = 0;
    }

    // Initialize the location
    toGlobal_ = std::vector<size_t>(nPoints_, 0);

    // Perform the iterations
    for (size_t iter = 0; iter < max_iter; iter++) {

        bool converged = true;
        
        // Assign each grid point to a cluster
        for (size_t i = 0; i < nPoints_; i++) {
            double min_dist = (points[i] - centroids_[toGlobal_[i]]).norm();
            for (size_t j = 0; j < n_clusters; j++) {
    
                // Skip this iteration
                if (j == toGlobal_[i]) {
                    continue;
                }
    
                double dist = (points[i] - centroids_[j]).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    if (toGlobal_[i] != j) {
                        toGlobal_[i] = j;
                        converged = false;
                    }
                }
            }
        }
    
        // Recompute the centroids
        for (size_t j = 0; j < n_clusters; j++) {
            Eigen::VectorXd sum = 0*one;
            size_t count = 0;
            for (size_t i = 0; i < nPoints_; i++) {
                if (toGlobal_[i] == j) {
                    sum += points[i];
                    count++;
                }
            }
            centroids_[j] = sum / count;
        }
    
        if (converged) {
            std::cout << "Converged after " << iter << " iterations.\n";
            return true;
        }
    }
    return false;
}

const size_t &cmp::cluster::GeometricCluster::operator[](size_t i) const
{
    return toGlobal_[i];
}

const Eigen::VectorXd &cmp::cluster::GeometricCluster::centroid(size_t i) const
{
    return centroids_[i];
}

size_t cmp::cluster::GeometricCluster::nClusters() const
{
    return nClusters_;
}

size_t cmp::cluster::GeometricCluster::nPoints() const
{
    return nPoints_;
}

size_t cmp::cluster::GeometricCluster::dim() const
{
    return dim_;
}
