#ifndef KLUSTER_H
#define KLUSTER_H

#include <cmp_defines.h>
#include <grid.h>

namespace cmp::cluster {

    class GeometricCluster {
        private:
            std::vector<size_t> toGlobal_;
            std::vector<Eigen::VectorXd> centroids_;

            size_t nClusters_;
            size_t nPoints_;
            size_t dim_;
        
        public:
            GeometricCluster() = default;

            bool compute(const std::vector<Eigen::VectorXd> &nPoints, size_t nClusters, std::default_random_engine &rng, size_t max_iter = 1000);
            const size_t &operator[](size_t i) const;
            const Eigen::VectorXd &centroid(size_t i) const;

            const std::vector<size_t> &getMembership() const {
                return toGlobal_;
            }

            size_t nClusters() const;
            size_t nPoints() const;
            size_t dim() const;


    };
}


#endif