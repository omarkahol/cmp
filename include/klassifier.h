#ifndef KLASSIFIER_H 
#define KLASSIFIER_H

#include <cmp_defines.h>
#include <list>

namespace cmp::classifier {
    class KNN {
        private:
            std::vector<std::vector<size_t>> neighbours_;
            size_t kNearestValue_;
            size_t nPoints_;
        
        public:
            KNN() = default;

            void compute(const std::vector<Eigen::VectorXd> &points, size_t k);
            
            const std::vector<size_t> &operator[](size_t i) const;
            size_t nPoints() const;
            size_t k() const;
    };
}

#endif