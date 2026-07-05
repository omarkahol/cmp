#include "poly.h"

void cmp::MultiIndex::set(size_t dimension, size_t totalDegree, double q) {
    if(dimension == 0) {
        throw std::invalid_argument("Dimension must be greater than 0.");
    }
    if(q <= 0.0 || q > 1.0) {
        throw std::invalid_argument("Hyperbolic truncation q-norm must be in (0, 1].");
    }

    indices.clear();
    Index current(dimension, 0);

    // Precalculate p^q to avoid doing it repeatedly in the recursion
    double maxQSum = std::pow(static_cast<double>(totalDegree), q);

    generateRecursive(current, 0, totalDegree, 0.0, maxQSum, q);
}

void cmp::MultiIndex::print() const {
    std::cout << "MultiIndex (size: " << indices.size() << "):" << std::endl;
    for(const auto& idx : indices) {
        std::cout << "[";
        for(size_t i = 0; i < idx.size(); ++i) {
            std::cout << idx[i];
            if(i < idx.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void cmp::MultiIndex::generateRecursive(Index& current, int pos, size_t maxDegree,
                                        double currentQSum, double maxQSum, double q) {
    // Base Case: We have reached the final dimension of the parameter space.
    if(pos == (int)current.size() - 1) {
        for(size_t k = 0; k <= maxDegree; ++k) {
            // Compute the contribution of the final dimension's degree: k^q
            double k_q = (k == 0) ? 0.0 : std::pow(static_cast<double>(k), q);

            // --- HYPERBOLIC TRUNCATION Q-NORM CONSTRAINT ---
            // Checks if the accumulated q-norm sum violates the threshold:
            // sum_{i=1}^d (alpha_i)^q <= p^q
            if(currentQSum + k_q <= maxQSum + 1e-9) {
                current[pos] = k;
                indices.push_back(current);
            } else {
                // MONOTONICITY OPTIMIZATION:
                // Since k^q is strictly monotonically increasing for k >= 0,
                // any larger k will also violate the constraint. We can break early.
                break;
            }
        }
        return;
    }

    // Recursive Case: We are at an intermediate dimension.
    for(size_t k = 0; k <= maxDegree; ++k) {
        double k_q = (k == 0) ? 0.0 : std::pow(static_cast<double>(k), q);

        // Verify if the current sub-basis degree satisfies the q-norm bound
        if(currentQSum + k_q <= maxQSum + 1e-9) {
            current[pos] = k;
            // Recurse to process the next dimension
            generateRecursive(current, pos + 1, maxDegree, currentQSum + k_q, maxQSum, q);
        } else {
            // Early break due to monotonicity of the q-norm contribution
            break;
        }
    }
}