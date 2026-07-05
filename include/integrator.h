#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <cmp_defines.h>

namespace cmp {

/**
 * @brief One-dimensional quadrature rule generator and integrator.
 *
 * @details Mathematical Formulation:
 * A 1D quadrature rule approximates the integral of a function $f(x)$ over an interval $[a, b]$ with a weight function $w(x)$ as:
 * \f[
 * \int_{a}^{b} w(x) f(x) \, dx \approx \sum_{i=0}^{N-1} w_i f(x_i)
 * \f]
 * where $x_i$ are the quadrature nodes (roots of the $N$-th order orthogonal polynomial associated with $w(x)$) and $w_i$ are the corresponding quadrature weights.
 * Using the Golub-Welsch algorithm, the nodes $x_i$ and weights $w_i$ are computed from the symmetric tridiagonal Jacobi matrix $J$ for the orthogonal polynomial sequence:
 * \f[
 * J = \begin{pmatrix}
 * a_0 & b_1 & 0 & \cdots & 0 \\
 * b_1 & a_1 & b_2 & \cdots & 0 \\
 * 0 & b_2 & a_2 & \ddots & \vdots \\
 * \vdots & \vdots & \ddots & \ddots & b_{N-1} \\
 * 0 & 0 & \cdots & b_{N-1} & a_{N-1}
 * \end{pmatrix}
 * \f]
 * The nodes $x_i$ are the eigenvalues of $J$. The weights $w_i$ are computed as:
 * \f[
 * w_i = \mu_0 v_{i,0}^2
 * \f]
 * where $v_{i,0}$ is the first component of the normalized $i$-th eigenvector of $J$, and $\mu_0 = \int_{a}^{b} w(x) \, dx$ is the total mass/zeroth moment of the weight function.
 *
 * @details Implementation Algorithm:
 * 1. Checks if the requested number of points $N$ is less than 1 (throwing std::invalid_argument).
 * 2. If $N == 1$, sets node $x_0 = 0.0$ and weight $w_0 = 1.0$.
 * 3. Otherwise, requests the Jacobi matrix $J$ of size $N \times N$ from `BasisType::getJacobiMatrix(numPoints)`.
 * 4. Computes the spectral decomposition of $J$ using `Eigen::SelfAdjointEigenSolver`.
 * 5. Assigns the resulting eigenvalues directly to `nodes`.
 * 6. Computes the weights `weights(i)` by squaring the first component of the $i$-th eigenvector (`solver.eigenvectors()(0, i)`).
 * 7. Provides `integrate(f)` which evaluates $\sum_i w_i f(x_i)$.
 */
template <typename BasisType>
class Quadrature1D {
  public:
    Eigen::VectorXd nodes;
    Eigen::VectorXd weights;

    Quadrature1D(int numPoints) {
        if(numPoints < 1) throw std::invalid_argument("Points must be >= 1");

        nodes.resize(numPoints);
        weights.resize(numPoints);

        if(numPoints == 1) {
            nodes(0) = 0.0;
            weights(0) = 1.0;
            return;
        }

        Eigen::MatrixXd J = BasisType::getJacobiMatrix(numPoints);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);
        nodes = solver.eigenvalues();
        Eigen::MatrixXd evecs = solver.eigenvectors();

        for(int i = 0; i < numPoints; ++i) {
            weights(i) =  evecs(0, i) * evecs(0, i);
        }
    }

    double integrate(std::function<double(double)> f) const {
        double result = 0.0;
        for(int i = 0; i < nodes.size(); ++i) {
            result += weights(i) * f(nodes(i));
        }
        return result;
    }
};

/**
 * @brief Multi-dimensional tensor product quadrature integrator.
 *
 * @details Mathematical Formulation:
 * A multi-dimensional integral over a tensor-product domain $\Omega = [a_1, b_1] \times \dots \times [a_d, b_d]$ is approximated by the tensor product of 1D quadrature rules:
 * \f[
 * \int_{\Omega} f(\mathbf{x}) \, d\mathbf{x} \approx \sum_{i_1=0}^{N-1} \dots \sum_{i_d=0}^{N-1} \left( \prod_{j=1}^{d} w_{i_j} \right) f(x_{i_1}, \dots, x_{i_d})
 * \f]
 * where $w_{i_j}$ and $x_{i_j}$ are the 1D quadrature weights and nodes along dimension $j$.
 * For physical domain integration over custom bounds $[p_{1,d}, p_{2,d}]$, a coordinate transformation maps the canonical nodes $\xi \in [-1, 1]$ (or the canonical domain of the basis) to $x \in [p_{1,d}, p_{2,d}]$ via:
 * \f[
 * x = \text{mapToPhysical}(\xi, p_{1,d}, p_{2,d})
 * \f]
 * The integration automatically incorporates any scaling factors (Jacobian of the transformation) if required by the basis.
 *
 * @details Implementation Algorithm:
 * 1. Instantiates a 1D quadrature rule `Quadrature1D<BasisType>` with the specified `pointsPerDim`.
 * 2. Allocates a grid node matrix of size $N^d \times d$ and a weight vector of size $N^d$, where $N$ is `pointsPerDim` and $d$ is `dim`.
 * 3. Iterates through all $N^d$ combinations using base-$N$ digit decomposition to reconstruct the index tuple $(i_0, \dots, i_{d-1})$.
 * 4. For each joint grid point $i$, assigns the $d$-th dimension coordinate to the corresponding 1D node, and multiplies the 1D weights to form the joint weight $W_i = \prod_{j=0}^{d-1} w_{i_j}$.
 * 5. Provides two `integrate` overloads:
 *    - Canonical `integrate(f)`: computes $\sum_{i} W_i f(\mathbf{x}_i)$.
 *    - Physical `integrate(f, p1, p2)`: maps canonical nodes to physical coordinates using `BasisType::mapToPhysical` for each dimension before evaluating $f$ and summing with weights.
 */
template <typename BasisType>
class TensorIntegrator {
  public:
    Eigen::MatrixXd gridNodes;
    Eigen::VectorXd gridWeights;

    TensorIntegrator(int dim, int pointsPerDim) {
        Quadrature1D<BasisType> gq1D(pointsPerDim);

        int totalPoints = std::pow(pointsPerDim, dim);
        gridNodes.resize(totalPoints, dim);
        gridWeights.resize(totalPoints);

        for(int i = 0; i < totalPoints; ++i) {
            int temp = i;
            double jointWeight = 1.0;

            for(int d = 0; d < dim; ++d) {
                int idx = temp % pointsPerDim;
                gridNodes(i, d) = gq1D.nodes(idx);
                jointWeight *= gq1D.weights(idx);
                temp /= pointsPerDim;
            }
            gridWeights(i) = jointWeight;
        }
    }

    /**
     * Canonical Integration: Integrates a function defined in the canonical domain using tensor product quadrature.
     * @param f: Function to integrate, defined in the canonical domain.
     * @return: The integral of the function over the canonical domain.
     */
    double integrate(const std::function<double(const Eigen::VectorXd&)> &f) const {
        double result = 0.0;
        for(int i = 0; i < gridNodes.rows(); ++i) {
            result += gridWeights(i) * f(gridNodes.row(i));
        }
        return result;
    }

    /**
     * Physical Integration: Integrates a function defined in the physical domain by automatically mapping to the canonical domain.
     * @param f: Function to integrate, defined in the physical domain.
     * @param p1: Lower bounds of the physical domain for each dimension.
     * @param p2: Upper bounds of the physical domain for each dimension.
     * @return: The integral of the function over the specified physical domain.
     */
    double integrate(const std::function<double(const Eigen::VectorXd&)> &f,
                     const Eigen::VectorXd& p1,
                     const Eigen::VectorXd& p2) const {

        int dim = gridNodes.cols();
        if(p1.size() != dim || p2.size() != dim) {
            throw std::invalid_argument("Parameter dimensions must match grid dimensions.");
        }

        double result = 0.0;
        Eigen::VectorXd physicalPoint(dim);

        for(int i = 0; i < gridNodes.rows(); ++i) {
            for(int d = 0; d < dim; ++d) {
                // Map from Canonical to Physical automatically
                physicalPoint(d) = BasisType::mapToPhysical(gridNodes(i, d), p1(d), p2(d));
            }
            result += gridWeights(i) * f(physicalPoint);
        }
        return result;
    }
};
}

#endif // INTEGRATOR_H