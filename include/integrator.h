#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <cmp_defines.h>

namespace cmp {

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