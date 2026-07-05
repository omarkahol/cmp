#ifndef POLY_H
#define POLY_H

#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>

#include <fstream>
#include <string>
#include <integrator.h>

/**
 * @addtogroup surrogate
 * @{
 */
namespace cmp {

// Basis for polynomial chaos expansion
/**
 * @brief Hermite orthogonal polynomial basis.
 * 
 * @details Mathematical Formulation
 * Probabilist's Hermite polynomials \f$H_n(\xi)\f$ are orthogonal with respect to the standard Gaussian density function \f$w(\xi) = \frac{1}{\sqrt{2\pi}} e^{-\xi^2/2}\f$. The recurrence relation is:
 * \f[
 * H_0(\xi) = 1, \quad H_1(\xi) = \xi, \quad H_{n+1}(\xi) = \xi H_n(\xi) - n H_{n-1}(\xi)
 * \f]
 * with \f$\mathbb{E}[H_m(\xi) H_n(\xi)] = n! \delta_{mn}\f$.
 * 
 * @details Implementation Algorithm
 * 1. `evaluate()` computes polynomial values using the 3-term recurrence relation.
 * 2. `getJacobiMatrix()` constructs the symmetric tridiagonal Jacobi matrix \f$\mathbf{J}\f$ with diagonal \f$\alpha_i = 0\f$ and off-diagonal \f$\beta_i = \sqrt{i}\f$ to evaluate quadrature nodes via spectral decomposition.
 */
struct HermiteBasis {
    static inline double evaluate(size_t deg, double xi) {
        if(deg == 0) return 1.0;
        if(deg == 1) return xi;
        double Hn_2 = 1.0, Hn_1 = xi, Hn = 0.0;
        for(size_t n = 2; n <= deg; ++n) {
            Hn = xi * Hn_1 - (n - 1) * Hn_2;
            Hn_2 = Hn_1;
            Hn_1 = Hn;
        }
        return Hn;
    }

    static inline double normSquared(size_t deg) {
        double norm = 1.0;
        for(size_t i = 2; i <= deg; ++i) norm *= i;
        return norm;
    }

    static inline Eigen::MatrixXd getJacobiMatrix(int numPoints) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(numPoints, numPoints);
        for(int i = 0; i < numPoints - 1; ++i) {
            double beta = std::sqrt(i + 1.0);
            J(i, i + 1) = beta;
            J(i + 1, i) = beta;
        }
        return J;
    }

    static inline double getPDFWeight() {
        return 1.0;
    }

    // Maps canonical N(0,1) root to physical N(mean, stdDev)
    static inline double mapToPhysical(double xi, double mean, double stdDev) {
        return mean + stdDev * xi;
    }

    // Maps physical x to canonical N(0,1) root
    static inline double mapToCanonical(double x, double mean, double stdDev) {
        return (x - mean) / stdDev;
    }
};

/**
 * @brief Legendre orthogonal polynomial basis.
 * 
 * @details Mathematical Formulation
 * Legendre polynomials \f$P_n(\xi)\f$ are orthogonal with respect to the uniform density function \f$w(\xi) = 1/2\f$ on \f$[-1, 1]\f$. The recurrence relation is:
 * \f[
 * (n+1) P_{n+1}(\xi) = (2n+1) \xi P_n(\xi) - n P_{n-1}(\xi)
 * \f]
 * with \f$\int_{-1}^1 P_m(\xi) P_n(\xi) d\xi = \frac{2}{2n+1} \delta_{mn}\f$.
 * 
 * @details Implementation Algorithm
 * 1. `evaluate()` computes polynomial values using Bonnet's recurrence relation.
 * 2. `getJacobiMatrix()` constructs the symmetric tridiagonal Jacobi matrix \f$\mathbf{J}\f$ with \f$\alpha_i = 0\f$ and off-diagonal \f$\beta_i = \frac{i}{\sqrt{(2i-1)(2i+1)}}\f$.
 */
struct LegendreBasis {
    static inline double evaluate(size_t deg, double xi) {
        if(deg == 0) return 1.0;
        if(deg == 1) return xi;
        double Pn_2 = 1.0, Pn_1 = xi, Pn = 0.0;
        for(size_t n = 2; n <= deg; ++n) {
            Pn = ((2 * n - 1) * xi * Pn_1 - (n - 1) * Pn_2) / n;
            Pn_2 = Pn_1;
            Pn_1 = Pn;
        }
        return Pn;
    }

    static inline double normSquared(size_t deg) {
        return 1.0 / (2.0 * deg + 1.0);
    }

    static inline Eigen::MatrixXd getJacobiMatrix(int numPoints) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(numPoints, numPoints);
        for(int i = 0; i < numPoints - 1; ++i) {
            double beta = std::sqrt((i + 1.0) * (i + 1.0) / ((2 * i + 1.0) * (2 * i + 3.0)));
            J(i, i + 1) = beta;
            J(i + 1, i) = beta;
        }
        return J;
    }

    static inline double getPDFWeight() {
        return 1.0;
    }

    // Maps canonical [-1, 1] root to physical [lowerBound, upperBound]
    static inline double mapToPhysical(double xi, double lowerBound, double upperBound) {
        return lowerBound + (upperBound - lowerBound) * (xi + 1.0) / 2.0;
    }

    // Maps physical x to canonical [-1, 1] root
    static inline double mapToCanonical(double x, double lowerBound, double upperBound) {
        return 2.0 * (x - lowerBound) / (upperBound - lowerBound) - 1.0;
    }
};

/**
 * @brief Chebyshev orthogonal polynomial basis.
 * 
 * @details Mathematical Formulation
 * Chebyshev polynomials of the first kind \f$T_n(\xi)\f$ are orthogonal with respect to the weight function \f$w(\xi) = \frac{1}{\sqrt{1-\xi^2}}\f$ on \f$[-1, 1]\f$. The recurrence relation is:
 * \f[
 * T_0(\xi) = 1, \quad T_1(\xi) = \xi, \quad T_{n+1}(\xi) = 2\xi T_n(\xi) - T_{n-1}(\xi)
 * \f]
 * 
 * @details Implementation Algorithm
 * 1. `evaluate()` computes polynomial values using the 3-term recurrence relation.
 * 2. `getJacobiMatrix()` constructs the symmetric tridiagonal Jacobi matrix \f$\mathbf{J}\f$ with \f$\alpha_i = 0\f$ and off-diagonal \f$\beta_1 = \frac{1}{\sqrt{2}}\f$, \f$\beta_i = 0.5\f$ for \f$i \ge 2\f$.
 */
struct ChebyshevBasis {
    static inline double evaluate(size_t deg, double xi) {
        if(deg == 0) return 1.0;
        if(deg == 1) return xi;
        double Tn_2 = 1.0, Tn_1 = xi, Tn = 0.0;
        for(size_t n = 2; n <= deg; ++n) {
            Tn = 2.0 * xi * Tn_1 - Tn_2;
            Tn_2 = Tn_1;
            Tn_1 = Tn;
        }
        return Tn;
    }

    static inline double normSquared(size_t deg) {
        constexpr double pi = 3.14159265358979323846;
        if(deg == 0) return pi;
        return pi / 2.0;
    }

    static inline Eigen::MatrixXd getJacobiMatrix(int numPoints) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(numPoints, numPoints);
        if(numPoints > 1) {
            J(0, 1) = 1.0 / std::sqrt(2.0);
            J(1, 0) = 1.0 / std::sqrt(2.0);
            for(int i = 1; i < numPoints - 1; ++i) {
                J(i, i + 1) = 0.5;
                J(i + 1, i) = 0.5;
            }
        }
        return J;
    }

    static inline double mapToPhysical(double xi, double lowerBound, double upperBound) {
        return lowerBound + (upperBound - lowerBound) * (xi + 1.0) / 2.0;
    }

    static inline double mapToCanonical(double x, double lowerBound, double upperBound) {
        return 2.0 * (x - lowerBound) / (upperBound - lowerBound) - 1.0;
    }
};

/**
 * @brief Laguerre orthogonal polynomial basis.
 * 
 * @details Mathematical Formulation
 * Laguerre polynomials \f$L_n(\xi)\f$ are orthogonal with respect to the exponential weight function \f$w(\xi) = e^{-\xi}\f$ on \f$[0, \infty)\f$. The recurrence relation is:
 * \f[
 * (n+1) L_{n+1}(\xi) = (2n + 1 - \xi) L_n(\xi) - n L_{n-1}(\xi)
 * \f]
 * 
 * @details Implementation Algorithm
 * 1. `evaluate()` computes polynomial values using the standard Laguerre recurrence relation.
 * 2. `getJacobiMatrix()` constructs the symmetric tridiagonal Jacobi matrix \f$\mathbf{J}\f$ with diagonal \f$\alpha_i = 2i + 1\f$ and off-diagonal \f$\beta_i = i + 1\f$.
 */
struct LaguerreBasis {
    static inline double evaluate(size_t deg, double xi) {
        if(deg == 0) return 1.0;
        if(deg == 1) return 1.0 - xi;
        double Ln_2 = 1.0, Ln_1 = 1.0 - xi, Ln = 0.0;
        for(size_t n = 1; n < deg; ++n) {
            Ln = ((2.0 * n + 1.0 - xi) * Ln_1 - n * Ln_2) / (n + 1.0);
            Ln_2 = Ln_1;
            Ln_1 = Ln;
        }
        return Ln;
    }

    static inline double normSquared(size_t deg) {
        return 1.0;
    }

    static inline Eigen::MatrixXd getJacobiMatrix(int numPoints) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(numPoints, numPoints);
        for(int i = 0; i < numPoints; ++i) {
            J(i, i) = 2.0 * i + 1.0;
            if(i < numPoints - 1) {
                double off_diag = static_cast<double>(i + 1);
                J(i, i + 1) = off_diag;
                J(i + 1, i) = off_diag;
            }
        }
        return J;
    }

    // Maps canonical root [0, inf) to physical
    static inline double mapToPhysical(double xi, double location, double scale) {
        return location + scale * xi;
    }

    // Maps physical x to canonical root [0, inf)
    static inline double mapToCanonical(double x, double location, double scale) {
        return (x - location) / scale;
    }
};

class MultiIndex {
  public:
    using Index = std::vector<size_t>;
    std::vector<Index> indices;

    MultiIndex() = default;
    void set(size_t dimension, size_t totalDegree, double q = 1.0);

    std::size_t size() const {
        return indices.size();
    }
    const Index& operator[](std::size_t i) const {
        return indices[i];
    }
    void print() const;

  private:
    void generateRecursive(Index& current, int pos, size_t remaining, double currentQSum, double maxQSum, double q);
};

/**
 * @class PolynomialExpansion
 * @brief Implements multi-dimensional Polynomial Chaos Expansion (PCE) for spectral surrogate modeling.
 *
 * @tparam Basis The orthogonal polynomial basis type (e.g., HermiteBasis for Gaussians, LegendreBasis for Uniforms).
 *
 * @details
 * ### Mathematical Foundations
 * Polynomial Chaos Expansion represents a stochastic model output \f$Y = f(\mathbf{X})\f$ as a spectral expansion
 * of multivariate orthogonal polynomials mapping canonical variables \f$\boldsymbol{\xi}\f$:
 * \f[ Y \approx \sum_{j=0}^{P-1} c_j \Psi_j(\boldsymbol{\xi}) \f]
 *
 * The multivariate polynomials \f$\Psi_j(\boldsymbol{\xi})\f$ are tensor products of univariate orthogonal polynomials:
 * \f[ \Psi_j(\boldsymbol{\xi}) = \prod_{i=1}^d \psi_{\alpha_i^{(j)}}(\xi_i) \f]
 * where \f$\boldsymbol{\alpha}^{(j)} = (\alpha_1^{(j)}, \dots, \alpha_d^{(j)})\f$ is the multi-index of polynomial degrees.
 *
 * Uni-dimensional orthogonal bases satisfy:
 * \f[ \int_{\Omega} \psi_k(x) \psi_l(x) w(x) dx = \gamma_k \delta_{kl} \f]
 * where \f$w(x)\f$ is the probability density function (PDF) weight of the canonical distribution.
 *
 * ### Implementation Algorithms
 * 1. **Canonical Mapping**: Physical samples \f$\mathbf{x}\f$ are mapped to canonical space \f$\boldsymbol{\xi}\f$
 *    using the basis-specific mappings (`mapToCanonical`).
 * 2. **Design Matrix Construction**: Computes the regression matrix \f$\mathbf{A} \in \mathbb{R}^{N \times P}\f$ where:
 *    \f[ A_{ij} = \Psi_j(\boldsymbol{\xi}^{(i)}) \f]
 * 3. **Ordinary Least Squares (OLS) Regression**:
 *    The expansion coefficients \f$\mathbf{c}\f$ are found by solving the linear least-squares problem \f$\mathbf{A}\mathbf{c} \approx \mathbf{y}\f$:
 *    \f[ \mathbf{c} = \left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{y} \f]
 *    We solve this system robustly using Column-Pivoted Householder QR decomposition (`ColPivHouseholderQR`).
 *
 * ### Constraints & Invariants
 * - **Sample Size**: The number of samples \f$N\f$ must satisfy \f$N > P\f$ (where \f$P = \text{multiIndex\_.size()}\f$)
 *   to avoid an underdetermined system and severe overfitting.
 * - **Parameter Ranges**: Bounding parameter vectors (e.g., mean/standard deviation or lower/upper bounds)
 *   must match the dimensionality of the input space.
 */
template <typename Basis>
class PolynomialExpansion {
  private:
    MultiIndex multiIndex_;
    Eigen::VectorXd coefficients_;
    Eigen::MatrixXd normalMatrixInverse_;
    Eigen::MatrixXd xObs_;
    Eigen::VectorXd yObs_;
    double residualVariance_ = 0.0;

    // Distribution parameters for each dimension
    Eigen::VectorXd p1_;
    Eigen::VectorXd p2_;

    // Vectorized evaluation of a block of data
    Eigen::MatrixXd computeDesignMatrix(const Eigen::Ref<const Eigen::MatrixXd>& X) const {
        const int numSamples = X.rows();
        const int numBasis = multiIndex_.size();
        const int dim = X.cols();

        Eigen::MatrixXd A(numSamples, numBasis);

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < numSamples; ++i) {
            for(int b = 0; b < numBasis; ++b) {
                double term = 1.0;
                for(int d = 0; d < dim; ++d) {
                    // Map the physical observation X to canonical space
                    double xi = Basis::mapToCanonical(X(i, d), p1_(d), p2_(d));
                    term *= Basis::evaluate(multiIndex_.indices[b][d], xi);
                }
                A(i, b) = term;
            }
        }
        return A;
    }

    template <typename MatrixType>
    void writeEigenMatrix(std::ofstream& out, const MatrixType& mat) const {
        typename MatrixType::Index rows = mat.rows();
        typename MatrixType::Index cols = mat.cols();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char*>(mat.data()), rows * cols * sizeof(typename MatrixType::Scalar));
    }

    template <typename MatrixType>
    void readEigenMatrix(std::ifstream& in, MatrixType& mat) {
        typename MatrixType::Index rows = 0, cols = 0;
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        mat.resize(rows, cols);
        in.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(typename MatrixType::Scalar));
    }

  public:
    PolynomialExpansion() = default;

    // Updated init method to capture bounding/distribution parameters
    void set(size_t dimension, size_t totalDegree, const Eigen::VectorXd& p1, const Eigen::VectorXd& p2, double q = 1.0) {
        if(p1.size() != dimension || p2.size() != dimension) {
            throw std::invalid_argument("Parameter vectors must match dimension.");
        }
        p1_ = p1;
        p2_ = p2;
        multiIndex_.set(dimension, totalDegree, q);
        coefficients_ = Eigen::VectorXd::Zero(multiIndex_.size());
    }

    void fit(const Eigen::Ref<const Eigen::MatrixXd>& samples, const Eigen::Ref<const Eigen::VectorXd>& values) {
        if(samples.rows() != values.size()) throw std::invalid_argument("Dimension mismatch.");

        xObs_ = samples;
        yObs_ = values;

        Eigen::MatrixXd A = computeDesignMatrix(samples);

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
        coefficients_ = qr.solve(values);

        normalMatrixInverse_ = (A.transpose() * A).completeOrthogonalDecomposition().pseudoInverse();
        Eigen::VectorXd residuals = values - A * coefficients_;

        if(samples.rows() > coefficients_.size()) {
            residualVariance_ = residuals.squaredNorm() / (samples.rows() - coefficients_.size());
        }
    }

    double getAnalyticalMean() const {
        if(coefficients_.size() == 0) return 0.0;
        return coefficients_(0);
    }

    double getAnalyticalVariance() const {
        double totalVar = 0.0;
        for(size_t i = 1; i < multiIndex_.size(); ++i) {
            double basisNormSq = 1.0;
            for(size_t d = 0; d < multiIndex_.indices[i].size(); ++d) {
                basisNormSq *= Basis::normSquared(multiIndex_.indices[i][d]);
            }
            totalVar += coefficients_(i) * coefficients_(i) * basisNormSq;
        }
        return totalVar;
    }

    Eigen::VectorXd getSobolMainIndices() const {
        double totalVar = getAnalyticalVariance();
        size_t dim = multiIndex_.indices[0].size();
        Eigen::VectorXd sobolIndices = Eigen::VectorXd::Zero(dim);

        if(totalVar < 1e-12) return sobolIndices;

        for(size_t i = 1; i < multiIndex_.size(); ++i) {
            int activeDim = -1;
            bool isMainEffect = true;

            for(size_t d = 0; d < dim; ++d) {
                if(multiIndex_.indices[i][d] > 0) {
                    if(activeDim == -1) activeDim = d;
                    else isMainEffect = false;
                }
            }

            if(isMainEffect && activeDim != -1) {
                double basisNormSq = 1.0;
                for(size_t d = 0; d < dim; ++d) {
                    basisNormSq *= Basis::normSquared(multiIndex_.indices[i][d]);
                }
                sobolIndices(activeDim) += coefficients_(i) * coefficients_(i) * basisNormSq;
            }
        }
        return sobolIndices / totalVar;
    }

    Eigen::VectorXd predictBatch(const Eigen::Ref<const Eigen::MatrixXd>& X) const {
        Eigen::MatrixXd A = computeDesignMatrix(X);
        return A * coefficients_;
    }

    std::pair<double, double> predict(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        Eigen::VectorXd phi = computeBasisRow(x);
        double mean = coefficients_.dot(phi);

        double variance = 0.0;
        if(residualVariance_ > 0.0 && normalMatrixInverse_.size() > 0) {
            variance = residualVariance_ * phi.dot(normalMatrixInverse_ * phi);
        }
        return {mean, variance};
    }

    std::pair<double, double> predictWithObs(const Eigen::Ref<const Eigen::VectorXd> &x, const double &yObs) const {
        if(normalMatrixInverse_.size() == 0 || yObs_.size() == 0) {
            throw std::runtime_error("predictWithObs requires a fitted model.");
        }
        Eigen::VectorXd phi = computeBasisRow(x);
        Eigen::VectorXd influence = normalMatrixInverse_ * phi;
        const double denom = 1.0 + phi.dot(influence);

        if(std::abs(denom) <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("Rank-one update is singular.");
        }

        const double innovation = yObs - phi.dot(coefficients_);
        Eigen::VectorXd gain = influence / denom;
        Eigen::VectorXd coeffsUpdated = coefficients_ + gain * innovation;
        Eigen::MatrixXd normalInvUpdated = normalMatrixInverse_ - (influence * influence.transpose()) / denom;

        const double mean = phi.dot(coeffsUpdated);
        double variance = 0.0;
        if(residualVariance_ > 0.0) {
            variance = residualVariance_ * phi.dot(normalInvUpdated * phi);
        }

        return {mean, variance};
    }

    std::pair<double, double> predictLOO(const size_t& i) const {
        if(i >= (size_t)xObs_.rows()) {
            throw std::out_of_range("LOO index out of range.");
        }
        if(normalMatrixInverse_.size() == 0) {
            throw std::runtime_error("LOO prediction requires a fitted normal matrix inverse.");
        }

        Eigen::VectorXd x = xObs_.row(i).transpose();
        Eigen::VectorXd phi = computeBasisRow(x);

        const double y_i = yObs_(i);
        const double residual_i = y_i - phi.dot(coefficients_);
        Eigen::VectorXd influence = normalMatrixInverse_ * phi;
        const double leverage_i = phi.dot(influence);
        const double denom = 1.0 - leverage_i;

        if(std::abs(denom) <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("LOO rank-one update is singular.");
        }

        Eigen::VectorXd coeffsLOO = coefficients_ - influence * (residual_i / denom);
        const double mean = phi.dot(coeffsLOO);

        double variance = 0.0;
        if(residualVariance_ > 0.0) {
            Eigen::MatrixXd normalInvLOO = normalMatrixInverse_ + (influence * influence.transpose()) / denom;
            variance = residualVariance_ * phi.dot(normalInvLOO * phi);
        }

        return {mean, variance};
    }

    Eigen::VectorXd computeBasisRow(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        if(multiIndex_.size() == 0) {
            throw std::runtime_error("Polynomial basis is not initialized.");
        }
        if(x.size() != (int)multiIndex_.indices[0].size()) {
            throw std::invalid_argument("Input vector size does not match the polynomial dimension.");
        }

        Eigen::VectorXd phi(multiIndex_.size());
        for(size_t i = 0; i < multiIndex_.size(); ++i) {
            double term = 1.0;
            for(size_t d = 0; d < (size_t)x.size(); ++d) {
                // Map the physical input vector to the canonical space
                double xi = Basis::mapToCanonical(x(d), p1_[d], p2_[d]);
                term *= Basis::evaluate(multiIndex_.indices[i][d], xi);
            }
            phi(i) = term;
        }
        return phi;
    }

    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if(!out.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // 1. Serialize domain parameters
        size_t p_size = p1_.size();
        out.write(reinterpret_cast<const char*>(&p_size), sizeof(p_size));
        if(p_size > 0) {
            out.write(reinterpret_cast<const char*>(p1_.data()), p_size * sizeof(double));
            out.write(reinterpret_cast<const char*>(p2_.data()), p_size * sizeof(double));
        }

        // 2. Serialize residual variance
        out.write(reinterpret_cast<const char*>(&residualVariance_), sizeof(residualVariance_));

        // 3. Serialize MultiIndex
        size_t numIndices = multiIndex_.indices.size();
        out.write(reinterpret_cast<const char*>(&numIndices), sizeof(numIndices));
        for(const auto& idx : multiIndex_.indices) {
            size_t dim = idx.size();
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            out.write(reinterpret_cast<const char*>(idx.data()), dim * sizeof(size_t));
        }

        // 4. Serialize Eigen Objects
        writeEigenMatrix(out, coefficients_);
        writeEigenMatrix(out, normalMatrixInverse_);
        writeEigenMatrix(out, xObs_);
        writeEigenMatrix(out, yObs_);

        out.close();
    }

    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if(!in.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        // 1. Deserialize domain parameters
        size_t p_size = 0;
        in.read(reinterpret_cast<char*>(&p_size), sizeof(p_size));
        p1_.resize(p_size);
        p2_.resize(p_size);
        if(p_size > 0) {
            in.read(reinterpret_cast<char*>(p1_.data()), p_size * sizeof(double));
            in.read(reinterpret_cast<char*>(p2_.data()), p_size * sizeof(double));
        }

        // 2. Deserialize residual variance
        in.read(reinterpret_cast<char*>(&residualVariance_), sizeof(residualVariance_));

        // 3. Deserialize MultiIndex
        size_t numIndices = 0;
        in.read(reinterpret_cast<char*>(&numIndices), sizeof(numIndices));
        multiIndex_.indices.clear();
        multiIndex_.indices.resize(numIndices);

        for(size_t i = 0; i < numIndices; ++i) {
            size_t dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            multiIndex_.indices[i].resize(dim);
            in.read(reinterpret_cast<char*>(multiIndex_.indices[i].data()), dim * sizeof(size_t));
        }

        // 4. Deserialize Eigen Objects
        readEigenMatrix(in, coefficients_);
        readEigenMatrix(in, normalMatrixInverse_);
        readEigenMatrix(in, xObs_);
        readEigenMatrix(in, yObs_);

        in.close();
    }

    size_t getNumBasisFunctions() const {
        return multiIndex_.size();
    }

    void project(const std::function<double(const Eigen::VectorXd&)>& model, int pointsPerDim) {
        if(multiIndex_.size() == 0) throw std::runtime_error("Basis not set.");
        int dim = multiIndex_.indices[0].size();

        cmp::TensorIntegrator<Basis> integrator(dim, pointsPerDim);
        int numEvals = integrator.gridNodes.rows();

        Eigen::VectorXd yVals(numEvals);
        for(int i = 0; i < numEvals; ++i) {
            // Evaluate the model using the mapped PHYSICAL nodes
            Eigen::VectorXd physicalPoint(dim);
            for(size_t d = 0; d < dim; ++d) {
                physicalPoint(d) = Basis::mapToPhysical(integrator.gridNodes(i, d), p1_[d], p2_[d]);
            }
            yVals(i) = model(physicalPoint);
        }

        // Compute coefficients via Galerkin projection
        for(size_t k = 0; k < multiIndex_.size(); ++k) {
            double integral_sum = 0.0;
            double basis_norm_squared = 1.0;

            for(size_t d = 0; d < dim; ++d) {
                basis_norm_squared *= Basis::normSquared(multiIndex_.indices[k][d]);
            }

            for(int i = 0; i < numEvals; ++i) {
                double psi_k = 1.0;
                for(size_t d = 0; d < dim; ++d) {
                    // Polynomials MUST be evaluated on the CANONICAL nodes
                    psi_k *= Basis::evaluate(multiIndex_.indices[k][d], integrator.gridNodes(i, d));
                }
                integral_sum += yVals(i) * psi_k * integrator.gridWeights(i);
            }
            coefficients_(k) = integral_sum / basis_norm_squared;
        }
        residualVariance_ = 0.0;
    }
};

} // namespace cmp
/** @} */

#endif // POLY_H