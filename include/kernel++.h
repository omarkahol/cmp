#ifndef KERNELPP_H
#define KERNELPP_H

#include "cmp_defines.h"

/**
 * @addtogroup probability
 * @{
 */
namespace cmp::kernel {

constexpr double TOL = 1e-12;


/**
 * @brief Abstract base class for multivariate KDE bandwidth matrices.
 * 
 * @details Mathematical Formulation
 * The bandwidth matrix \f$\mathbf{H} \in \mathbb{R}^{D \times D}\f$ acts as a linear transformation scaling spatial distances:
 * \f[
 * \mathbf{z} = \mathbf{H}^{-1} (\mathbf{x}_1 - \mathbf{x}_2)
 * \f]
 * The smoothing kernel scales inversely with the determinant: \f$\frac{1}{\det(\mathbf{H})}\f$.
 * 
 * @details Implementation Algorithm
 * Defines virtual methods to compute coordinate scaling (`apply`), matrix determinant (`determinant`), and parameter gradients for NLopt optimization.
 */
class Bandwidth {
  public:
    virtual ~Bandwidth() = default;
    virtual Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const = 0;
    virtual double determinant() const = 0;
    virtual size_t size() const = 0;

    // Gradient methods
    virtual Eigen::VectorXd gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t &i) const = 0;
    virtual double gradientOfLogDeterminant(const size_t &i) const = 0;

    // Construct for NLopt routines
    virtual void setFromVector(const Eigen::VectorXd &params) = 0;
    virtual Eigen::VectorXd getParams() const = 0;

};

/**
 * @brief Isotropic bandwidth matrix (single scalar parameter h).
 * 
 * @details Mathematical Formulation
 * Defines a spherical scaling matrix \f$\mathbf{H} = h \mathbf{I}_D\f$.
 * The transformation scales inputs uniformly:
 * \f[
 * \mathbf{z} = \frac{1}{h} (\mathbf{x}_1 - \mathbf{x}_2)
 * \f]
 * The determinant is \f$\det(\mathbf{H}) = h^D\f$.
 */
class IsotropicBandwidth : public Bandwidth {
  private:
    double h_{1.0};
    size_t dim_{1};
  public:
    IsotropicBandwidth() = delete;

    /**
     * @brief Constructs an IsotropicBandwidth object.
     * @param h Smoothing bandwidth scalar.
     * @param dim Dimension of the data.
     */
    IsotropicBandwidth(const double &h, const size_t &dim) : h_(h), dim_(dim) {};

    /**
     * @brief Scales the distance between two vectors by 1/h.
     */
    Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

    /**
     * @brief Computes the determinant of the bandwidth matrix (h^D).
     */
    double determinant() const override;

    /**
     * @brief Returns the dimension of the space.
     */
    size_t size() const override {
        return dim_;
    };

    /**
     * @brief Computes the gradient of the scaling transformation with respect to h.
     */
    Eigen::VectorXd gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t &i) const override;

    /**
     * @brief Computes the gradient of the log-determinant (D / h).
     */
    double gradientOfLogDeterminant(const size_t &i) const override;

    /**
     * @brief Sets the bandwidth parameter from a parameter vector.
     */
    void setFromVector(const Eigen::VectorXd &params) {
        h_ = params(0);
    };

    /**
     * @brief Returns the bandwidth parameter vector.
     */
    Eigen::VectorXd getParams() const {
        return Eigen::VectorXd::Constant(1, h_);
    };

    /**
     * @brief Factory method for creating an IsotropicBandwidth.
     */
    static std::shared_ptr<Bandwidth> make(const double &h, const size_t &dim) {
        return std::make_shared<IsotropicBandwidth>(h, dim);
    };

};

/**
 * @brief Diagonal bandwidth matrix (independent parameter per dimension).
 * 
 * @details Mathematical Formulation
 * Represents a diagonal matrix \f$\mathbf{H} = \text{diag}(h_1, \dots, h_D)\f$.
 * Coordinates are scaled independently:
 * \f[
 * z_d = \frac{x_{1,d} - x_{2,d}}{h_d}
 * \f]
 * The determinant is \f$\det(\mathbf{H}) = \prod_{d=1}^D h_d\f$.
 */
class DiagonalBandwidth : public Bandwidth {
  private:
    Eigen::VectorXd h_;
    double det_;
  public:
    DiagonalBandwidth() = delete;

    /**
     * @brief Constructs a DiagonalBandwidth from a parameter vector.
     */
    DiagonalBandwidth(const Eigen::VectorXd &h) : h_(h) {
        det_ = h_.prod();
    };

    /**
     * @brief Scales the coordinate distances independently by 1 / h_d.
     */
    Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

    /**
     * @brief Computes the determinant of the diagonal bandwidth matrix.
     */
    double determinant() const override;


    /**
     * @brief Returns the dimension of the space.
     */
    size_t size() const override {
        return h_.size();
    };

    // Gradient methods
    Eigen::VectorXd gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t &i) const override;

    double gradientOfLogDeterminant(const size_t &i) const override;

    void setFromVector(const Eigen::VectorXd &params) override;

    Eigen::VectorXd getParams() const override;

    // Make method
    static std::shared_ptr<Bandwidth> make(const Eigen::VectorXd &h) {
        return std::make_shared<DiagonalBandwidth>(h);
    };

};

// We implement a full bandwidth based on the Cholesky decomposition
/**
 * @brief Full covariance bandwidth matrix parameterized by its Cholesky factor.
 * 
 * @details Mathematical Formulation
 * Represents a full bandwidth matrix parameterized via lower triangular Cholesky factor \f$\mathbf{L}\f$ where:
 * \f[
 * \mathbf{H} = \mathbf{L} \mathbf{L}^T
 * \f]
 * The transformation projects distances as:
 * \f[
 * \mathbf{z} = \mathbf{L}^{-1} (\mathbf{x}_1 - \mathbf{x}_2)
 * \f]
 * The determinant is computed from the diagonal elements of the Cholesky factor:
 * \f[
 * \det(\mathbf{H}) = \prod_{d=1}^D L_{dd}^2
 * \f]
 */
class FullBandwidth : public Bandwidth {

  private:
    Eigen::LLT<Eigen::MatrixXd> L_;
    double det_;
    size_t dim_;

  public:
    FullBandwidth() = delete;

    /**
     * @brief Constructs a FullBandwidth from a positive definite scale matrix.
     */
    FullBandwidth(const Eigen::MatrixXd &LLT) : dim_(LLT.rows()) {
        if(LLT.rows() != LLT.cols()) {
            throw std::invalid_argument("Full bandwidth expects a square matrix.");
        }
        L_ = Eigen::LLT<Eigen::MatrixXd>(LLT);
        if(L_.info() != Eigen::Success) {
            throw std::invalid_argument("The provided matrix is not positive definite.");
        }

        // Compute determinant as product of diagonal entries of the Cholesky factor
        // directly to avoid hitting Eigen internals that trigger deprecated enum
        // bitwise operations.
        det_ = 1.0;
        for(size_t i = 0; i < static_cast<size_t>(L_.matrixL().rows()); ++i) {
            det_ *= L_.matrixL()(i, i);
        }
    };

    /**
     * @brief Computes the determinant of the bandwidth matrix.
     */
    double determinant() const override;

    /**
     * @brief Returns the dimension of the space.
     */
    size_t size() const {
        return dim_;
    };

    /**
     * @brief Maps a flat index to row and column indices of the Cholesky factor.
     */
    std::pair<size_t, size_t> indexToRowCol(const size_t &i) const;

    /**
     * @brief Projects the distance vector using the inverse of the Cholesky factor L.
     */
    Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

    /**
     * @brief Computes the gradient of the scaling transformation with respect to parameter i.
     */
    Eigen::VectorXd gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t &i) const override;

    /**
     * @brief Computes the gradient of the log-determinant.
     */
    double gradientOfLogDeterminant(const size_t &i) const override;

    /**
     * @brief Sets the bandwidth parameters from a parameter vector.
     */
    void setFromVector(const Eigen::VectorXd &params) override;

    /**
     * @brief Gets the bandwidth parameters as a vector.
     */
    Eigen::VectorXd getParams() const override;

    /**
     * @brief Factory method for creating a FullBandwidth.
     */
    static std::shared_ptr<Bandwidth> make(const Eigen::MatrixXd &LLT) {
        return std::make_shared<FullBandwidth>(LLT);
    };

    /**
     * @brief Returns the full density bandwidth matrix.
     */
    Eigen::MatrixXd matrix() const {
        return L_.matrixL().toDenseMatrix() * L_.matrixL().transpose().toDenseMatrix();

    };

};



// Implement Kernels
/**
 * @brief Abstract base class for KDE density kernel functions.
 * 
 * @details Mathematical Formulation
 * A multivariate kernel \f$K: \mathbb{R}^D \to [0,\infty)\f$ defines localized weights for smoothing observations, satisfying:
 * \f[
 * \int_{\mathbb{R}^D} K(\mathbf{z}) d\mathbf{z} = 1.0, \quad \int_{\mathbb{R}^D} \mathbf{z} K(\mathbf{z}) d\mathbf{z} = \mathbf{0}
 * \f]
 * 
 * @details Implementation Algorithm
 * Defines virtual methods for evaluating the kernel value (`eval`), computing the dimension-dependent normalizing scalar (`normalizationConstant`), and calculating gradients (`applyToGradient`).
 */
class Kernel {
  public:
    virtual ~Kernel() = default;
    virtual double eval(const Eigen::VectorXd& z) const = 0;
    virtual double normalizationConstant(const size_t &dim) const = 0;
    virtual double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const = 0;


};

/**
 * @brief Standard Gaussian density kernel.
 * 
 * @details Mathematical Formulation
 * The standard multivariate Gaussian density kernel:
 * \f[
 * K(\mathbf{z}) = (2\pi)^{-D/2} \exp\left(-\frac{1}{2} \|\mathbf{z}\|_2^2 \right)
 * \f]
 */
class Gaussian : public Kernel {
  public:

    Gaussian() = default;

    /**
     * @brief Evaluates the unnormalized Gaussian kernel.
     */
    double eval(const Eigen::VectorXd& z_diff) const override;

    /**
     * @brief Returns the normalization constant for a given dimension.
     */
    double normalizationConstant(const size_t &dim) const override;

    /**
     * @brief Computes the gradient of the kernel evaluation.
     */
    double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const override;

    /**
     * @brief Factory method for creating a Gaussian kernel.
     */
    static std::shared_ptr<Kernel> make() {
        return std::make_shared<Gaussian>();
    };
};

/**
 * @brief Epanechnikov (parabolic) density kernel.
 * 
 * @details Mathematical Formulation
 * The multivariate Epanechnikov kernel:
 * \f[
 * K(\mathbf{z}) \propto \begin{cases} 1 - \|\mathbf{z}\|_2^2 & \text{if } \|\mathbf{z}\|_2 \le 1 \\ 0 & \text{otherwise} \end{cases}
 * \f]
 */
class Epanechnikov : public Kernel {
  public:

    Epanechnikov() = default;

    /**
     * @brief Evaluates the unnormalized Epanechnikov kernel.
     */
    double eval(const Eigen::VectorXd& z_diff) const override;

    /**
     * @brief Returns the normalization constant for a given dimension.
     */
    double normalizationConstant(const size_t &dim) const override;

    /**
     * @brief Computes the gradient of the kernel evaluation.
     */
    double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const override;

    /**
     * @brief Factory method for creating an Epanechnikov kernel.
     */
    static std::shared_ptr<Kernel> make() {
        return std::make_shared<Epanechnikov>();
    };
};

/**
 * @brief Rectangular (uniform) density kernel.
 * 
 * @details Mathematical Formulation
 * The multivariate uniform kernel:
 * \f[
 * K(\mathbf{z}) \propto \begin{cases} 1 & \text{if } \|\mathbf{z}\|_\infty \le 1/2 \\ 0 & \text{otherwise} \end{cases}
 * \f]
 */
class Uniform : public Kernel {
  public:

    Uniform() = default;

    /**
     * @brief Evaluates the unnormalized uniform kernel.
     */
    double eval(const Eigen::VectorXd& z_diff) const override;

    /**
     * @brief Returns the normalization constant for a given dimension.
     */
    double normalizationConstant(const size_t &dim) const override;

    /**
     * @brief Computes the gradient of the kernel evaluation (always 0 since it is flat).
     */
    double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const override;

    /**
     * @brief Factory method for creating a Uniform kernel.
     */
    static std::shared_ptr<Kernel> make() {
        return std::make_shared<Uniform>();
    };
};



} // namespace cmp::kernel

/** @} */

#endif