#ifndef KERNELPP_H
#define KERNELPP_H

#include "cmp_defines.h"

namespace cmp::kernel {

constexpr double TOL = 1e-12;


// Implement Bandiwdths
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

class IsotropicBandwidth : public Bandwidth {
  private:
    double h_{1.0};
    size_t dim_{1};
  public:
    IsotropicBandwidth() = delete;
    IsotropicBandwidth(const double &h, const size_t &dim) : h_(h), dim_(dim) {};

    // This is just a scaling by 1/h to the difference between the two points
    Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

    // Computes the determinant of the transformation x' = (1/h) * x
    double determinant() const override;

    // Returns the dimension of the data
    size_t size() const override {
        return dim_;
    };

    // Gradient methods
    Eigen::VectorXd gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t &i) const override;

    double gradientOfLogDeterminant(const size_t &i) const override;

    void setFromVector(const Eigen::VectorXd &params) {
        h_ = params(0);
    };

    Eigen::VectorXd getParams() const {
        return Eigen::VectorXd::Constant(1, h_);
    };

    // Make method
    static std::shared_ptr<Bandwidth> make(const double &h, const size_t &dim) {
        return std::make_shared<IsotropicBandwidth>(h, dim);
    };

};

// Here we implement a very simple diagonal bandwidth (anisotropic)
class DiagonalBandwidth : public Bandwidth {
  private:
    Eigen::VectorXd h_;
    double det_;
  public:
    DiagonalBandwidth() = delete;
    DiagonalBandwidth(const Eigen::VectorXd &h) : h_(h) {
        det_ = h_.prod();
    };

    // Simple scaling
    Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

    double determinant() const override;


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
class FullBandwidth : public Bandwidth {

  private:
    Eigen::LLT<Eigen::MatrixXd> L_;
    double det_;
    size_t dim_;

  public:
    FullBandwidth() = delete;

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

    double determinant() const override;

    size_t size() const {
        return dim_;
    };

    std::pair<size_t, size_t> indexToRowCol(const size_t &i) const;

    Eigen::VectorXd apply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

    // Gradient methods
    Eigen::VectorXd gradientOfApply(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const size_t &i) const override;

    double gradientOfLogDeterminant(const size_t &i) const override;

    void setFromVector(const Eigen::VectorXd &params) override;

    Eigen::VectorXd getParams() const override;

    // Make method
    static std::shared_ptr<Bandwidth> make(const Eigen::MatrixXd &LLT) {
        return std::make_shared<FullBandwidth>(LLT);
    };

    // Full matrix
    Eigen::MatrixXd matrix() const {
        return L_.matrixL().toDenseMatrix() * L_.matrixL().transpose().toDenseMatrix();

    };

};



// Implement Kernels
/**
 * Now we implement a kernel virtual class and derive a few kernels from it.
 */

class Kernel {
  public:
    virtual ~Kernel() = default;
    virtual double eval(const Eigen::VectorXd& z) const = 0;
    virtual double normalizationConstant(const size_t &dim) const = 0;
    virtual double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const = 0;


};

class Gaussian : public Kernel {
  public:

    Gaussian() = default;

    double eval(const Eigen::VectorXd& z_diff) const override;

    double normalizationConstant(const size_t &dim) const override;

    double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const override;

    // Make method
    static std::shared_ptr<Kernel> make() {
        return std::make_shared<Gaussian>();
    };
};

class Epanechnikov : public Kernel {
  public:

    Epanechnikov() = default;

    double eval(const Eigen::VectorXd& z_diff) const override;

    double normalizationConstant(const size_t &dim) const override;

    double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const override;

    // Make method
    static std::shared_ptr<Kernel> make() {
        return std::make_shared<Epanechnikov>();
    };
};

// Rectangular (uniform) kernel
class Uniform : public Kernel {
  public:

    Uniform() = default;

    double eval(const Eigen::VectorXd& z_diff) const override;

    double normalizationConstant(const size_t &dim) const override;

    double applyToGradient(const Eigen::VectorXd& z, const Eigen::VectorXd& grad_z_i) const override;

    // Make method
    static std::shared_ptr<Kernel> make() {
        return std::make_shared<Uniform>();
    };
};



} // namespace cmp::kernel

#endif