#ifndef SCALER_H
#define SCALER_H

#include <cmp_defines.h>


namespace cmp::scaler {

constexpr double TOL = 1e-10;

/**
 * @class Scaler
 * @brief Template virtual base class for feature scaling and transformation.
 * 
 * @details \b Mathematical \b Formulation
 * Define a transformation mapping \f$ T: \mathbb{R}^d \to \mathbb{R}^d \f$ and its inverse \f$ T^{-1} \f$:
 * \f[ y = T(x) \f]
 * \f[ x = T^{-1}(y) \f]
 * Often represented linearly with intercept \f$ \mu \f$ and scaling factor/matrix \f$ S \f$:
 * \f[ T(x) = S^{-1}(x - \mu) \f]
 * \f[ T^{-1}(y) = S y + \mu \f]
 * 
 * @details \b Implementation \b Algorithm
 * Pure virtual interface. Concrete implementations define the fitting, transforming, and inverse transforming steps.
 */
class Scaler {
  public:

    virtual Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const = 0;
    virtual Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const = 0;

    virtual Eigen::VectorXd getIntercept() const = 0;
    virtual Eigen::MatrixXd getScale() const = 0;

    virtual void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) = 0;
    virtual Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) = 0;
};

/**
 * @class StandardScaler
 * @brief Standardizes features by removing the mean and scaling to unit variance (using Cholesky decomposition).
 * 
 * @details \b Mathematical \b Formulation
 * Given data matrix \f$ X \in \mathbb{R}^{n \times d} \f$, let \f$ \mu \in \mathbb{R}^d \f$ be the column-wise mean and \f$ \Sigma \in \mathbb{R}^{d \times d} \f$ be the covariance matrix.
 * We decompose the covariance matrix via the lower Cholesky factor \f$ L \f$ such that:
 * \f[ \Sigma = L L^T \f]
 * The transformation maps:
 * \f[ y = L^{-1}(x - \mu) \f]
 * The inverse transformation maps:
 * \f[ x = L y + \mu \f]
 * 
 * @details \b Implementation \b Algorithm
 * 1. \b Fit: Calculate mean vector \f$ \mu \f$ and covariance matrix \f$ \Sigma \f$ of \f$ X \f$, then perform Cholesky LLT decomposition to obtain \f$ L \f$.
 * 2. \b Transform: Solve the lower triangular system \f$ L y = x - \mu \f$ to compute standardized data.
 * 3. \b Inverse \b Transform: Compute \f$ L y + \mu \f$ via matrix multiplication and vector addition.
 */
class StandardScaler : public Scaler {
  private:
    Eigen::VectorXd mean_;
    Eigen::LLT<Eigen::MatrixXd> lltDecomposition_;

  public:
    StandardScaler() = default;
    StandardScaler(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::MatrixXd> &scale) : mean_(mean), lltDecomposition_(scale) {};
    StandardScaler(const StandardScaler &other) = default;
    StandardScaler(StandardScaler &&other) = default;

    ~StandardScaler() = default;

    StandardScaler &operator=(const StandardScaler &other) = default;
    StandardScaler &operator=(StandardScaler &&other) = default;

    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    Eigen::VectorXd getIntercept() const override {
        return mean_;
    };
    Eigen::MatrixXd getScale() const override {
        return lltDecomposition_.matrixL();
    };

    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override;
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override;
};

// PCA scaler
class PCA : public Scaler {
  private:
    Eigen::VectorXd mean_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver_;
    Eigen::MatrixXd sqrtCov_;
    Eigen::MatrixXd sqrtCovInv_;

    size_t nComponents_;

    void eigenDecomposition();


  public:
    PCA(size_t nComponents) : nComponents_(nComponents) {};
    PCA(const PCA &other) = default;
    PCA(PCA &&other) = default;

    ~PCA() = default;

    PCA &operator=(const PCA &other) = default;
    PCA &operator=(PCA &&other) = default;

    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    Eigen::VectorXd getIntercept() const override {
        return mean_;
    };
    Eigen::MatrixXd getScale() const override {
        return sqrtCov_;
    };

    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override;
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override;

    void resize(size_t nComponents);

    Eigen::VectorXd getEigenvalues() const {
        return eigenSolver_.eigenvalues();
    };
    Eigen::MatrixXd getEigenvectors() const {
        return eigenSolver_.eigenvectors();
    };
};

// Dummy vector scaler
class DummyScaler : public Scaler {
  private:
    size_t dim_;
  public:
    DummyScaler() = default;

    DummyScaler(const DummyScaler &other) = default;
    DummyScaler(DummyScaler &&other) = default;

    ~DummyScaler() = default;

    DummyScaler &operator=(const DummyScaler &other) = default;
    DummyScaler &operator=(DummyScaler &&other) = default;

    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override {
        return data;
    };
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override {
        return data;
    };

    Eigen::VectorXd getIntercept() const override {
        return Eigen::VectorXd::Zero(dim_);
    };
    Eigen::MatrixXd getScale() const override {
        return Eigen::MatrixXd::Identity(dim_, dim_);
    };

    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override {
        dim_ = data.cols();
    };

    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override {
        dim_ = data.cols();
        return data;
    };

    void setDim(size_t dim) {
        dim_ = dim;
    };

};

// Elliptic vector scaler
class EllipticScaler : public Scaler {
  private:
    Eigen::VectorXd mean_;
    Eigen::VectorXd std_;

  public:
    EllipticScaler() = default;
    EllipticScaler(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::VectorXd> &std) : mean_(mean), std_(std) {};
    EllipticScaler(const EllipticScaler &other) = default;
    EllipticScaler(EllipticScaler &&other) = default;

    ~EllipticScaler() = default;

    EllipticScaler &operator=(const EllipticScaler &other) = default;
    EllipticScaler &operator=(EllipticScaler &&other) = default;
    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;
    Eigen::VectorXd getIntercept() const override {
        return mean_;
    };
    Eigen::MatrixXd getScale() const override {
        Eigen::MatrixXd scale = Eigen::MatrixXd::Identity(mean_.size(), mean_.size());
        for(int i = 0; i < mean_.size(); i++) {
            scale(i, i) = std_[i];
        }
        return scale;
    };
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override;
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override;
    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    };
    void setStd(const Eigen::Ref<const Eigen::VectorXd> &std) {
        std_ = std;
    };
};

// Min-Max vector scaler
class MinMaxScaler : public Scaler {
  private:
    Eigen::VectorXd min_;
    Eigen::VectorXd max_;
    Eigen::VectorXd data_min_;
    Eigen::VectorXd data_max_;

  public:
    MinMaxScaler() = default;
    MinMaxScaler(const Eigen::Ref<const Eigen::VectorXd> &min, const Eigen::Ref<const Eigen::VectorXd> &max) : min_(min), max_(max) {};
    MinMaxScaler(const MinMaxScaler &other) = default;
    MinMaxScaler(MinMaxScaler &&other) = default;
    ~MinMaxScaler() = default;
    MinMaxScaler &operator=(const MinMaxScaler &other) = default;
    MinMaxScaler &operator=(MinMaxScaler &&other) = default;
    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    Eigen::VectorXd getDataMin() const {
        return data_min_;
    };
    Eigen::VectorXd getDataMax() const {
        return data_max_;
    };

    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override;
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override;

    Eigen::VectorXd getIntercept() const override;
    Eigen::MatrixXd getScale() const override;
};
}


#endif // SCALER_H