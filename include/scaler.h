#ifndef SCALER_H
#define SCALER_H

#include <cmp_defines.h>


/**
 * @addtogroup surrogate
 * @{
 */
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

/**
 * @class PCA
 * @brief Principal Component Analysis (PCA) feature scaler and dimension reducer.
 * 
 * @details Mathematical Formulation
 * Centers data matrix \f$\mathbf{X} \in \mathbb{R}^{N \times D}\f$ around column mean \f$\boldsymbol{\mu}\f$ and computes the eigendecomposition of the sample covariance matrix \f$\boldsymbol{\Sigma}\f$:
 * \f[
 * \boldsymbol{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T
 * \f]
 * where \f$\mathbf{V}\f$ contains orthogonal eigenvectors and \f$\mathbf{\Lambda}\f$ contains eigenvalues in descending order.
 * The transformation projects \f$\mathbf{x}\f$ into a lower-dimensional latent space of size \f$M\f$:
 * \f[
 * \mathbf{y} = \mathbf{\Lambda}_M^{-1/2} \mathbf{V}_M^T (\mathbf{x} - \boldsymbol{\mu})
 * \f]
 * where \f$\mathbf{V}_M\f$ and \f$\mathbf{\Lambda}_M\f$ represent truncated eigenvector and eigenvalue matrices.
 * The inverse transformation reconstructs the point:
 * \f[
 * \mathbf{x} = \mathbf{V}_M \mathbf{\Lambda}_M^{1/2} \mathbf{y} + \boldsymbol{\mu}
 * \f]
 * 
 * @details Implementation Algorithm
 * 1. `fit()` computes the mean \f$\boldsymbol{\mu}\f$ and covariance \f$\boldsymbol{\Sigma}\f$ of the training matrix, then runs `Eigen::SelfAdjointEigenSolver`.
 * 2. `transform()` centers the query vector, projects it onto the principal eigenvectors, and scales it by the inverse square root of eigenvalues.
 * 3. `inverseTransform()` rescales by square root eigenvalues, projects back to physical coordinates, and adds the mean.
 */
class PCA : public Scaler {
  private:
    Eigen::VectorXd mean_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver_;
    Eigen::MatrixXd sqrtCov_;
    Eigen::MatrixXd sqrtCovInv_;

    size_t nComponents_;

    /**
     * @brief Internal helper to run eigenvalues and eigenvectors solvers.
     */
    void eigenDecomposition();


  public:
    /**
     * @brief Constructs a PCA scaler with the specified number of components.
     * @param nComponents Number of principal components to retain.
     */
    PCA(size_t nComponents) : nComponents_(nComponents) {};
    PCA(const PCA &other) = default;
    PCA(PCA &&other) = default;

    ~PCA() = default;

    PCA &operator=(const PCA &other) = default;
    PCA &operator=(PCA &&other) = default;

    /**
     * @brief Projects physical data onto the principal components.
     */
    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    /**
     * @brief Reconstructs physical data from the principal components.
     */
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    Eigen::VectorXd getIntercept() const override {
        return mean_;
    };
    Eigen::MatrixXd getScale() const override {
        return sqrtCov_;
    };

    /**
     * @brief Fits the PCA model to the training dataset.
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override;

    /**
     * @brief Fits the PCA model and returns the projected dataset.
     */
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override;

    /**
     * @brief Resizes the number of principal components.
     */
    void resize(size_t nComponents);

    /**
     * @brief Returns the eigenvalues of the covariance matrix.
     */
    Eigen::VectorXd getEigenvalues() const {
        return eigenSolver_.eigenvalues();
    };

    /**
     * @brief Returns the eigenvectors of the covariance matrix.
     */
    Eigen::MatrixXd getEigenvectors() const {
        return eigenSolver_.eigenvectors();
    };
};

/**
 * @brief Dummy scaler that leaves input data unchanged.
 */
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

    /**
     * @brief Returns the input data unchanged.
     */
    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override {
        return data;
    };

    /**
     * @brief Returns the input data unchanged.
     */
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override {
        return data;
    };

    Eigen::VectorXd getIntercept() const override {
        return Eigen::VectorXd::Zero(dim_);
    };
    Eigen::MatrixXd getScale() const override {
        return Eigen::MatrixXd::Identity(dim_, dim_);
    };

    /**
     * @brief Learns the dimension of the training dataset.
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override {
        dim_ = data.cols();
    };

    /**
     * @brief Learns the dimension and returns the unchanged dataset.
     */
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override {
        dim_ = data.cols();
        return data;
    };

    /**
     * @brief Sets the data dimensionality.
     */
    void setDim(size_t dim) {
        dim_ = dim;
    };

};

/**
 * @brief Scaler that standardizes features independently (diagonal variance scaling).
 * 
 * @details Mathematical Formulation
 * Standardizes each feature coordinate independently:
 * \f[
 * y_j = \frac{x_j - \mu_j}{\sigma_j}
 * \f]
 */
class EllipticScaler : public Scaler {
  private:
    Eigen::VectorXd mean_;
    Eigen::VectorXd std_;

  public:
    EllipticScaler() = default;

    /**
     * @brief Constructs an EllipticScaler with specified mean and standard deviations.
     */
    EllipticScaler(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::VectorXd> &std) : mean_(mean), std_(std) {};
    EllipticScaler(const EllipticScaler &other) = default;
    EllipticScaler(EllipticScaler &&other) = default;

    ~EllipticScaler() = default;

    EllipticScaler &operator=(const EllipticScaler &other) = default;
    EllipticScaler &operator=(EllipticScaler &&other) = default;

    /**
     * @brief Standardizes the input data vector.
     */
    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    /**
     * @brief Unstandardizes the input data vector back to physical space.
     */
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

    /**
     * @brief Fits the mean and standard deviation of each coordinate of the training data.
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) override;

    /**
     * @brief Fits the scaler and returns standardized data.
     */
    Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) override;

    /**
     * @brief Explicitly sets the mean parameters.
     */
    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    };

    /**
     * @brief Explicitly sets the standard deviation scaling parameters.
     */
    void setStd(const Eigen::Ref<const Eigen::VectorXd> &std) {
        std_ = std;
    };
};

/**
 * @class MinMaxScaler
 * @brief Linearly scales features to a target bounding box range.
 * 
 * @details Mathematical Formulation
 * Scales each feature vector component to live within range \f$[a_j, b_j]\f$ (typically \f$[0, 1]\f$):
 * \f[
 * y_j = a_j + \frac{x_j - x_{\text{min}, j}}{x_{\text{max}, j} - x_{\text{min}, j}} (b_j - a_j)
 * \f]
 * where \f$x_{\text{min}, j}\f$ and \f$x_{\text{max}, j}\f$ are the training data's minimum and maximum values for feature \f$j\f$.
 * 
 * @details Implementation Algorithm
 * 1. `fit()` computes column-wise minima and maxima vectors of the training matrix.
 * 2. `transform()` and `inverseTransform()` apply element-wise division and scaling multiplication.
 */
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


/** @} */

#endif // SCALER_H