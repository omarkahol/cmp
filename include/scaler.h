#ifndef CMP_SCALER_H
#define CMP_SCALER_H

#include <cmp_defines.h>

/**
 * @addtogroup surrogate
 * @{
 */
namespace cmp::scaler {

/**
 * @brief Global tolerance value used for numerical stability and zero-checks.
 */
constexpr double TOL = 1e-10;

/**
 * @class Scaler
 * @brief Template virtual base class for feature scaling and transformation.
 * * @details
 * ### Mathematical Formulation
 * Define a transformation mapping \f$ T: \mathbb{R}^d \to \mathbb{R}^d \f$ and its inverse \f$ T^{-1} \f$:
 * \f[ y = T(x) \f]
 * \f[ x = T^{-1}(y) \f]
 * Often represented linearly with intercept \f$ \mu \f$ and scaling factor/matrix \f$ S \f$:
 * \f[ T(x) = S^{-1}(x - \mu) \f]
 * \f[ T^{-1}(y) = S y + \mu \f]
 * * ### Implementation Algorithm
 * Pure virtual interface. Concrete implementations define the fitting, transforming, and inverse transforming steps.
 */
class Scaler {
  public:
    /**
     * @brief Transforms the input data from physical space to the scaled latent space.
     * @param data A column vector \f$ x \f$ representing a single data point in physical space.
     * @return The scaled data vector \f$ y \f$.
     */
    virtual Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const = 0;

    /**
     * @brief Reconstructs the data from the scaled latent space back to physical space.
     * @param data A column vector \f$ y \f$ in the scaled space.
     * @return The unscaled physical data vector \f$ x \f$.
     */
    virtual Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const = 0;

    /**
     * @brief Retrieves the intercept vector \f$ \mu \f$ used in the transformation.
     * @return The intercept vector.
     */
    virtual Eigen::VectorXd getIntercept() const = 0;

    /**
     * @brief Retrieves the scaling matrix \f$ S \f$ used in the transformation.
     * @return The scaling matrix.
     */
    virtual Eigen::MatrixXd getScale() const = 0;

    /**
     * @brief Learns the scaling parameters (e.g., mean, variance) from the training data.
     * @param data A matrix \f$ X \in \mathbb{R}^{n \times d} \f$ of training samples.
     */
    virtual void fit(const Eigen::Ref<const Eigen::MatrixXd> &data) = 0;

    /**
     * @brief Fits the scaling parameters to the data and subsequently transforms the data.
     * @param data A matrix \f$ X \in \mathbb{R}^{n \times d} \f$ of training samples.
     * @return The transformed dataset matrix \f$ Y \f$.
     */
    virtual Eigen::MatrixXd fit_transform(const Eigen::Ref<const Eigen::MatrixXd> &data) = 0;
};

/**
 * @class StandardScaler
 * @brief Standardizes features by removing the mean and scaling to unit variance using Cholesky decomposition.
 * * @details
 * ### Mathematical Formulation
 * Given data matrix \f$ X \in \mathbb{R}^{n \times d} \f$, let \f$ \mu \in \mathbb{R}^d \f$ be the column-wise mean and \f$ \Sigma \in \mathbb{R}^{d \times d} \f$ be the covariance matrix.
 * We decompose the covariance matrix via the lower Cholesky factor \f$ L \f$ such that:
 * \f[ \Sigma = L L^T \f]
 * The transformation maps:
 * \f[ y = L^{-1}(x - \mu) \f]
 * The inverse transformation maps:
 * \f[ x = L y + \mu \f]
 * * ### Implementation Algorithm
 * 1. **Fit**: Calculate mean vector \f$ \mu \f$ and covariance matrix \f$ \Sigma \f$ of \f$ X \f$, then perform Cholesky LLT decomposition to obtain \f$ L \f$.
 * 2. **Transform**: Solve the lower triangular system \f$ L y = x - \mu \f$ to compute standardized data.
 * 3. **Inverse Transform**: Compute \f$ L y + \mu \f$ via matrix multiplication and vector addition.
 */
class StandardScaler : public Scaler {
  private:
    /** @brief The empirical mean vector \f$ \mu \f$ of the training dataset. */
    Eigen::VectorXd mean_;

    /** @brief The LLT (Cholesky) decomposition of the covariance matrix \f$ \Sigma \f$. */
    Eigen::LLT<Eigen::MatrixXd> lltDecomposition_;

  public:
    /** @brief Default constructor. */
    StandardScaler() = default;

    /**
     * @brief Constructs a StandardScaler with a pre-computed mean and covariance scale.
     * @param mean Pre-computed mean vector \f$ \mu \f$.
     * @param scale Pre-computed covariance matrix \f$ \Sigma \f$ (will be decomposed internally).
     */
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
 * * @details
 * ### Mathematical Formulation
 * Centers data matrix \f$\mathbf{X} \in \mathbb{R}^{N \times D}\f$ around column mean \f$\boldsymbol{\mu}\f$ and computes the eigendecomposition of the sample covariance matrix \f$\boldsymbol{\Sigma}\f$:
 * \f[
 * \boldsymbol{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T
 * \f]
 * where \f$\mathbf{V}\f$ contains orthogonal eigenvectors and \f$\mathbf{\Lambda}\f$ contains eigenvalues in descending order.
 * The transformation projects \f$\mathbf{x}\f$ into a lower-dimensional latent space of size \f$M\f$:
 * \f[
 * \mathbf{y} = \mathbf{\Lambda}_M^{-1/2} \mathbf{V}_M^T (\mathbf{x} - \boldsymbol{\mu})
 * \f]
 * The inverse transformation reconstructs the point:
 * \f[
 * \mathbf{x} = \mathbf{V}_M \mathbf{\Lambda}_M^{1/2} \mathbf{y} + \boldsymbol{\mu}
 * \f]
 * * ### Implementation Algorithm
 * 1. `fit()` computes the mean \f$\boldsymbol{\mu}\f$ and covariance \f$\boldsymbol{\Sigma}\f$ of the training matrix, then runs `Eigen::SelfAdjointEigenSolver`.
 * 2. `transform()` centers the query vector, projects it onto the principal eigenvectors, and scales it by the inverse square root of eigenvalues.
 * 3. `inverseTransform()` rescales by square root eigenvalues, projects back to physical coordinates, and adds the mean.
 */
class PCA : public Scaler {
  private:
    /** @brief The empirical mean vector \f$ \boldsymbol{\mu} \f$. */
    Eigen::VectorXd mean_;

    /** @brief Eigen solver storing the eigenvectors \f$ \mathbf{V} \f$ and eigenvalues \f$ \mathbf{\Lambda} \f$. */
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver_;

    /** @brief Precomputed square root of the truncated eigenvalues matrix \f$ \mathbf{\Lambda}_M^{1/2} \f$. */
    Eigen::MatrixXd sqrtCov_;

    /** @brief Precomputed inverse square root of the truncated eigenvalues matrix \f$ \mathbf{\Lambda}_M^{-1/2} \f$. */
    Eigen::MatrixXd sqrtCovInv_;

    /** @brief The number of principal components \f$ M \f$ to retain. */
    size_t nComponents_;

    /**
     * @brief Internal helper to execute the eigendecomposition on the current covariance matrix.
     */
    void eigenDecomposition();

  public:
    /**
     * @brief Constructs a PCA scaler targeting a specific latent dimensionality.
     * @param nComponents Number of principal components \f$ M \f$ to retain.
     */
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

    /**
     * @brief Resizes the number of retained principal components \f$ M \f$ after fitting.
     * @param nComponents The new number of components to retain.
     */
    void resize(size_t nComponents);

    /**
     * @brief Returns the eigenvalues \f$ \mathbf{\Lambda} \f$ of the covariance matrix.
     * @return A vector of eigenvalues.
     */
    Eigen::VectorXd getEigenvalues() const {
        return eigenSolver_.eigenvalues();
    };

    /**
     * @brief Returns the eigenvectors \f$ \mathbf{V} \f$ of the covariance matrix.
     * @return A matrix whose columns are the eigenvectors.
     */
    Eigen::MatrixXd getEigenvectors() const {
        return eigenSolver_.eigenvectors();
    };
};

/**
 * @class DummyScaler
 * @brief A pass-through scaler that leaves input data unchanged.
 * * @details
 * ### Mathematical Formulation
 * The transformation is the identity function mapping \f$ \mathbb{R}^d \to \mathbb{R}^d \f$:
 * \f[ T(x) = x \f]
 * \f[ T^{-1}(y) = y \f]
 * The intercept \f$ \mu \f$ is the zero vector, and the scale \f$ S \f$ is the identity matrix \f$ I \f$.
 */
class DummyScaler : public Scaler {
  private:
    /** @brief The dimensionality \f$ d \f$ of the data vectors. */
    size_t dim_;

  public:
    /** @brief Default constructor. */
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

    /**
     * @brief Explicitly sets the data dimensionality.
     * @param dim The dimension \f$ d \f$.
     */
    void setDim(size_t dim) {
        dim_ = dim;
    };
};

/**
 * @class EllipticScaler
 * @brief Scaler that standardizes features independently using diagonal variance scaling.
 * * @details
 * ### Mathematical Formulation
 * Standardizes each feature coordinate \f$ j \f$ independently:
 * \f[
 * y_j = \frac{x_j - \mu_j}{\sigma_j}
 * \f]
 * where \f$ \mu_j \f$ is the mean and \f$ \sigma_j \f$ is the standard deviation of feature \f$ j \f$.
 * The inverse transformation is:
 * \f[
 * x_j = y_j \sigma_j + \mu_j
 * \f]
 */
class EllipticScaler : public Scaler {
  private:
    /** @brief The empirical mean vector \f$ \boldsymbol{\mu} \f$ of the training dataset. */
    Eigen::VectorXd mean_;

    /** @brief The standard deviation vector \f$ \boldsymbol{\sigma} \f$ for each feature. */
    Eigen::VectorXd std_;

  public:
    /** @brief Default constructor. */
    EllipticScaler() = default;

    /**
     * @brief Constructs an EllipticScaler with specified mean and standard deviations.
     * @param mean Pre-computed mean vector \f$ \boldsymbol{\mu} \f$.
     * @param std Pre-computed standard deviation vector \f$ \boldsymbol{\sigma} \f$.
     */
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

    /**
     * @brief Explicitly overrides the mean parameters \f$ \boldsymbol{\mu} \f$.
     * @param mean The new mean vector.
     */
    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    };

    /**
     * @brief Explicitly overrides the standard deviation parameters \f$ \boldsymbol{\sigma} \f$.
     * @param std The new standard deviation vector.
     */
    void setStd(const Eigen::Ref<const Eigen::VectorXd> &std) {
        std_ = std;
    };
};

/**
 * @class MinMaxScaler
 * @brief Linearly scales features to a target bounding box range.
 * * @details
 * ### Mathematical Formulation
 * Scales each feature vector component to live within a target range \f$[a_j, b_j]\f$ (typically \f$[0, 1]\f$):
 * \f[
 * y_j = a_j + \frac{x_j - x_{\text{min}, j}}{x_{\text{max}, j} - x_{\text{min}, j}} (b_j - a_j)
 * \f]
 * where \f$x_{\text{min}, j}\f$ and \f$x_{\text{max}, j}\f$ are the training data's empirical minimum and maximum values for feature \f$j\f$.
 * * ### Implementation Algorithm
 * 1. `fit()` computes column-wise minima and maxima vectors of the training matrix.
 * 2. `transform()` and `inverseTransform()` apply element-wise division and scaling multiplication.
 */
class MinMaxScaler : public Scaler {
  private:
    /** @brief Target minimum value bounding vector \f$ \mathbf{a} \f$. */
    Eigen::VectorXd min_;

    /** @brief Target maximum value bounding vector \f$ \mathbf{b} \f$. */
    Eigen::VectorXd max_;

    /** @brief Empirical minimum value vector \f$ \mathbf{x}_{\text{min}} \f$ observed in the training dataset. */
    Eigen::VectorXd data_min_;

    /** @brief Empirical maximum value vector \f$ \mathbf{x}_{\text{max}} \f$ observed in the training dataset. */
    Eigen::VectorXd data_max_;

  public:
    /** @brief Default constructor. */
    MinMaxScaler() = default;

    /**
     * @brief Constructs a MinMaxScaler bounded to a specific target range.
     * @param min The target minimum bounding vector \f$ \mathbf{a} \f$.
     * @param max The target maximum bounding vector \f$ \mathbf{b} \f$.
     */
    MinMaxScaler(const Eigen::Ref<const Eigen::VectorXd> &min, const Eigen::Ref<const Eigen::VectorXd> &max) : min_(min), max_(max) {};

    MinMaxScaler(const MinMaxScaler &other) = default;
    MinMaxScaler(MinMaxScaler &&other) = default;
    ~MinMaxScaler() = default;
    MinMaxScaler &operator=(const MinMaxScaler &other) = default;
    MinMaxScaler &operator=(MinMaxScaler &&other) = default;

    Eigen::VectorXd transform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;
    Eigen::VectorXd inverseTransform(const Eigen::Ref<const Eigen::VectorXd> &data) const override;

    /**
     * @brief Retrieves the empirical minimum vector \f$ \mathbf{x}_{\text{min}} \f$ found during fitting.
     * @return The minimum data vector.
     */
    Eigen::VectorXd getDataMin() const {
        return data_min_;
    };

    /**
     * @brief Retrieves the empirical maximum vector \f$ \mathbf{x}_{\text{max}} \f$ found during fitting.
     * @return The maximum data vector.
     */
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