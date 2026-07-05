#ifndef CMP_KDE_H
#define CMP_KDE_H

#include <cmp_defines.h>
#include <statistics.h>
#include <kernel++.h>
#include <optimization.h>
#include <distribution.h>


/**
 * @addtogroup probability
 * @{
 */
namespace cmp::density {

// Here we implement a KDE class that uses the above bandwidth and kernel classes
/**
 * @brief Multivariate Kernel Density Estimation (KDE) class.
 * 
 * @details Mathematical Formulation
 * Evaluates the multivariate probability density function estimate of a random vector \f$\mathbf{x} \in \mathbb{R}^D\f$ based on observations \f$\{\mathbf{x}_i\}_{i=1}^N\f$:
 * \f[
 * \hat{f}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N \frac{1}{\det(\mathbf{H})} K\left( \mathbf{H}^{-1} (\mathbf{x} - \mathbf{x}_i) \right)
 * \f]
 * where \f$\mathbf{H}\f$ is the bandwidth matrix and \f$K\f$ is the multi-dimensional kernel function.
 * - Scott's Rule: \f$H_{dd} = N^{-1/(D+4)} \hat{\sigma}_d\f$
 * - Silverman's Rule: \f$H_{dd} = \left(\frac{4}{D+2}\right)^{1/(D+4)} N^{-1/(D+4)} \hat{\sigma}_d\f$
 * where \f$\hat{\sigma}_d\f$ is the standard deviation of coordinate \f$d\f$.
 * 
 * @details Implementation Algorithm
 * 1. `condition()` registers the dataset references.
 * 2. `eval()` averages kernel evaluations over all conditioning observation points, scaled by the determinant of the bandwidth matrix.
 */
class KDE {
  private:
    std::shared_ptr<kernel::Bandwidth> bandwidth_{nullptr};
    std::shared_ptr<kernel::Kernel> kernel_{nullptr};

    // Owning or non-owning data storage
    Eigen::MatrixXd data_{0, 0};
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> pData_;
    bool isOwning_{false};

    // Dimension and number of points
    size_t dim_{0};
    size_t nPoints_{0};
  public:

    /**
     * @brief Default constructor for KDE.
     * Creates a KDE with identity bandwidth and Gaussian kernel in 1D.
     */
    KDE();

    /**
     * @brief Constructor for KDE with specified bandwidth and kernel.
     * @param bandwidth Shared pointer to the bandwidth object.
     * @param kernel Shared pointer to the kernel object.
     */
    KDE(std::shared_ptr<kernel::Bandwidth> bandwidth, std::shared_ptr<kernel::Kernel> kernel);

    // Copy and move constructors
    KDE(const KDE &other);
    KDE(KDE &&other) noexcept;

    // Copy and move assignment operators
    KDE &operator=(const KDE &other);
    KDE &operator=(KDE &&other) noexcept;

    /**
     * @brief Set the bandwidth and kernel for the KDE.
     * @param bandwidth Shared pointer to the bandwidth object.
     * @param kernel Shared pointer to the kernel object.
     */
    void set(const std::shared_ptr<kernel::Bandwidth> &bandwidth, const std::shared_ptr<kernel::Kernel> &kernel);

    /**
     * @brief Condition the KDE on the provided data.
     * @param data The data matrix where each row is a data point.
     * @param copyData If true, the data is copied into the KDE object; if false, a reference is kept.
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &data, bool copyData = false);

    /**
     * @brief Evaluate the KDE at a given point.
     * @param x The point at which to evaluate the density estimate.
     * @return The estimated density at point x.
     */
    double eval(const Eigen::VectorXd &x) const;

  public:
    enum class BandWidthSelectionMethod {
        SCOTT,
        SILVERMAN
    };

};

/**
 * @brief Computes the bandwidth matrix using standard rules of thumb.
 * 
 * @param method Selection rule (SCOTT or SILVERMAN).
 * @param xObs Observation data matrix.
 * @return Calculated bandwidth matrix.
 */
Eigen::MatrixXd bandwidthSelectionRule(const KDE::BandWidthSelectionMethod &method, Eigen::Ref<Eigen::MatrixXd> xObs);

/**
 * @brief Optimizes the bandwidth matrix using K-Fold cross-validation over the observations.
 * 
 * @param kf KFold cross-validation partitioning helper.
 * @param xObs Observation data matrix.
 * @param kernel The density kernel to use.
 * @param initialBandwidth Input/output bandwidth object to optimize.
 * @param min Minimum bounds for bandwidth parameters.
 * @param max Maximum bounds for bandwidth parameters.
 * @param alg The optimization algorithm.
 * @param tol Optimization relative tolerance.
 */
void bandwidthOptimizationCrossValidation(const cmp::statistics::KFold& kf, Eigen::Ref<Eigen::MatrixXd> xObs, std::shared_ptr<cmp::kernel::Kernel> kernel, std::shared_ptr<cmp::kernel::Bandwidth> initialBandwidth, const double &min, const double &max, nlopt::algorithm alg, const double & tol = 1e-4);



} // namespace cmp::density

/** @} */

#endif // CMP_KDE_H