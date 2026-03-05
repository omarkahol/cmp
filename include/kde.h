#ifndef CMP_KDE_H
#define CMP_KDE_H

#include <cmp_defines.h>
#include <statistics.h>
#include <kernel++.h>
#include <optimization.h>
#include <distribution.h>


namespace cmp::density {

// Here we implement a KDE class that uses the above bandwidth and kernel classes
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

Eigen::MatrixXd bandwidthSelectionRule(const KDE::BandWidthSelectionMethod &method, Eigen::Ref<Eigen::MatrixXd> xObs);

void bandwidthOptimizationCrossValidation(const cmp::statistics::KFold& kf, Eigen::Ref<Eigen::MatrixXd> xObs, std::shared_ptr<cmp::kernel::Kernel> kernel, std::shared_ptr<cmp::kernel::Bandwidth> initialBandwidth, const double &min, const double &max, nlopt::algorithm alg, const double & tol = 1e-4);



} // namespace cmp::density

#endif // CMP_KDE_H