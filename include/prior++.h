#ifndef PRIORPP_H
#define PRIORPP_H

#include <cmp_defines.h>
#include <distribution.h>

/**
 * @addtogroup probability
 * @{
 */
namespace cmp::prior {
/**
 * @class Prior
 * @brief Base class for prior probability distributions.
 *
 * @details \b Mathematical \b Formulation
 * Represents a log-prior probability density function \f$ \log p(\theta) \f$ over parameters \f$ \theta \in \mathbb{R}^d \f$.
 * Provides virtual interfaces for evaluation, gradient, and Hessian computation:
 * - Value: \f$ f(\theta) = \log p(\theta) \f$
 * - Gradient: \f$ \nabla_i f(\theta) = \frac{\partial \log p(\theta)}{\partial \theta_i} \f$
 * - Hessian: \f$ \mathcal{H}_{ij} f(\theta) = \frac{\partial^2 \log p(\theta)}{\partial \theta_i \partial \theta_j} \f$
 *
 * @details \b Implementation \b Algorithm
 * Pure virtual interface specifying the contract for prior evaluation. Concrete subclasses must implement `eval`, `evalGradient`, and `evalHessian`.
 */
class Prior {
  public:
    virtual ~Prior() = default;
    virtual double eval(const Eigen::VectorXd &par) const = 0;
    virtual double evalGradient(const Eigen::VectorXd &par, const size_t &i) const = 0;
    virtual double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const = 0;
};

/**
 * @class Product
 * @brief Represents the product (sum of logs) of two independent prior distributions.
 *
 * @details \b Mathematical \b Formulation
 * For independent priors \f$ p_1(\theta) \f$ and \f$ p_2(\theta) \f$, the joint prior is:
 * \f[ \log p(\theta) = \log p_1(\theta) + \log p_2(\theta) \f]
 * Derivatives are additive due to the linearity of differentiation:
 * \f[ \nabla_i \log p(\theta) = \nabla_i \log p_1(\theta) + \nabla_i \log p_2(\theta) \f]
 * \f[ \mathcal{H}_{ij} \log p(\theta) = \mathcal{H}_{ij} \log p_1(\theta) + \mathcal{H}_{ij} \log p_2(\theta) \f]
 *
 * @details \b Implementation \b Algorithm
 * Evaluates the log-prior, gradient, and Hessian by summing the results of the `leftPrior_` and `rightPrior_` objects.
 */
class Product : public Prior {
  private:
    std::shared_ptr<Prior> leftPrior_;  ///< Left factor independent prior.
    std::shared_ptr<Prior> rightPrior_; ///< Right factor independent prior.
  public:

    Product() = default;
    Product(const Product&) = default;
    Product(Product&&) = default;
    Product& operator=(const Product&) = default;
    Product& operator=(Product&&) = default;

    Product(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2) : leftPrior_(p1), rightPrior_(p2) {};

    double eval(const Eigen::VectorXd &par) const {
        return leftPrior_->eval(par) + rightPrior_->eval(par);
    }

    double evalGradient(const Eigen::VectorXd &par, const size_t &i) const {
        return leftPrior_->evalGradient(par, i) + rightPrior_->evalGradient(par, i);
    }

    double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const {
        return leftPrior_->evalHessian(par, i, j) + rightPrior_->evalHessian(par, i, j);
    }

    static std::shared_ptr<Prior> make(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2) {
        return std::make_shared<Product>(p1, p2);
    }
};

/**
 * @class Uniform
 * @brief Represents an improper flat (uniform) prior.
 *
 * @details \b Mathematical \b Formulation
 * A uniform prior assumes constant probability density:
 * \f[ p(\theta) \propto 1 \implies \log p(\theta) = C \f]
 * For convenience, the constant \f$ C \f$ is set to 0. All derivatives are zero:
 * \f[ \nabla_i \log p(\theta) = 0 \f]
 * \f[ \mathcal{H}_{ij} \log p(\theta) = 0 \f]
 *
 * @details \b Implementation \b Algorithm
 * Returns constant 0.0 for all evaluation, gradient, and Hessian queries.
 */
class Uniform : public Prior {
  public:
    Uniform(const Uniform&) = default;
    Uniform(Uniform&&) = default;
    Uniform& operator=(const Uniform&) = default;
    Uniform& operator=(Uniform&&) = default;

    Uniform() = default;

    double eval(const Eigen::VectorXd &par) const {
        return 0;
    }

    double evalGradient(const Eigen::VectorXd &par, const size_t &i) const {
        return 0;
    }

    double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const {
        return 0;
    }

    static std::shared_ptr<Prior> make() {
        return std::make_shared<Uniform>();
    }
};

/**
 * @class FromDistribution
 * @brief Adapts a univariate probability distribution to act as a prior on a single coordinate.
 *
 * @details \b Mathematical \b Formulation
 * Applies a univariate distribution \f$ \mathcal{D} \f$ to a specific parameter dimension \f$ k \f$:
 * \f[ \log p(\theta) = \log p_{\mathcal{D}}(\theta_k) \f]
 * The gradient is non-zero only for coordinate \f$ k \f$:
 * \f[ \frac{\partial \log p(\theta)}{\partial \theta_i} = \begin{cases} \frac{d \log p_{\mathcal{D}}(\theta_k)}{d \theta_k} & \text{if } i = k \\ 0 & \text{otherwise} \end{cases} \f]
 * The Hessian is non-zero only for diagonal entry \f$ (k, k) \f$:
 * \f[ \frac{\partial^2 \log p(\theta)}{\partial \theta_i \partial \theta_j} = \begin{cases} \frac{d^2 \log p_{\mathcal{D}}(\theta_k)}{d \theta_k^2} & \text{if } i = j = k \\ 0 & \text{otherwise} \end{cases} \f]
 *
 * @details \b Implementation \b Algorithm
 * Accesses parameter coordinate \f$ \theta_k \f$ and delegates log-PDF and its first/second derivative calculations to the underlying `dist_` object.
 */
template <typename DistType>
class FromDistribution : public Prior {
  private:
    DistType dist_;      ///< Underlying univariate distribution.
    size_t paramIndex_;  ///< Parameter vector coordinate index.

  public:
// Delete default constructor since we need a valid distribution
    FromDistribution() = delete;

    FromDistribution(const FromDistribution&) = default;
    FromDistribution(FromDistribution&&) = default;
    FromDistribution& operator=(const FromDistribution&) = default;
    FromDistribution& operator=(FromDistribution&&) = default;

    /**
     * @brief Constructs a Prior from a Univariate Distribution
     * @param dist The distribution instance (e.g., NormalDistribution)
     * @param paramIndex The index in the Eigen::VectorXd this prior applies to
     */
    FromDistribution(const DistType& dist, size_t paramIndex)
        : dist_(dist), paramIndex_(paramIndex) {}

    double eval(const Eigen::VectorXd &par) const override {
        // Ensure index is within bounds (optional, but safe)
        if(paramIndex_ >= par.size()) {
            throw std::out_of_range("Parameter index out of bounds in FromDistribution::eval");
        }
        return dist_.logPDF(par(paramIndex_));
    }

    double evalGradient(const Eigen::VectorXd &par, const size_t &i) const override {
        if(i == paramIndex_) {
            return dist_.dLogPDF(par(paramIndex_));
        }
        // If taking the gradient with respect to a different parameter, it's 0
        return 0.0;
    }

    double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const override {
        if(i == paramIndex_ && j == paramIndex_) {
            return dist_.ddLogPDF(par(paramIndex_));
        }
        // Cross-derivatives and derivatives for other parameters are 0
        return 0.0;
    }

// Factory method matching your other Prior classes
    static std::shared_ptr<Prior> make(const DistType& dist, size_t paramIndex) {
        return std::make_shared<FromDistribution<DistType>>(dist, paramIndex);
    }
};

/**
 * @brief Helper template function to create a FromDistribution prior.
 *
 * @tparam DistType The type of the univariate distribution.
 * @param dist The distribution.
 * @param paramIndex The index of the parameter coordinate.
 * @return Shared pointer to the created Prior.
 */
template <typename DistType>
std::shared_ptr<Prior> make(const DistType& dist, size_t paramIndex) {
    return FromDistribution<DistType>::make(dist, paramIndex);
}

/**
 * @brief Multiplies two prior distributions (returns a Product prior).
 *
 * @param p1 First prior distribution.
 * @param p2 Second prior distribution.
 * @return Shared pointer to the Product prior.
 */
std::shared_ptr<Prior> operator*(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2);

}


/** @} */

#endif