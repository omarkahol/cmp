#ifndef MEANPP_H
#define MEANPP_H

#include "cmp_defines.h"
#include <memory>

/**
 * @addtogroup probability
 * @{
 */
namespace cmp::mean {
/**
 * @brief Abstract base class for Gaussian Process prior mean functions.
 * 
 * @details Mathematical Formulation
 * The mean function \f$m: \mathbb{R}^D \to \mathbb{R}\f$ defines the expected value of the GP prior at any input \f$\mathbf{x}\f$:
 * \f[
 * m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]
 * \f]
 * Commonly configured as a constant \f$m(\mathbf{x}) = c\f$ or zero \f$m(\mathbf{x}) = 0\f$.
 * 
 * @details Implementation Algorithm
 * Provides virtual interfaces for mean evaluations (`eval`), first-order gradients (`evalGradient`) with respect to a parameter \f$p_i\f$, and second-order Hessians (`evalHessian`).
 */
class Mean {
  public:
    virtual ~Mean() = default;
    virtual double eval(const Eigen::VectorXd &x, const Eigen::VectorXd &par) const = 0;
    virtual double evalGradient(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const size_t &i) const = 0;
    virtual double evalHessian(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const size_t &i, const size_t &j) const = 0;
};

/**
 * @brief Represents a constant mean function.
 * 
 * @details Mathematical Formulation
 * The mean function returns a constant value from the hyperparameters:
 * \f[
 * m(\mathbf{x}) = \theta_{\text{index}}
 * \f]
 */
class Constant : public Mean {
  private:
    size_t index_;
  public:

    Constant(const Constant&) = default;
    Constant(Constant&&) = default;
    Constant& operator=(const Constant&) = default;
    Constant& operator=(Constant&&) = default;

    /**
     * @brief Constructs a Constant mean function using the hyperparameter at index c.
     */
    Constant(const size_t &c) : index_(c) {};

    /**
     * @brief Evaluates the constant mean.
     */
    double eval(const Eigen::VectorXd& x, const Eigen::VectorXd &par) const {
        return par(index_);
    };

    /**
     * @brief Evaluates the partial derivative of the constant mean function.
     */
    double evalGradient(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i) const {
        if(i == index_) {
            return 1.0;
        } else {
            return 0.0;
        }
    };

    /**
     * @brief Evaluates the second-order partial derivative (Hessian) of the constant mean function.
     */
    double evalHessian(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i, const size_t &j) const {
        return 0;
    };

    /**
     * @brief Factory method for creating a Constant mean function.
     */
    static std::shared_ptr<Mean> make(const size_t &c) {
        return std::make_shared<Constant>(c);
    };

};

/**
 * @brief Represents a constant zero mean function.
 * 
 * @details Mathematical Formulation
 * The mean is zero everywhere:
 * \f[
 * m(\mathbf{x}) = 0
 * \f]
 */
class Zero : public Mean {
  public:

    Zero() = default;
    Zero(const Zero&) = default;
    Zero(Zero&&) = default;
    Zero& operator=(const Zero&) = default;
    Zero& operator=(Zero&&) = default;

    /**
     * @brief Evaluates the zero mean function.
     */
    double eval(const Eigen::VectorXd& x, const Eigen::VectorXd &par) const {
        return 0.0;
    };

    /**
     * @brief Evaluates the partial derivative of the zero mean function.
     */
    double evalGradient(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i) const {
        return 0.0;
    };

    /**
     * @brief Evaluates the second-order partial derivative (Hessian) of the zero mean function.
     */
    double evalHessian(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i, const size_t &j) const {
        return 0.0;
    };

    /**
     * @brief Factory method for creating a Zero mean function.
     */
    static std::shared_ptr<Mean> make() {
        return std::make_shared<Zero>();
    };

};

}

/** @} */

#endif