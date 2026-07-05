#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <cmp_defines.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/bessel.hpp>

/**
 * @addtogroup probability
 * @{
 */
namespace cmp::covariance {
/**
 * @brief Abstract base class for all covariance (kernel) functions.
 * 
 * @details Mathematical Formulation
 * A covariance function (kernel) \f$k: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}\f$ defines the covariance between GP values at any two inputs \f$\mathbf{x}_1, \mathbf{x}_2\f$:
 * \f[
 * \text{Cov}(f(\mathbf{x}_1), f(\mathbf{x}_2)) = k(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta})
 * \f]
 * where \f$\boldsymbol{\theta}\f$ is the vector of kernel hyperparameters. The function must be symmetric and positive semi-definite:
 * \f[
 * \sum_{i=1}^n \sum_{j=1}^n c_i c_j k(\mathbf{x}_i, \mathbf{x}_j) \ge 0 \quad \forall c_i \in \mathbb{R}
 * \f]
 * 
 * @details Implementation Algorithm
 * Provides a virtual interface for evaluating the covariance value (`eval`), its first-order gradient (`evalGradient`) with respect to a hyperparameter \f$\theta_i\f$, and its second-order Hessian (`evalHessian`) with respect to hyperparameters \f$\theta_i, \theta_j\f$.
 */
class Covariance {
  public:
    virtual ~Covariance() = default;
    virtual double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const = 0;
    virtual double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const = 0;
    virtual double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const = 0;
};

/**
 * @brief Represents the sum of two covariance functions.
 * 
 * @details Mathematical Formulation
 * For two covariance kernels \f$k_1\f$ and \f$k_2\f$, the sum kernel is:
 * \f[
 * k_{\text{sum}}(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta}) = k_1(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta}) + k_2(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta})
 * \f]
 */
class Sum : public Covariance {
  private:
    std::shared_ptr<Covariance> leftCovariance_;  ///< Left operand covariance function.
    std::shared_ptr<Covariance> rightCovariance_; ///< Right operand covariance function.
  public:

    Sum() = default;
    Sum(const Sum&) = default;
    Sum(Sum&&) = default;
    Sum& operator=(const Sum&) = default;
    Sum& operator=(Sum&&) = default;

    /**
     * @brief Constructs a sum covariance function from two kernels.
     */
    Sum(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) : leftCovariance_(k1), rightCovariance_(k2) {};

    /**
     * @brief Evaluates the sum kernel.
     */
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return leftCovariance_->eval(x1, x2, par) + rightCovariance_->eval(x1, x2, par);
    }

    /**
     * @brief Evaluates the partial derivative of the sum kernel.
     */
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return leftCovariance_->evalGradient(x1, x2, par, i) + rightCovariance_->evalGradient(x1, x2, par, i);
    }

    /**
     * @brief Evaluates the second-order partial derivative of the sum kernel.
     */
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return leftCovariance_->evalHessian(x1, x2, par, i, j) + rightCovariance_->evalHessian(x1, x2, par, i, j);
    }

    /**
     * @brief Factory method to create a shared pointer to a sum covariance.
     */
    static std::shared_ptr<Covariance> make(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) {
        return std::make_shared<Sum>(k1, k2);
    };
};

/**
 * @brief Represents the product of two covariance functions.
 * 
 * @details Mathematical Formulation
 * For two covariance kernels \f$k_1\f$ and \f$k_2\f$, the product kernel is:
 * \f[
 * k_{\text{prod}}(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta}) = k_1(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta}) \times k_2(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta})
 * \f]
 */
class Product : public Covariance {
  private:
    std::shared_ptr<Covariance> leftCovariance_;  ///< Left operand covariance function.
    std::shared_ptr<Covariance> rightCovariance_; ///< Right operand covariance function.

  public:

    Product(const Product&) = default;
    Product(Product&&) = default;
    Product& operator=(const Product&) = default;
    Product& operator=(Product&&) = default;

    /**
     * @brief Constructs a product covariance function from two kernels.
     */
    Product(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) : leftCovariance_(k1), rightCovariance_(k2) {};

    /**
     * @brief Evaluates the product kernel.
     */
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return leftCovariance_->eval(x1, x2, par) * rightCovariance_->eval(x1, x2, par);
    }

    /**
     * @brief Evaluates the partial derivative of the product kernel.
     */
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {

        // Smart eval for first term
        double d1 = leftCovariance_->evalGradient(x1, x2, par, i);
        if(std::abs(d1) < TOL) {
        } else {
            d1 *= rightCovariance_->eval(x1, x2, par);
        }

        // Smart eval for second term
        double d2 = rightCovariance_->evalGradient(x1, x2, par, i);
        if(std::abs(d2) < TOL) {
        } else {
            d2 *= leftCovariance_->eval(x1, x2, par);
        }

        // Return the result
        return d1 + d2;
    }

    /**
     * @brief Evaluates the second-order partial derivative of the product kernel.
     */
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {

        // Smart eval of the first term
        double d1 = leftCovariance_->evalGradient(x1, x2, par, i);
        if(std::abs(d1) < TOL) {
        } else {
            d1 = (d1 + leftCovariance_->evalHessian(x1, x2, par, i, j)) * rightCovariance_->eval(x1, x2, par);
        }

        // Smart eval for second term
        double d2 = rightCovariance_->evalGradient(x1, x2, par, i);
        if(std::abs(d2) < TOL) {
        } else {
            d2 = (d2 + rightCovariance_->evalHessian(x1, x2, par, i, j)) * leftCovariance_->eval(x1, x2, par);
        }

        // Smart eval for the third term
        return d1 + d2;
    }

    /**
     * @brief Factory method to create a shared pointer to a product covariance.
     */
    static std::shared_ptr<Covariance> make(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) {
        return std::make_shared<Product>(k1, k2);
    };
};

/**
 * @brief Represents a custom user-defined covariance function using std::function wrappers.
 */
class Custom : public Covariance {
  private:
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval_;                                                 ///< Custom kernel evaluation function.
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient_;                           ///< Custom gradient evaluation function.
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian_;            ///< Custom Hessian evaluation function.
  public:

    Custom(const Custom&) = default;
    Custom(Custom&&) = default;
    Custom& operator=(const Custom&) = default;
    Custom& operator=(Custom&&) = default;

    /**
     * @brief Constructs a custom covariance kernel with evaluation, gradient, and Hessian functions.
     */
    Custom(std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval,
           std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient,
           std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian) : eval_(eval), evalGradient_(evalGradient), evalHessian_(evalHessian) {};
    
    /**
     * @brief Evaluates the custom kernel.
     */
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return eval_(x1, x2, par);
    }

    /**
     * @brief Evaluates the custom kernel's gradient.
     */
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return evalGradient_(x1, x2, par, i);
    }

    /**
     * @brief Evaluates the custom kernel's Hessian.
     */
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return evalHessian_(x1, x2, par, i, j);
    }

    /**
     * @brief Factory method for creating a shared pointer to a custom covariance.
     */
    std::shared_ptr<Custom> make(std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval,
                                 std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient,
                                 std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian) {
        return std::make_shared<Custom>(eval, evalGradient, evalHessian);
    };
};

/**
 * @brief Represents a constant scale covariance function.
 * 
 * @details Mathematical Formulation
 * For a constant kernel, the covariance is independent of the input points:
 * \f[
 * k_{\text{const}}(\mathbf{x}_1, \mathbf{x}_2; \boldsymbol{\theta}) = \theta_{\text{index}}^2
 * \f]
 */
class Constant : public Covariance {
  private:
    size_t index_; ///< Hyperparameter parameter index.
  public:

    Constant(const Constant&) = default;
    Constant(Constant&&) = default;
    Constant& operator=(const Constant&) = default;
    Constant& operator=(Constant&&) = default;

    /**
     * @brief Constructs a Constant covariance function using the hyperparameter at index.
     */
    Constant(const size_t &index) : index_(index) {};

    /**
     * @brief Evaluates the Constant kernel.
     */
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return std::pow(par(index_), 2);
    };

    /**
     * @brief Evaluates the partial derivative of the Constant kernel.
     */
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        if(i == index_) {
            return 2 * par(index_);
        } else {
            return 0;
        }
    };

    /**
     * @brief Evaluates the second-order partial derivative of the Constant kernel.
     */
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        if(i == index_ && j == index_) {
            return 2;
        } else {
            return 0;
        }
    };

    /**
     * @brief Factory method for creating a constant covariance.
     */
    static std::shared_ptr<Covariance> make(const size_t &c) {
        return std::make_shared<Constant>(c);
    };

};

/**
 * @brief Represents a Linear covariance function.
 * 
 * @details Mathematical Formulation
 * If `indexX_` is -1, computes the full dot product:
 * \f[
 * k_{\text{lin}}(\mathbf{x}_1, \mathbf{x}_2) = \mathbf{x}_1^T \mathbf{x}_2
 * \f]
 * Otherwise, evaluates only for the coordinate at indexX_:
 * \f[
 * k_{\text{lin}}(\mathbf{x}_1, \mathbf{x}_2) = x_{1,\text{index}} \times x_{2,\text{index}}
 * \f]
 */
class Linear : public Covariance {
  private:
    int indexX_; ///< Coordinate index to project (-1 for full vector inner product).
  public:

    Linear(const Linear&) = default;
    Linear(Linear&&) = default;
    Linear& operator=(const Linear&) = default;
    Linear& operator=(Linear&&) = default;

    /**
     * @brief Constructs a Linear covariance function.
     * @param indexX Dimension index to evaluate, or -1 for the full inner product.
     */
    Linear(const int &indexX = -1) : indexX_(indexX) {};

    /**
     * @brief Evaluates the Linear kernel.
     */
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        if(indexX_ == -1) {
            return x1.dot(x2);
        } else {
            return x1(indexX_) * x2(indexX_);
        }
    };

    /**
     * @brief Evaluates the partial derivative of the Linear kernel.
     */
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return 0;
    };

    /**
     * @brief Evaluates the second-order partial derivative of the Linear kernel.
     */
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return 0;
    };

    /**
     * @brief Factory method for creating a linear covariance.
     */
    static std::shared_ptr<Covariance> make(const int &i) {
        return std::make_shared<Linear>(i);
    };

};

/**
 * @brief Represents an Inverse covariance function.
 * 
 * @details Mathematical Formulation
 * If `indexX_` is -1, computes:
 * \f[
 * k_{\text{inv}}(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\mathbf{x}_1^T \mathbf{x}_2 + 1}
 * \f]
 * Otherwise, evaluates only for the coordinate at indexX_:
 * \f[
 * k_{\text{inv}}(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{x_{1,\text{index}} \times x_{2,\text{index}} + 1}
 * \f]
 */
class Inverse : public Covariance {
  private:
    int indexX_; ///< Coordinate index to project (-1 for full vector dot product).
  public:

    Inverse(const Inverse&) = default;
    Inverse(Inverse&&) = default;
    Inverse& operator=(const Inverse&) = default;
    Inverse& operator=(Inverse&&) = default;

    /**
     * @brief Constructs an Inverse covariance function.
     * @param indexX Dimension index to evaluate, or -1 for the full dot product.
     */
    Inverse(const int &indexX = -1) : indexX_(indexX) {};

    /**
     * @brief Evaluates the Inverse kernel.
     */
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        if(indexX_ == -1) {
            return 1.0 / (x1.dot(x2) + 1);
        } else {
            return 1.0 / (x1(indexX_) * x2(indexX_) + 1);
        }
    };

    /**
     * @brief Evaluates the partial derivative of the Inverse kernel.
     */
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return 0;
    };

    /**
     * @brief Evaluates the second-order partial derivative of the Inverse kernel.
     */
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return 0;
    };

    /**
     * @brief Factory method for creating an inverse covariance.
     */
    static std::shared_ptr<Covariance> make(const int &i) {
        return std::make_shared<Inverse>(i);
    };

};

/**
 * @brief Squared Exponential (RBF / Gaussian) covariance function.
 * 
 * @details Mathematical Formulation
 * Evaluates the squared exponential kernel between two inputs \f$\mathbf{x}_1, \mathbf{x}_2\f$:
 * \f[
 * k(\mathbf{x}_1, \mathbf{x}_2) = \exp\left( -\frac{1}{2} \left(\frac{d}{\ell}\right)^2 \right)
 * \f]
 * where \f$d\f$ is the distance (Euclidean norm \f$\|\mathbf{x}_1 - \mathbf{x}_2\|_2\f$ if isotropic, or absolute coordinate distance \f$|x_{1,i} - x_{2,i}|\f$ if restricted to dimension \f$i\f$), and \f$\ell = \theta_{\text{index}}\f$ is the lengthscale hyperparameter.
 * 
 * @details Implementation Algorithm
 * 1. Computes the distance \f$d\f$ based on the component index configuration.
 * 2. Computes the exponential kernel value.
 * 3. Evaluates analytical first-order and second-order derivatives with respect to the lengthscale \f$\ell\f$.
 */
class SquaredExponential : public Covariance {
  private:
    size_t index_;  ///< Hyperparameter index for the lengthscale parameter.
    int indexX_;    ///< Dimension index to evaluate, or -1 for the full isotropic kernel.
  public:

    SquaredExponential(const SquaredExponential&) = default;
    SquaredExponential(SquaredExponential&&) = default;
    SquaredExponential& operator=(const SquaredExponential&) = default;
    SquaredExponential& operator=(SquaredExponential&&) = default;

    SquaredExponential(const size_t &index, const int &indexX = -1) : index_(index), indexX_(indexX) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }
        return std::exp(-0.5 * std::pow(d / par(index_), 2));
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }

        // Compute the gradient
        if(i == index_) {
            return std::exp(-0.5 * std::pow(d / par(index_), 2)) * std::pow(d, 2) / std::pow(par(index_), 3);
        } else {
            return 0;
        }
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }

        // Compute the hessian
        if(i == index_ && j == index_) {
            return (std::exp(-0.5 * std::pow(d / par(index_), 2)) * std::pow(d, 2) / std::pow(par(index_), 4)) * (-3 + std::pow(d, 2) / std::pow(par(index_), 3));
        } else {
            return 0;
        }
    };

    static std::shared_ptr<Covariance> make(const size_t &l, const int &i) {
        return std::make_shared<SquaredExponential>(l, i);
    };

};

/**
 * @brief Matérn covariance function with parameter nu = 5/2.
 * 
 * @details Mathematical Formulation
 * Evaluates the Matérn 5/2 kernel between two inputs \f$\mathbf{x}_1, \mathbf{x}_2\f$:
 * \f[
 * k(\mathbf{x}_1, \mathbf{x}_2) = \left( 1 + \frac{\sqrt{5}d}{\ell} + \frac{5d^2}{3\ell^2} \right) \exp\left( -\frac{\sqrt{5}d}{\ell} \right)
 * \f]
 * where \f$d\f$ is the distance (isotropic or coordinate-based) and \f$\ell = \theta_{\text{index}}\f$ is the lengthscale hyperparameter. This kernel yields processes that are twice differentiable.
 * 
 * @details Implementation Algorithm
 * 1. Computes the distance \f$d\f$.
 * 2. Computes the polynomial coefficients and exponential scaling terms.
 * 3. Evaluates analytical first-order and second-order derivatives with respect to \f$\ell\f$.
 */
class Matern52 : public Covariance {
  private:
    size_t index_;  ///< Hyperparameter index for the lengthscale parameter.
    int indexX_;    ///< Dimension index to evaluate, or -1 for the full isotropic kernel.
  public:

    Matern52(const Matern52&) = default;
    Matern52(Matern52&&) = default;
    Matern52& operator=(const Matern52&) = default;
    Matern52& operator=(Matern52&&) = default;

    Matern52(const size_t &l, const int &i = -1) : index_(l), indexX_(i) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }
        double l = par(index_);
        double c_1 = sqrt(5.) * d / l;
        double c_2 = (5. / 3.) * pow(d / l, 2);
        return (1 + c_1 + c_2) * exp(-c_1);
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }

        // Compute the gradient
        if(i == index_) {
            double l = par(index_);
            return (5.*std::pow(d, 2) * (std::sqrt(5) * d + l)) / (3.*std::exp((std::sqrt(5) * d) / l) * std::pow(l, 4));
        } else {
            return 0;
        }
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }

        // Compute the hessian
        if(i == index_ && j == index_) {
            double l = par(index_);
            return (5 * std::pow(d, 2) * (5 * std::pow(d, 2) - 3 * std::sqrt(5) * d * l - 3 * std::pow(l, 2))) / (3.*std::exp(std::sqrt(5) * d / l) * std::pow(l, 6));
        } else {
            return 0;
        }
    };

    static std::shared_ptr<Covariance> make(const size_t &l, const int &i) {
        return std::make_shared<Matern52>(l, i);
    };
};

/**
 * @brief General Matérn covariance function.
 * 
 * @details Mathematical Formulation
 * Evaluates the Matérn covariance function between two inputs \f$\mathbf{x}_1, \mathbf{x}_2\f$:
 * \f[
 * k(\mathbf{x}_1, \mathbf{x}_2) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{\sqrt{2\nu}d}{\ell} \right)^\nu K_\nu\left( \frac{\sqrt{2\nu}d}{\ell} \right)
 * \f]
 * where \f$d\f$ is the distance, \f$\ell = \theta_{\text{index}}\f$ is the lengthscale hyperparameter, \f$\nu\f$ is the smoothness parameter, \f$\Gamma\f$ is the Gamma function, and \f$K_\nu\f$ is the modified Bessel function of the second kind.
 * 
 * @details Implementation Algorithm
 * 1. Computes distance \f$d\f$ and scaling parameter \f$r = \frac{\sqrt{2\nu}d}{\ell}\f$.
 * 2. Evaluates the term using `std::tgamma` and Boost's `boost::math::cyl_bessel_k` implementation of the modified Bessel function.
 * 3. Evaluates analytical derivatives with respect to the lengthscale \f$\ell\f$.
 */
class Matern : public Covariance {
  private:
    size_t index_;  ///< Hyperparameter index for the lengthscale parameter.
    int indexX_;    ///< Dimension index to evaluate, or -1 for the full isotropic kernel.
    double nu_;     ///< Smoothness parameter nu.
  public:

    Matern(const Matern&) = default;
    Matern(Matern&&) = default;
    Matern& operator=(const Matern&) = default;
    Matern& operator=(Matern&&) = default;

    Matern(const size_t &index, const double &nu, const int &indexX = -1) : index_(index), indexX_(indexX), nu_(nu) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }
        double l = par(index_);
        double r = std::sqrt(2 * nu_) * d / l;
        return (1.0 / (std::tgamma(nu_) * std::pow(2, nu_ - 1))) * std::pow(r, nu_) * boost::math::cyl_bessel_k(nu_, r);
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }

        // Compute the gradient
        if(i == index_) {
            double l = par(index_);
            double r = std::sqrt(2 * nu_) * d / l;
            return (std::pow(r, nu_) * boost::math::cyl_bessel_k(nu_ - 1, r) * (-std::sqrt(2 * nu_) * d / std::pow(l, 2)) - nu_ * std::pow(r, nu_) * boost::math::cyl_bessel_k(nu_, r) / l) / (std::tgamma(nu_) * std::pow(2, nu_ - 1));
        } else {
            return 0;
        }
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }

        // Compute the hessian
        if(i == index_ && j == index_) {
            double l = par(index_);
            double r = std::sqrt(2 * nu_) * d / l;
            return (std::pow(r, nu_) * boost::math::cyl_bessel_k(nu_ - 2, r) * std::pow(-std::sqrt(2 * nu_) * d / std::pow(l, 2), 2)
                    + 2 * nu_ * std::pow(r, nu_) * boost::math::cyl_bessel_k(nu_ - 1, r) * (-std::sqrt(2 * nu_) * d / std::pow(l, 3))
                    + nu_ * (nu_ + 1) * std::pow(r, nu_) * boost::math::cyl_bessel_k(nu_, r) / std::pow(l, 2)) / (std::tgamma(nu_) * std::pow(2, nu_ - 1));
        } else {
            return 0;
        }
    };
    static std::shared_ptr<Covariance> make(const size_t &l, const double &nu, const int &i) {
        return std::make_shared<Matern>(l, nu, i);
    };
};

/**
 * @brief White noise covariance function.
 * 
 * @details Mathematical Formulation
 * Evaluates the Kronecker delta kernel between two inputs \f$\mathbf{x}_1, \mathbf{x}_2\f$:
 * \f[
 * k(\mathbf{x}_1, \mathbf{x}_2) = \begin{cases} 1.0 & \text{if } d < \text{tol} \\ 0.0 & \text{otherwise} \end{cases}
 * \f]
 * where \f$d\f$ is the distance between the two points.
 * 
 * @details Implementation Algorithm
 * Computes distance \f$d\f$ and compares it to a small tolerance threshold `tol_`.
 */
class WhiteNoise : public Covariance {
  private:
    int indexX_{-1};    ///< Dimension index to evaluate, or -1 for the full isotropic kernel.
    double tol_{1e-10}; ///< Distance tolerance threshold.
  public:
    WhiteNoise(const WhiteNoise&) = default;
    WhiteNoise(WhiteNoise&&) = default;
    WhiteNoise& operator=(const WhiteNoise&) = default;
    WhiteNoise& operator=(WhiteNoise&&) = default;
    WhiteNoise(const size_t &indexX = -1, double tol = 1e-10) : indexX_(indexX), tol_(tol) {};

    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        double d;
        if(indexX_ == -1) {
            d = (x1 - x2).norm();
        } else {
            d = std::abs(x1(indexX_) - x2(indexX_));
        }
        if(d < tol_) {
            return 1.0;
        } else {
            return 0.0;
        }
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return 0.0;
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return 0.0;
    };

    static std::shared_ptr<Covariance> make(const int &i = -1, double tol = 1e-10) {
        return std::make_shared<WhiteNoise>(i, tol);
    };

};

// Inline operator helpers — define here to ensure they live in the same namespace
inline std::shared_ptr<Covariance> operator+(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) {
    return Sum::make(k1, k2);
}

inline std::shared_ptr<Covariance> operator*(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) {
    return Product::make(k1, k2);
}




}

/** @} */

#endif