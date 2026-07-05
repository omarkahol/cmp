#ifndef PRIORPP_H
#define PRIORPP_H

#include <cmp_defines.h>
#include <distribution.h>

namespace cmp::prior {
class Prior {
  public:
    virtual ~Prior() = default;
    virtual double eval(const Eigen::VectorXd &par) const = 0;
    virtual double evalGradient(const Eigen::VectorXd &par, const size_t &i) const = 0;
    virtual double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const = 0;
};

class Product : public Prior {
  private:
    std::shared_ptr<Prior> leftPrior_;
    std::shared_ptr<Prior> rightPrior_;
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

template <typename DistType>
class FromDistribution : public Prior {
  private:
    DistType dist_;
    size_t paramIndex_;

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

template <typename DistType>
std::shared_ptr<Prior> make(const DistType& dist, size_t paramIndex) {
    return FromDistribution<DistType>::make(dist, paramIndex);
}

std::shared_ptr<Prior> operator*(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2);

}


#endif