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

    class FromDistribution : public Prior {
        private:
            std::shared_ptr<cmp::distribution::UnivariateDistribution> dist_;
            std::size_t index_;
        public:
            FromDistribution(const FromDistribution&) = default;
            FromDistribution(FromDistribution&&) = default;
            FromDistribution& operator=(const FromDistribution&) = default;
            FromDistribution& operator=(FromDistribution&&) = default;

            FromDistribution(std::shared_ptr<cmp::distribution::UnivariateDistribution> dist, std::size_t index) : dist_(dist), index_(index) {};

            double eval(const Eigen::VectorXd &par) const {
                return dist_->logPDF(par(index_));
            }

            double evalGradient(const Eigen::VectorXd &par, const size_t &i) const {
                if (i == index_) {
                    return dist_->dLogPDF(par(index_));
                } else {
                    return 0;
                }
            }

            double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const {
                if (i == index_ && j == index_) {
                    return dist_->ddLogPDF(par(index_));
                } else {
                    return 0;
                }
            }
            
            static std::shared_ptr<Prior> make(std::shared_ptr<cmp::distribution::UnivariateDistribution> dist, std::size_t i) {
                return std::make_shared<FromDistribution>(dist,i);
            }
    };

    class Product : public Prior {
        private:
            std::shared_ptr<Prior> leftPrior_;
            std::shared_ptr<Prior> rightPrior_;
        public:

            Product () = default;
            Product(const Product&) = default;
            Product(Product&&) = default;
            Product& operator=(const Product&) = default;
            Product& operator=(Product&&) = default;

            Product(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2) : leftPrior_(p1), rightPrior_(p2) {};

            double eval(const Eigen::VectorXd &par) const {
                return leftPrior_->eval(par) + rightPrior_->eval(par);
            }

            double evalGradient(const Eigen::VectorXd &par, const size_t &i) const {
                return leftPrior_->evalGradient(par,i) + rightPrior_->evalGradient(par,i);
            }

            double evalHessian(const Eigen::VectorXd &par, const size_t &i, const size_t &j) const {
                return leftPrior_->evalHessian(par,i,j) + rightPrior_->evalHessian(par,i,j);
            }

            static std::shared_ptr<Prior> make(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2) {
                return std::make_shared<Product>(p1,p2);
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

    std::shared_ptr<Prior> operator*(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2);

}

#endif