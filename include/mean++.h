#ifndef MEANPP_H
#define MEANPP_H

#include "cmp_defines.h"
#include <memory>

namespace cmp::mean {
    class Mean {
        public:
            virtual ~Mean() = default;
            virtual double eval(const Eigen::VectorXd &x, const Eigen::VectorXd &par) const = 0;
            virtual double evalGradient(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const size_t &i) const = 0;
            virtual double evalHessian(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const size_t &i, const size_t &j) const = 0;
    };

    class Constant : public Mean {
        private:
            size_t index_;
        public:
            
            Constant(const Constant&) = default;
            Constant(Constant&&) = default;
            Constant& operator=(const Constant&) = default;
            Constant& operator=(Constant&&) = default;

            Constant(const size_t &c) : index_(c) {};
            double eval(const Eigen::VectorXd& x, const Eigen::VectorXd &par) const {
                return par(index_);
            };
            double evalGradient(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i) const {
                if (i == index_) {
                    return 1.0;
                } else {
                    return 0.0;
                }
            };
            double evalHessian(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i, const size_t &j) const {
                return 0;
            };

            static std::shared_ptr<Mean> make(const size_t &c) {
                return std::make_shared<Constant>(c);
            };

        };

    class Zero : public Mean {
        public:
            
            Zero() = default;
            Zero(const Zero&) = default;
            Zero(Zero&&) = default;
            Zero& operator=(const Zero&) = default;
            Zero& operator=(Zero&&) = default;

            double eval(const Eigen::VectorXd& x, const Eigen::VectorXd &par) const {
                return 0.0;
            };
            double evalGradient(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i) const {
                return 0.0;
            };
            double evalHessian(const Eigen::VectorXd& x, const Eigen::VectorXd &log_par, const size_t &i, const size_t &j) const {
                return 0.0;
            };

            static std::shared_ptr<Mean> make() {
                return std::make_shared<Zero>();
            };

        };

}

#endif