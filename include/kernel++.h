#ifndef KERNELPP_H
#define KERNELPP_H

#include "cmp_defines.h"
#define TOL 1e-8

namespace cmp::kernel {

    class Kernel {
        public:
            virtual ~Kernel() = default;
            virtual double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const = 0;
            virtual double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const = 0;
            virtual double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const = 0;
        };

    class Sum : public Kernel {
        private:
            std::shared_ptr<Kernel> leftKernel_;
            std::shared_ptr<Kernel> rightKernel_;
        public:

            Sum() = default;
            Sum(const Sum&) = default;
            Sum(Sum&&) = default;
            Sum& operator=(const Sum&) = default;
            Sum& operator=(Sum&&) = default;

            Sum(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2) : leftKernel_(k1), rightKernel_(k2) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                return leftKernel_->eval(x1,x2,par) + rightKernel_->eval(x1,x2,par);
            }
            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                return leftKernel_->evalGradient(x1,x2,par,i) + rightKernel_->evalGradient(x1,x2,par,i);
            }
            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                return leftKernel_->evalHessian(x1,x2,par,i,j) + rightKernel_->evalHessian(x1,x2,par,i,j);
            }

            static std::shared_ptr<Kernel> make(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2) {
                return std::make_shared<Sum>(k1,k2);
            }; 
    };

    class Product : public Kernel {
        std::shared_ptr<Kernel> leftKernel_;
        std::shared_ptr<Kernel> rightKernel_;

        public:

            Product(const Product&) = default;
            Product(Product&&) = default;
            Product& operator=(const Product&) = default;
            Product& operator=(Product&&) = default;

            Product(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2) : leftKernel_(k1), rightKernel_(k2) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                return leftKernel_->eval(x1,x2,par) * rightKernel_->eval(x1,x2,par);
            }
            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {

                // Smart eval for first term
                double d1 = leftKernel_->evalGradient(x1,x2,par,i);
                if (std::abs(d1) < TOL) {
                } else {
                    d1*= rightKernel_->eval(x1,x2,par);
                }

                // Smart eval for second term
                double d2 = rightKernel_->evalGradient(x1,x2,par,i);
                if (std::abs(d2) < TOL) {
                } else {
                    d2*= leftKernel_->eval(x1,x2,par);
                }

                // Return the result
                return d1 + d2;
            }
            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                
                // Smart eval of the first term
                double d1 = leftKernel_->evalGradient(x1,x2,par,i);
                if (std::abs(d1) < TOL) {
                } else {
                    d1 = (d1+leftKernel_->evalHessian(x1,x2,par,i,j))*rightKernel_->eval(x1,x2,par);
                }
                
                // Smart eval for second term
                double d2 = rightKernel_->evalGradient(x1,x2,par,i);
                if (std::abs(d2) < TOL) {
                } else {
                    d2 = (d2+rightKernel_->evalHessian(x1,x2,par,i,j))*leftKernel_->eval(x1,x2,par);
                }

                // Smart eval for the third term
                return d1 + d2;
            }

            static std::shared_ptr<Kernel> make(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2) {
                return std::make_shared<Product>(k1,k2);
            };
    };

    class Nugget: public Kernel {
        private:
            std::shared_ptr<Kernel> kernel_;
            double nugget_;
        public:

            Nugget(const Nugget&) = default;
            Nugget(Nugget&&) = default;
            Nugget& operator=(const Nugget&) = default;
            Nugget& operator=(Nugget&&) = default;

            Nugget(std::shared_ptr<Kernel> k, const double &nugget) : kernel_(k), nugget_(nugget) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                return kernel_->eval(x1,x2,par) + nugget_*double(x1.isApprox(x2));
            }
            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                return kernel_->evalGradient(x1,x2,par,i);
            }
            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                return kernel_->evalHessian(x1,x2,par,i,j);
            }

            static std::shared_ptr<Kernel> make(std::shared_ptr<Kernel> k, const double &nugget_val=1e-3) {
                return std::make_shared<Nugget>(k,nugget_val);
            };

    };

    class Custom : public Kernel {
        private:
            std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval_;
            std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient_;
            std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian_;
        public:

            Custom(const Custom&) = default;
            Custom(Custom&&) = default;
            Custom& operator=(const Custom&) = default;
            Custom& operator=(Custom&&) = default;

            Custom(std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval,
                   std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient,
                   std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian) : eval_(eval), evalGradient_(evalGradient), evalHessian_(evalHessian) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                return eval_(x1,x2,par);
            }
            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                return evalGradient_(x1,x2,par,i);
            }
            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                return evalHessian_(x1,x2,par,i,j);
            }

            std::shared_ptr<Custom> make(std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval,
                                         std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient,
                                         std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian) {
                return std::make_shared<Custom>(eval,evalGradient,evalHessian);
            };
    };

    class Constant : public Kernel {
        private:
            size_t index_;
        public:

            Constant(const Constant&) = default;
            Constant(Constant&&) = default;
            Constant& operator=(const Constant&) = default;
            Constant& operator=(Constant&&) = default;

            Constant(const size_t &index) : index_(index) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                return std::pow(par(index_),2);
            };
            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                if (i == index_) {
                    return 2*par(index_);
                } else {
                    return 0;
                }
            };
            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                if (i == index_ && j == index_) {
                    return 2;
                } else {
                    return 0;
                }
            };
            
            static std::shared_ptr<Kernel> make(const size_t &c) {
                return std::make_shared<Constant>(c);
            };
            
    };

    class Linear : public Kernel {
        private:
            int indexX_;
        public:

            Linear(const Linear&) = default;
            Linear(Linear&&) = default;
            Linear& operator=(const Linear&) = default;
            Linear& operator=(Linear&&) = default;

            Linear(const int &indexX=-1) : indexX_(indexX) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                if (indexX_ == -1) {
                    return x1.dot(x2);
                } else {
                    return x1(indexX_)*x2(indexX_);
                }
            };

            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                return 0;
            };

            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                return 0;
            };

            static std::shared_ptr<Kernel> make(const int &i) {
                return std::make_shared<Linear>(i);
            };

    };

    class SquaredExponential : public Kernel {
        private:
            size_t index_;
            int indexX_;
        public:

            SquaredExponential(const SquaredExponential&) = default;
            SquaredExponential(SquaredExponential&&) = default;
            SquaredExponential& operator=(const SquaredExponential&) = default;
            SquaredExponential& operator=(SquaredExponential&&) = default;

            SquaredExponential(const size_t &index, const int &indexX=-1) : index_(index), indexX_(indexX) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                double d;
                if (indexX_ == -1) {
                    d = (x1-x2).norm();
                } else {
                    d = std::abs(x1(indexX_)-x2(indexX_));
                }
                return std::exp(-0.5*std::pow(d/par(index_),2));
            };

            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                double d;
                if (indexX_ == -1) {
                    d = (x1-x2).norm();
                } else {
                    d = std::abs(x1(indexX_)-x2(indexX_));
                }

                // Compute the gradient
                if (i == index_) {
                    return std::exp(-0.5*std::pow(d/par(index_),2))*std::pow(d,2)/std::pow(par(index_),3);
                } else {
                    return 0;
                }
            };

            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                double d;
                if (indexX_ == -1) {
                    d = (x1-x2).norm();
                } else {
                    d = std::abs(x1(indexX_)-x2(indexX_));
                }

                // Compute the hessian
                if (i == index_ && j == index_) {
                    return (std::exp(-0.5*std::pow(d/par(index_),2))*std::pow(d,2)/std::pow(par(index_),4))*(-3+std::pow(d,2)/std::pow(par(index_),3));
                } else {
                    return 0;
                }
            };

            static std::shared_ptr<Kernel> make(const size_t &l, const int &i) {
                return std::make_shared<SquaredExponential>(l,i);
            };

    };

    class Matern52 : public Kernel {
        private:
            size_t index_;
            int indexX_;
        public:

            Matern52(const Matern52&) = default;
            Matern52(Matern52&&) = default;
            Matern52& operator=(const Matern52&) = default;
            Matern52& operator=(Matern52&&) = default;

            Matern52(const size_t &l, const int &i=-1) : index_(l), indexX_(i) {};
            double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
                double d;
                if (indexX_ == -1) {
                    d = (x1-x2).norm();
                } else {
                    d = std::abs(x1(indexX_)-x2(indexX_));
                }
                double l = par(index_);
                double c_1 = sqrt(5.)*d/l;
                double c_2 = (5./3.)*pow(d/l,2);
                return (1+c_1+c_2)*exp(-c_1);
            };

            double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
                double d;
                if (indexX_ == -1) {
                    d = (x1-x2).norm();
                } else {
                    d = std::abs(x1(indexX_)-x2(indexX_));
                }

                // Compute the gradient
                if (i == index_) {
                    double l = par(index_);
                    return (5.*std::pow(d,2)*(std::sqrt(5)*d + l))/(3.*std::exp((std::sqrt(5)*d)/l)*std::pow(l,4));
                } else {
                    return 0;
                }
            };

            double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
                double d;
                if (indexX_ == -1) {
                    d = (x1-x2).norm();
                } else {
                    d = std::abs(x1(indexX_)-x2(indexX_));
                }

                // Compute the hessian
                if (i == index_ && j == index_) {
                    double l = par(index_);
                    return (5*std::pow(d,2)*(5*std::pow(d,2) - 3*std::sqrt(5)*d*l - 3*std::pow(l,2)))/(3.*std::exp(std::sqrt(5)*d/l)*std::pow(l,6));
                } else {
                    return 0;
                }
            };

            static std::shared_ptr<Kernel> make(const size_t &l, const int &i) {
                return std::make_shared<Matern52>(l,i);
            };
    };


    std::shared_ptr<Kernel> operator+(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2);

    std::shared_ptr<Kernel> operator*(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2);
}

#endif