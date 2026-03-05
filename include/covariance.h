#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <cmp_defines.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/bessel.hpp>

namespace cmp::covariance {
class Covariance {
  public:
    virtual ~Covariance() = default;
    virtual double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const = 0;
    virtual double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const = 0;
    virtual double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const = 0;
};

class Sum : public Covariance {
  private:
    std::shared_ptr<Covariance> leftCovariance_;
    std::shared_ptr<Covariance> rightCovariance_;
  public:

    Sum() = default;
    Sum(const Sum&) = default;
    Sum(Sum&&) = default;
    Sum& operator=(const Sum&) = default;
    Sum& operator=(Sum&&) = default;

    Sum(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) : leftCovariance_(k1), rightCovariance_(k2) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return leftCovariance_->eval(x1, x2, par) + rightCovariance_->eval(x1, x2, par);
    }
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return leftCovariance_->evalGradient(x1, x2, par, i) + rightCovariance_->evalGradient(x1, x2, par, i);
    }
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return leftCovariance_->evalHessian(x1, x2, par, i, j) + rightCovariance_->evalHessian(x1, x2, par, i, j);
    }

    static std::shared_ptr<Covariance> make(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) {
        return std::make_shared<Sum>(k1, k2);
    };
};

class Product : public Covariance {
    std::shared_ptr<Covariance> leftCovariance_;
    std::shared_ptr<Covariance> rightCovariance_;

  public:

    Product(const Product&) = default;
    Product(Product&&) = default;
    Product& operator=(const Product&) = default;
    Product& operator=(Product&&) = default;

    Product(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) : leftCovariance_(k1), rightCovariance_(k2) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return leftCovariance_->eval(x1, x2, par) * rightCovariance_->eval(x1, x2, par);
    }
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

    static std::shared_ptr<Covariance> make(std::shared_ptr<Covariance> k1, std::shared_ptr<Covariance> k2) {
        return std::make_shared<Product>(k1, k2);
    };
};

class Custom : public Covariance {
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
        return eval_(x1, x2, par);
    }
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return evalGradient_(x1, x2, par, i);
    }
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return evalHessian_(x1, x2, par, i, j);
    }

    std::shared_ptr<Custom> make(std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> eval,
                                 std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&)> evalGradient,
                                 std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const size_t&, const size_t&)> evalHessian) {
        return std::make_shared<Custom>(eval, evalGradient, evalHessian);
    };
};

class Constant : public Covariance {
  private:
    size_t index_;
  public:

    Constant(const Constant&) = default;
    Constant(Constant&&) = default;
    Constant& operator=(const Constant&) = default;
    Constant& operator=(Constant&&) = default;

    Constant(const size_t &index) : index_(index) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        return std::pow(par(index_), 2);
    };
    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        if(i == index_) {
            return 2 * par(index_);
        } else {
            return 0;
        }
    };
    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        if(i == index_ && j == index_) {
            return 2;
        } else {
            return 0;
        }
    };

    static std::shared_ptr<Covariance> make(const size_t &c) {
        return std::make_shared<Constant>(c);
    };

};

class Linear : public Covariance {
  private:
    int indexX_;
  public:

    Linear(const Linear&) = default;
    Linear(Linear&&) = default;
    Linear& operator=(const Linear&) = default;
    Linear& operator=(Linear&&) = default;

    Linear(const int &indexX = -1) : indexX_(indexX) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        if(indexX_ == -1) {
            return x1.dot(x2);
        } else {
            return x1(indexX_) * x2(indexX_);
        }
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return 0;
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return 0;
    };

    static std::shared_ptr<Covariance> make(const int &i) {
        return std::make_shared<Linear>(i);
    };

};

class Inverse : public Covariance {
  private:
    int indexX_;
  public:

    Inverse(const Inverse&) = default;
    Inverse(Inverse&&) = default;
    Inverse& operator=(const Inverse&) = default;
    Inverse& operator=(Inverse&&) = default;

    Inverse(const int &indexX = -1) : indexX_(indexX) {};
    double eval(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par) const {
        if(indexX_ == -1) {
            return 1.0 / (x1.dot(x2) + 1);
        } else {
            return 1.0 / (x1(indexX_) * x2(indexX_) + 1);
        }
    };

    double evalGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i) const {
        return 0;
    };

    double evalHessian(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd& par, const size_t &i, const size_t &j) const {
        return 0;
    };

    static std::shared_ptr<Covariance> make(const int &i) {
        return std::make_shared<Inverse>(i);
    };

};

class SquaredExponential : public Covariance {
  private:
    size_t index_;
    int indexX_;
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

class Matern52 : public Covariance {
  private:
    size_t index_;
    int indexX_;
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

class Matern : public Covariance {
  private:
    size_t index_;
    int indexX_;
    double nu_;
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

class WhiteNoise : public Covariance {
  private:
    int indexX_{-1};
    double tol_{1e-10};
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

#endif