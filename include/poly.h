/**
 * @file poly.h
 * @brief Header file for Polynomial Chaos Expansion (PCE) related functions and classes.
 * @author Omar Kahol
 */

#ifndef POLY_H
#define POLY_H

#include <cmp_defines.h>


namespace cmp {

class MultiIndex {
  public:
    using Index = std::vector<size_t>;

    std::vector<Index> indices;

    MultiIndex() = default;

    void set(size_t dimension, size_t totalDegree);

    std::size_t size() const {
        return indices.size();
    }

    const Index& operator[](std::size_t i) const {
        return indices[i];
    }

    void print() const;

  private:
    void generateRecursive(Index& current, int pos, int remaining);
};


// Virtual Class for Polynomial Basis (Hermite, Legendre, etc.)
class PolynomialBasis {
  public:
    virtual double evaluate(const size_t &deg, const double &x) const = 0;
    virtual ~PolynomialBasis() = default;
};

class HermitePolynomial : public PolynomialBasis {
  public:
    double evaluate(const size_t &deg, const double &x) const;

    static std::shared_ptr<PolynomialBasis> make() {
        return std::make_shared<HermitePolynomial>();
    }
};
class LegendrePolynomial : public PolynomialBasis {
  public:
    double evaluate(const size_t &deg, const double &x) const;

    static std::shared_ptr<PolynomialBasis> make() {
        return std::make_shared<LegendrePolynomial>();
    }
};

class PolynomialExpansion {
  private:
    MultiIndex multiIndex_;
    Eigen::VectorXd coefficients_;
    Eigen::MatrixXd coefficientsCovariance_;
    std::shared_ptr<PolynomialBasis> basis_;
    Eigen::MatrixXd xObs_;

  public:
    PolynomialExpansion() = default;

    void set(size_t dimension, size_t totalDegree, std::shared_ptr<PolynomialBasis> basis) {
        multiIndex_.set(dimension, totalDegree);
        basis_ = basis;
        coefficients_ = Eigen::VectorXd::Zero(multiIndex_.size());
    }

    void setCoefficients(const Eigen::Ref<const Eigen::VectorXd> &coeffs) {
        if(coeffs.size() != coefficients_.size()) {
            throw std::invalid_argument("Coefficient vector size does not match the number of basis functions.");
        }
        coefficients_ = coeffs;
    }

    std::pair<double, double> predict(const Eigen::Ref<const Eigen::VectorXd> &x) const;

    std::pair<double, double> predictLOO(const size_t &i) const;


    void fit(const Eigen::Ref<const Eigen::MatrixXd> &samples, const Eigen::Ref<const Eigen::VectorXd> &values);
};

}


#endif // POLY_H