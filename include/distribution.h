#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include "cmp_defines.h"
#include <cmath>
#include <vector>
#include <memory>
#include <random>
#include <stdexcept>
#include <limits>
#include <Eigen/Dense>

namespace cmp::distribution {


// ==============================================================================
// HELPER FUNCTIONS
// ==============================================================================

inline double erfinv(float x) {
    double tt1, tt2, lnx, sgn;
    sgn = (x < 0) ? -1.0f : 1.0f;
    x = 1 - x * x;
    lnx = std::log(x);
    tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
    tt2 = 1 / (0.147) * lnx;
    return (sgn * std::sqrt(-tt1 + std::sqrt(tt1 * tt1 - tt2)));
}

inline double CDF_normal(const double &x) {
    return 0.5 * (1 + std::erf(x / std::sqrt(2)));
}

inline double sech(const double &x) {
    return 1.0 / std::cosh(x);
}

// ==============================================================================
// CRTP BASE CLASSES
// ==============================================================================

template <typename Derived>
class UnivariateDistribution {
  public:
    double logPDF(const double &x) const {
        return static_cast<const Derived*>(this)->logPDF(x);
    }
    double dLogPDF(const double &x) const {
        return static_cast<const Derived*>(this)->dLogPDF(x);
    }
    double ddLogPDF(const double &x) const {
        return static_cast<const Derived*>(this)->ddLogPDF(x);
    }
    double CDF(const double &x) const {
        return static_cast<const Derived*>(this)->CDF(x);
    }
    double quantile(const double &p) const {
        return static_cast<const Derived*>(this)->quantile(p);
    }
    double sample(std::default_random_engine &rng) {
        return static_cast<Derived*>(this)->sample(rng);
    }
};

template <typename Derived>
class MultivariateDistribution {
  public:
    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        return static_cast<const Derived*>(this)->logPDF(x);
    }
    Eigen::VectorXd sample(std::default_random_engine &rng) {
        return static_cast<Derived*>(this)->sample(rng);
    }

    Eigen::MatrixXd toCanonical(const Eigen::MatrixXd &x) const {
        return static_cast<const Derived*>(this)->toCanonical(x);
    }
    Eigen::MatrixXd fromCanonical(const Eigen::MatrixXd &x) const {
        return static_cast<const Derived*>(this)->fromCanonical(x);
    }

    size_t dimension() const {
        return static_cast<const Derived*>(this)->dimension();
    }
};

template <typename Derived>
class ProposalDistribution : public MultivariateDistribution<Derived> {
  public:
    double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) {
        return static_cast<Derived*>(this)->logJumpPDF(jump);
    }
    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma) {
        return static_cast<Derived*>(this)->sample(rng, gamma);
    }
    double squaredMahalanobis(const Eigen::Ref<const Eigen::VectorXd> &jump) const {
        return static_cast<const Derived*>(this)->squaredMahalanobis(jump);
    }
    Eigen::VectorXd sample(std::default_random_engine &rng) {
        return static_cast<Derived*>(this)->sample(rng, 1.0);
    }
    Eigen::VectorXd get() const {
        return static_cast<Derived*>(this)->get();
    }
    void set(const Eigen::Ref<const Eigen::VectorXd> &x) {
        static_cast<Derived*>(this)->set(x);
    }
};

// ==============================================================================
// UNIVARIATE DISTRIBUTIONS
// ==============================================================================

class NormalDistribution : public UnivariateDistribution<NormalDistribution> {
  private:
    double mean_{0.0};
    double std_{1.0};
    std::normal_distribution<double> distN_;
  public:
    NormalDistribution(double mean, double sd): mean_(mean), std_(sd), distN_(0., 1.) {}
    NormalDistribution() = default;

    static double logPDF(const double &res, const double &std) {
        return -0.5 * std::log(2 * M_PI) - std::log(std) - 0.5 * std::pow(res / std, 2);
    }

    double logPDF(const double &x) const {
        return logPDF(x - mean_, std_);
    }
    double dLogPDF(const double &x) const {
        return -(x - mean_) / std::pow(std_, 2);
    }
    double ddLogPDF(const double &x) const {
        return -1 / std::pow(std_, 2);
    }
    double CDF(const double &x) const {
        return 0.5 * (1 + std::erf((x - mean_) / (std_ * std::sqrt(2))));
    }
    double quantile(const double &p) const {
        return mean_ + std_ * std::sqrt(2) * erfinv(2 * p - 1);
    }
    double sample(std::default_random_engine &rng) {
        return distN_(rng) * std_ + mean_;
    }

    void setMean(double mean) {
        mean_ = mean;
    }
    void setStd(double std) {
        std_ = std;
    }
    double mean() const {
        return mean_;
    }
    double std() const {
        return std_;
    }
};


class UniformDistribution : public UnivariateDistribution<UniformDistribution> {
  private:
    double lowerBound_{0.0};
    double upperBound_{1.0};
    std::uniform_real_distribution<double> distU_{0., 1.};
  public:
    UniformDistribution(double a, double b): lowerBound_(a), upperBound_(b), distU_(0., 1.) {}
    UniformDistribution() = default;

    double logPDF(const double &x) const {
        if(x < lowerBound_ || x > upperBound_) return -std::numeric_limits<double>::infinity();
        return -std::log(upperBound_ - lowerBound_);
    }
    double dLogPDF(const double &x) const {
        return 0.0;
    }
    double ddLogPDF(const double &x) const {
        return 0.0;
    }
    double CDF(const double &x) const {
        if(x < lowerBound_) return 0.0;
        if(x > upperBound_) return 1.0;
        return (x - lowerBound_) / (upperBound_ - lowerBound_);
    }
    double quantile(const double &p) const {
        return lowerBound_ + p * (upperBound_ - lowerBound_);
    }
    double sample(std::default_random_engine &rng) {
        return lowerBound_ + distU_(rng) * (upperBound_ - lowerBound_);
    }
    void setLowerBound(double a) {
        lowerBound_ = a;
    }
    void setUpperBound(double b) {
        upperBound_ = b;
    }
};


class InverseGammaDistribution : public UnivariateDistribution<InverseGammaDistribution> {
  private:
    double alpha_;
    double beta_;
    std::gamma_distribution<double> distGamma_;
  public:
    InverseGammaDistribution(double alpha, double beta): alpha_(alpha), beta_(beta), distGamma_(alpha, 1 / beta) {}
    InverseGammaDistribution() = default;

    double logPDF(const double &x) const {
        return -(alpha_ + 1) * std::log(x) - beta_ / x + alpha_ * std::log(beta_) - std::log(std::tgamma(alpha_));
    }
    double dLogPDF(const double &x) const {
        return (beta_ - (alpha_ + 1) * x) / std::pow(x, 2);
    }
    double ddLogPDF(const double &x) const {
        return (-2 * beta_ + x + x * alpha_) / std::pow(x, 3);
    }
    double CDF(const double &x) const {
        return 0.0;    // To-do
    }
    double quantile(const double &p) const {
        return 0.0;    // To-do
    }
    double sample(std::default_random_engine &rng) {
        return 1 / distGamma_(rng);
    }
    void setAlpha(double alpha) {
        alpha_ = alpha;
    }
    void setBeta(double beta) {
        beta_ = beta;
    }
};

class GammaDistribution : public UnivariateDistribution<GammaDistribution> {
  private:
    double alpha_; // Shape parameter (often denoted as k)
    double beta_;  // Rate parameter (often denoted as theta = 1/beta)
    std::gamma_distribution<double> distGamma_;

  public:
    // std::gamma_distribution takes (shape, scale), so we pass (alpha, 1/beta)
    GammaDistribution(double alpha, double beta)
        : alpha_(alpha), beta_(beta), distGamma_(alpha, 1.0 / beta) {
        if(alpha_ <= 0.0 || beta_ <= 0.0) {
            throw std::invalid_argument("Gamma parameters alpha and beta must be strictly positive.");
        }
    }

    GammaDistribution() = default;

    double logPDF(const double &x) const {
        if(x <= 0.0) return -std::numeric_limits<double>::infinity();

        // Using std::lgamma is much more numerically stable than std::log(std::tgamma())
        return alpha_ * std::log(beta_) - std::lgamma(alpha_) + (alpha_ - 1.0) * std::log(x) - beta_ * x;
    }

    double dLogPDF(const double &x) const {
        if(x <= 0.0) return 0.0;
        return (alpha_ - 1.0) / x - beta_;
    }

    double ddLogPDF(const double &x) const {
        if(x <= 0.0) return 0.0;
        return -(alpha_ - 1.0) / std::pow(x, 2);
    }

    double CDF(const double &x) const {
        return 0.0;    // To-do: Requires regularized lower incomplete gamma function
    }

    double quantile(const double &p) const {
        return 0.0;    // To-do: Requires inverse incomplete gamma function
    }

    double sample(std::default_random_engine &rng) {
        return distGamma_(rng);
    }

    // Setters need to recreate the std::gamma_distribution to update internal state
    void setAlpha(double alpha) {
        alpha_ = alpha;
        distGamma_ = std::gamma_distribution<double>(alpha_, 1.0 / beta_);
    }

    void setBeta(double beta) {
        beta_ = beta;
        distGamma_ = std::gamma_distribution<double>(alpha_, 1.0 / beta_);
    }

    // Getters are useful when extracting parameters in the DPMM
    double getAlpha() const {
        return alpha_;
    }
    double getBeta() const {
        return beta_;
    }
};

class BetaDistribution : public UnivariateDistribution<BetaDistribution> {
  private:
    double alpha_;
    double beta_;

    // We use std::gamma_distribution internally with a rate of 1.0 for fast sampling
    std::gamma_distribution<double> distGammaAlpha_;
    std::gamma_distribution<double> distGammaBeta_;

  public:
    BetaDistribution(double alpha, double beta)
        : alpha_(alpha), beta_(beta),
          distGammaAlpha_(alpha, 1.0),
          distGammaBeta_(beta, 1.0) {
        if(alpha_ <= 0.0 || beta_ <= 0.0) {
            throw std::invalid_argument("Beta parameters alpha and beta must be strictly positive.");
        }
    }

    BetaDistribution() = default;

    double logPDF(const double &x) const {
        // Beta distribution is strictly defined on the interval (0, 1)
        if(x <= 0.0 || x >= 1.0) return -std::numeric_limits<double>::infinity();

        // Calculate the log Beta function: ln(B(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
        double logBetaFunc = std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_);

        return (alpha_ - 1.0) * std::log(x) + (beta_ - 1.0) * std::log(1.0 - x) - logBetaFunc;
    }

    double dLogPDF(const double &x) const {
        if(x <= 0.0 || x >= 1.0) return 0.0;
        return (alpha_ - 1.0) / x - (beta_ - 1.0) / (1.0 - x);
    }

    double ddLogPDF(const double &x) const {
        if(x <= 0.0 || x >= 1.0) return 0.0;
        return -(alpha_ - 1.0) / std::pow(x, 2) - (beta_ - 1.0) / std::pow(1.0 - x, 2);
    }

    double CDF(const double &x) const {
        return 0.0; // To-do: Requires regularized incomplete beta function
    }

    double quantile(const double &p) const {
        return 0.0; // To-do: Requires inverse incomplete beta function
    }

    double sample(std::default_random_engine &rng) {
        double x = distGammaAlpha_(rng);
        double y = distGammaBeta_(rng);

        // Prevent division by zero in the extremely rare case both evaluate to 0.0
        if(x + y == 0.0) return 0.5;

        return x / (x + y);
    }

    // Setters update the internal gamma generators
    void setAlpha(double alpha) {
        if(alpha <= 0.0) throw std::invalid_argument("Alpha must be > 0");
        alpha_ = alpha;
        distGammaAlpha_ = std::gamma_distribution<double>(alpha_, 1.0);
    }

    void setBeta(double beta) {
        if(beta <= 0.0) throw std::invalid_argument("Beta must be > 0");
        beta_ = beta;
        distGammaBeta_ = std::gamma_distribution<double>(beta_, 1.0);
    }

    double getAlpha() const {
        return alpha_;
    }
    double getBeta() const {
        return beta_;
    }
};


class LogNormalDistribution : public UnivariateDistribution<LogNormalDistribution> {
  private:
    double mean_{0.0};
    double std_{1.0};
    std::normal_distribution<double> distN_;
  public:
    LogNormalDistribution(double mu, double sigma): mean_(mu), std_(sigma), distN_(0, 1) {}
    LogNormalDistribution() = default;

    double logPDF(const double &x) const {
        return -0.5 * std::pow((std::log(x) - mean_) / std_, 2) - 0.5 * std::log(2 * M_PI);
    }
    double dLogPDF(const double &x) const {
        return (mean_ - std::log(x)) / (x * std::pow(std_, 2));
    }
    double ddLogPDF(const double &x) const {
        return (-1 - mean_ + std::log(x)) / std::pow(x * std_, 2);
    }
    double CDF(const double &x) const {
        return 0.5 * (1 + std::erf((std::log(x) - mean_) / (std_ * std::sqrt(2))));
    }
    double quantile(const double &p) const {
        return std::exp(mean_ + std_ * std::sqrt(2) * erfinv(2 * p - 1));
    }
    double sample(std::default_random_engine &rng) {
        return std::exp(distN_(rng) * std_ + mean_);
    }
    void setMean(double mu) {
        mean_ = mu;
    }
    void setStd(double sigma) {
        std_ = sigma;
    }
};


class StudentDistribution : public UnivariateDistribution<StudentDistribution> {
  private:
    double dofs_;
    double mean_;
    double std_;
    std::student_t_distribution<double> distN_;
    std::normal_distribution<double> normDist_{0.0, 1.0};
  public:
    StudentDistribution(double nu, double mu, double sigma): dofs_(nu), mean_(mu), std_(sigma), distN_(nu) {}
    StudentDistribution() = default;

    double logPDF(const double &x) const {
        return std::log(std::tgammal(0.5 * (dofs_ + 1)) - std::log(std::tgammal(0.5 * dofs_)) - 0.5 * std::log(M_PI * dofs_)) - std_ - 0.5 * (dofs_ + 1) * std::log(1 + std::pow(x - mean_, 2) / (dofs_ * std_ * std_));
    }
    double dLogPDF(const double &x) const {
        return -(x - mean_) * (1 + dofs_) / (std_ * std_ * dofs_ + std::pow(x - mean_, 2));
    }
    double ddLogPDF(const double &x) const {
        return (1 + dofs_) * (std::pow(x - mean_, 2) - dofs_ * std_ * std_) / std::pow((std::pow(x - mean_, 2) + dofs_ * std_ * std_), 2);
    }
    double CDF(const double &x) const {
        return 0.0;    // To-do
    }
    double quantile(const double &p) const {
        return 0.0;    // To-do
    }
    double sample(std::default_random_engine &rng) {
        double z = normDist_(rng);
        double chi = 0.0;
        for(size_t j = 0; j < dofs_; j++) {
            chi += std::pow(normDist_(rng), 2);
        }
        return mean_ + z * std_ / std::sqrt(chi / dofs_);
    }
    void setDoFs(double dofs) {
        dofs_ = dofs;
    }
    void setMean(double mu) {
        mean_ = mu;
    }
    void setStd(double sigma) {
        std_ = sigma;
    }
};


class PowerLawDistribution : public UnivariateDistribution<PowerLawDistribution> {
  private:
    double degree_;
    double lowerBound_;
    std::uniform_real_distribution<double> distU_;
  public:
    PowerLawDistribution(double alpha, double lowerBound = 1.0): degree_(alpha), distU_(0., 1.), lowerBound_(lowerBound) {}
    PowerLawDistribution() = default;

    double logPDF(const double &x) const {
        return std::log(lowerBound_) - degree_ * std::log(x);
    }
    double dLogPDF(const double &x) const {
        return -degree_ / x;
    }
    double ddLogPDF(const double &x) const {
        return degree_ / std::pow(x, 2);
    }
    double CDF(const double &t) const {
        return 1.0 - std::pow(lowerBound_ / t, degree_);
    }
    double quantile(const double &q) const {
        return lowerBound_ / std::pow(1 - q, 1 / degree_);
    }
    double sample(std::default_random_engine &rng) {
        return quantile(distU_(rng));
    }
    void setDegree(double degree) {
        degree_ = degree;
    }
};


class SmoothUniformDistribution : public UnivariateDistribution<SmoothUniformDistribution> {
  private:
    double lowerBound_;
    double upperBound_;
    double std_;
    std::uniform_real_distribution<double> distU_;
    std::normal_distribution<double> distN_;
  public:
    SmoothUniformDistribution(double lowerBound, double upperBound, double sigma): lowerBound_(lowerBound), upperBound_(upperBound), std_(sigma), distU_(0, 1), distN_(0, 1) {}
    SmoothUniformDistribution() = default;

    double logPDF(const double &x) const {
        return std::log((CDF_normal((upperBound_ - x) / std_) - CDF_normal((lowerBound_ - x) / std_)) / (upperBound_ - lowerBound_));
    }
    double dLogPDF(const double &x) const {
        return (std::pow(M_E, -std::pow(lowerBound_ - x, 2) / (2.*std::pow(std_, 2))) - std::pow(M_E, -std::pow(upperBound_ - x, 2) / (2.*std::pow(std_, 2)))) / ((-lowerBound_ + upperBound_) * std::sqrt(2 * M_PI) * std_);
    }
    double ddLogPDF(const double &x) const {
        return ((lowerBound_ - x) / std::pow(M_E, std::pow(lowerBound_ - x, 2) / (2.*std::pow(std_, 2))) +
                (-upperBound_ + x) / std::pow(M_E, std::pow(upperBound_ - x, 2) / (2.*std::pow(std_, 2)))) /
               ((-lowerBound_ + upperBound_) * std::sqrt(2 * M_PI) * std::pow(std_, 3));
    }
    double CDF(const double &t) const {
        return (lowerBound_ - upperBound_ + (-std::pow(M_E, -std::pow(lowerBound_ - t, 2) / (2.*std::pow(std_, 2))) +
                                             std::pow(M_E, -std::pow(upperBound_ - t, 2) / (2.*std::pow(std_, 2)))) *
                std::sqrt(2 / M_PI) * std_ + (-lowerBound_ + t) * std::erf((lowerBound_ - t) / (std::sqrt(2) * std_)) +
                (upperBound_ - t) * std::erf((upperBound_ - t) / (std::sqrt(2) * std_))) / (2.*(lowerBound_ - upperBound_));
    }
    double quantile(const double &p) const {
        if(p < 0 || p > 1) return std::numeric_limits<double>::quiet_NaN();
        double x = 0.5 * (lowerBound_ + upperBound_);
        for(size_t i = 0; i < 5; i++) {
            double f = CDF(x) - p;
            double df = std::exp(logPDF(x));
            x = x - f / df;
        }
        return x;
    }
    double sample(std::default_random_engine &rng) {
        return lowerBound_ + (upperBound_ - lowerBound_) * distU_(rng) + distN_(rng) * std_;
    }

    void setLowerBound(double a) {
        lowerBound_ = a;
    }
    void setUpperBound(double b) {
        upperBound_ = b;
    }
    void setStd(double sigma) {
        std_ = sigma;
    }
};

// ==============================================================================
// MULTIVARIATE DISTRIBUTIONS & PROPOSALS
// ==============================================================================

class MultivariateNormalDistribution : public ProposalDistribution<MultivariateNormalDistribution> {
  private:
    Eigen::VectorXd mean_;
    Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition_;
    std::normal_distribution<double> distN_;
  public:

    MultivariateNormalDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::MatrixXd> &cov)
        : mean_(mean), distN_(0., 1.) {
        ldltDecomposition_.compute(cov);
    }
    template<typename MatrixType>
    MultivariateNormalDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::LDLT<MatrixType> &ldltDecomposition)
        : mean_(mean), ldltDecomposition_(ldltDecomposition), distN_(0., 1.) {}

    MultivariateNormalDistribution() = default;

    static double logPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition) {
        Eigen::VectorXd alpha = ldltDecomposition.solve(res);
        return -0.5 * res.dot(alpha) - 0.5 * (ldltDecomposition.vectorD().array().abs().log()).sum() - 0.5 * res.size() * std::log(2 * M_PI);
    }
    static double dLogPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::Ref<const Eigen::MatrixXd> &cov_gradient, const Eigen::Ref<const Eigen::VectorXd> &mean_gradient) {
        Eigen::VectorXd alpha = ldltDecomposition.solve(res);
        Eigen::MatrixXd alpha_alpha_t = alpha * alpha.transpose();
        return 0.5 * (alpha_alpha_t * cov_gradient - ldltDecomposition.solve(cov_gradient)).trace() + res.dot(ldltDecomposition.solve(mean_gradient));
    }
    static double ddLogPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::Ref<const Eigen::MatrixXd> &cov_gradient_l, const Eigen::Ref<const Eigen::MatrixXd> &cov_gradient_k, const Eigen::Ref<const Eigen::MatrixXd> &cov_hessian) {
        Eigen::VectorXd alpha = ldltDecomposition.solve(res);
        Eigen::MatrixXd alpha_alpha_t = alpha * alpha.transpose();
        auto a_l  = ldltDecomposition.solve(cov_gradient_l);
        auto a_k  = ldltDecomposition.solve(cov_gradient_k);
        auto sym_tens = 0.5 * ((a_l * alpha_alpha_t) + (a_l * alpha_alpha_t).transpose());
        double H1 = (alpha_alpha_t*cov_hessian).trace();
        double H2 = (ldltDecomposition.solve(cov_hessian)).trace();
        double H3 = (sym_tens * cov_gradient_k).trace();
        double H4 = (a_l * a_k).trace();
        return 0.5 * H1 - 0.5 * H2 - H3 + 0.5 * H4;
    }

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        return logPDF(mean_ - x, ldltDecomposition_);
    }
    double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) {
        return logPDF(jump, ldltDecomposition_);
    }
    double squaredMahalanobis(const Eigen::Ref<const Eigen::VectorXd> &jump) const {
        return jump.dot(ldltDecomposition_.solve(jump));
    }

    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma = 1.0) {
        Eigen::VectorXd z = Eigen::VectorXd::Zero(mean_.size());
        for(size_t i = 0; i < mean_.size(); i++) {
            z(i) = distN_(rng) * std::sqrt(std::abs(ldltDecomposition_.vectorD()(i)));
        }
        return mean_ + (ldltDecomposition_.transpositionsP().transpose() * (ldltDecomposition_.matrixL() * z)) * gamma;
    }

    Eigen::VectorXd get() const {
        return mean_;
    }
    void set(const Eigen::Ref<const Eigen::VectorXd> &x) {
        mean_ = x;
    }
    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    }
    void setLdltDecomposition(const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition) {
        ldltDecomposition_ = ldltDecomposition;
    }

    // Implement canonical transformation
    Eigen::MatrixXd toCanonical(const Eigen::MatrixXd& x) const {
        Eigen::MatrixXd centered = x.colwise() - mean_;
        Eigen::MatrixXd canonical = ldltDecomposition_.solve(centered);
        return canonical;
    }

    Eigen::MatrixXd toPhysical(const Eigen::MatrixXd& z) const {
        Eigen::MatrixXd original = ldltDecomposition_.matrixL() * z;
        original = ldltDecomposition_.transpositionsP().transpose() * original;
        original.colwise() += mean_;
        return original;
    }

    size_t dimension() const {
        return mean_.size();
    }

    static MultivariateNormalDistribution canonical(const size_t dim) {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(dim, dim);
        return MultivariateNormalDistribution(mean, cov);
    }
};


class MultivariateMixtureDistribution : public MultivariateDistribution<MultivariateMixtureDistribution> {
  private:
    std::vector<std::shared_ptr<MultivariateNormalDistribution>> components_;
    std::vector<double> weights_;
  public:
    MultivariateMixtureDistribution(
        const std::vector<std::shared_ptr<MultivariateNormalDistribution>>& components,
        const std::vector<double>& weights)
        : components_(components), weights_(weights) {}

    Eigen::VectorXd sample(std::default_random_engine &rng) {
        std::discrete_distribution<size_t> dist_weights(weights_.begin(), weights_.end());
        size_t idx = dist_weights(rng);
        return components_[idx]->sample(rng);
    }

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        double pdf = 0.0;
        for (size_t i = 0; i < components_.size(); ++i) {
            pdf += weights_[i] * std::exp(components_[i]->logPDF(x));
        }
        return std::log(pdf);
    }

    size_t dimension() const {
        return components_.empty() ? 0 : components_[0]->dimension();
    }
};


class MultivariateStudentDistribution : public ProposalDistribution<MultivariateStudentDistribution> {
  private:
    Eigen::VectorXd mean_;
    Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition_;
    double dofs_;
    std::normal_distribution<double> distN_;
  public:

    template <typename LDLTDerived>
    MultivariateStudentDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::LDLT<LDLTDerived> &ldltDecomposition, double nu)
        : mean_(mean), dofs_(nu), distN_(0., 1.) {
        Eigen::VectorXd d = ldltDecomposition.vectorD();
        Eigen::MatrixXd D = d.asDiagonal();
        Eigen::MatrixXd L = ldltDecomposition.matrixL().toDenseMatrix();
        Eigen::MatrixXd cov = L * D * L.transpose();
        ldltDecomposition_.compute(cov);
    }
    MultivariateStudentDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::MatrixXd> &cov, double nu)
        : mean_(mean), dofs_(nu), distN_(0., 1.) {
        ldltDecomposition_.compute(cov);
    }
    MultivariateStudentDistribution() = default;

    static double logPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const double &nu) {
        size_t dim = res.size();
        Eigen::VectorXd alpha = ldltDecomposition.solve(res);
        double quad = res.dot(alpha);
        double logDet = ldltDecomposition.vectorD().array().log().sum();
        double logGammaTerm = std::lgamma(0.5 * (nu + dim)) - std::lgamma(0.5 * nu);
        double logNormTerm = -0.5 * dim * std::log(nu * M_PI);
        double logDetTerm = -0.5 * logDet;
        double logQuadTerm = -0.5 * (nu + dim) * std::log(1 + quad / nu);
        return logGammaTerm + logNormTerm + logDetTerm + logQuadTerm;
    }

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        return logPDF(mean_ - x, ldltDecomposition_, dofs_);
    }
    double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) {
        return logPDF(jump, ldltDecomposition_, dofs_);
    }
    double squaredMahalanobis(const Eigen::Ref<const Eigen::VectorXd> &jump) const {
        return jump.dot(ldltDecomposition_.solve(jump));
    }

    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma = 1.0) {
        Eigen::VectorXd z = Eigen::VectorXd::Zero(mean_.size());
        for(size_t i = 0; i < mean_.size(); i++) {
            z(i) = distN_(rng);
            double chi = 0.0;
            for(size_t j = 0; j < dofs_; j++) {
                chi += std::pow(distN_(rng), 2);
            }
            z(i) = std::sqrt(std::abs(ldltDecomposition_.vectorD()(i))) * z(i) / std::sqrt(chi / dofs_);
        }
        return mean_ + (ldltDecomposition_.transpositionsP().transpose() * (ldltDecomposition_.matrixL() * z)) * gamma;
    }

    Eigen::VectorXd get() const {
        return mean_;
    }
    void set(const Eigen::Ref<const Eigen::VectorXd> &x) {
        mean_ = x;
    }
    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    }
    void setLdltDecomposition(const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition) {
        ldltDecomposition_ = ldltDecomposition;
    }
    void setDoFs(double nu) {
        dofs_ = nu;
    }

    Eigen::MatrixXd toCanonical(const Eigen::MatrixXd& x) const {
        Eigen::MatrixXd centered = x.colwise() - mean_;
        Eigen::MatrixXd canonical = ldltDecomposition_.solve(centered);
        return canonical;
    }

    Eigen::MatrixXd toPhysical(const Eigen::MatrixXd& z) const {
        Eigen::MatrixXd original = ldltDecomposition_.matrixL() * z;
        original = ldltDecomposition_.transpositionsP().transpose() * original;
        original.colwise() += mean_;
        return original;
    }

    size_t dimension() const {
        return mean_.size();
    }

    static MultivariateStudentDistribution canonical(const size_t dim, const double nu) {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(dim, dim);
        return MultivariateStudentDistribution(mean, cov, nu);
    }
};


class MultivariateUniformDistribution : public MultivariateDistribution<MultivariateUniformDistribution> {
  private:
    Eigen::VectorXd lowerBound_;
    Eigen::VectorXd upperBound_;

    // FIX 2: Standard random generator must span [0, 1] to correctly scale physical bounds
    std::uniform_real_distribution<double> distU_;

  public:

    MultivariateUniformDistribution(const Eigen::Ref<const Eigen::VectorXd> &lowerBound,
                                    const Eigen::Ref<const Eigen::VectorXd> &upperBound)
        : lowerBound_(lowerBound), upperBound_(upperBound), distU_(0.0, 1.0) {}

    MultivariateUniformDistribution() = default;

    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma = 1.0) {
        Eigen::VectorXd rv(lowerBound_.size());
        for(size_t i = 0; i < lowerBound_.size(); i++) {
            // Now samples exactly between [lowerBound, upperBound]
            rv(i) = lowerBound_(i) + (upperBound_(i) - lowerBound_(i)) * distU_(rng) * gamma;
        }
        return rv;
    }

    Eigen::MatrixXd toCanonical(const Eigen::MatrixXd& x) const {

        Eigen::RowVectorXd a = lowerBound_.transpose();
        Eigen::RowVectorXd b = upperBound_.transpose();

        Eigen::MatrixXd scaled = (x.rowwise() - a).array().rowwise() / (b - a).array();

        // 2. Do the global scalar math on the resulting array: 2.0 * scaled - 1.0
        return (2.0 * scaled.array() - 1.0).matrix();
    }

    Eigen::MatrixXd toPhysical(const Eigen::MatrixXd& z) const {

        Eigen::RowVectorXd a = lowerBound_.transpose();
        Eigen::RowVectorXd b = upperBound_.transpose();

        Eigen::MatrixXd original = (z.array() + 1.0) / 2.0;
        original = original.array().rowwise() * (b - a).array();
        original.rowwise() += a;

        return original;
    }

    size_t dimension() const {
        return lowerBound_.size();
    }

    static MultivariateUniformDistribution canonical(const size_t dim) {
        Eigen::VectorXd lowerBound = -Eigen::VectorXd::Ones(dim);
        Eigen::VectorXd upperBound =  Eigen::VectorXd::Ones(dim);
        return MultivariateUniformDistribution(lowerBound, upperBound);
    }
};


class UniformSphereDistribution : public MultivariateDistribution<UniformSphereDistribution> {
  private:
    std::normal_distribution<double> distN_{0., 1.};
    std::size_t dim_{1};
  public:

    UniformSphereDistribution(size_t dim): dim_(dim), distN_(0., 1.) {}
    UniformSphereDistribution() = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        return 0.0;
    }
    Eigen::VectorXd sample(std::default_random_engine &rng) {
        Eigen::VectorXd x(dim_);
        for(size_t i = 0; i < dim_; i++) {
            x(i) = distN_(rng);
        }
        return x / x.norm();
    }


    Eigen::MatrixXd toCanonical(const Eigen::MatrixXd& x) const {
        Eigen::MatrixXd canonical = x.normalized();
        return canonical;
    }

    Eigen::MatrixXd toPhysical(const Eigen::MatrixXd& z) const {
        Eigen::MatrixXd original = z.normalized();
        return original;
    }

    size_t dimension() const {
        return dim_;
    }
};


class NormalInverseWishartDistribution : public MultivariateDistribution<NormalInverseWishartDistribution> {
  private:
    Eigen::VectorXd mean_;
    double kappa_;
    double nu_;
    size_t dim_;
    Eigen::LDLT<Eigen::MatrixXd> covLDLT_;
    double logDeterminant_;
    std::normal_distribution<double> distN_{0., 1.};

  public:

    NormalInverseWishartDistribution(
        const Eigen::Ref<const Eigen::VectorXd> &mean,
        double kappa, double nu,
        const Eigen::Ref<const Eigen::MatrixXd> &psi
    ) : mean_(mean), kappa_(kappa), nu_(nu - mean.size() + 1), dim_(mean.size()) {
        if(psi.rows() != psi.cols() || psi.rows() != dim_) throw std::invalid_argument("Scale matrix Ψ must be square and match the dimensionality of the mean");
        if(kappa_ <= 0) throw std::invalid_argument("κ must be positive");
        if(nu <= dim_ - 1) throw std::invalid_argument("ν must be greater than dimension - 1");

        double scaling = (kappa_ + 1.0) / (kappa_ * nu_);
        Eigen::MatrixXd scaledPsi = scaling * psi;
        covLDLT_.compute(scaledPsi);
        if(covLDLT_.info() != Eigen::Success) throw std::invalid_argument("Scaled covariance matrix must be positive definite");
        logDeterminant_ = covLDLT_.vectorD().array().log().sum();
    }
    NormalInverseWishartDistribution() = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const {
        Eigen::VectorXd res = x - mean_;
        return MultivariateStudentDistribution::logPDF(res, covLDLT_, nu_);
    }

    Eigen::VectorXd sample(std::default_random_engine &rng) {
        const auto &L = covLDLT_.matrixL();
        const auto &P = covLDLT_.transpositionsP();
        const Eigen::VectorXd D = covLDLT_.vectorD();

        Eigen::VectorXd z = Eigen::VectorXd::Zero(dim_);
        for(size_t i = 0; i < dim_; ++i) {
            z(i) = distN_(rng) * std::sqrt(std::abs(D(i)));
        }
        Eigen::VectorXd y = P.transpose() * (L * z);

        std::chi_squared_distribution<double> distChi(nu_);
        double s = distChi(rng);
        return mean_ + y / std::sqrt(s / nu_);
    }

    const Eigen::VectorXd &mean() const {
        return mean_;
    }
    const double &kappa() const {
        return kappa_;
    }
    const double &nu() const {
        return nu_;
    }
    const Eigen::MatrixXd covariance() const {
        return covLDLT_.reconstructedMatrix();
    }

    Eigen::MatrixXd toCanonical(const Eigen::MatrixXd& x) const {
        Eigen::MatrixXd centered = x.colwise() - mean_;
        Eigen::MatrixXd canonical = covLDLT_.solve(centered);
        return canonical;
    }

    Eigen::MatrixXd toPhysical(const Eigen::MatrixXd& z) const {
        Eigen::MatrixXd original = covLDLT_.matrixL() * z;
        original = covLDLT_.transpositionsP().transpose() * original;
        original.colwise() += mean_;
        return original;
    }

    size_t dimension() const {
        return dim_;
    }

    static NormalInverseWishartDistribution canonical(const size_t dim, double kappa, double nu) {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        Eigen::MatrixXd psi = Eigen::MatrixXd::Identity(dim, dim);
        return NormalInverseWishartDistribution(mean, kappa, nu, psi);
    }

    /**
     * @brief Computes an empirical Normal-Inverse-Wishart prior from the provided dataset. Useful for initializing the prior in a Dirichlet Process Mixture Model (DPMM) when you have observed data.
     * @param data The observed data matrix.
     * @param expected_clusters The expected number of clusters.
     * @param kappa0 The prior precision parameter.
     * @return The computed empirical Normal-Inverse-Wishart prior.
     */
    static NormalInverseWishartDistribution empiricalPrior(const Eigen::MatrixXd &data, double expected_clusters, double kappa0) {

        int D = static_cast<int>(data.cols());
        double N = static_cast<double>(data.rows());

        // 1. Compute empirical global mean (1 x D) -> Convert to Column Vector (D x 1)
        Eigen::VectorXd mu0 = data.colwise().mean().transpose();

        // 2. Set degrees of freedom to the weakest valid setting
        double nu0 = D + 2.0;

        // 3. Compute empirical covariance matrix (D x D)
        // Centering the data: subtract the mean row from every row
        Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

        // Covariance = (Centered^T * Centered) / (N - 1)
        Eigen::MatrixXd global_cov = (centered.transpose() * centered) / (N - 1.0);

        // 4. Scale Psi0 (the scale matrix) based on expected cluster density
        // We multiply by (nu0 - D - 1) to ensure the expected value of the prior covariance
        // matches the fraction of the global variance exactly.
        double scale_factor = (nu0 - static_cast<double>(D) - 1.0) / expected_clusters;
        Eigen::MatrixXd Psi0 = global_cov * scale_factor;

        // Fix potential numerical edge-cases where columns have zero variance
        Psi0.diagonal().array() += 1e-6;

        // 5. Return the constructed instance
        // (Assuming your constructor takes mu0, kappa0, nu0, and Psi0)
        return NormalInverseWishartDistribution(mu0, kappa0, nu0, Psi0);
    }
};

} // namespace cmp::distribution

#endif // DISTRIBUTION_HPP