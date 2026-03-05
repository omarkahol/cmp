#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "cmp_defines.h"

namespace cmp::distribution {

class UnivariateDistribution {
  public:

    /**
     * @brief Evaluate the log-pdf of the distribution at point x
     *
     * @param x The point at which to evaluate the log-pdf
     * @return double
     */
    virtual double logPDF(const double &x) = 0;

    /**
     * @brief Evaluate the first derivative of the log-pdf
     *
     * @param x Point at which to evaluate the derivative
     * @return double
     */
    virtual double dLogPDF(const double &x) = 0;

    /**
     * @brief Evaluate the second derivative of the log-pdf
     *
     * @param x Point at which to evaluate the derivative
     * @return double
     */
    virtual double ddLogPDF(const double &x) = 0;

    /**
     * @brief Evaluate the CDF of the distribution at point x
     *
     * @param x Point at which to evaluate the CDF
     * @return double
     */
    virtual double CDF(const double &x) = 0;

    /**
     * @brief Evaluate the quantile function of the distribution
     *
     * @param p The probability at which to evaluate the quantile
     * @return double
     */
    virtual double quantile(const double &p) = 0;

    /**
     * @brief Sample from the distribution
     *
     * @return double
     */
    virtual double sample(std::default_random_engine &rng) = 0;
};

class MultivariateDistribution {
  public:

    /**
     * Accepts a Eigen::Ref to avoid unnecessary copies
     * @brief Evaluate the log-pdf of the distribution at point x
     */
    virtual double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const = 0;

    /**
     * @brief Generate a sample from the distribution
     *
     * @return Eigen::VectorXd
     */
    virtual Eigen::VectorXd sample(std::default_random_engine &rng) = 0;
};

class ProposalDistribution : public MultivariateDistribution {
  public:
    /**
     * @brief Evaluate the log of the jumping distribution PDF at point jump
     *
     * @param jump The jump vector
     * @return double
     */
    virtual double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) = 0;

    /**
     * @brief Generate a sample from the distribution
     * @param gamma Scaling factor for the covariance of the proposal distribution
     *
     * @return Eigen::VectorXd
     */
    virtual Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma) = 0;

    Eigen::VectorXd sample(std::default_random_engine &rng) override {
        return sample(rng, 1.0);
    }

    virtual Eigen::VectorXd get() = 0;

    /**
     * @brief Set the current parameter
     * @param x The new parameter
     */
    virtual void set(const Eigen::Ref<const Eigen::VectorXd> &x) = 0;

};


// Univariate normal distribution
class NormalDistribution: public UnivariateDistribution {
  private:
    double mean_{0.0};
    double std_{1.0};
    std::normal_distribution<double> distN_;

  public:

    // Constructors
    NormalDistribution(double mean, double sd): mean_(mean), std_(sd), distN_(0., 1.) {}
    NormalDistribution() = default;
    NormalDistribution(const NormalDistribution &other) = default;
    NormalDistribution(NormalDistribution &&other) = default;

    // Destructor
    ~NormalDistribution() = default;

    // Assignment operators
    NormalDistribution &operator=(const NormalDistribution &other) = default;
    NormalDistribution &operator=(NormalDistribution &&other) = default;

    double logPDF(const double &x) override;

    static double logPDF(const double &res, const double &std);

    /**
     * @brief Evaluate the first derivative of the log-pdf
     *
     * @param x Point at which to evaluate the derivative
     * @return double
     */
    double dLogPDF(const double &x) override;

    /**
     * @brief Evaluate the second derivative of the log-pdf
     *
     * @param x Point at which to evaluate the derivative
     * @return double
     */
    double ddLogPDF(const double &x) override;

    double CDF(const double &x) override;
    double quantile(const double &p) override;
    double sample(std::default_random_engine &rng) override;

    void setMean(double mean) {
        mean_ = mean;
    };
    void setStd(double std) {
        std_ = std;
    };

    double mean() const {
        return mean_;
    };
    double std() const {
        return std_;
    };

    static std::shared_ptr<UnivariateDistribution> make(double mean, double std) {
        return std::make_shared<NormalDistribution>(mean, std);
    };
};

// Multivariate normal distribution
class MultivariateNormalDistribution: public ProposalDistribution {
  private:
    Eigen::VectorXd mean_;
    Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition_;
    std::normal_distribution<double> distN_;
  public:

    // Constructors

    // Convenience: construct directly from a covariance matrix (compute LDLT internally)
    MultivariateNormalDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::MatrixXd> &cov)
        : mean_(mean), distN_(0., 1.) {
        ldltDecomposition_.compute(cov);
    }

    // Overload: accept an LDLT decomposition already using Eigen::MatrixXd and store it directly
    template<typename MatrixType>
    MultivariateNormalDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::LDLT<MatrixType> &ldltDecomposition)
        : mean_(mean), ldltDecomposition_(ldltDecomposition), distN_(0., 1.) {}

    // Note: no separate rvalue overload for mean to avoid overload ambiguity with Eigen::Ref
    // The Ref-taking overload above will accept both lvalues and temporaries; copying/moving
    // Eigen objects should be done explicitly by callers if needed for performance.

    MultivariateNormalDistribution() = default;

    // Destructor
    ~MultivariateNormalDistribution() = default;

    // Special members - defaulted explicitly for clarity. This avoids accidental deletion
    // of the copy constructor/assignment when moving/assigning is declared elsewhere.
    MultivariateNormalDistribution(const MultivariateNormalDistribution &other) = default;
    MultivariateNormalDistribution(MultivariateNormalDistribution &&other) = default;
    MultivariateNormalDistribution &operator=(const MultivariateNormalDistribution &other) = default;
    MultivariateNormalDistribution &operator=(MultivariateNormalDistribution &&other) = default;


    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override;
    static double logPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition);

    static double dLogPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::Ref<const Eigen::MatrixXd> &cov_gradient, const Eigen::Ref<const Eigen::VectorXd> &mean_gradient);
    static double ddLogPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::Ref<const Eigen::MatrixXd> &cov_gradient_l, const Eigen::Ref<const Eigen::MatrixXd> &cov_gradient_k, const Eigen::Ref<const Eigen::MatrixXd> &cov_hessian);

    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma = 1.0) override;

    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    };
    void setLdltDecomposition(const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition) {
        ldltDecomposition_ = ldltDecomposition;
    };

    Eigen::VectorXd get() override;
    void set(const Eigen::Ref<const Eigen::VectorXd> &x) override;
    double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) override;

};

// Uniform sphere distribution
class UniformSphereDistribution: public MultivariateDistribution {
  private:
    std::normal_distribution<double> distN_{0., 1.};
    std::size_t dim_{1};
  public:
    UniformSphereDistribution(size_t dim): dim_(dim), distN_(0., 1.) {}
    UniformSphereDistribution() = default;

    // Destructor
    ~UniformSphereDistribution() = default;

    // Assignment operators
    UniformSphereDistribution &operator=(const UniformSphereDistribution &other) = default;
    UniformSphereDistribution &operator=(UniformSphereDistribution &&other) = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override {
        return 0.0;
    };

    Eigen::VectorXd sample(std::default_random_engine &rng) override;
};

// Uniform distribution
class UniformDistribution: public UnivariateDistribution {
  private:
    double lowerBound_{0.0};
    double upperBound_{1.0};
    std::uniform_real_distribution<double> distN_{0., 1.};
  public:
    UniformDistribution(double a, double b): lowerBound_(a), upperBound_(b), distN_(0., 1.) {}
    UniformDistribution() = default;

    // Destructor
    ~UniformDistribution() = default;

    // Assignment operators
    UniformDistribution &operator=(const UniformDistribution &other) = default;
    UniformDistribution &operator=(UniformDistribution &&other) = default;

    double logPDF(const double &x) override;
    double dLogPDF(const double &x) {
        return 0.0;
    };
    double ddLogPDF(const double &x) {
        return 0.0;
    };
    double CDF(const double & x) override;
    double quantile(const double & p) override;
    double sample(std::default_random_engine &rng) override;

    void setLowerBound(double a) {
        lowerBound_ = a;
    };
    void setUpperBound(double b) {
        upperBound_ = b;
    };

    static std::shared_ptr<UnivariateDistribution> make(double a, double b) {
        return std::make_shared<UniformDistribution>(a, b);
    };
};

// Inverse gamma distribution
class InverseGammaDistribution: public UnivariateDistribution {
  private:
    double alpha_;
    double beta_;
    std::gamma_distribution<double> distGamma_;
  public:
    InverseGammaDistribution(double alpha, double beta): alpha_(alpha), beta_(beta), distGamma_(alpha, 1 / beta) {}
    InverseGammaDistribution() = default;

    // Destructor
    ~InverseGammaDistribution() = default;

    // Assignment operators
    InverseGammaDistribution &operator=(const InverseGammaDistribution &other) = default;
    InverseGammaDistribution &operator=(InverseGammaDistribution &&other) = default;

    double logPDF(const double & x) override;
    double dLogPDF(const double & x) override;
    double ddLogPDF(const double & x) override;
    double CDF(const double & x) override;
    double quantile(const double & p) override;
    double sample(std::default_random_engine &rng) override;

    void setAlpha(double alpha) {
        alpha_ = alpha;
    };
    void setBeta(double beta) {
        beta_ = beta;
    };

    static std::shared_ptr<UnivariateDistribution> make(double alpha, double beta) {
        return std::make_shared<InverseGammaDistribution>(alpha, beta);
    };
};

// Multivariate normal distribution
class MultivariateStudentDistribution: public ProposalDistribution {
  private:
    Eigen::VectorXd mean_;
    Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition_;
    double dofs_;
    std::normal_distribution<double> distN_;
  public:

    // Constructors
    // Templated LDLT constructor to accept LDLT objects for different matrix instantiations
    template <typename LDLTDerived>
    MultivariateStudentDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::LDLT<LDLTDerived> &ldltDecomposition, double nu)
        : mean_(mean), dofs_(nu), distN_(0., 1.) {
        Eigen::VectorXd d = ldltDecomposition.vectorD();
        Eigen::MatrixXd D = d.asDiagonal();
        Eigen::MatrixXd L = ldltDecomposition.matrixL().toDenseEigen::MatrixXd();
        Eigen::MatrixXd cov = L * D * L.transpose();
        ldltDecomposition_.compute(cov);
    }

    // Convenience constructor: accept covariance matrix and compute LDLT internally
    MultivariateStudentDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::MatrixXd> &cov, double nu)
        : mean_(mean), dofs_(nu), distN_(0., 1.) {
        ldltDecomposition_.compute(cov);
    }

    MultivariateStudentDistribution() = default;

    // Destructor
    ~MultivariateStudentDistribution() = default;

    // Special members - default explicitly
    MultivariateStudentDistribution(const MultivariateStudentDistribution &other) = default;
    MultivariateStudentDistribution(MultivariateStudentDistribution &&other) = default;
    MultivariateStudentDistribution &operator=(const MultivariateStudentDistribution &other) = default;
    MultivariateStudentDistribution &operator=(MultivariateStudentDistribution &&other) = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override;
    static double logPDF(const Eigen::Ref<const Eigen::VectorXd> &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const double &nu);

    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma = 1.0) override;

    void setMean(const Eigen::Ref<const Eigen::VectorXd> &mean) {
        mean_ = mean;
    };
    void setLdltDecomposition(const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition) {
        ldltDecomposition_ = ldltDecomposition;
    };
    void setDoFs(double nu) {
        dofs_ = nu;
    };

    Eigen::VectorXd get() override;
    void set(const Eigen::Ref<const Eigen::VectorXd> &x) override;
    double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) override;

};

// Multivariate uniform distribution

class MultivariateUniformDistribution: public ProposalDistribution {
  private:
    Eigen::VectorXd mean_;
    Eigen::VectorXd size_;
    std::uniform_real_distribution<double> distN_;
  public:
    MultivariateUniformDistribution(const Eigen::Ref<const Eigen::VectorXd> &mean, const Eigen::Ref<const Eigen::VectorXd> &size): mean_(mean), size_(size), distN_(0, 1) {}
    MultivariateUniformDistribution() = default;

    // Destructor
    ~MultivariateUniformDistribution() = default;

    // Assignment operators
    MultivariateUniformDistribution &operator=(const MultivariateUniformDistribution &other) = default;
    MultivariateUniformDistribution &operator=(MultivariateUniformDistribution &&other) = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override;
    double logJumpPDF(const Eigen::Ref<const Eigen::VectorXd> &jump) override;
    Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma = 1.0) override;

    Eigen::VectorXd get() override;
    void set(const Eigen::Ref<const Eigen::VectorXd> &x) override;
};


// Univariate log-normal distribution
class LogNormalDistribution: public UnivariateDistribution {
  private:
    double mean_{0.0};
    double std_{1.0};
    std::normal_distribution<double> distN_;
  public:
    LogNormalDistribution(double mu, double sigma): mean_(mu), std_(sigma), distN_(0, 1) {}
    LogNormalDistribution() = default;

    double logPDF(const double & x) override;
    double dLogPDF(const double & x) override;
    double ddLogPDF(const double & x) override;
    double CDF(const double & x) override;
    double quantile(const double & p) override;
    double sample(std::default_random_engine &rng) override;

    void setMean(double mu) {
        mean_ = mu;
    };
    void setStd(double sigma) {
        std_ = sigma;
    };

    static std::shared_ptr<UnivariateDistribution> make(double mu, double sigma) {
        return std::make_shared<LogNormalDistribution>(mu, sigma);
    };
};

// Univariate t distribution
class StudentDistribution: public UnivariateDistribution {
  private:
    double dofs_;
    double mean_;
    double std_;
    std::student_t_distribution<double> distN_;
  public:
    StudentDistribution(double nu, double mu, double sigma): dofs_(nu), mean_(mu), std_(sigma), distN_(nu) {}
    StudentDistribution() = default;

    // Destructor
    ~StudentDistribution() = default;

    // Assignment operators
    StudentDistribution &operator=(const StudentDistribution &other) = default;
    StudentDistribution &operator=(StudentDistribution &&other) = default;

    double logPDF(const double & x) override;
    double dLogPDF(const double & x) override;
    double ddLogPDF(const double & x) override;
    double CDF(const double & x) override;
    double quantile(const double & p) override;
    double sample(std::default_random_engine &rng) override;

    void setDoFs(double dofs) {
        dofs_ = dofs;
    };
    void setMean(double mu) {
        mean_ = mu;
    };
    void setStd(double sigma) {
        std_ = sigma;
    };

    static std::shared_ptr<UnivariateDistribution> make(double nu, double mu, double sigma) {
        return std::make_shared<StudentDistribution>(nu, mu, sigma);
    };
};

// Power law distribution
class PowerLawDistribution: public UnivariateDistribution {
  private:
    double degree_;
    double lowerBound_;
    std::uniform_real_distribution<double> distU_;

  public:
    PowerLawDistribution(double alpha, double lowerBound = 1.0): degree_(alpha), distU_(0., 1.), lowerBound_(lowerBound) {}
    PowerLawDistribution() = default;

    // Destructor
    ~PowerLawDistribution() = default;

    // Assignment operators
    PowerLawDistribution &operator=(const PowerLawDistribution &other) = default;
    PowerLawDistribution &operator=(PowerLawDistribution &&other) = default;

    double logPDF(const double & x) override;
    double dLogPDF(const double & x) override;
    double ddLogPDF(const double & x) override;
    double CDF(const double & x) override;
    double quantile(const double & p) override;
    double sample(std::default_random_engine &rng) override;

    void setDegree(double degree) {
        degree_ = degree;
    };

    static std::shared_ptr<UnivariateDistribution> make(double degree, double lowerBound = 1.0) {
        return std::make_shared<PowerLawDistribution>(degree, lowerBound);
    };
};


// Smoothed uniform distribution with normal
class SmoothUniformDistribution: public UnivariateDistribution {
  private:
    double lowerBound_;
    double upperBound_;
    double std_;
    std::uniform_real_distribution<double> distU_;
    std::normal_distribution<double> distN_;
  public:
    SmoothUniformDistribution(double lowerBound, double upperBound, double sigma): lowerBound_(lowerBound), upperBound_(upperBound), std_(sigma), distU_(0, 1), distN_(0, 1) {}
    SmoothUniformDistribution() = default;

    // Destructor
    ~SmoothUniformDistribution() = default;

    // Assignment operators
    SmoothUniformDistribution &operator=(const SmoothUniformDistribution &other) = default;
    SmoothUniformDistribution &operator=(SmoothUniformDistribution &&other) = default;

    double logPDF(const double & x) override;
    double dLogPDF(const double & x) override;
    double ddLogPDF(const double & x) override;
    double CDF(const double & x) override;
    double quantile(const double & p) override;
    double sample(std::default_random_engine &rng) override;

    void setLowerBound(double a) {
        lowerBound_ = a;
    };
    void setUpperBound(double b) {
        upperBound_ = b;
    };
    void setStd(double sigma) {
        std_ = sigma;
    };

    static std::shared_ptr<UnivariateDistribution> make(double a, double b, double sigma) {
        return std::make_shared<SmoothUniformDistribution>(a, b, sigma);
    };
};

class MultivariateMixtureDistribution: public MultivariateDistribution {
  private:
    std::vector<std::shared_ptr<MultivariateDistribution>> components_;
    std::vector<double> weights_;
    std::discrete_distribution<size_t> distDiscrete_;

  public:
    MultivariateMixtureDistribution(const std::vector<std::shared_ptr<MultivariateDistribution>> &components, const std::vector<double> &weights)
        : components_(components), weights_(weights), distDiscrete_(weights.begin(), weights.end()) {
        if(components.size() != weights.size()) {
            throw std::invalid_argument("Number of components must match number of weights");
        }
    }

    MultivariateMixtureDistribution() = default;

// Destructor
    ~MultivariateMixtureDistribution() = default;

// Assignment operators
    MultivariateMixtureDistribution &operator=(const MultivariateMixtureDistribution &other) = default;
    MultivariateMixtureDistribution &operator=(MultivariateMixtureDistribution &&other) = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override {
        double pdf = 0.0;
        for(size_t i = 0; i < components_.size(); i++) {
            pdf += weights_[i] * std::exp(components_[i]->logPDF(x));
        }
        return std::log(pdf);
    }

    Eigen::VectorXd sample(std::default_random_engine &rng) override {
        size_t idx = distDiscrete_(rng);
        return components_[idx]->sample(rng);
    }
};

class ComponentDistribution : public MultivariateDistribution {
  private:
    std::vector<std::shared_ptr<UnivariateDistribution>> components_;
    Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition_;
  public:
    ComponentDistribution(const std::vector<std::shared_ptr<UnivariateDistribution>> &components, const Eigen::Ref<const Eigen::MatrixXd> &cov)
        : components_(components) {
        ldltDecomposition_.compute(cov);
    }
    ComponentDistribution() = default;
    ~ComponentDistribution() = default;
    ComponentDistribution &operator=(const ComponentDistribution &other) = default;
    ComponentDistribution &operator=(ComponentDistribution &&other) = default;

    Eigen::VectorXd sample(std::default_random_engine &rng) override {
        size_t dim = components_.size();
        Eigen::VectorXd sample(dim);
        for(size_t i = 0; i < dim; i++) {
            sample(i) = components_[i]->sample(rng);
        }
        // Apply the covariance structure
        Eigen::VectorXd z = ldltDecomposition_.matrixL() * sample;
        return z;
    }

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override {
        double log_pdf = 0.0;
        for(size_t i = 0; i < components_.size(); i++) {
            log_pdf += components_[i]->logPDF(x(i));
        }
        return log_pdf;
    }
};

class UnivariateMixtureDistribution: public UnivariateDistribution {
  private:
    std::vector<std::shared_ptr<UnivariateDistribution>> components_;
    std::vector<double> weights_;
    std::discrete_distribution<size_t> distDiscrete_;

  public:
    UnivariateMixtureDistribution(const std::vector<std::shared_ptr<UnivariateDistribution>> &components, const std::vector<double> &weights)
        : components_(components), weights_(weights), distDiscrete_(weights.begin(), weights.end()) {
        if(components.size() != weights.size()) {
            throw std::invalid_argument("Number of components must match number of weights");
        }
    }

    UnivariateMixtureDistribution() = default;

    // Destructor
    ~UnivariateMixtureDistribution() = default;

    // Assignment operators
    UnivariateMixtureDistribution &operator=(const UnivariateMixtureDistribution &other) = default;
    UnivariateMixtureDistribution &operator=(UnivariateMixtureDistribution &&other) = default;

    double logPDF(const double & x) override {
        double log_pdf = 0.0;
        for(size_t i = 0; i < components_.size(); i++) {
            log_pdf += std::log(weights_[i]) + components_[i]->logPDF(x);
        }
        return std::log(log_pdf);
    }

    double dLogPDF(const double & x) override {
        double dlog_pdf = 0.0;
        for(size_t i = 0; i < components_.size(); i++) {
            dlog_pdf += weights_[i] * std::exp(components_[i]->logPDF(x)) * components_[i]->dLogPDF(x);
        }
        return dlog_pdf / std::exp(logPDF(x));
    }

    double ddLogPDF(const double & x) override {
        double ddlog_pdf = 0.0;
        for(size_t i = 0; i < components_.size(); i++) {
            ddlog_pdf += weights_[i] * std::exp(components_[i]->logPDF(x)) * (components_[i]->ddLogPDF(x) + std::pow(components_[i]->dLogPDF(x), 2));
        }
        double dlog_pdf = dLogPDF(x);
        return (ddlog_pdf / std::exp(logPDF(x))) - std::pow(dlog_pdf, 2);
    }

    double CDF(const double & x) override {
        double cdf = 0.0;
        for(size_t i = 0; i < components_.size(); i++) {
            cdf += weights_[i] * components_[i]->CDF(x);
        }
        return cdf;
    }

    double quantile(const double & p) override {
        // Numerical inversion of the CDF using bisection method
        double lower = -1e6;
        double upper = 1e6;
        double mid;
        double tol = 1e-6;
        while(upper - lower > tol) {
            mid = (lower + upper) / 2.0;
            if(CDF(mid) < p) {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        return mid;
    }

    double sample(std::default_random_engine &rng) override {
        size_t idx = distDiscrete_(rng);
        return components_[idx]->sample(rng);
    }
};


class NormalInverseWishartDistribution: public MultivariateDistribution {
  private:
    Eigen::VectorXd mean_;                            // prior mean (d)
    double kappa_;                                    // prior scaling
    double nu_;                                       // prior degrees of freedom
    size_t dim_;                                      // dimensionality (d)
    Eigen::LDLT<Eigen::MatrixXd> covLDLT_;           // prior scale matrix (d x d)
    double logDeterminant_;                           // log determinant of the scale matrix

    // For sampling
    std::normal_distribution<double> distN_{0., 1.};

  public:
    NormalInverseWishartDistribution(
        const Eigen::Ref<const Eigen::VectorXd> &mean,
        double kappa,
        double nu,
        const Eigen::Ref<const Eigen::MatrixXd> &psi  // scale matrix of the InvWishart
    )
        : mean_(mean),
          kappa_(kappa),
          nu_(nu - mean.size() + 1),   // transform to the Student-t degrees of freedom
          dim_(mean.size()) {
        if(psi.rows() != psi.cols() || psi.rows() != dim_) {
            throw std::invalid_argument("Scale matrix Ψ must be square and match the dimensionality of the mean");
        }
        if(kappa_ <= 0) {
            throw std::invalid_argument("κ must be positive");
        }
        if(nu <= dim_ - 1) {
            throw std::invalid_argument("ν must be greater than dimension - 1");
        }

        // Scale Ψ into the predictive covariance Σ = ((κ+1)/(κ * ν')) Ψ
        double scaling = (kappa_ + 1.0) / (kappa_ * nu_);

        // Compute scaled covariance
        Eigen::MatrixXd scaledPsi = scaling * psi;

        // LDLT decomposition for numerical stability
        covLDLT_.compute(scaledPsi);
        if(covLDLT_.info() != Eigen::Success) {
            throw std::invalid_argument("Scaled covariance matrix must be positive definite");
        }

        // Precompute log|Σ|
        logDeterminant_ = covLDLT_.vectorD().array().log().sum();
    }


    NormalInverseWishartDistribution() = default;

    ~NormalInverseWishartDistribution() = default;

    NormalInverseWishartDistribution &operator=(const NormalInverseWishartDistribution &other) = default;
    NormalInverseWishartDistribution &operator=(NormalInverseWishartDistribution &&other) = default;

    double logPDF(const Eigen::Ref<const Eigen::VectorXd> &x) const override;
    Eigen::VectorXd sample(std::default_random_engine &rng) override;


    // Getters
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
};

} // namespace cmp::distribution

#endif