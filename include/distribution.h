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
            virtual double logPDF(const Eigen::VectorXd &x) = 0;
            
            /**
             * @brief Generate a sample from the distribution
             * 
             * @return Eigen::VectorXd
             */
            virtual Eigen::VectorXd sample(std::default_random_engine &rng)=0;
    };

    class ProposalDistribution : public MultivariateDistribution {
        public:

            virtual double logJumpPDF(const Eigen::VectorXd &jump) = 0;
            
            /**
             * @brief Generate a sample from the distribution
             * 
             * @return Eigen::VectorXd
             */
            virtual Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma)=0;

            Eigen::VectorXd sample(std::default_random_engine &rng) override {
                return sample(rng,1.0);
            }

            virtual Eigen::VectorXd get() = 0;
            virtual void set(const Eigen::VectorXd &x) = 0;

    };


    // Univariate normal distribution
    class NormalDistribution: public UnivariateDistribution {
        private:
            double mean_{0.0};
            double std_{1.0};
            std::normal_distribution<double> distN_;

        public:

            // Constructors
            NormalDistribution(double mean, double sd): mean_(mean), std_(sd), distN_(0.,1.) {}
            NormalDistribution() = default;

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
            
            void setMean(double mean){mean_ = mean;};
            void setStd(double std){std_ = std;};

            double mean() const {return mean_;};
            double std() const {return std_;};

            static std::shared_ptr<UnivariateDistribution> make(double mean, double std) {
                return std::make_shared<NormalDistribution>(mean,std);
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
            MultivariateNormalDistribution(Eigen::VectorXd mean, Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition): mean_(mean), ldltDecomposition_(ldltDecomposition), distN_(0., 1.) {}
            MultivariateNormalDistribution() = default;

            // Destructor
            ~MultivariateNormalDistribution() = default;

            // Assignment operators
            MultivariateNormalDistribution &operator=(const MultivariateNormalDistribution &other) = default;
            MultivariateNormalDistribution &operator=(MultivariateNormalDistribution &&other) = default;


            double logPDF(const Eigen::VectorXd &x) override;
            static double logPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition);

            static double dLogPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::MatrixXd &cov_gradient, const Eigen::VectorXd &mean_gradient);
            static double ddLogPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::MatrixXd &cov_gradient_l, const Eigen::MatrixXd &cov_gradient_k, const Eigen::MatrixXd &cov_hessian);

            Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma=1.0) override;
            
            void setMean(Eigen::VectorXd mean){mean_ = mean;};
            void setLdltDecomposition(Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition){ldltDecomposition_ = ldltDecomposition;};

            Eigen::VectorXd get() override;
            void set(const Eigen::VectorXd &x) override;
            double logJumpPDF(const Eigen::VectorXd &jump) override;

    };

    // Uniform sphere distribution
    class UniformSphereDistribution: public MultivariateDistribution {
        private:
            std::normal_distribution<double> distN_{0.,1.};
            std::size_t dim_{1};
        public:
            UniformSphereDistribution(size_t dim): dim_(dim), distN_(0.,1.) {}
            UniformSphereDistribution() = default;

            // Destructor
            ~UniformSphereDistribution() = default;

            // Assignment operators
            UniformSphereDistribution &operator=(const UniformSphereDistribution &other) = default;
            UniformSphereDistribution &operator=(UniformSphereDistribution &&other) = default;

            double logPDF(const Eigen::VectorXd &x) override {return 0.0;};

            Eigen::VectorXd sample(std::default_random_engine &rng) override;
    };

    // Uniform distribution
    class UniformDistribution: public UnivariateDistribution {
        private:
            double lowerBound_{0.0};
            double upperBound_{1.0};
            std::uniform_real_distribution<double> distN_{0.,1.};
        public:
            UniformDistribution(double a, double b): lowerBound_(a), upperBound_(b), distN_(0.,1.) {}
            UniformDistribution() = default;

            // Destructor
            ~UniformDistribution() = default;

            // Assignment operators
            UniformDistribution &operator=(const UniformDistribution &other) = default;
            UniformDistribution &operator=(UniformDistribution &&other) = default;

            double logPDF(const double &x) override;
            double dLogPDF(const double &x) {return 0.0;};
            double ddLogPDF(const double &x) {return 0.0;};
            double CDF(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void setLowerBound(double a){lowerBound_ = a;};
            void setUpperBound(double b){upperBound_ = b;};

            static std::shared_ptr<UnivariateDistribution> make(double a, double b) {
                return std::make_shared<UniformDistribution>(a,b);
            };
    };

    // Inverse gamma distribution
    class InverseGammaDistribution: public UnivariateDistribution {
        private:
            double alpha_;
            double beta_;
            std::gamma_distribution<double> distGamma_;
        public:
            InverseGammaDistribution(double alpha, double beta): alpha_(alpha), beta_(beta), distGamma_(alpha,1/beta) {}
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

            void setAlpha(double alpha){alpha_ = alpha;};
            void setBeta(double beta){beta_ = beta;};

            static std::shared_ptr<UnivariateDistribution> make(double alpha, double beta) {
                return std::make_shared<InverseGammaDistribution>(alpha,beta);
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

            MultivariateStudentDistribution(Eigen::VectorXd mean, Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition, double nu): mean_(mean), ldltDecomposition_(ldltDecomposition), dofs_(nu), distN_(0., 1.) {}
            MultivariateStudentDistribution() = default;

            // Destructor
            ~MultivariateStudentDistribution() = default;

            // Assignment operators
            MultivariateStudentDistribution &operator=(const MultivariateStudentDistribution &other) = default;
            MultivariateStudentDistribution &operator=(MultivariateStudentDistribution &&other) = default;

            double logPDF(const Eigen::VectorXd &x) override;
            static double logPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const double &nu);

            Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma=1.0) override;
            
            void setMean(Eigen::VectorXd mean){mean_ = mean;};
            void setLdltDecomposition(Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition){ldltDecomposition_ = ldltDecomposition;};
            void setDoFs(double nu){dofs_ = nu;};

            Eigen::VectorXd get() override;
            void set(const Eigen::VectorXd &x) override;
            double logJumpPDF(const Eigen::VectorXd &jump) override;

    };

    // Multivariate uniform distribution

    class MultivariateUniformDistribution: public ProposalDistribution {
        private:
            Eigen::VectorXd mean_;
            Eigen::VectorXd size_;
            std::uniform_real_distribution<double> distN_;
        public:
            MultivariateUniformDistribution(Eigen::VectorXd mean, Eigen::VectorXd size): mean_(mean), size_(size_), distN_(0,1) {}
            MultivariateUniformDistribution() = default;

            // Destructor
            ~MultivariateUniformDistribution() = default;

            // Assignment operators
            MultivariateUniformDistribution &operator=(const MultivariateUniformDistribution &other) = default;
            MultivariateUniformDistribution &operator=(MultivariateUniformDistribution &&other) = default;

            double logPDF(const Eigen::VectorXd &x) override;
            double logJumpPDF(const Eigen::VectorXd &jump) override;
            Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma=1.0) override;

            Eigen::VectorXd get() override;
            void set(const Eigen::VectorXd &x) override;
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

            void setMean(double mu){mean_ = mu;};
            void setStd(double sigma){std_ = sigma;};

            static std::shared_ptr<UnivariateDistribution> make(double mu, double sigma) {
                return std::make_shared<LogNormalDistribution>(mu,sigma);
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

            void setDoFs(double dofs){dofs_ = dofs;};
            void setMean(double mu){mean_ = mu;};
            void setStd(double sigma){std_ = sigma;};

            static std::shared_ptr<UnivariateDistribution> make(double nu, double mu, double sigma) {
                return std::make_shared<StudentDistribution>(nu,mu,sigma);
            };
    };

    // Power law distribution
    class PowerLawDistribution: public UnivariateDistribution {
        private:
            double degree_;
            double lowerBound_;
            std::uniform_real_distribution<double> distU_;
            
        public:
            PowerLawDistribution(double alpha, double lowerBound = 1.0): degree_(alpha), distU_(0.,1.), lowerBound_(lowerBound) {}
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

            void setDegree(double degree){degree_ = degree;};

            static std::shared_ptr<UnivariateDistribution> make(double degree, double lowerBound = 1.0) {
                return std::make_shared<PowerLawDistribution>(degree,lowerBound);
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
            SmoothUniformDistribution(double lowerBound, double upperBound, double sigma): lowerBound_(lowerBound), upperBound_(upperBound), std_(sigma), distU_(0,1), distN_(0,1) {}
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

            void setLowerBound(double a){lowerBound_ = a;};
            void setUpperBound(double b){upperBound_ = b;};
            void setStd(double sigma){std_ = sigma;};

            static std::shared_ptr<UnivariateDistribution> make(double a, double b, double sigma) {
                return std::make_shared<SmoothUniformDistribution>(a,b,sigma);
            };
    };

}

    



#endif