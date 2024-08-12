#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "cmp_defines.h"

namespace cmp {

    class univariate_distribution {
        public:

            /**
             * @brief Evaluate the log-pdf of the distribution at point x
             * 
             * @param x The point at which to evaluate the log-pdf
             * @return double 
             */
            virtual double log_pdf(const double &x) = 0;

            /**
             * @brief Evaluate the cdf of the distribution at point x
             * 
             * @param x Point at which to evaluate the cdf
             * @return double
             */
            virtual double cdf(const double &x) = 0;

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
            virtual double sample() = 0;
    };

    class multivariate_distribution {
        public:
            virtual double log_pdf(const Eigen::VectorXd &x) = 0;
            
            /**
             * @brief Generate a sample from the distribution
             * 
             * @return Eigen::VectorXd 
             */
            virtual Eigen::VectorXd sample() = 0;

            /**
             * @brief Jump to a new point in the parameter space centered at x
             * 
             * @param x The center of the jump
             * @param gamma The size of the jump
             * @return Eigen::VectorXd 
             */
            virtual Eigen::VectorXd jump(const Eigen::VectorXd &x, double gamma = 1.0) = 0;

            virtual double log_jump_prob(const Eigen::VectorXd &jump, double gamma = 1.0)=0;
    };


    // Univariate normal distribution
    class normal_distribution: public univariate_distribution {
        private:
            double m_mean{0.0};
            double m_std{1.0};
            std::normal_distribution<double> m_dist;
            std::default_random_engine m_rng;
        public:
            normal_distribution(double mean, double sd, std::default_random_engine rng): m_mean(mean), m_std(sd), m_rng(rng), m_dist(0.,1.) {}
            
            double log_pdf(const double &x) override;
            
            /**
             * @brief Evaluate the first derivative of the log-pdf
             * 
             * @param x Point at which to evaluate the derivative
             * @return double
             */
            double d_log_pdf(const double &x);

            /**
             * @brief Evaluate the second derivative of the log-pdf
             * 
             * @param x Point at which to evaluate the derivative
             * @return double
             */
            double dd_log_pdf(const double &x);
            
            double cdf(const double &x) override;
            double quantile(const double &p) override;
            double sample() override;
            
            void set_mean(double mean){m_mean = mean;};
            void set_std(double std){m_std = std;};
    };

    // Multivariate normal distribution
    class multivariate_normal_distribution: public multivariate_distribution {
        private:
            Eigen::VectorXd m_mean;
            Eigen::LDLT<Eigen::MatrixXd> m_ldlt;
            std::default_random_engine m_rng;
            std::normal_distribution<double> m_dist;
        public:

            multivariate_normal_distribution(Eigen::VectorXd mean, Eigen::LDLT<Eigen::MatrixXd> cov_ldlt, std::default_random_engine rng): m_mean(mean), m_rng(rng), m_ldlt(cov_ldlt), m_dist(0., 1.) {}
            multivariate_normal_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override;
            static double log_pdf(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt);

            static double log_pdf_gradient(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const Eigen::MatrixXd &cov_gradient, const Eigen::VectorXd &mean_gradient);
            static double log_pdf_hessian(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const Eigen::MatrixXd &cov_gradient_l, const Eigen::MatrixXd &cov_gradient_k, const Eigen::MatrixXd &cov_hessian);

            Eigen::VectorXd sample() override;

            Eigen::VectorXd jump(const Eigen::VectorXd &x, double gamma = 1.0) override;

            double log_jump_prob(const Eigen::VectorXd &jump, double gamma = 1.0) override;
            
            void set_mean(Eigen::VectorXd mean){m_mean = mean;};
            void set_cov_ldlt(Eigen::LDLT<Eigen::MatrixXd> cov_ldlt){m_ldlt = cov_ldlt;};

    };

    // Uniform sphere distribution
    class uniform_sphere_distribution: public multivariate_distribution {
        private:
            std::default_random_engine m_rng;
            std::normal_distribution<double> m_dist;
            std::size_t m_dim;
        public:
            uniform_sphere_distribution(size_t dim, std::default_random_engine rng): m_dim(dim), m_rng(rng), m_dist(0.,1.) {}
            uniform_sphere_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override {return 0.0;};

            Eigen::VectorXd sample() override;

            Eigen::VectorXd jump(const Eigen::VectorXd &x, double gamma = 1.0) override {return x;};
            double log_jump_prob(const Eigen::VectorXd &jump, double gamma = 1.0) {return 0.0;};


    };

    // Uniform distribution
    class uniform_distribution: public univariate_distribution {
        private:
            double m_a;
            double m_b;
            std::uniform_real_distribution<double> m_dist;
            std::default_random_engine m_rng;
            double m_size;
        public:
            uniform_distribution(double a, double b, std::default_random_engine rng): m_a(a), m_b(b), m_rng(rng), m_dist(0.,1.), m_size(m_b-m_a) {}
            uniform_distribution() = default;

            double log_pdf(const double &x) override;
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample() override;

            void set_a(double a){m_a = a;};
            void set_b(double b){m_b = b;};
    };

    // Inverse gamma distribution
    class inverse_gamma_distribution: public univariate_distribution {
        private:
            double m_alpha;
            double m_beta;
            std::gamma_distribution<double> m_gamma;
            std::default_random_engine m_rng;
        public:
            inverse_gamma_distribution(double alpha, double beta, std::default_random_engine rng): m_alpha(alpha), m_beta(beta), m_rng(rng), m_gamma(alpha,1/beta) {}
            inverse_gamma_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample() override;

            void set_alpha(double alpha){m_alpha = alpha;};
            void set_beta(double beta){m_beta = beta;};
    };

    // Multivariate normal distribution
    class multivariate_t_distribution: public multivariate_distribution {
        private:
            Eigen::VectorXd m_mean;
            Eigen::LDLT<Eigen::MatrixXd> m_ldlt;
            double m_nu;
            std::default_random_engine m_rng;
            std::normal_distribution<double> m_dist;
        public:

            multivariate_t_distribution(Eigen::VectorXd mean, Eigen::LDLT<Eigen::MatrixXd> cov_ldlt, double nu, std::default_random_engine rng): m_mean(mean), m_rng(rng), m_ldlt(cov_ldlt), m_nu(nu), m_dist(0., 1.) {}
            multivariate_t_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override;
            static double log_pdf(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, double nu);

            Eigen::VectorXd sample() override;

            Eigen::VectorXd jump(const Eigen::VectorXd &x, double gamma = 1.0) override;

            double log_jump_prob(const Eigen::VectorXd &jump, double gamma = 1.0) override;
            
            void set_mean(Eigen::VectorXd mean){m_mean = mean;};
            void set_cov_ldlt(Eigen::LDLT<Eigen::MatrixXd> cov_ldlt){m_ldlt = cov_ldlt;};
            void set_nu(double nu){m_nu = nu;};

    };

    // Univariate log-normal distribution
    class log_normal_distribution: public univariate_distribution {
        private:
            double m_mu;
            double m_sigma;
            std::normal_distribution<double> m_dist;
            std::default_random_engine m_rng;
        public:
            log_normal_distribution(double mu, double sigma, std::default_random_engine rng): m_mu(mu), m_sigma(sigma), m_rng(rng), m_dist(0, 1) {}
            log_normal_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample() override;

            void set_mu(double mu){m_mu = mu;};
            void set_sigma(double sigma){m_sigma = sigma;};
    };

    // Univariate t distribution
    class t_distribution: public univariate_distribution {
        private:
            double m_nu;
            double m_mu;
            double m_sigma;
            std::student_t_distribution<double> m_dist;
            std::default_random_engine m_rng;
        public:
            t_distribution(double nu, double mu, double sigma, std::default_random_engine rng): m_nu(nu), m_mu(mu), m_sigma(sigma), m_rng(rng), m_dist(nu) {}
            t_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample() override;

            void set_nu(double nu){m_nu = nu;};
            void set_mu(double mu){m_mu = mu;};
            void set_sigma(double sigma){m_sigma = sigma;};
    };

    // Univariate Cauchy distribution
    class cauchy_distribution: public univariate_distribution {
        private:
            double m_mu;
            double m_sigma;
            std::cauchy_distribution<double> m_dist;
            std::default_random_engine m_rng;
        public:
            cauchy_distribution(double mu, double sigma, std::default_random_engine rng): m_mu(mu), m_sigma(sigma), m_rng(rng), m_dist(0, 1) {}
            cauchy_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample() override;

            void set_mu(double mu){m_mu = mu;};
            void set_sigma(double sigma){m_sigma = sigma;};
    };
}

    



#endif