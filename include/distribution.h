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
            virtual double sample(std::default_random_engine &rng) = 0;
    };

    class multivariate_distribution {
        public:
            virtual double log_pdf(const Eigen::VectorXd &x) = 0;
            
            /**
             * @brief Generate a sample from the distribution
             * 
             * @return Eigen::VectorXd
             */
            virtual Eigen::VectorXd sample(std::default_random_engine &rng)=0;
    };

    class proposal_distribution {
        public:
            
            virtual double log_pdf(const Eigen::VectorXd &x) = 0;

            virtual double log_jump_pdf(const Eigen::VectorXd &jump) = 0;
            
            /**
             * @brief Generate a sample from the distribution
             * 
             * @return Eigen::VectorXd
             */
            virtual Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma)=0;

            virtual Eigen::VectorXd get() = 0;
            virtual void set(const Eigen::VectorXd &x) = 0;

    };


    // Univariate normal distribution
    class normal_distribution: public univariate_distribution {
        private:
            double m_mean{0.0};
            double m_std{1.0};
            std::normal_distribution<double> m_dist;

        public:
            normal_distribution(double mean, double sd): m_mean(mean), m_std(sd), m_dist(0.,1.) {}
            
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
            double sample(std::default_random_engine &rng) override;
            
            void set_mean(double mean){m_mean = mean;};
            void set_std(double std){m_std = std;};
    };

    // Multivariate normal distribution
    class multivariate_normal_distribution: public proposal_distribution {
        private:
            Eigen::VectorXd m_mean;
            Eigen::LDLT<Eigen::MatrixXd> m_ldlt;
            std::normal_distribution<double> m_dist;
        public:

            multivariate_normal_distribution(Eigen::VectorXd mean, Eigen::LDLT<Eigen::MatrixXd> cov_ldlt): m_mean(mean), m_ldlt(cov_ldlt), m_dist(0., 1.) {}
            multivariate_normal_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override;
            static double log_pdf(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt);

            static double log_pdf_gradient(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const Eigen::MatrixXd &cov_gradient, const Eigen::VectorXd &mean_gradient);
            static double log_pdf_hessian(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const Eigen::MatrixXd &cov_gradient_l, const Eigen::MatrixXd &cov_gradient_k, const Eigen::MatrixXd &cov_hessian);

            Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma=1.0) override;
            
            void set_mean(Eigen::VectorXd mean){m_mean = mean;};
            void set_cov_ldlt(Eigen::LDLT<Eigen::MatrixXd> cov_ldlt){m_ldlt = cov_ldlt;};

            Eigen::VectorXd get() override;
            void set(const Eigen::VectorXd &x) override;
            double log_jump_pdf(const Eigen::VectorXd &jump) override;

    };

    // Uniform sphere distribution
    class uniform_sphere_distribution: public multivariate_distribution {
        private:
            std::normal_distribution<double> m_dist;
            std::size_t m_dim;
        public:
            uniform_sphere_distribution(size_t dim): m_dim(dim), m_dist(0.,1.) {}
            uniform_sphere_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override {return 0.0;};

            Eigen::VectorXd sample(std::default_random_engine &rng) override;
    };

    // Uniform distribution
    class uniform_distribution: public univariate_distribution {
        private:
            double m_a;
            double m_b;
            std::uniform_real_distribution<double> m_dist;
            double m_size;
        public:
            uniform_distribution(double a, double b): m_a(a), m_b(b), m_dist(0.,1.), m_size(m_b-m_a) {}
            uniform_distribution() = default;

            double log_pdf(const double &x) override;
            double d_log_pdf(const double &x) {return 0.0;};
            double dd_log_pdf(const double &x) {return 0.0;};
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void set_a(double a){m_a = a;};
            void set_b(double b){m_b = b;};
    };

    // Inverse gamma distribution
    class inverse_gamma_distribution: public univariate_distribution {
        private:
            double m_alpha;
            double m_beta;
            std::gamma_distribution<double> m_gamma;
        public:
            inverse_gamma_distribution(double alpha, double beta): m_alpha(alpha), m_beta(beta), m_gamma(alpha,1/beta) {}
            inverse_gamma_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void set_alpha(double alpha){m_alpha = alpha;};
            void set_beta(double beta){m_beta = beta;};
    };

    // Multivariate normal distribution
    class multivariate_t_distribution: public proposal_distribution {
        private:
            Eigen::VectorXd m_mean;
            Eigen::LDLT<Eigen::MatrixXd> m_ldlt;
            double m_nu;
            std::normal_distribution<double> m_dist;
        public:

            multivariate_t_distribution(Eigen::VectorXd mean, Eigen::LDLT<Eigen::MatrixXd> cov_ldlt, double nu): m_mean(mean), m_ldlt(cov_ldlt), m_nu(nu), m_dist(0., 1.) {}
            multivariate_t_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override;
            static double log_pdf(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const double &nu);

            Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma=1.0) override;
            
            void set_mean(Eigen::VectorXd mean){m_mean = mean;};
            void set_cov_ldlt(Eigen::LDLT<Eigen::MatrixXd> cov_ldlt){m_ldlt = cov_ldlt;};
            void set_nu(double nu){m_nu = nu;};

            Eigen::VectorXd get() override;
            void set(const Eigen::VectorXd &x) override;
            double log_jump_pdf(const Eigen::VectorXd &jump) override;

    };

    // Multivariate uniform distribution

    class multivariate_uniform_distribution: public proposal_distribution {
        private:
            Eigen::VectorXd m_mean;
            Eigen::VectorXd m_eps;
            std::uniform_real_distribution<double> m_dist;
        public:
            multivariate_uniform_distribution(Eigen::VectorXd mean, Eigen::VectorXd eps): m_mean(mean), m_eps(eps), m_dist(0,1) {}
            multivariate_uniform_distribution() = default;

            double log_pdf(const Eigen::VectorXd &x) override;
            double log_jump_pdf(const Eigen::VectorXd &jump) override;
            Eigen::VectorXd sample(std::default_random_engine &rng, const double &gamma=1.0) override;

            Eigen::VectorXd get() override;
            void set(const Eigen::VectorXd &x) override;
    };


    // Univariate log-normal distribution
    class log_normal_distribution: public univariate_distribution {
        private:
            double m_mu;
            double m_sigma;
            std::normal_distribution<double> m_dist;
        public:
            log_normal_distribution(double mu, double sigma): m_mu(mu), m_sigma(sigma), m_dist(0, 1) {}
            log_normal_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

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
        public:
            t_distribution(double nu, double mu, double sigma): m_nu(nu), m_mu(mu), m_sigma(sigma), m_dist(nu) {}
            t_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

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
        public:
            cauchy_distribution(double mu, double sigma): m_mu(mu), m_sigma(sigma), m_dist(0, 1) {}
            cauchy_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void set_mu(double mu){m_mu = mu;};
            void set_sigma(double sigma){m_sigma = sigma;};
    };


    // Smooth uniform distribution
    class smooth_uniform_distribution_sigmoid: public univariate_distribution {
        private:
            double m_a;
            double m_b;
            double m_k;
            std::uniform_real_distribution<double> m_dist;
        public:
            smooth_uniform_distribution_sigmoid(double a, double b, double k): m_a(a), m_b(b), m_k(k), m_dist(0,1) {}
            smooth_uniform_distribution_sigmoid() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void set_a(double a){m_a = a;};
            void set_b(double b){m_b = b;};
            void set_k(double k){m_k = k;};
    };

    // Power law distribution
    class power_law_distribution: public univariate_distribution {
        private:
            double m_alpha;
            double m_tol;
            std::uniform_real_distribution<double> m_dist;
            
        public:
            power_law_distribution(double alpha, double tol = 1.0): m_alpha(alpha), m_dist(0.,1.), m_tol(tol) {}
            power_law_distribution() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void set_alpha(double alpha){m_alpha = alpha;};
    };


    // Smoothed uniform distribution with normal
    class smooth_uniform_distribution_normal: public univariate_distribution {
        private:
            double m_a;
            double m_b;
            double m_sigma;
            std::uniform_real_distribution<double> m_dist_u;
            std::normal_distribution<double> m_dist_n;
        public:
            smooth_uniform_distribution_normal(double a, double b, double sigma): m_a(a), m_b(b), m_sigma(sigma), m_dist_u(0,1), m_dist_n(0,1) {}
            smooth_uniform_distribution_normal() = default;

            double log_pdf(const double & x) override;
            double d_log_pdf(const double & x);
            double dd_log_pdf(const double & x);
            double cdf(const double & x) override;
            double quantile(const double & p) override;
            double sample(std::default_random_engine &rng) override;

            void set_a(double a){m_a = a;};
            void set_b(double b){m_b = b;};
            void set_sigma(double sigma){m_sigma = sigma;};
    };

}

    



#endif