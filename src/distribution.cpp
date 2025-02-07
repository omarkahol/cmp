#include <distribution.h>

/**
 * @brief Implementation of the inverse error function. 
 * Based on "A handy approximation for the error function and its inverse" by Sergei Winitzki
 * 
 * @param x 
 * @return double 
 */
double erfinv(float x){
   double tt1, tt2, lnx, sgn;
   sgn = (x < 0) ? -1.0f : 1.0f;

   x = 1-x*x;       // x = 1 - x*x;
   lnx = log(x);

   tt1 = 2/(M_PI*0.147) + 0.5f * lnx;
   tt2 = 1/(0.147) * lnx;

   return (sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));

}

double cmp::normal_distribution::log_pdf(const double & x) {
    return -0.5*log(2*M_PI) - log(m_std) - 0.5*pow((x-m_mean)/m_std,2);
}

double cmp::normal_distribution::d_log_pdf(const double & x)
{
    return -(x-m_mean)/pow(m_std,2);
}

double cmp::normal_distribution::dd_log_pdf(const double & x)
{
    return -1/pow(m_std,2);
}

double cmp::normal_distribution::cdf(const double & x)
{
    return 0.5*(1+erf((x-m_mean)/(m_std*sqrt(2))));
}

double cmp::normal_distribution::quantile(const double & p) {
    return m_mean + m_std*sqrt(2)*erfinv(2*p-1);
}

double cmp::normal_distribution::sample(std::default_random_engine &rng) {
    return m_dist(rng)*m_std + m_mean;
}

double cmp::multivariate_normal_distribution::log_pdf(const Eigen::VectorXd &x)
{
    return log_pdf(m_mean-x, m_ldlt);
}

double cmp::multivariate_normal_distribution::log_jump_pdf(const Eigen::VectorXd &jump)
{
    return log_pdf(jump, m_ldlt);
}

double cmp::multivariate_normal_distribution::log_pdf(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt)
{
    // Solve linear system
    Eigen::VectorXd alpha = cov_ldlt.solve(res);

    return -0.5 * res.dot(alpha) - 0.5 * (cov_ldlt.vectorD().array().abs().log()).sum() - 0.5 * res.size() * log(2 * M_PI);
}

double cmp::multivariate_normal_distribution::log_pdf_gradient(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const Eigen::MatrixXd &cov_gradient, const Eigen::VectorXd &mean_gradient)
{
    // Useful quantities (see Rasmussen for reference)
    Eigen::VectorXd alpha = cov_ldlt.solve(res);
    Eigen::MatrixXd alpha_alpha_t = alpha * alpha.transpose();

    return 0.5 * (alpha_alpha_t * cov_gradient - cov_ldlt.solve(cov_gradient)).trace() + res.dot(cov_ldlt.solve(mean_gradient));
}

double cmp::multivariate_normal_distribution::log_pdf_hessian(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const Eigen::MatrixXd &cov_gradient_l, const Eigen::MatrixXd &cov_gradient_k, const Eigen::MatrixXd &cov_hessian) {
    // Required quantities for the computation
    Eigen::VectorXd alpha = cov_ldlt.solve(res);
    Eigen::MatrixXd alpha_alpha_t = alpha * alpha.transpose();

    // Compute the remaining symbols
    auto a_l  = cov_ldlt.solve(cov_gradient_l);
    auto a_k  = cov_ldlt.solve(cov_gradient_k);
    auto sym_tens = 0.5*((a_l*alpha_alpha_t) + (a_l*alpha_alpha_t).transpose());

    // Compute the 4 contributions to the hessian (I think that taking the trace 4 times is faster than adding and then taking the trace(?))
    double H1 = (alpha_alpha_t*cov_hessian).trace();
    double H2 = (cov_ldlt.solve(cov_hessian)).trace();
    double H3 = (sym_tens*cov_gradient_k).trace();
    double H4 = (a_l*a_k).trace();

    // return the weighted sum of the contributions
    return 0.5*H1 - 0.5*H2 - H3 + 0.5*H4;
}

Eigen::VectorXd cmp::multivariate_normal_distribution::sample(std::default_random_engine &rng, const double &gamma)
{
    // Generate a sample
    Eigen::VectorXd z = Eigen::VectorXd::Zero(m_mean.size());
    for (size_t i = 0; i < m_mean.size(); i++) {
        z(i) = m_dist(rng)*sqrt(abs(m_ldlt.vectorD()(i)));
    }

    return m_mean + (m_ldlt.transpositionsP().transpose()*(m_ldlt.matrixL() * z))*gamma;
}

Eigen::VectorXd cmp::multivariate_normal_distribution::get() {
    return m_mean;
}

void cmp::multivariate_normal_distribution::set(const Eigen::VectorXd &x) {
    m_mean = x;
}

Eigen::VectorXd cmp::uniform_sphere_distribution::sample(std::default_random_engine &rng) {
    // Sample a normal distribution
    Eigen::VectorXd x(m_dim);
    for (size_t i = 0; i < m_dim; i++) {
        x(i) = m_dist(rng);
    }

    // Normalize the vector
    return x/x.norm();
}

double cmp::uniform_distribution::log_pdf(const double & x)
{
   if (x < m_a || x > m_b) {
       return -std::numeric_limits<double>::infinity();
   } else {
       return -log(m_size);
   }
}

double cmp::uniform_distribution::cdf(const double & x)
{
    if (x < m_a) {
        return 0.0;
    } else if (x > m_b) {
        return 1.0;
    } else {
        return (x-m_a)/m_size;
    }
}

double cmp::uniform_distribution::quantile(const double & p)
{
    return m_a + p*m_size;
}

double cmp::uniform_distribution::sample(std::default_random_engine &rng)
{
    return  m_dist(rng)*m_size + m_a;
}

double cmp::inverse_gamma_distribution::log_pdf(const double & x)
{
    return - (m_alpha + 1)*log(x) - m_beta/x + m_alpha*log(m_beta) - std::log(std::tgamma(m_alpha));
}

double cmp::inverse_gamma_distribution::d_log_pdf(const double & x) {
    return (m_beta-(m_alpha+1)*x)/std::pow(x,2);
}  
double cmp::inverse_gamma_distribution::dd_log_pdf(const double & x) {

    return (-2*m_beta+x+x*m_alpha)/std::pow(x,3);
}

double cmp::inverse_gamma_distribution::cdf(const double & x)
{
    // To-do implement gamma regularized
    return 0.0;
}

double cmp::inverse_gamma_distribution::quantile(const double & p)
{
    // To-do implement inverse gamma regularized
    return 0.0;
}

double cmp::inverse_gamma_distribution::sample(std::default_random_engine &rng)
{
    double x = m_gamma(rng);
    return 1/x;
}

double cmp::multivariate_t_distribution::log_pdf(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt, const double &nu)
{
    // Solve linear system
    Eigen::VectorXd alpha = cov_ldlt.solve(res);
    double det = (cov_ldlt.vectorD().array().abs().log()).sum();
    size_t dim = res.size();

    return std::log(std::tgammal(0.5*(nu + dim)) - std::log(std::tgammal(0.5*nu)) - 0.5*dim*std::log(M_PI*nu)) - 0.5*det -0.5*(nu+dim)*std::log(1+res.dot(alpha)/nu);
}

double cmp::multivariate_t_distribution::log_pdf(const Eigen::VectorXd &x)
{
    return log_pdf(m_mean-x, m_ldlt, m_nu);
}

double cmp::multivariate_t_distribution::log_jump_pdf(const Eigen::VectorXd &jump)
{
    return log_pdf(jump, m_ldlt, m_nu);
}

Eigen::VectorXd cmp::multivariate_t_distribution::sample(std::default_random_engine &rng, const double &gamma)
{
    // Sample from a T distribution
    Eigen::VectorXd z = Eigen::VectorXd::Zero(m_mean.size());
    for (size_t i = 0; i < m_mean.size(); i++) {
        z(i) = m_dist(rng);

        // Generate a sample from a chi-square distribution
        double chi = 0.0;
        for (size_t j = 0; j < m_nu; j++) {
            chi += std::pow(m_dist(rng),2);
        }

        z(i) = sqrt(abs(m_ldlt.vectorD()(i)))*z(i)/std::sqrt(chi/m_nu);
    }

    return m_mean + (m_ldlt.transpositionsP().transpose()*(m_ldlt.matrixL() * z))*gamma;
}

Eigen::VectorXd cmp::multivariate_t_distribution::get()
{
    return m_mean;
} 

void cmp::multivariate_t_distribution::set(const Eigen::VectorXd &x)
{
    m_mean = x;
}

double cmp::log_normal_distribution::log_pdf(const double &x)
{
    return -0.5 * pow((log(x) - m_mu) / m_sigma, 2) - 0.5 * log(2 * M_PI);
}

double cmp::log_normal_distribution::d_log_pdf(const double &x)
{
    return (m_mu-log(x))/(x*pow(m_sigma,2));;
}

double cmp::log_normal_distribution::dd_log_pdf(const double &x)
{
    return (-1-m_mu+log(x))/pow(x*m_sigma,2);
}

double cmp::log_normal_distribution::cdf(const double &x)
{
    return 0.5*(1+std::erf((log(x)-m_mu)/(m_sigma*sqrt(2))));
}

double cmp::log_normal_distribution::quantile(const double &p)
{
    return std::exp(m_mu + m_sigma*sqrt(2)*erfinv(2*p-1));
}

double cmp::log_normal_distribution::sample(std::default_random_engine &rng)
{
    return std::exp(m_dist(rng)*m_sigma + m_mu);
}

double cmp::t_distribution::log_pdf(const double &x)
{
    return std::log(std::tgammal(0.5*(m_nu + 1)) - std::log(std::tgammal(0.5*m_nu)) - 0.5*std::log(M_PI*m_nu)) - m_sigma -0.5*(m_nu+1)*std::log(1+pow(x-m_mu,2)/(m_nu*m_sigma*m_sigma));
}

double cmp::t_distribution::d_log_pdf(const double &x)
{
    return -(x-m_mu)*(1+m_nu)/(m_sigma*m_sigma*m_nu + pow(x-m_mu,2));
}

double cmp::t_distribution::dd_log_pdf(const double &x)
{
    return (1+m_nu)*(pow(x-m_mu,2)-m_nu*m_sigma*m_sigma)/pow((pow(x-m_mu,2)+m_nu*m_sigma*m_sigma),2);
}

double cmp::t_distribution::cdf(const double &x)
{
    // To-do implement
    return 0.0;
}

double cmp::t_distribution::quantile(const double &p)
{   
    // To-do implement
    return 0.0;
}

double cmp::t_distribution::sample(std::default_random_engine &rng)
{   
    
    // Sample a normal variate
    double z = m_dist(rng);

    // Sample a chi-square variate
    double chi = 0.0;
    for (size_t j = 0; j < m_nu; j++) {
        chi += std::pow(m_dist(rng),2);
    }

    // Return the sample
    return m_mu + z*m_sigma/std::sqrt(chi/m_nu);
}

double cmp::cauchy_distribution::log_pdf(const double &x)
{
    return -log(M_PI) - log(m_sigma) - log(1 + pow((x - m_mu)/m_sigma, 2));
}

double cmp::cauchy_distribution::d_log_pdf(const double &x)
{
    // derivative of the log pdf
    return -2*(x - m_mu)/(pow(m_sigma,2) + pow(x - m_mu,2));
}

double cmp::cauchy_distribution::dd_log_pdf(const double &x)
{
    return -2*(pow(m_sigma,2) - pow(x - m_mu,2))/pow(pow(m_sigma,2) + pow(x - m_mu,2),2);
}

double cmp::cauchy_distribution::cdf(const double &x)
{
    return 0.5 + atan((x - m_mu)/m_sigma)/M_PI;
}

double cmp::cauchy_distribution::quantile(const double &p)
{
    return m_mu + m_sigma*tan(M_PI*(p - 0.5));
}

double cmp::cauchy_distribution::sample(std::default_random_engine &rng)
{
    return m_dist(rng);

}

double sech(const double &x)
{
    return 1.0/std::cosh(x);
}

double cmp::smooth_uniform_distribution_sigmoid::log_pdf(const double &x)
{

    double p = 0.5*std::tanh(m_k*(x-m_a))- 0.5*std::tanh(m_k*(x-m_b));
    double int_const = ((-1.*std::pow(M_E,2*m_a*m_k) + 
        1.*std::pow(M_E,2*m_b*m_k))*
      (1.*m_a*m_k - 1.*m_b*m_k) + 
     (0.5*std::pow(M_E,2*m_a*m_k) - 
        0.5*std::pow(M_E,2.*m_a*m_k) - 
        0.5*std::pow(M_E,2*m_b*m_k) + 
        0.5*std::pow(M_E,2.*m_b*m_k))*
      std::log(1. + 1.*std::pow(M_E,2.*m_a*m_k))\
      + (-0.5*std::pow(M_E,2*m_a*m_k) + 
        0.5*std::pow(M_E,2.*m_a*m_k) + 
        0.5*std::pow(M_E,2*m_b*m_k) - 
        0.5*std::pow(M_E,2.*m_b*m_k))*
      std::log(1. + 1.*std::pow(M_E,2.*m_b*m_k)))/
   ((std::pow(M_E,2*m_a*m_k) - 
       1.*std::pow(M_E,2*m_b*m_k))*m_k);
    
    if (std::isnan(int_const)) {
        int_const = 1.0;
    }

    return std::log(p) - std::log(int_const);
    
}


double cmp::smooth_uniform_distribution_sigmoid::d_log_pdf(const double &x)
{
    return (0.5*m_k*std::pow(sech(m_k*(-m_a + x)),2) - 
     0.5*m_k*std::pow(sech(m_k*(-m_b + x)),2)
     )/(0.5*std::tanh(m_k*(-m_a + x)) - 
     0.5*std::tanh(m_k*(-m_b + x)));
}

double cmp::smooth_uniform_distribution_sigmoid::dd_log_pdf(const double &x)
{
    return (std::pow(m_k,2)*(0.5 - 0.5*std::cosh(2*(-m_a + m_b)*m_k) + 
       0.125*std::cosh(2*m_k*(-m_a + x)) - 
       0.125*std::cosh(2*m_k*(m_a - 2*m_b + x)) + 
       0.125*std::cosh(2*m_k*(-m_b + x)) - 
       0.125*std::cosh(2*m_k*(-2*m_a + m_b + x)))*
     std::pow(sech(m_k*(-m_a + x)),4)*std::pow(sech(m_k*(-m_b + x)),4))/
   std::pow(1.*std::tanh(m_k*(-m_a + x)) - 1.*std::tanh(m_k*(-m_b + x)),2);
}

double cmp::smooth_uniform_distribution_sigmoid::cdf(const double &t)
{
    return ((1.*std::pow(M_E,2.*m_a*m_k) - 
        1.*std::pow(M_E,2.*m_b*m_k))*
      (1.*m_a*m_k - 1.*m_b*m_k) + 
     (-0.5*std::pow(M_E,2.*m_a*m_k) + 
        0.5*std::pow(M_E,2.*m_b*m_k))*
      std::log(1. + std::pow(M_E,2.*m_a*m_k)) + 
     (0.5*std::pow(M_E,2.*m_a*m_k) - 
        0.5*std::pow(M_E,2.*m_b*m_k))*
      std::log(1. + std::pow(M_E,2.*m_b*m_k)) + 
     (-0.5*std::pow(M_E,2*m_a*m_k) + 
        0.5*std::pow(M_E,2*m_b*m_k))*
      (1.*std::log(std::cosh(m_k*(m_a - t))*
           sech(m_a*m_k)) - 
        1.*std::log(std::cosh(m_k*(m_b - t))*
           sech(m_b*m_k))))/
   ((1.*std::pow(M_E,2*m_a*m_k) - 
        1.*std::pow(M_E,2*m_b*m_k))*
      (1.*m_a*m_k - 1.*m_b*m_k) + 
     (-0.5*std::pow(M_E,2*m_a*m_k) + 
        0.5*std::pow(M_E,2.*m_a*m_k) + 
        0.5*std::pow(M_E,2*m_b*m_k) - 
        0.5*std::pow(M_E,2.*m_b*m_k))*
      std::log(1. + 1.*std::pow(M_E,2.*m_a*m_k))\
      + (0.5*std::pow(M_E,2*m_a*m_k) - 
        0.5*std::pow(M_E,2.*m_a*m_k) - 
        0.5*std::pow(M_E,2*m_b*m_k) + 
        0.5*std::pow(M_E,2.*m_b*m_k))*
      std::log(1. + 1.*std::pow(M_E,2.*m_b*m_k)));
}

double cmp::smooth_uniform_distribution_sigmoid::quantile(const double &p)
{

    // Check if the quantile is outside the bounds
    if (p < 0 || p > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Use Newton's method to find the quantile
    double x = 0.5*(m_a+m_b);
    
    // Perform 5 Newton iterations
    for (size_t i = 0; i < 5; i++) {
        double f = cdf(x) - p;
        double df = std::exp(log_pdf(x));
        x = x - f/df;
    }

    return x;
}

double cmp::smooth_uniform_distribution_sigmoid::sample(std::default_random_engine &rng)
{
    return quantile(m_dist(rng));

}

double cmp::power_law_distribution::log_pdf(const double &x)
{
    return log(m_tol) - m_alpha*log(x);
}

double cmp::power_law_distribution::d_log_pdf(const double &x)
{
    return -m_alpha/x;
}

double cmp::power_law_distribution::dd_log_pdf(const double &x)
{
    return m_alpha/pow(x,2);
}

double cmp::power_law_distribution::cdf(const double &t)
{
    return 1.0 - pow(m_tol/t,m_alpha);
}

double cmp::power_law_distribution::quantile(const double &q)
{
    return m_tol/pow(1-q,1/m_alpha);
}

double cmp::power_law_distribution::sample(std::default_random_engine &rng)
{
    return quantile(m_dist(rng));
}

double cmp::multivariate_uniform_distribution::log_pdf(const Eigen::VectorXd &x) {
    return log_jump_pdf(x-m_mean);
}

double cmp::multivariate_uniform_distribution::log_jump_pdf(const Eigen::VectorXd &jump) {
    for (size_t i = 0; i < jump.size(); i++) {
        if (jump(i) < -m_eps(i)*0.5 || jump(i) > m_eps(i)*0.5) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    return 0.0;
}
Eigen::VectorXd cmp::multivariate_uniform_distribution::sample(std::default_random_engine &rng, const double &gamma) {
    Eigen::VectorXd rv(m_mean.size());
    for (size_t i = 0; i < m_mean.size(); i++) {
        rv(i) = m_mean(i) - 0.5*m_eps(i)*gamma + m_eps(i)*m_dist(rng)*gamma;
    }
    return rv;
}

Eigen::VectorXd cmp::multivariate_uniform_distribution::get() {
    return m_mean;
}
void cmp::multivariate_uniform_distribution::set(const Eigen::VectorXd &x){
    m_mean = x;
}

double cdf_normal(const double &x)
{
    return 0.5*(1+erf(x/sqrt(2)));
}

double cmp::smooth_uniform_distribution_normal::log_pdf(const double &x)
{
    return std::log((cdf_normal((m_b-x)/m_sigma) - cdf_normal((m_a-x)/m_sigma))/(m_b-m_a));
}

double cmp::smooth_uniform_distribution_normal::d_log_pdf(const double &x)
{
    return (std::pow(M_E,-std::pow(m_a - x,2)/(2.*std::pow(m_sigma,2))) - std::pow(M_E,-std::pow(m_b - x,2)/(2.*std::pow(m_sigma,2))))/((-m_a + m_b)*std::sqrt(2*M_PI)*m_sigma);
}

double cmp::smooth_uniform_distribution_normal::dd_log_pdf(const double &x)
{
    return ((m_a - x)/
      std::pow(M_E,std::pow(m_a - x,2)/
        (2.*std::pow(m_sigma,2))) + 
     (-m_b + x)/
      std::pow(M_E,std::pow(m_b - x,2)/
        (2.*std::pow(m_sigma,2))))/
   ((-m_a + m_b)*std::sqrt(2*M_PI)*std::pow(m_sigma,3));
}

double cmp::smooth_uniform_distribution_normal::cdf(const double &t)
{
    return (m_a - m_b + (-std::pow(M_E,
          -std::pow(m_a - t,2)/
           (2.*std::pow(m_sigma,2))) + 
        std::pow(M_E,
         -std::pow(m_b - t,2)/
          (2.*std::pow(m_sigma,2))))*
      std::sqrt(2/M_PI)*m_sigma + 
     (-m_a + t)*std::erf((m_a - t)/
        (std::sqrt(2)*m_sigma)) + 
     (m_b - t)*std::erf((m_b - t)/
        (std::sqrt(2)*m_sigma)))/(2.*(m_a - m_b));
}

double cmp::smooth_uniform_distribution_normal::quantile(const double &p)
{
    // Check if the quantile is outside the bounds
    if (p < 0 || p > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Use Newton's method to find the quantile
    double x = 0.5*(m_a+m_b);
    
    // Perform 5 Newton iterations
    for (size_t i = 0; i < 5; i++) {
        double f = cdf(x) - p;
        double df = std::exp(log_pdf(x));
        x = x - f/df;
    }

    return x;
}

double cmp::smooth_uniform_distribution_normal::sample(std::default_random_engine &rng)
{
    return m_a + (m_b-m_a)*m_dist_u(rng) + m_dist_n(rng)*m_sigma;
}
