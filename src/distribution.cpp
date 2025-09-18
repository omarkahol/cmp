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

double cmp::distribution::NormalDistribution::logPDF(const double &res, const double &std)
{
    return -0.5*log(2*M_PI) - log(std) - 0.5*pow(res/std,2);
}

double cmp::distribution::NormalDistribution::logPDF(const double & x) {
    return logPDF(x-mean_,std_);
}

double cmp::distribution::NormalDistribution::dLogPDF(const double & x)
{
    return -(x-mean_)/pow(std_,2);
}

double cmp::distribution::NormalDistribution::ddLogPDF(const double & x)
{
    return -1/pow(std_,2);
}

double cmp::distribution::NormalDistribution::CDF(const double & x)
{
    return 0.5*(1+erf((x-mean_)/(std_*sqrt(2))));
}

double cmp::distribution::NormalDistribution::quantile(const double & p) {
    return mean_ + std_*sqrt(2)*erfinv(2*p-1);
}

double cmp::distribution::NormalDistribution::sample(std::default_random_engine &rng) {
    return distN_(rng)*std_ + mean_;
}

// Normal distribution

double cmp::distribution::MultivariateNormalDistribution::logPDF(const Eigen::VectorXd &x)
{
    return logPDF(mean_-x, ldltDecomposition_);
}

double cmp::distribution::MultivariateNormalDistribution::logJumpPDF(const Eigen::VectorXd &jump)
{
    return logPDF(jump, ldltDecomposition_);
}

double cmp::distribution::MultivariateNormalDistribution::logPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition)
{
    // Solve linear system
    Eigen::VectorXd alpha = ldltDecomposition.solve(res);

    return -0.5 * res.dot(alpha) - 0.5 * (ldltDecomposition.vectorD().array().abs().log()).sum() - 0.5 * res.size() * log(2 * M_PI);
}

double cmp::distribution::MultivariateNormalDistribution::dLogPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::MatrixXd &cov_gradient, const Eigen::VectorXd &mean_gradient)
{
    // Useful quantities (see Rasmussen for reference)
    Eigen::VectorXd alpha = ldltDecomposition.solve(res);
    Eigen::MatrixXd alpha_alpha_t = alpha * alpha.transpose();

    return 0.5 * (alpha_alpha_t * cov_gradient - ldltDecomposition.solve(cov_gradient)).trace() + res.dot(ldltDecomposition.solve(mean_gradient));
}

double cmp::distribution::MultivariateNormalDistribution::ddLogPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const Eigen::MatrixXd &cov_gradient_l, const Eigen::MatrixXd &cov_gradient_k, const Eigen::MatrixXd &cov_hessian) {
    
    // Required quantities for the computation
    Eigen::VectorXd alpha = ldltDecomposition.solve(res);
    Eigen::MatrixXd alpha_alpha_t = alpha * alpha.transpose();

    // Compute the remaining symbols
    auto a_l  = ldltDecomposition.solve(cov_gradient_l);
    auto a_k  = ldltDecomposition.solve(cov_gradient_k);
    auto sym_tens = 0.5*((a_l*alpha_alpha_t) + (a_l*alpha_alpha_t).transpose());

    // Compute the 4 contributions to the hessian (I think that taking the trace 4 times is faster than adding and then taking the trace(?))
    double H1 = (alpha_alpha_t*cov_hessian).trace();
    double H2 = (ldltDecomposition.solve(cov_hessian)).trace();
    double H3 = (sym_tens*cov_gradient_k).trace();
    double H4 = (a_l*a_k).trace();

    // return the weighted KernelSum of the contributions
    return 0.5*H1 - 0.5*H2 - H3 + 0.5*H4;
}

Eigen::VectorXd cmp::distribution::MultivariateNormalDistribution::sample(std::default_random_engine &rng, const double &gamma)
{
    // Generate a sample
    Eigen::VectorXd z = Eigen::VectorXd::Zero(mean_.size());
    for (size_t i = 0; i < mean_.size(); i++) {
        z(i) = distN_(rng)*sqrt(abs(ldltDecomposition_.vectorD()(i)));
    }

    return mean_ + (ldltDecomposition_.transpositionsP().transpose()*(ldltDecomposition_.matrixL() * z))*gamma;
}

Eigen::VectorXd cmp::distribution::MultivariateNormalDistribution::get() {
    return mean_;
}

void cmp::distribution::MultivariateNormalDistribution::set(const Eigen::VectorXd &x) {
    mean_ = x;
}

Eigen::VectorXd cmp::distribution::UniformSphereDistribution::sample(std::default_random_engine &rng) {
    // Sample a normal distribution
    Eigen::VectorXd x(dim_);
    for (size_t i = 0; i < dim_; i++) {
        x(i) = distN_(rng);
    }

    // Normalize the vector
    return x/x.norm();
}

double cmp::distribution::UniformDistribution::logPDF(const double & x)
{
   if (x < lowerBound_ || x > upperBound_) {
       return -std::numeric_limits<double>::infinity();
   } else {
       return -log(upperBound_-lowerBound_);
   }
}

double cmp::distribution::UniformDistribution::CDF(const double & x)
{
    if (x < lowerBound_) {
        return 0.0;
    } else if (x > upperBound_) {
        return 1.0;
    } else {
        return (x-lowerBound_)/(upperBound_-lowerBound_);
    }
}

double cmp::distribution::UniformDistribution::quantile(const double & p)
{
    return lowerBound_ + p*(upperBound_-lowerBound_);
}

double cmp::distribution::UniformDistribution::sample(std::default_random_engine &rng)
{
    return  distN_(rng)*(upperBound_-lowerBound_); + lowerBound_;
}

double cmp::distribution::InverseGammaDistribution::logPDF(const double & x)
{
    return - (alpha_ + 1)*log(x) - beta_/x + alpha_*log(beta_) - std::log(std::tgamma(alpha_));
}

double cmp::distribution::InverseGammaDistribution::dLogPDF(const double & x) {
    return (beta_-(alpha_+1)*x)/std::pow(x,2);
}  
double cmp::distribution::InverseGammaDistribution::ddLogPDF(const double & x) {

    return (-2*beta_+x+x*alpha_)/std::pow(x,3);
}

double cmp::distribution::InverseGammaDistribution::CDF(const double & x)
{
    // To-do implement gamma regularized
    return 0.0;
}

double cmp::distribution::InverseGammaDistribution::quantile(const double & p)
{
    // To-do implement inverse gamma regularized
    return 0.0;
}

double cmp::distribution::InverseGammaDistribution::sample(std::default_random_engine &rng)
{
    double x = distGamma_(rng);
    return 1/x;
}

double cmp::distribution::MultivariateStudentDistribution::logPDF(const Eigen::VectorXd &res, const Eigen::LDLT<Eigen::MatrixXd> &ldltDecomposition, const double &nu)
{
    // Solve linear system
    Eigen::VectorXd alpha = ldltDecomposition.solve(res);
    double det = (ldltDecomposition.vectorD().array().abs().log()).sum();
    size_t dim = res.size();

    return std::log(std::tgammal(0.5*(nu + dim)) - std::log(std::tgammal(0.5*nu)) - 0.5*dim*std::log(M_PI*nu)) - 0.5*det -0.5*(nu+dim)*std::log(1+res.dot(alpha)/nu);
}

double cmp::distribution::MultivariateStudentDistribution::logPDF(const Eigen::VectorXd &x)
{
    return logPDF(mean_-x, ldltDecomposition_, dofs_);
}

double cmp::distribution::MultivariateStudentDistribution::logJumpPDF(const Eigen::VectorXd &jump)
{
    return logPDF(jump, ldltDecomposition_, dofs_);
}

Eigen::VectorXd cmp::distribution::MultivariateStudentDistribution::sample(std::default_random_engine &rng, const double &gamma)
{
    // Sample from a T distribution
    Eigen::VectorXd z = Eigen::VectorXd::Zero(mean_.size());
    for (size_t i = 0; i < mean_.size(); i++) {
        z(i) = distN_(rng);

        // Generate a sample from a chi-square distribution
        double chi = 0.0;
        for (size_t j = 0; j < dofs_; j++) {
            chi += std::pow(distN_(rng),2);
        }

        z(i) = sqrt(abs(ldltDecomposition_.vectorD()(i)))*z(i)/std::sqrt(chi/dofs_);
    }

    return mean_ + (ldltDecomposition_.transpositionsP().transpose()*(ldltDecomposition_.matrixL() * z))*gamma;
}

Eigen::VectorXd cmp::distribution::MultivariateStudentDistribution::get()
{
    return mean_;
} 

void cmp::distribution::MultivariateStudentDistribution::set(const Eigen::VectorXd &x)
{
    mean_ = x;
}

double cmp::distribution::LogNormalDistribution::logPDF(const double &x)
{
    return -0.5 * pow((log(x) - mean_) / std_, 2) - 0.5 * log(2 * M_PI);
}

double cmp::distribution::LogNormalDistribution::dLogPDF(const double &x)
{
    return (mean_-log(x))/(x*pow(std_,2));;
}

double cmp::distribution::LogNormalDistribution::ddLogPDF(const double &x)
{
    return (-1-mean_+log(x))/pow(x*std_,2);
}

double cmp::distribution::LogNormalDistribution::CDF(const double &x)
{
    return 0.5*(1+std::erf((log(x)-mean_)/(std_*sqrt(2))));
}

double cmp::distribution::LogNormalDistribution::quantile(const double &p)
{
    return std::exp(mean_ + std_*sqrt(2)*erfinv(2*p-1));
}

double cmp::distribution::LogNormalDistribution::sample(std::default_random_engine &rng)
{
    return std::exp(distN_(rng)*std_ + mean_);
}

double cmp::distribution::StudentDistribution::logPDF(const double &x)
{
    return std::log(std::tgammal(0.5*(dofs_ + 1)) - std::log(std::tgammal(0.5*dofs_)) - 0.5*std::log(M_PI*dofs_)) - std_ -0.5*(dofs_+1)*std::log(1+pow(x-mean_,2)/(dofs_*std_*std_));
}

double cmp::distribution::StudentDistribution::dLogPDF(const double &x)
{
    return -(x-mean_)*(1+dofs_)/(std_*std_*dofs_ + pow(x-mean_,2));
}

double cmp::distribution::StudentDistribution::ddLogPDF(const double &x)
{
    return (1+dofs_)*(pow(x-mean_,2)-dofs_*std_*std_)/pow((pow(x-mean_,2)+dofs_*std_*std_),2);
}

double cmp::distribution::StudentDistribution::CDF(const double &x)
{
    // To-do implement
    return 0.0;
}

double cmp::distribution::StudentDistribution::quantile(const double &p)
{   
    // To-do implement
    return 0.0;
}

double cmp::distribution::StudentDistribution::sample(std::default_random_engine &rng)
{   
    
    // Sample a normal variate
    double z = distN_(rng);

    // Sample a chi-square variate
    double chi = 0.0;
    for (size_t j = 0; j < dofs_; j++) {
        chi += std::pow(distN_(rng),2);
    }

    // Return the sample
    return mean_ + z*std_/std::sqrt(chi/dofs_);
}

double sech(const double &x)
{
    return 1.0/std::cosh(x);
}

double cmp::distribution::PowerLawDistribution::logPDF(const double &x)
{
    return log(lowerBound_) - degree_*log(x);
}

double cmp::distribution::PowerLawDistribution::dLogPDF(const double &x)
{
    return -degree_/x;
}

double cmp::distribution::PowerLawDistribution::ddLogPDF(const double &x)
{
    return degree_/pow(x,2);
}

double cmp::distribution::PowerLawDistribution::CDF(const double &t)
{
    return 1.0 - pow(lowerBound_/t,degree_);
}

double cmp::distribution::PowerLawDistribution::quantile(const double &q)
{
    return lowerBound_/pow(1-q,1/degree_);
}

double cmp::distribution::PowerLawDistribution::sample(std::default_random_engine &rng)
{
    return quantile(distU_(rng));
}

double cmp::distribution::MultivariateUniformDistribution::logPDF(const Eigen::VectorXd &x) {
    return logJumpPDF(x-mean_);
}

double cmp::distribution::MultivariateUniformDistribution::logJumpPDF(const Eigen::VectorXd &jump) {
    for (size_t i = 0; i < jump.size(); i++) {
        if (jump(i) < -size_(i)*0.5 || jump(i) > size_(i)*0.5) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    return 0.0;
}
Eigen::VectorXd cmp::distribution::MultivariateUniformDistribution::sample(std::default_random_engine &rng, const double &gamma) {
    Eigen::VectorXd rv(mean_.size());
    for (size_t i = 0; i < mean_.size(); i++) {
        rv(i) = mean_(i) - 0.5*size_(i)*gamma + size_(i)*distN_(rng)*gamma;
    }
    return rv;
}

Eigen::VectorXd cmp::distribution::MultivariateUniformDistribution::get() {
    return mean_;
}
void cmp::distribution::MultivariateUniformDistribution::set(const Eigen::VectorXd &x){
    mean_ = x;
}

double CDF_normal(const double &x)
{
    return 0.5*(1+erf(x/sqrt(2)));
}

double cmp::distribution::SmoothUniformDistribution::logPDF(const double &x)
{
    return std::log((CDF_normal((upperBound_-x)/std_) - CDF_normal((lowerBound_-x)/std_))/(upperBound_-lowerBound_));
}

double cmp::distribution::SmoothUniformDistribution::dLogPDF(const double &x)
{
    return (std::pow(M_E,-std::pow(lowerBound_ - x,2)/(2.*std::pow(std_,2))) - std::pow(M_E,-std::pow(upperBound_ - x,2)/(2.*std::pow(std_,2))))/((-lowerBound_ + upperBound_)*std::sqrt(2*M_PI)*std_);
}

double cmp::distribution::SmoothUniformDistribution::ddLogPDF(const double &x)
{
    return ((lowerBound_ - x)/
      std::pow(M_E,std::pow(lowerBound_ - x,2)/
        (2.*std::pow(std_,2))) + 
     (-upperBound_ + x)/
      std::pow(M_E,std::pow(upperBound_ - x,2)/
        (2.*std::pow(std_,2))))/
   ((-lowerBound_ + upperBound_)*std::sqrt(2*M_PI)*std::pow(std_,3));
}

double cmp::distribution::SmoothUniformDistribution::CDF(const double &t)
{
    return (lowerBound_ - upperBound_ + (-std::pow(M_E,
          -std::pow(lowerBound_ - t,2)/
           (2.*std::pow(std_,2))) + 
        std::pow(M_E,
         -std::pow(upperBound_ - t,2)/
          (2.*std::pow(std_,2))))*
      std::sqrt(2/M_PI)*std_ + 
     (-lowerBound_ + t)*std::erf((lowerBound_ - t)/
        (std::sqrt(2)*std_)) + 
     (upperBound_ - t)*std::erf((upperBound_ - t)/
        (std::sqrt(2)*std_)))/(2.*(lowerBound_ - upperBound_));
}

double cmp::distribution::SmoothUniformDistribution::quantile(const double &p)
{
    // Check if the quantile is outside the bounds
    if (p < 0 || p > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Use Newton's method to find the quantile
    double x = 0.5*(lowerBound_+upperBound_);
    
    // Perform 5 Newton iterations
    for (size_t i = 0; i < 5; i++) {
        double f = CDF(x) - p;
        double df = std::exp(logPDF(x));
        x = x - f/df;
    }

    return x;
}

double cmp::distribution::SmoothUniformDistribution::sample(std::default_random_engine &rng)
{
    return lowerBound_ + (upperBound_-lowerBound_)*distU_(rng) + distN_(rng)*std_;
}
