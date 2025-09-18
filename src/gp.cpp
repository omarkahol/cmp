#include "gp.h"
using namespace cmp::gp;


/*
FUNCTIONS FOR THE KERNEL
*/
Eigen::MatrixXd GaussianProcess::covariance(Eigen::VectorXd par) const
{
    
    // Computation of the kernel matrix
    Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(nObs_, nObs_);
    for (size_t i = 0; i < nObs_; i++) {
        for (size_t j = i; j < nObs_; j++) {
            kernel_mat(i, j) = pKernel_->eval(XObs_[i], XObs_[j], par);

            // Kernel matrix is symmetric
            if (i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat + nugget_ * Eigen::MatrixXd::Identity(nObs_, nObs_);
}

Eigen::MatrixXd GaussianProcess::covarianceGradient(Eigen::VectorXd par,const int &n) const{

    // Initialize the kernel derivative
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(nObs_, nObs_);
    
    // Fill the matrix that contains the kernel derivative
    for (size_t i = 0; i < nObs_; i++) {
        for (size_t j = i; j < nObs_; j++) {
            
            k_der(i, j) = pKernel_->evalGradient(XObs_[i], XObs_[j], par, n);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

Eigen::MatrixXd GaussianProcess::covarianceHessian(Eigen::VectorXd par,const int &l, const int &k) const {

    // Initialize the kernel hessian
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(nObs_, nObs_);
    
    // Fill the matrix that contains the kernel hessian
    for (size_t i = 0; i < nObs_; i++) {
        for (size_t j = i; j < nObs_; j++) {
            
            k_der(i, j) = pKernel_->evalHessian(XObs_[i], XObs_[j], par, l, k);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}


/*
FUNCTIONS FOR THE MEAN
*/
Eigen::VectorXd GaussianProcess::priorMean(Eigen::VectorXd par) const{
    Eigen::VectorXd y(nObs_);

    for (size_t i=0; i<nObs_; i++) {
        y(i) = pMean_->eval(XObs_[i],par);
    }

    return y;
}

Eigen::VectorXd GaussianProcess::priorMeanGradient(Eigen::VectorXd par,const int &i) const
{
     Eigen::VectorXd y(nObs_);
    for (size_t j=0; j<nObs_; j++) {
        y(j) = pMean_->evalGradient(XObs_[j], par, i);
    }

    return y;
}

Eigen::VectorXd GaussianProcess::residual(Eigen::VectorXd par) const
{

    Eigen::VectorXd res(nObs_);
    for (size_t i=0; i<nObs_; i++) {
        res(i) = YObs_[i]-pMean_->eval(XObs_[i],par);
    }
    return res;
}



void GaussianProcess::setParameters(const Eigen::VectorXd &par)
{

    // Store the parameters and scale them
    par_ = par;
    
    // Compute the covariance matrix
    Eigen::MatrixXd cov = covariance(par);

    // Compute the decomposition
    covDecomposition_.compute(cov);

    // Store the residuals
    residual_ = residual(par_);

    // Compute the solution of [K]x=r for fast prediction
    alpha_ = covDecomposition_.solve(residual_);

    // Compute the inverse of the covariance matrix and store its diagonal
    diagCovInverse_ = covDecomposition_.solve(Eigen::MatrixXd::Identity(nObs_,nObs_)).diagonal();

}

void GaussianProcess::fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const method &method, const nlopt::algorithm &alg, const double &tol_rel)
{
    // Convert Eigen::VectorXd to std::vector
    std::vector<double> x0_v = vxd_to_v(x0);
    std::vector<double> lb_v = vxd_to_v(lb);
    std::vector<double> ub_v = vxd_to_v(ub);

    // Initialize the algorithm
    nlopt::opt optimizer(alg, x0.size());

    // Set the optimization parameters
    optimizer.set_ftol_rel(tol_rel);
    optimizer.set_lower_bounds(lb_v);
    optimizer.set_upper_bounds(ub_v);

    // Set the optimization function
    switch (method) {
        case MLE:
            optimizer.set_max_objective(opt_fun_gp_mle, static_cast<void*>(this));
            break;
        case MAP:
            optimizer.set_max_objective(opt_fun_gp_map, static_cast<void*>(this));
            break;
        case MLOO:
            optimizer.set_max_objective(opt_fun_gp_mloo, static_cast<void*>(this));
            break;
        case MLOOP:
            optimizer.set_max_objective(opt_fun_gp_mloop, static_cast<void*>(this));
            break;
        default:
            throw std::runtime_error("Unknown optimization method");
    }

    double f_val=0.0;
    try {
        optimizer.optimize(x0_v, f_val);
        setParameters(v_to_vxd(x0_v));
    } catch (const std::runtime_error &err) {
        std::cout << err.what() << "\n";
        std::cout << "Local optimization failed. Keeping initial value." << std::endl;
        setParameters(x0);
    }


}

double GaussianProcess::predict(const Eigen::VectorXd &x) const {

    // Compute the dot product (see Rasmussen and Williams)
    double k_star_dot_alpha = 0.0;
    for(size_t i=0; i<nObs_; i++) {
        k_star_dot_alpha += pKernel_->eval(x,XObs_[i],par_)*alpha_(i);
    }

    return pMean_->eval(x,par_) + k_star_dot_alpha;
}

double cmp::gp::GaussianProcess::predictLooCV(const size_t &i) const
{
    return YObs_[i] - alpha_(i)/diagCovInverse_(i);
}

double GaussianProcess::predictVariance(const Eigen::VectorXd &x) const {

    // Compute the kernel matrix
    Eigen::VectorXd k_star = Eigen::VectorXd::Zero(nObs_);
    for(size_t i=0; i<nObs_; i++) {
        k_star(i) = pKernel_->eval(x,XObs_[i],par_);
    }

    // Compute the variance
    return pKernel_->eval(x,x,par_) - k_star.dot(covDecomposition_.solve(k_star)) + nugget_;
}

double cmp::gp::GaussianProcess::predictVarianceLooCV(const size_t &i) const
{
    return 1.0/diagCovInverse_(i);
}

cmp::distribution::NormalDistribution cmp::gp::GaussianProcess::predictiveDistribution(const Eigen::VectorXd &x) const
{

    double k_star_dot_alpha = 0.0;
    Eigen::VectorXd k_star = Eigen::VectorXd::Zero(nObs_);
    for (size_t i = 0; i < nObs_; i++) {
        k_star(i) = pKernel_->eval(x, XObs_[i], par_);
        k_star_dot_alpha += k_star(i)*alpha_(i);
    }

    // Solve the system [k] x = {k_star}
    Eigen::VectorXd k_star_sol = covDecomposition_.solve(k_star);

    // Compute the prediction
    double var = pKernel_->eval(x,x,par_) - k_star.dot(k_star_sol) + nugget_;
    double mean = pMean_->eval(x,par_) + k_star_dot_alpha;

    return cmp::distribution::NormalDistribution(mean,std::sqrt(var));
}

cmp::distribution::MultivariateNormalDistribution cmp::gp::GaussianProcess::predictiveDistribution(const std::vector<Eigen::VectorXd> &x_pts) const
{
    // Predictive mean and covariance
    auto [mean, cov] = predictiveMeanAndCovariance(x_pts);

    // Return the distribution
    return cmp::distribution::MultivariateNormalDistribution(mean,cov.ldlt());
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> cmp::gp::GaussianProcess::predictiveMeanAndCovariance(const std::vector<Eigen::VectorXd> &x_pts) const {
    size_t n_pts = x_pts.size();

    // Initialize variables
    Eigen::MatrixXd k_star(nObs_, n_pts);
    Eigen::MatrixXd k_xx(n_pts, n_pts);
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(x_pts.size());

    for (size_t i = 0; i < n_pts; i++) {
        
        // Evaluate the prior mean
        mean(i) = pMean_->eval(x_pts[i],par_);

        // Fill k_star and compute the mean
        for (size_t j = 0; j < nObs_; j++) {
            k_star(j, i) = pKernel_->eval(XObs_[j], x_pts[i], par_);
            mean(i) += k_star(j,i)*alpha_(j);
        }
        
        // Fill k_xy
        for (size_t j = i; j < n_pts; j++) {
            k_xx(i, j) = pKernel_->eval(x_pts[i], x_pts[j], par_);

            // Tensor must be symmetric
            if (i!=j){
                k_xx(j,i) = k_xx(i,j);
            }
        }

    }

    auto cov = k_xx - k_star.transpose() * covDecomposition_.solve(k_star) + nugget_*Eigen::MatrixXd::Identity(n_pts,n_pts);

    return std::make_pair(mean, cov);
}

cmp::distribution::NormalDistribution cmp::gp::GaussianProcess::predictiveDistributionLooCV(const size_t &i) const
{
    double mean = YObs_[i] - alpha_(i)/diagCovInverse_(i);
    double std = 1.0/std::sqrt(diagCovInverse_(i));

    return cmp::distribution::NormalDistribution(mean, std);
}

double GaussianProcess::logLikelihood() const
{
    return cmp::distribution::MultivariateNormalDistribution::logPDF(residual_,covDecomposition_);
}

double GaussianProcess::logLikelihoodLooCV(const size_t &i) const
{
    return cmp::distribution::NormalDistribution::logPDF(YObs_[i]-predictLooCV(i),std::sqrt(predictVarianceLooCV(i)));
}

Eigen::MatrixXd GaussianProcess::expectedVarianceImprovement(const std::vector<Eigen::VectorXd> &x_pts, double nu) const {

    // Sizes of the matrices
    size_t n_pts = x_pts.size();

    // Initialize all the variables here
    Eigen::MatrixXd Kop(nObs_, n_pts);
    Eigen::MatrixXd variance_reduction_matrix(n_pts, n_pts);
    Eigen::VectorXd Kno(nObs_);
    Eigen::VectorXd Kpn(n_pts);


    // Fill the two matrices
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t j = 0; j <nObs_; j++) {
            Kop(j, i) = pKernel_->eval(XObs_[j], x_pts[i], par_);
        }
    }

    // Select a point and compute the variance reduction assuming that it is observed
    for (size_t sel = 0; sel<n_pts; sel++) {

        // Compute Kno
        for (size_t i = 0; i < nObs_; i++) {
            Kno(i) = pKernel_->eval(XObs_[i], x_pts[sel], par_);
        }

        // Compute Kpn
        for (size_t i = 0; i < n_pts; i++) {
            Kpn(i) = pKernel_->eval(x_pts[i], x_pts[sel], par_);
        }

        // Compute rh (see Formula) 
        Eigen::VectorXd Ksol = covDecomposition_.solve(Kno);
        Eigen::VectorXd rho = Kpn - Kop.transpose()*Ksol;

        // Compute sigma_n
        double sigma_n = abs(pKernel_->eval(x_pts[sel], x_pts[sel], par_) - Kno.dot(Ksol)) + nu;

        // The variance reduction is computed here
        for(size_t i=0; i<n_pts; i++) {
            variance_reduction_matrix(sel,i) = rho(i)*rho(i)/(sigma_n);
        }
    }

    return variance_reduction_matrix;
}

double GaussianProcess::predictiveCovariance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{
    // Compute k_star1
    Eigen::VectorXd k_star1 = Eigen::VectorXd::Zero(nObs_);
    for(size_t i=0; i<nObs_; i++) {
        k_star1(i) = pKernel_->eval(x1,XObs_[i],par_);
    }

    // Compute k_star2
    Eigen::VectorXd k_star2 = Eigen::VectorXd::Zero(nObs_);
    for(size_t i=0; i<nObs_; i++) {
        k_star2(i) = pKernel_->eval(x2,XObs_[i],par_);
    }

    // Compute the covariance
    return pKernel_->eval(x1,x2,par_) - k_star1.dot(covDecomposition_.solve(k_star2));
}

Eigen::VectorXd GaussianProcess::expectedVarianceImprovement(const std::vector<Eigen::VectorXd> &x_pts, const std::vector<Eigen::VectorXd> &new_x_obs, double nu) const{
    
    // Define the two kernel evaluation matrices
    size_t n_pts = x_pts.size();
    size_t n_new_obs = new_x_obs.size();

    Eigen::MatrixXd k_no = Eigen::MatrixXd::Zero(n_new_obs, nObs_);
    Eigen::MatrixXd k_nn = Eigen::MatrixXd::Zero(n_new_obs, n_new_obs);
    Eigen::MatrixXd k_np = Eigen::MatrixXd::Zero(n_new_obs, n_pts);
    Eigen::MatrixXd k_op = Eigen::MatrixXd::Zero(nObs_, n_pts);


    // Fill the matrices
    for (size_t i = 0; i < n_new_obs; i++) {
        for (size_t j = 0; j < nObs_; j++) {
            k_no(i, j) = pKernel_->eval(new_x_obs[i], XObs_[j], par_);
        }
        for (size_t j = i; j < n_new_obs; j++) {
            k_nn(i, j) = pKernel_->eval(new_x_obs[i], new_x_obs[j], par_);

            // The matrix is symmetric
            if (i != j) {
                k_nn(j, i) = k_nn(i, j);
            }
        }
    }

    k_nn = k_nn - k_no*covDecomposition_.solve(k_no.transpose()) + nu*Eigen::MatrixXd::Identity(n_new_obs,n_new_obs);

    for (size_t i = 0; i < n_pts; i++) {
        for (size_t j = 0; j < n_new_obs; j++) {
            k_np(j, i) = pKernel_->eval(new_x_obs[j], x_pts[i], par_);
        }
        for (size_t j = 0; j < nObs_; j++) {
            k_op(j, i) = pKernel_->eval(XObs_[j], x_pts[i], par_);
        }
    }

    // Compute rho
    Eigen::MatrixXd rho = k_np-k_no*covDecomposition_.solve(k_op);
    Eigen::MatrixXd rho_sol = k_nn.ldlt().solve(rho);


    // Compute the expected variance reuduction
    Eigen::VectorXd evi(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        evi(i) = rho.col(i).dot(rho_sol.col(i));
    }

    return evi;
    
}
