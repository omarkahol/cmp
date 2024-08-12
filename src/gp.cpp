#include "gp.h"
#include "optimization.h"
using namespace cmp;

void gp::set_obs(const std::vector<Eigen::VectorXd> &x_obs, const std::vector<double> &y_obs)
{
    if (x_obs.size() != y_obs.size()) {
        spdlog::error("x_obs and y_obs sizes do not match!\nx_obs has size {0:d} and y_obs has size {1:d}",x_obs.size(),y_obs.size());
    } else {
        m_x_obs = x_obs;
        m_y_obs = y_obs;
    }
}

Eigen::MatrixXd gp::covariance(const Eigen::VectorXd &par) const{
    
    // Number of points
    size_t n_pts = m_x_obs.size();
    
    // Computation of the kernel matrix
    Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_pts, n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t j = i; j < n_pts; j++) {
        
            kernel_mat(i, j) = m_kernel(m_x_obs[i], m_x_obs[j], par);

            // Kernel matrix is symmetric
            if (i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat;
}

Eigen::VectorXd gp::evaluate_mean(const Eigen::VectorXd &par) const{
    Eigen::VectorXd y(m_x_obs.size());

    for (size_t i=0; i<m_x_obs.size(); i++) {
        y(i) = m_mean(m_x_obs.at(i),par);
    }

    return y;
}

Eigen::VectorXd cmp::gp::evaluate_mean_gradient(const Eigen::VectorXd &par, int i) const {
    
    Eigen::VectorXd y(m_x_obs.size());
    for (size_t j=0; j<m_x_obs.size(); j++) {
        y(j) = m_mean_gradient(m_x_obs.at(j), par, i);
    }

    return y;
}

Eigen::VectorXd gp::residual(const Eigen::VectorXd &par) const{
    
    Eigen::VectorXd res(m_y_obs.size());
    for (size_t i=0; i<m_y_obs.size(); i++) {
        res(i) = m_y_obs[i]-m_mean(m_x_obs[i],par);
    }
    return res;
}

double gp::loglikelihood(Eigen::VectorXd const &res, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const {
    return cmp::multivariate_normal_distribution::log_pdf(res,ldlt);
}

Eigen::VectorXd gp::par_opt(Eigen::VectorXd x0, double ftol_rel, nlopt::algorithm alg) const {
    opt_routine(opt_fun_gp, (void*)this, x0, m_lb_par, m_ub_par, ftol_rel, alg);
    return x0;
}

Eigen::VectorXd gp::par_opt_loo(Eigen::VectorXd x0, double ftol_rel) const {
    opt_routine(opt_fun_gp_loo, (void*)this, x0, m_lb_par, m_ub_par, ftol_rel, nlopt::LN_SBPLX);
    return x0;
}

double gp::prediction_mean(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const Eigen::VectorXd &alpha) const {

    // Compute the dot product (see Rasmussen and Williams)
    double k_star_dot_alpha = 0.0;
    for(size_t i=0; i<m_x_obs.size(); i++) {
        k_star_dot_alpha += m_kernel(x,m_x_obs.at(i),par)*alpha(i);
    }

    return m_mean(x,par) + k_star_dot_alpha;
}

double gp::prediction_variance(const Eigen::VectorXd &x, const Eigen::VectorXd &par, const Eigen::LDLT<Eigen::MatrixXd> &ldlt) const{
    
    // Define the two kernel evaluation matrices
    size_t n_obs = m_x_obs.size();

    Eigen::VectorXd k_star = Eigen::VectorXd::Zero(n_obs);
    for (size_t i = 0; i < n_obs; i++) {
        k_star(i) = m_kernel(x, m_x_obs[i], par);
    }

    // Solve the system [k] x = {k_star}
    Eigen::VectorXd k_star_sol = ldlt.solve(k_star);
    ldlt.solveInPlace(k_star_sol);

    // Compute the dot product between k_star and the solution
    return m_kernel(x,x,par) - k_star.dot(k_star_sol);
}


Eigen::VectorXd gp::draw_sample(const std::vector<Eigen::VectorXd> &x_pts, Eigen::VectorXd const &par, const Eigen::LDLT<Eigen::MatrixXd> &ldlt, const Eigen::VectorXd &res, std::default_random_engine &rng) const {

    std::normal_distribution<double> dist_n(0, 1);
    size_t n_obs = m_x_obs.size();
    size_t n_pts = x_pts.size();

    // Initialize variables
    Eigen::VectorXd gp_mean(n_pts);
    Eigen::MatrixXd k_star(n_obs, n_pts);
    Eigen::MatrixXd k_xy(n_pts, n_pts);

    for (size_t i = 0; i < n_pts; i++) {

        gp_mean(i) = m_mean(x_pts[i],par);
        
        // Fill k_star
        for (size_t j = 0; j < n_obs; j++) {
            k_star(j, i) = m_kernel(m_x_obs[j], x_pts[i], par);
        }
        
        // Fill k_xy
        for (size_t j = i; j < n_pts; j++) {
            k_xy(i, j) = m_kernel(x_pts[i], x_pts[j], par);

            // Tensor must be symmetric
            if (i!=j){
                k_xy(j,i) = k_xy(i,j);
            }
        }

    }

    // Use GP predictive equations for the evaluation of the mean and variance
    Eigen::VectorXd prediction_mean = gp_mean + k_star.transpose()*ldlt.solve(res);
    Eigen::MatrixXd prediction_var = k_xy - k_star.transpose() * ldlt.solve(k_star);
    
    // Setup the solver and compute the eigen-decomposition of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(prediction_var, Eigen::ComputeEigenvectors);
    Eigen::VectorXd eigen_val = solver.eigenvalues();

    // Compute the sqrt of the eigenvalue matrix
    for (unsigned i = 0; i < eigen_val.rows(); i++)
        eigen_val(i) = sqrt(fabs(eigen_val(i)));
    
    // Extract a uniform sample and multiply it by the sqrt of the eigenvalues
    Eigen::VectorXd sample(prediction_mean.size());
    for (int i = 0; i < sample.size(); i++)
        sample(i) = dist_n(rng) * eigen_val(i);
    
    // A single random sample ~ N(mean, cov)
    return prediction_mean + solver.eigenvectors() * sample;
}

Eigen::MatrixXd gp::predict(const std::vector<Eigen::VectorXd> &x_pts, const Eigen::VectorXd &par, const Eigen::LDLT<Eigen::MatrixXd> &ldlt, const Eigen::VectorXd &res) const{
    
    // Observation and prediction sizes
    size_t n_obs = m_x_obs.size();
    size_t n_pred = x_pts.size();
    
    // Initialize variables
    Eigen::MatrixXd k_star(n_obs, n_pred);
    Eigen::VectorXd k_xy_diag(n_pred);
    Eigen::VectorXd mean(n_pred);

    // Fill the matrices
    for (size_t i = 0; i < n_pred; i++) {
        k_xy_diag(i) = m_kernel(x_pts[i], x_pts[i], par);
        mean(i) = m_mean(x_pts[i], par);
        for (size_t j = 0; j < n_obs; j++) {
            k_star(j, i) = m_kernel(x_pts[i], m_x_obs[j], par);
        }
    }

    // Define the matrix 
    Eigen::MatrixXd mean_var(n_pred,2);

    mean_var.col(0) = mean + k_star.transpose() * ldlt.solve(res);
    mean_var.col(1) = k_xy_diag - (k_star.transpose() * ldlt.solve(k_star)).diagonal();

    return mean_var;

}


Eigen::MatrixXd gp::covariance_gradient(const Eigen::VectorXd &par, const int &n) const{
    size_t n_obs = m_x_obs.size();

    // Initialize the kernel derivative
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(n_obs, n_obs);
    
    // Fill the matrix that contains the kernel derivative
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = i; j < n_obs; j++) {
            
            k_der(i, j) = m_kernel_gradient(m_x_obs.at(i), m_x_obs.at(j), par, n);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

Eigen::MatrixXd gp::covariance_hessian(const Eigen::VectorXd &par, const int &l, const int &k) const {
    size_t n_obs = m_x_obs.size();

    // Initialize the kernel hessian
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(n_obs, n_obs);
    
    // Fill the matrix that contains the kernel hessian
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = i; j < n_obs; j++) {
            
            k_der(i, j) = m_kernel_hessian(m_x_obs.at(i), m_x_obs.at(j), par, l, k);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

Eigen::MatrixXd cmp::gp::compute_variance_reduction(const std::vector<Eigen::VectorXd> &x_pts, const Eigen::LDLT<Eigen::MatrixXd> &ldlt, const Eigen::VectorXd &par) const {

    // Sizes of the matrices
    size_t n_obs = m_x_obs.size();
    size_t n_pts = x_pts.size();

    // Initialize all the variables here
    Eigen::MatrixXd k_star(n_obs, n_pts);
    Eigen::MatrixXd variance_reduction_matrix(n_pts, n_pts);
    Eigen::VectorXd Kno(n_obs);
    Eigen::VectorXd Kpn(n_pts);


    // Fill the two matrices
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t j = 0; j <n_obs; j++) {
            k_star(j, i) = m_kernel(m_x_obs[j], x_pts[i], par);
        }
    }
    Eigen::MatrixXd K_sol = ldlt.solve(k_star);

    // Select a point and compute the variance reduction assuming that it is observed
    for (size_t sel = 0; sel<n_pts; sel++) {

        // Compute Kno
        for (size_t i = 0; i < n_obs; i++) {
            Kno(i) = m_kernel(m_x_obs[i], x_pts[sel], par);
        }

        // Compute Kpn
        for (size_t i = 0; i < n_pts; i++) {
            Kpn(i) = m_kernel(x_pts[i], x_pts[sel], par);
        }

        // Compute rh (see Formula) 
        Eigen::VectorXd rho = K_sol.transpose()*Kno - Kpn;

        // Compute sigma_n
        double sigma_n = m_kernel(x_pts[sel], x_pts[sel], par) - Kno.dot(ldlt.solve(Kno));

        // The variance reduction is computed here
        for(size_t i=0; i<n_pts; i++) {
            variance_reduction_matrix(sel,i) = rho(i)*rho(i)/sigma_n;
        }
    }

    return variance_reduction_matrix;

}

