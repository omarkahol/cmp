#include "gp.h"
#include "optimization.h"
using namespace cmp;


/*
FUNCTIONS FOR THE KERNEL
*/
Eigen::MatrixXd gp::covariance(Eigen::VectorXd par) const
{
    
    // Computation of the kernel matrix
    Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(m_size, m_size);
    for (size_t i = 0; i < m_size; i++) {
        for (size_t j = i; j < m_size; j++) {
            kernel_mat(i, j) = m_kernel(m_x_obs->at(i), m_x_obs->at(j), par);

            // Kernel matrix is symmetric
            if (i != j) {
                kernel_mat(j, i) = kernel_mat(i, j);
            }
        }
    }
    return kernel_mat;
}

Eigen::MatrixXd gp::covariance_grad(Eigen::VectorXd par,const int &n) const{

    // Initialize the kernel derivative
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(m_size, m_size);
    
    // Fill the matrix that contains the kernel derivative
    for (size_t i = 0; i < m_size; i++) {
        for (size_t j = i; j < m_size; j++) {
            
            k_der(i, j) = m_kernel_grad(m_x_obs->at(i), m_x_obs->at(j), par, n);
            
            // The matrix is symmetric
            if (i != j) {
                k_der(j, i) = k_der(i, j);
            }
        }
    }

    return k_der;
}

Eigen::MatrixXd gp::covariance_hess(Eigen::VectorXd par,const int &l, const int &k) const {

    // Initialize the kernel hessian
    Eigen::MatrixXd k_der = Eigen::MatrixXd::Zero(m_size, m_size);
    
    // Fill the matrix that contains the kernel hessian
    for (size_t i = 0; i < m_size; i++) {
        for (size_t j = i; j < m_size; j++) {
            
            k_der(i, j) = m_kernel_hess(m_x_obs->at(i), m_x_obs->at(j), par, l, k);
            
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
Eigen::VectorXd gp::mean(Eigen::VectorXd par) const{
    Eigen::VectorXd y(m_size);

    for (size_t i=0; i<m_size; i++) {
        y(i) = m_mean(m_x_obs->at(i),par);
    }

    return y;
}

Eigen::VectorXd cmp::gp::mean_grad(Eigen::VectorXd par,const int &i) const
{
     Eigen::VectorXd y(m_size);
    for (size_t j=0; j<m_size; j++) {
        y(j) = m_mean_grad(m_x_obs->at(j), par, i);
    }

    return y;
}

Eigen::VectorXd gp::residual(Eigen::VectorXd par) const
{

    Eigen::VectorXd res(m_size);
    for (size_t i=0; i<m_size; i++) {
        res(i) = m_y_obs->at(i)-m_mean(m_x_obs->at(i),par);
    }
    return res;
}



void cmp::gp::set_params(const Eigen::VectorXd &par)
{
    m_par = par;
    
    // Compute the covariance matrix
    Eigen::MatrixXd cov = covariance(par);

    // Compute the decomposition
    m_cov_ldlt.compute(cov);

    // Compute the solution of [K]x=r for fast prediction
    m_alpha = m_cov_ldlt.solve(residual(par));

}

void cmp::gp::fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const cmp::method &method, const nlopt::algorithm &alg, const double &tol_rel)
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
        case cmp::MLE:
            optimizer.set_max_objective(opt_fun_gp_mle, static_cast<void*>(this));
            break;
        case cmp::MAP:
            optimizer.set_max_objective(opt_fun_gp_map, static_cast<void*>(this));
            break;
        case cmp::MLOO:
            optimizer.set_max_objective(opt_fun_gp_mloo, static_cast<void*>(this));
            break;
        case cmp::MLOOP:
            optimizer.set_max_objective(opt_fun_gp_mloop, static_cast<void*>(this));
            break;
        default:
            throw std::runtime_error("Unknown optimization method");
    }

    double f_val=0.0;
    try {
        optimizer.optimize(x0_v, f_val);
        set_params(v_to_vxd(x0_v));
    } catch (const std::runtime_error &err) {
        std::cout << err.what() << "\n";
        std::cout << "Local optimization failed. Keeping initial value." << std::endl;
        set_params(x0);
    }


}

double gp::predictive_mean(Eigen::VectorXd x) const {

    // Scale the input
    m_x_obs->transform(x);

    // Compute the dot product (see Rasmussen and Williams)
    double k_star_dot_alpha = 0.0;
    for(size_t i=0; i<m_size; i++) {
        k_star_dot_alpha += m_kernel(x,m_x_obs->at(i),m_par)*m_alpha(i);
    }

    double pred = m_mean(x,m_par) + k_star_dot_alpha;
    m_y_obs->inverse_transform(pred);
    return pred;
}

double gp::predictive_var(Eigen::VectorXd x) const {

    // Scale the input
    m_x_obs->transform(x);

    Eigen::VectorXd k_star = Eigen::VectorXd::Zero(m_size);
    for (size_t i = 0; i < m_size; i++) {
        k_star(i) = m_kernel(x, m_x_obs->at(i), m_par);
    }

    // Solve the system [k] x = {k_star}
    Eigen::VectorXd k_star_sol = m_cov_ldlt.solve(k_star);

    // Compute the dot product between k_star and the solution
    double pred = m_kernel(x,x,m_par) - k_star.dot(k_star_sol);
    return pred*std::pow(m_y_obs->get_scale(),2);
}


std::vector<Eigen::VectorXd> gp::sample(std::vector<Eigen::VectorXd> x_pts, std::default_random_engine &rng, const size_t &n) const {

    // Scale the input
    m_x_obs->transform(x_pts);

    std::normal_distribution<double> dist_n(0, 1);
    size_t n_pts = x_pts.size();

    // Initialize variables
    Eigen::VectorXd gp_mean(n_pts);
    Eigen::MatrixXd k_star(m_size, n_pts);
    Eigen::MatrixXd k_xy(n_pts, n_pts);

    for (size_t i = 0; i < n_pts; i++) {

        gp_mean(i) = m_mean(x_pts[i],m_par);
        
        // Fill k_star
        for (size_t j = 0; j < m_size; j++) {
            k_star(j, i) = m_kernel(m_x_obs->at(j), x_pts.at(i), m_par);
        }
        
        // Fill k_xy
        for (size_t j = i; j < n_pts; j++) {
            k_xy(i, j) = m_kernel(x_pts.at(i), x_pts.at(j), m_par);

            // Tensor must be symmetric
            if (i!=j){
                k_xy(j,i) = k_xy(i,j);
            }
        }

    }

    // Use GP predictive equations for the evaluation of the mean and variance
    Eigen::VectorXd prediction_mean = gp_mean + k_star.transpose()*m_alpha;
    Eigen::MatrixXd prediction_var = k_xy - k_star.transpose() * m_cov_ldlt.solve(k_star);
    
    // Setup the solver and compute the eigen-decomposition of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(prediction_var, Eigen::ComputeEigenvectors);
    Eigen::VectorXd eigen_val = solver.eigenvalues();

    // Compute the sqrt of the eigenvalue matrix
    for (unsigned i = 0; i < eigen_val.rows(); i++)
        eigen_val(i) = sqrt(fabs(eigen_val(i)));
    
    // Extract a uniform sample and multiply it by the sqrt of the eigenvalues
    std::vector<Eigen::VectorXd> sample(n, Eigen::VectorXd::Zero(n_pts));
    for (int i = 0; i < n; i++) {
        for (size_t j=0; j<n_pts; j++) {
            sample[i](j) = eigen_val(j)*dist_n(rng);
        }
        sample[i] = prediction_mean + solver.eigenvectors() * sample[i];

        // Inverse transform the sample
        for (size_t j=0; j<n_pts; j++) {
            m_y_obs->inverse_transform(sample[i](j));
        }
    }
    
    
    return sample;
}

Eigen::MatrixXd cmp::gp::expected_variance_improvement(std::vector<Eigen::VectorXd> x_pts, double nu) const {

    // Scale the input
    m_x_obs->transform(x_pts);

    // Sizes of the matrices
    size_t n_pts = x_pts.size();

    // Initialize all the variables here
    Eigen::MatrixXd Kop(m_size, n_pts);
    Eigen::MatrixXd variance_reduction_matrix(n_pts, n_pts);
    Eigen::VectorXd Kno(m_size);
    Eigen::VectorXd Kpn(n_pts);


    // Fill the two matrices
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t j = 0; j <m_size; j++) {
            Kop(j, i) = m_kernel(m_x_obs->at(j), x_pts[i], m_par);
        }
    }

    // Select a point and compute the variance reduction assuming that it is observed
    for (size_t sel = 0; sel<n_pts; sel++) {

        // Compute Kno
        for (size_t i = 0; i < m_size; i++) {
            Kno(i) = m_kernel(m_x_obs->at(i), x_pts[sel], m_par);
        }

        // Compute Kpn
        for (size_t i = 0; i < n_pts; i++) {
            Kpn(i) = m_kernel(x_pts[i], x_pts[sel], m_par);
        }

        // Compute rh (see Formula) 
        Eigen::VectorXd Ksol = m_cov_ldlt.solve(Kno);
        Eigen::VectorXd rho = Kpn - Kop.transpose()*Ksol;

        // Compute sigma_n
        double sigma_n = abs(m_kernel(x_pts[sel], x_pts[sel], m_par) - Kno.dot(Ksol)) + nu;

        // The variance reduction is computed here
        for(size_t i=0; i<n_pts; i++) {
            variance_reduction_matrix(sel,i) = rho(i)*rho(i)/(sigma_n);
        }
    }

    return variance_reduction_matrix*m_y_obs->get_scale()*m_y_obs->get_scale();
}

Eigen::VectorXd cmp::gp::expected_variance_improvement(std::vector<Eigen::VectorXd> x_pts, const std::vector<Eigen::VectorXd> &new_x_obs, double nu) const{
    
    // Scale the input
    m_x_obs->transform(x_pts);
    
    // Define the two kernel evaluation matrices
    size_t n_pts = x_pts.size();
    size_t n_new_obs = new_x_obs.size();

    Eigen::MatrixXd k_no = Eigen::MatrixXd::Zero(n_new_obs, m_size);
    Eigen::MatrixXd k_nn = Eigen::MatrixXd::Zero(n_new_obs, n_new_obs);
    Eigen::MatrixXd k_np = Eigen::MatrixXd::Zero(n_new_obs, n_pts);
    Eigen::MatrixXd k_op = Eigen::MatrixXd::Zero(m_size, n_pts);


    // Fill the matrices
    for (size_t i = 0; i < n_new_obs; i++) {
        for (size_t j = 0; j < m_size; j++) {
            k_no(i, j) = m_kernel(new_x_obs[i], m_x_obs->at(j), m_par);
        }
        for (size_t j = i; j < n_new_obs; j++) {
            k_nn(i, j) = m_kernel(new_x_obs[i], new_x_obs[j], m_par);

            // The matrix is symmetric
            if (i != j) {
                k_nn(j, i) = k_nn(i, j);
            }
        }
    }

    k_nn = k_nn - k_no*m_cov_ldlt.solve(k_no.transpose()) + nu*Eigen::MatrixXd::Identity(n_new_obs,n_new_obs);

    for (size_t i = 0; i < n_pts; i++) {
        for (size_t j = 0; j < n_new_obs; j++) {
            k_np(j, i) = m_kernel(new_x_obs[j], x_pts[i], m_par);
        }
        for (size_t j = 0; j < m_size; j++) {
            k_op(j, i) = m_kernel(m_x_obs->at(j), x_pts[i], m_par);
        }
    }

    // Compute rho
    Eigen::MatrixXd rho = k_np-k_no*m_cov_ldlt.solve(k_op);
    Eigen::MatrixXd rho_sol = k_nn.ldlt().solve(rho);


    // Compute the expected variance reuduction
    Eigen::VectorXd evi(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        evi(i) = rho.col(i).dot(rho_sol.col(i));
    }

    return evi*m_y_obs->get_scale()*m_y_obs->get_scale();
    
}
