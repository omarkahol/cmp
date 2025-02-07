#include <multi_gp.h>

void cmp::multi_gp::set_observations(cmp::vector_scaler *x_obs, cmp::vector_scaler *y_obs)
{
    m_x_obs = x_obs;
    m_y_obs = y_obs;

    std::cout << "Setting the observations" << std::endl;

    // Create the GPs and save their observations
    for (size_t i=0; i<m_dim; i++) {
        m_gps[i] = gp();
        m_scalers[i].fit(m_y_obs,i);
        m_gps[i].set_observations(m_x_obs,&m_scalers[i]);
    }
}

void cmp::multi_gp::set_kernel(kernel_t kernel)
{
    for (auto &gp : m_gps) {
        gp.set_kernel(kernel);
    }
}

void cmp::multi_gp::set_mean(model_t mean)
{
    for (auto &gp : m_gps) {
        gp.set_mean(mean);
    }
}

void cmp::multi_gp::set_log_prior(prior_t log_prior)
{
    for (auto &gp : m_gps) {
        gp.set_log_prior(log_prior);
    }
}

void cmp::multi_gp::set_mean_grad(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> mean_grad)
{
    for (auto &gp : m_gps) {
        gp.set_mean_grad(mean_grad);
    }
}

void cmp::multi_gp::set_kernel_grad(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> kernel_grad)
{
    for (auto &gp : m_gps) {
        gp.set_kernel_grad(kernel_grad);
    }
}

void cmp::multi_gp::set_log_prior_grad(std::function<double(Eigen::VectorXd const &, int i)> log_prior_grad)
{
    for (auto &gp : m_gps) {
        gp.set_log_prior_grad(log_prior_grad);
    }
}

void cmp::multi_gp::set_kernel_hess(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i, int j)> kernel_hess)
{
    for (auto &gp : m_gps) {
        gp.set_kernel_hess(kernel_hess);
    }
}

void cmp::multi_gp::set_log_prior_hess(std::function<double(Eigen::VectorXd const &, int i, int j)> log_prior_hess)
{
    for (auto &gp : m_gps) {
        gp.set_log_prior_hess(log_prior_hess);
    }
}

void cmp::multi_gp::fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const cmp::method &method, const nlopt::algorithm &alg, const double &tol_rel)
{
    // Fit the GPs
    for (auto &gp : m_gps) {
        gp.fit(x0,lb,ub,method,alg,tol_rel);
    }
}

Eigen::VectorXd cmp::multi_gp::predictive_mean(Eigen::VectorXd x) const
{
    Eigen::VectorXd pred(m_dim);
    for (size_t i=0; i<m_dim; i++) {
        pred[i] = m_gps[i].predictive_mean(x);
    }

    // Scale the prediction
    m_y_obs->inverse_transform(pred);
    return pred;
}

Eigen::MatrixXd cmp::multi_gp::predictive_var(Eigen::VectorXd x) const
{
    Eigen::VectorXd pred(m_dim);
    for (size_t i=0; i<m_dim; i++) {
        pred[i] = m_gps[i].predictive_var(x);
    }

    // Scale the prediction
    return m_y_obs->get_scale()*pred.asDiagonal()*m_y_obs->get_scale().transpose();
}


