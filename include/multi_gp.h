#ifndef MULTI_GP_H
#define MULTI_GP_H

#include "gp.h"

namespace cmp {
    class multi_gp {

    private:

        // Base components
        std::vector<gp> m_gps;
        std::vector<cmp::component_scaler> m_scalers;

        // Observations
        vector_scaler *m_y_obs;
        vector_scaler *m_x_obs;
        size_t m_dim;

    public:

        multi_gp(size_t dim) : m_dim(dim) {
            m_gps = std::vector<gp>(dim);
            m_scalers = std::vector<cmp::component_scaler>(dim);
        }

        /**
         * @brief Set the values of the observations
         * 
         * @param x_obs the observation points
         * @param y_obs the observation values
         */
        void set_observations(cmp::vector_scaler *x_obs, cmp::vector_scaler *y_obs);

        void set_kernel(kernel_t kernel);

        void set_mean(model_t mean);

        void set_log_prior(prior_t log_prior);

        void set_mean_grad(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> mean_grad);

        void set_kernel_grad(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> kernel_grad);

        void set_log_prior_grad(std::function<double(Eigen::VectorXd const &, int i)> log_prior_grad);

        void set_kernel_hess(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i, int j)> kernel_hess);

        void set_log_prior_hess(std::function<double(Eigen::VectorXd const &, int i, int j)> log_prior_hess);

        void fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,  const cmp::method &method = cmp::MLE, const nlopt::algorithm &alg = nlopt::LN_SBPLX, const double &tol_rel = 1e-3);

        Eigen::VectorXd predictive_mean(Eigen::VectorXd x) const;

        Eigen::MatrixXd predictive_var(Eigen::VectorXd x) const;

        gp &operator[](const int &i) {
            return m_gps.at(i);
        }
    
};


}

#endif