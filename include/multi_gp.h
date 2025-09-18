#ifndef MULTI_GP_H
#define MULTI_GP_H

#include "gp.h"

namespace cmp::gp {
    class MultiOutputGaussianProcess {

        private:

            // Base components
            std::vector<GaussianProcess> gps_;
            size_t nGPs_;
            

        public:

            MultiOutputGaussianProcess() = default;

            /**
             * @brief Set the values of the observations
             * 
             * @param x_obs the observation points
             * @param y_obs the observation values
             */
            void setObservations(const std::vector<Eigen::VectorXd> &xObs, const std::vector<Eigen::VectorXd> &yObs) {
                
                nGPs_ = yObs[0].size();
                gps_.resize(nGPs_);

                for (size_t i = 0; i < nGPs_; i++) {
                    gps_[i] = GaussianProcess();
                    gps_[i].setObservations(xObs, get_column(yObs, i));
                }
            }

            /**
             * Set the GP
             * @param kernel The kernel function
             * @param mean The mean function
             * @param prior The prior function
             * 
             */
            void set(const std::shared_ptr<kernel::Kernel> &kernel, const std::shared_ptr<mean::Mean> &mean, const std::shared_ptr<prior::Prior> &prior) {
                for (size_t i = 0; i < nGPs_; i++) {
                    gps_[i].set(kernel, mean, prior);
                }
            }

            void fit(const Eigen::VectorXd &x0, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,  const method &method = MLE, const nlopt::algorithm &alg = nlopt::LN_SBPLX, const double &tol_rel = 1e-3);

            cmp::distribution::MultivariateNormalDistribution predictiveDistribution(const Eigen::VectorXd &x) const;

            Eigen::VectorXd predict(const Eigen::VectorXd &x) const;

            Eigen::MatrixXd predictVariance(const Eigen::VectorXd &x, const int &i) const;

            GaussianProcess &operator[](const int &i) {
                return gps_.at(i);
            }

            size_t size() const {
                return nGPs_;
            }
};


}

#endif