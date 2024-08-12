#include <finite_diff.h>


namespace cmp {
    

    std::vector<double> external_stencil(const accuracy &accuracy)
    {
        switch (accuracy) {
            case SECOND:
                return { { 1, -1 } };
            case FOURTH:
                return { { 1, -8, 8, -1 } };
            case SIXTH:
                return { { -1, 9, -45, 45, -9, 1 } };
            case EIGHTH:
                return { { 3, -32, 168, -672, 672, -168, 32, -3 } };
            default:
                spdlog::error("Invalid accuracy order");
            }
    }

    std::vector<double> internal_stencil(const accuracy &accuracy)
    {
        switch (accuracy) {
            case SECOND:
                return { { 1, -1 } };
            case FOURTH:
                return { { -2, -1, 1, 2 } };
            case SIXTH:
                return { { -3, -2, -1, 1, 2, 3 } };
            case EIGHTH:
                return { { -4, -3, -2, -1, 1, 2, 3, 4 } };
            default:
                spdlog::error("Invalid accuracy order. Using order 2");
                return { { 1, -1 } };
            }
    }

    double denominator(const accuracy &accuracy){
        switch (accuracy) {
            case SECOND:
                return 2;
            case FOURTH:
                return 12;
            case SIXTH:
                return 60;
            case EIGHTH:
                return 840;
            default:
                spdlog::error("Invalid accuracy order. Using order 2");
                return 2;
        }
    }


    double fd_gradient(const Eigen::VectorXd &x_0, const std::function<double(const Eigen::VectorXd&)> fun, const int &i, const accuracy accuracy, const double h) {
        
        // Check on the components
        if (i<0 || i>x_0.size()-1) {
            return 0.0;
        }

        const std::vector<double> external_coeffs = external_stencil(accuracy);
        const std::vector<double> internal_coeffs = internal_stencil(accuracy);
        
        double denom = denominator(accuracy)*h;
        denom *= denom;

        const int n_steps = external_coeffs.size();


        Eigen::VectorXd x_step = x_0;
        double grad = 0.0;

        for(int l = 0; l<n_steps; l++) {
            x_step[i] += internal_coeffs[l] * h;
            grad += external_coeffs[l] * fun(x_step);
            x_step[i] = x_0[i];
        }

        return grad/denom;
    }


    double fd_hessian(const Eigen::VectorXd &x_0, const std::function<double(const Eigen::VectorXd&)> fun, const int &i, const int &j, const accuracy accuracy, const double h) {
        
        // Check on the components
        if (i<0 || i>x_0.size()-1) {
            return 0.0;
        }
        if (j<0 || j>x_0.size()-1) {
            return 0.0;
        }
        
        const std::vector<double> external_coeffs = external_stencil(accuracy);
        const std::vector<double> internal_coeffs = internal_stencil(accuracy);
        double denom = denominator(accuracy)*h;
        denom *= denom;

        const int n_steps = external_coeffs.size();


        Eigen::VectorXd x_step = x_0;
        double hess = 0.0;

        for(int l = 0; l<n_steps; l++) {
            for (int k=0; k<n_steps; k++) {
                
                x_step[l] += internal_coeffs[l] * h;
                x_step[k] += internal_coeffs[k] * h;

                hess += external_coeffs[l] * external_coeffs[k] * fun(x_step);
                
                x_step[k] = x_0[k];
                x_step[l] = x_0[l];
            }
        }

        return hess/denom;
    }




}