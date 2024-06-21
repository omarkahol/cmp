#include "utils.h"
#include <io.h>
#include <optimization.h>
#include <finite_diff.h>

using namespace cmp;

double cmp::v_mean(const vector_t &v) {
    int n = v.size();
    double mean=0.0;
    for(int i=0; i<n; i++) {
        mean += v(i);
    }
    return mean/double(n);
}

double cmp::v_std(const vector_t &v) {
    int n = v.size();
    double mean=v_mean(v);
    double std = 0;
    for(int i=0; i<n; i++) {
        std += pow(v(i)-mean,2);
    }
    return sqrt(std/double(n));
}

std::vector<double> cmp::get_column(const std::vector<vector_t> &data, const int &index) {
    int rows = data.size();
    std::vector<double> my_col(rows);

    for (int i=0; i<rows; i++) {
        my_col[i] = data[i](index);
    }

    return my_col;
}

std::pair<vector_t, Eigen::LLT<matrix_t>> cmp::normalize(std::vector<vector_t> &grid) {
    
    int rows = grid.size();
    int cols = grid[0].size();

    vector_t mean = vector_t::Zero(cols);
    matrix_t cov = matrix_t::Zero(cols,cols);

    for (int i=0; i<rows; i++) {
        mean += grid[i];
        cov += grid[i]*grid[i].transpose();
    }

    // Rescale to compute the actual mean and covariance
    mean = mean / double(rows);
    cov = (cov / double(rows)) - mean*mean.transpose();

    // compute the Cholesky decomposition
    Eigen::LLT<matrix_t> cov_llt(cov);
    auto L = cov_llt.matrixL();

    for (int i=0; i<rows; i++){
        grid[i] = L.solve(grid[i] - mean);
    }

    return std::make_pair(mean,cov_llt);
}

std::pair<double, double> cmp::normalize(std::vector<double> &grid) {
    
    int rows = grid.size();

    double mean = 0.0;
    double cov = 0.0;

    for (int i=0; i<rows; i++) {
        mean += grid[i];
        cov += grid[i]*grid[i];
    }

    // Rescale to compute the actual mean and covariance
    mean = mean / double(rows);
    cov = (cov / double(rows - 1) - mean*mean);

    double std = sqrt(cov);

    for (int i=0; i<rows; i++){
        grid[i] = (grid[i] - mean)/std;
    }

    return std::make_pair(mean,std);
}

void cmp::scale(vector_t &v, const std::pair<vector_t, Eigen::LLT<matrix_t>> &scale) {
    v = scale.second.matrixL().solve(v-scale.first);
}

void cmp::un_scale(vector_t &v, const std::pair<vector_t, Eigen::LLT<matrix_t>> &scale) {
    v = scale.second.matrixL()*v + scale.first;
}

void cmp::un_scale(double &v, const std::pair<double, double> &scale) {
    v = scale.second*v + scale.first;
}

laplace_object cmp::gaussian_approximation(const score_t &score, const vector_t & par_0, const vector_t &par_lb, const vector_t &par_ub, const double &tol) {

        vector_t par_opt = par_0;

        std::pair<const score_t &, void *> my_pair(score,nullptr);
        void *p_data = (void *) &my_pair;
        
        auto my_fun = [](const std::vector<double> &x, std::vector<double> &grad, void *data)  {
            std::pair<const score_t &, void *> *my_pair = (std::pair<const score_t &, void *>*)data;
            vector_t x_v = v_to_vxd(x);
            return my_pair->first(x_v);
        };

        // Perform the optimization
        double fmap = cmp::opt_routine(my_fun,p_data,par_opt,par_lb,par_ub,tol,nlopt::LN_SBPLX);

        // Return this value as a laplace object
        laplace_object return_value;
        return_value.map = fmap;
        return_value.arg_map = par_opt;

        // Compute the Hessian using finite difference
        int dim_par = par_0.size();
        matrix_t cov(dim_par,dim_par);


        // Evaluate the hessian
        for(int i=0; i<dim_par; i++) {
            for (int j=i; j<dim_par; j++) {
                cov(i,j) = -fd_hessian(par_opt,score,i,j);

                if (i != j) {
                    cov(j,i) = cov(i,j);
                }
            }
        }

        // Compute the LDLT decomposition
        return_value.cov_llt = Eigen::LLT<matrix_t>(cov);

        return return_value;
}