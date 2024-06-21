#ifndef UTILS_H
#define UTILS_H

#include <cmp_defines.h>

namespace cmp {
    /**
     * @brief Compute the mean of a vector
     * 
     * @param v A vector containing some data
     * @return The mean of the vector 
     */
    double v_mean(const vector_t &v);

    /**
     * @brief Compute the standard deviation of the vector
     * 
     * @param v A vector containing data
     * @return The standard deviation of the data 
     */
    double v_std(const vector_t &v);

    /**
     * @brief Get the column object
     * 
     * @param data The data
     * @param index The index of the colum
     * @return The required colum of the vector 
     */
    std::vector<double> get_column(const std::vector<vector_t> &data, const int &index);

    /**
     * @brief Normalize a data-set (subtract the mean and divide by the std)
     * 
     * @param grid The vector to be normalized
     * @param scale A pair containing the mean and the cholesky decomposition of the covariance used
     * @return a normalized version of the grid
     */
    std::pair<vector_t, Eigen::LLT<matrix_t>> normalize(std::vector<vector_t> &grid);

    /**
     * @brief Normalize a data-set (subtract the mean and divide by the std)
     * 
     * @param grid The vector to be normalized
     * @param scale A pair containing the mean and the cholesky decomposition of the covariance used
     * @return a normalized version of the grid
     */
    std::pair<double, double> normalize(std::vector<double> &grid);

    /**
     * @brief Scale a vector according to its mean and std
     * 
     * @param v A vector
     * @param scale A pair containing the mean and L matrix of the covariance
     */
    void scale(vector_t &v, const std::pair<vector_t, Eigen::LLT<matrix_t>> &scale);

    /**
     * @brief Un-normalize a vector
     * 
     * @param v The vector
     * @param scale The mean and cholesky decomposition of the covariance
     */
    void un_scale(vector_t &v, const std::pair<vector_t, Eigen::LLT<matrix_t>> &scale);

    /**
     * @brief Un-normalize a number
     * 
     * @param v The number
     * @param scale The mean and cov
     */
    void un_scale(double &v, const std::pair<double, double> &scale);


    struct laplace_object {
        double map;
        vector_t arg_map;
        Eigen::LLT<matrix_t> cov_llt;
    };

    laplace_object gaussian_approximation(const score_t &score, const vector_t & par_0, const vector_t &par_lb, const vector_t &par_ub, const double &tol);




    

}


#endif