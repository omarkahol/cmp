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
    double v_mean(const Eigen::VectorXd &v);

    /**
     * @brief Compute the standard deviation of the vector
     * 
     * @param v A vector containing data
     * @return The standard deviation of the data 
     */
    double v_std(const Eigen::VectorXd &v);

    /**
     * @brief Get the column object
     * 
     * @param data The data
     * @param index The index of the colum
     * @return The required colum of the vector 
     */
    std::vector<double> get_column(const std::vector<Eigen::VectorXd> &data, const int &index);

    /**
     * @brief Normalize a data-set (subtract the mean and divide by the std)
     * 
     * @param grid The vector to be normalized
     * @param scale A pair containing the mean and the cholesky decomposition of the covariance used
     * @return a normalized version of the grid
     */
    std::pair<Eigen::VectorXd, Eigen::LLT<Eigen::MatrixXd>> normalize(std::vector<Eigen::VectorXd> &grid);

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
    void scale(Eigen::VectorXd &v, const std::pair<Eigen::VectorXd, Eigen::LLT<Eigen::MatrixXd>> &scale);

    void scale(double &v, const std::pair<double, double> &scale);

    /**
     * @brief Un-normalize a vector
     * 
     * @param v The vector
     * @param scale The mean and cholesky decomposition of the covariance
     */
    void un_scale(Eigen::VectorXd &v, const std::pair<Eigen::VectorXd, Eigen::LLT<Eigen::MatrixXd>> &scale);

    /**
     * @brief Un-normalize a number
     * 
     * @param v The number
     * @param scale The mean and cov
     */
    void un_scale(double &v, const std::pair<double, double> &scale);

     /**
     * @brief Convert a matrix into a std::vector<Eigen::VectorXd>
     * 
     * @param data The matrix to convert
     * @return the data in a different format 
     */
    std::vector<Eigen::VectorXd> matrix_to_vvxd(const Eigen::MatrixXd &data);

    /**
    * Convert a std::vector to an Eigen::VectorXd.
    * @param v A c++ std::vector containing doubles
    * @return the same vector converted into an eigen::VectorXd
    */
    Eigen::VectorXd v_to_vxd(const std::vector<double> &v);

    /**
    * Convert an Eigen::VectorXd to an c++ std::vector.
    * @param v An Eigen::VectorXd containing doubles
    * @return the same vector converted into a c++ std::vector.
    */
    std::vector<double> vxd_to_v(const Eigen::VectorXd &v);

    /**
     * @brief Transform a vector of doubles to a vector of scalars (Eigen::vectorXd of dimension 1)
     * 
     * @param v the vector to be transformed
     * @return std::vector<Eigen::VectorXd> The same vector in the requested format
     */
    std::vector<Eigen::VectorXd> v_to_vvxd(const std::vector<double> &v);

    /**
    * Write the contents of a matrix into a file. 
    * @param data A matrix containing the data.
    * @param o_file An open and valid file in which to write the contents.
    */
    void write_vector(const std::vector<Eigen::VectorXd> &data, std::ofstream &o_file);

    /**
     * @brief Write data points to a file
     * 
     * @param x The points (as a vector)
     * @param y The data (as a matrix)
     * @param o_file The output file
     */
    void write_data(const std::vector<Eigen::VectorXd> &x, const Eigen::MatrixXd &y, std::ofstream &o_file);

    /**
    * Read the contents of a file. Note, the file must be organized in rows containing data in the form of numbers.
    * As an example, a valid file is \n 
    * 
    * 0.1 1.2 3.0 \n 
    * 0.2 1.3 3.1 \n 
    * ... \n 
    * 5.4 7.8 9.0 \n 
    * \n 
    * The separator must be a blank space
    * @param i_file An open and valid file stream.
    * @return A std::vector containin
    * 
    * g each row of the file as a Eigen::VectorXd.
    */
    std::vector<Eigen::VectorXd> read_vector(std::ifstream &i_file);




    

}


#endif