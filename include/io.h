/**
 * Input-output routines
*/


#ifndef IO_H
#define IO_H

#include "cmp_defines.h"

namespace cmp {
    /**
    * Convert a std::vector to an Eigen::VectorXd.
    * @param v A c++ std::vector containg doubles
    * @return the same vector connverted into an eigen::VectorXd
    */
    vector_t v_to_vxd(const std::vector<double> &v);

    /**
    * Convert an Eigen::VectorXd to an c++ std::vector.
    * @param v An Eigen::VectorXd containg doubles
    * @return the same vector connverted into a c++ std::vector.
    */
    std::vector<double> vxd_to_v(const vector_t &v);

    /**
     * @brief Transform a vector of doubles to a vector of scalars (Eigen::vectorXd of dimension 1)
     * 
     * @param v the vector to be transformed
     * @return std::vector<vector_t> The same vector in the requested format
     */
    std::vector<vector_t> v_to_vvxd(const std::vector<double> &v);

    /**
    * Write the contents of a matrix into a file. 
    * @param data A matrix containing the data.
    * @param o_file An open and valid file in which to write the contents.
    */
    void write_vector(const std::vector<vector_t> &data, std::ofstream &o_file);

    /**
     * @brief Write data points to a file
     * 
     * @param x The points (as a vector)
     * @param y The data (as a matrix)
     * @param o_file The output file
     */
    void write_data(const std::vector<vector_t> &x, const matrix_t &y, std::ofstream &o_file);

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
    * @return A std::vector containg each row of the file as a Eigen::VectorXd.
    */
    std::vector<vector_t> read_vector(std::ifstream &i_file);
}

#endif