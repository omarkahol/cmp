#ifndef IO_H
#define IO_H

#include <cmp_defines.h>

namespace cmp {

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
std::vector<Eigen::VectorXd> read_vector(std::ifstream &i_file, std::string delimiter = ",", size_t header = 0);

}


#endif