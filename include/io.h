#ifndef IO_H
#define IO_H

#include <cmp_defines.h>

/**
 * @addtogroup core
 * @{
 */
namespace cmp {

/**
 * @brief Serializes a list of vector coordinates to a file stream.
 * 
 * @details Mathematical Formulation
 * Serializes a set of vectors \f$\{\mathbf{x}_i\}_{i=1}^M\f$ where \f$\mathbf{x}_i \in \mathbb{R}^N\f$ into a structured grid representation in text format:
 * \f[
 * \mathbf{x}_i^T = [x_{i,1}, x_{i,2}, \dots, x_{i,N}]
 * \f]
 * 
 * @details Implementation Algorithm
 * Streams each vector to the output file stream `o_file` line-by-line, separating coordinates with spaces.
 * 
 * @param data A matrix containing the data.
 * @param o_file An open and valid file in which to write the contents.
 */
void write_vector(const std::vector<Eigen::VectorXd> &data, std::ofstream &o_file);

/**
 * @brief Write data points and evaluations to a file.
 * 
 * @details Mathematical Formulation
 * Serializes input vectors \f$\mathbf{x}_i \in \mathbb{R}^D\f$ along with corresponding multivariate observations \f$\mathbf{y}_i \in \mathbb{R}^Q\f$:
 * \f[
 * \text{Row}_i = [x_{i,1}, \dots, x_{i,D}, y_{i,1}, \dots, y_{i,Q}]
 * \f]
 * 
 * @details Implementation Algorithm
 * Loops over the vector entries, streaming coordinates of \f$\mathbf{x}_i\f$ followed by the corresponding row coordinates of the matrix \f$\mathbf{y}_i\f$ into `o_file`.
 *
 * @param x The points (as a vector)
 * @param y The data (as a matrix)
 * @param o_file The output file
 */
void write_data(const std::vector<Eigen::VectorXd> &x, const Eigen::MatrixXd &y, std::ofstream &o_file);

/**
 * @brief Read the contents of a delimited text file into vector collections.
 * 
 * @details Mathematical Formulation
 * Parses text lines representing coordinate records to rebuild the set \f$\{\mathbf{x}_i\}_{i=1}^M \subset \mathbb{R}^N\f$.
 * 
 * @details Implementation Algorithm
 * 1. Skips the specified number of `header` rows.
 * 2. Parses each line using `std::getline`, splitting coordinates based on the delimiter.
 * 3. Converts string tokens to `double` using standard converters, aggregates them into `Eigen::VectorXd` rows, and returns them as a standard vector.
 * 
 * @param i_file An open and valid file stream.
 * @param delimiter The delimiter string separating columns (default is ",")
 * @param header The number of header rows to skip (default is 0)
 * @return A std::vector containing each row of the file as a Eigen::VectorXd.
 */
std::vector<Eigen::VectorXd> read_vector(std::ifstream &i_file, std::string delimiter = ",", size_t header = 0);

}


/** @} */

#endif