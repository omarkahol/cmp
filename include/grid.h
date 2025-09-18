#ifndef GRID_H
#define GRID_H

#include <utils.h>
#include "cmp_defines.h"

namespace cmp::grid {

    /**
     * @brief Generate a uniform grid with n point per dimension
     * 
     * @param lowerBound The lower bound of the interval
     * @param upperBound The upper bound of the interval
     * @param n  The number of points per dimension
     * @return std::vector<Eigen::VectorXd> A vector containing all points
     * 
     * @note This function used the function gridElement to compute a standard grid
     * [0,n-1]^d and then it uses a linear transformation to remap it to the desired bounds
     */
    std::vector<Eigen::VectorXd> gridUniform(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n);

    /**
     * @brief Generate a Latin HypercupperBounde grid with n points in total
     * 
     * @param lowerBound The lower bound of the interval
     * @param upperBound The upper bound of the interval
     * @param n  The total number of points
     * @return std::vector<Eigen::VectorXd> A vector containing all points
     */
    std::vector<Eigen::VectorXd> gridLHS(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n, std::default_random_engine &rng);

    /**
     * @brief Computes the element #index of a standard uniform grid. 
     * A standard uniform grid of dimension d and size n is the grid of points [0,n-1]^d.
     * 
     * @param index The index of the element to be constructed 
     * @param n_pts The desired number of points per dimension 
     * @param dim   The dimension of the grid
     * @return std::vector<int>, the standard grid element required
     * 
     * @note As an example we suppose that n_pts = 3 and dim = 2. The standard grid is 
     * (0,0), (0,1), (0,2); (1,0), (1,1), (1,2); (2,0), (2,1), (2,2)
     * So the element 5 is (1,2)
     */
    std::vector<int> gridElement(int index, const int n_pts, const int dim);

    /**
     * @brief Generate a 1-dimensional Halton sequence
     * 
     * @param base Base of the Halton sequence
     * @param n_pts Length of the sequence
     * @return Eigen::VectorXd A vector containing the sequence
     * 
     * @note Based on the pseudo-code in https://en.wikipedia.org/wiki/Halton_sequence
     */
    Eigen::VectorXd haltonSequence(int base, int n_pts);

    /**
     * @brief Generate a grid of points in the interval [lowerBound,upperBound] using QMC sampling based of the Halton sequence
     * 
     * @param lowerBound The lower bound of the hypercupperBounde
     * @param upperBound The upper bound of the hypercupperBounde
     * @param n The dimension of the grid
     * @return std::vector<Eigen::VectorXd> The grid points
     */
    std::vector<Eigen::VectorXd> gridQMC(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n);

    /**
     * @brief generate a grid sampling from a uniform distribution
     * 
     * @param lowerBound The lower bound of the hypercupperBounde
     * @param upperBound The upper bound of the hypercupperBounde
     * @param n The dimension of the grid
     * @param rng A random number generator
     * @return std::vector<Eigen::VectorXd> The grid points
     */
    std::vector<Eigen::VectorXd> gridMonteCarlo(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n, std::default_random_engine &rng);
};


#endif