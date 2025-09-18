#include "grid.h"
#include <boost/math/special_functions/prime.hpp>
#include <random>
using namespace cmp;

std::vector<Eigen::VectorXd> cmp::grid::gridUniform(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n) {

    //Get the number of parameters and the number of points of the grid
    int n_par = lowerBound.size();
    int n_points = pow(n, n_par);

    //this contains the parameter vector to be added to the grid
    Eigen::VectorXd par(n_par);

    //Index will be updated to contain numbers from 0 to n-1
    std::vector<int> index(n_par);

    // Generate the vector containing the integration points and fill it
    std::vector<Eigen::VectorXd> grid_points(n_points);
    for (int i = 0; i < n_points; i++) {

        //Generate current indices for the parameters
        index = gridElement(i, n, n_par);
        
        for (int j = 0; j < n_par; j++) {

            // perform a linear mapping to make the point in the interval [lowerBound, upperBound]
            par(j) = lowerBound(j) + (index[j] + 0.5) * (upperBound(j) - lowerBound(j)) / double(n);
        }

        grid_points[i] = par;
    }

    return grid_points;
};

std::vector<int> cmp::grid::gridElement(int index, const int n_pts, const int dim){

    /*
    Visualization grid 2x4 so dim = 2 and n_pts = 4
     
    (0,0) (0,1) (0,2) (0,3)
    (1,0) (1,1) (1,2) (1,3)
    (2,0) (2,1) (2,2) (2,3)
    (3,0) (3,1) (3,2) (3,3)

    Note that we start to fill by the last dimension for convention.

    Element 7 is 
    1) Do     7%4 = 3
    2) Set    (.,3)
    3) Update (7-3)/4 = 1
    ... repeat
    1) Do 1%4=1
    2) Set (1,3)
    3) Update ...
    Stop
    */

    // This is the result container
    std::vector<int> element(dim);
     
    for (size_t i=0; i<dim; i++) {
        
        element[dim-1-i] = index%n_pts;  // Take the remainder and save it (step 1 and 2)
        index = (index-element[dim-1-i])/n_pts; //Increase the dimension number and update (step 3)

    }
    return element;
};

std::vector<Eigen::VectorXd> cmp::grid::gridLHS(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n, std::default_random_engine &rng) {

    // Initialize the data
    size_t dim = lowerBound.size();

    // For each dimension, pick a random permutation of [0,... n-1]
    std::vector<Eigen::VectorXd> perm_1n(dim);
    for (int i=0; i<dim; i++) {
        
        // Create an array containing [0,1,... n-1]
        Eigen::VectorXd array(n);
        for (size_t i=0; i<n; i++)  {
            array(i) = i;
        }
        
        //shuffle it
        std::shuffle(array.begin(),array.end(),rng);
        perm_1n[i] = array;
    }
    
    // Generate the grid points
    std::vector<Eigen::VectorXd> grid_points(n);
    for (int i = 0; i < n; i++) {
        grid_points[i] = Eigen::VectorXd::Zero(dim);
        for (int j = 0; j < dim; j++) {
            // The first part lowerBound(j) + (upperBound(j) - lowerBound(j)) * is just a linear transformation that transforms the interval [0,1]
            // in the desired interval. The second generates the LH in [0,1]
            grid_points[i](j) = lowerBound(j) + (upperBound(j) - lowerBound(j)) * perm_1n[j](i) / double(n);
        }
    }

    return grid_points;
}

Eigen::VectorXd cmp::grid::haltonSequence(int first_el, int length) {

    // Interval bounds 
    int lowerBound = 0;
    int upperBound = 1;

    // Sequence elements
    Eigen::VectorXd sequence(length);
    int x = 0;
    int y = 0;
    for(int i=0; i<length; i++) {
        
        x = upperBound-lowerBound;
        if (x==1) {
            lowerBound = 1;
            upperBound *= first_el;
        } else {
            y = upperBound/first_el;
            while (x<=y){
                y = y/first_el;
            }
            lowerBound = (first_el+1)*y-x;
        }

        sequence(i) = lowerBound/double(upperBound);
    }
    return sequence;
}

std::vector<Eigen::VectorXd> cmp::grid::gridQMC(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n) {
    
    // Dimension
    size_t dim = lowerBound.size();

    // For each dimension, generate a Halton sequence
    std::vector<Eigen::VectorXd> halton_sequences(dim);
    for (int i=0; i<dim; i++) {
        halton_sequences[i] = grid::haltonSequence(boost::math::prime(i),n);
    }
    
    // Generate the grid points
    std::vector<Eigen::VectorXd> grid_points(n);
    for (int i = 0; i < n; i++) {
        grid_points[i] = Eigen::VectorXd::Zero(dim);
        for (int j = 0; j < dim; j++) {
            // The first part lowerBound(j) + (upperBound(j) - lowerBound(j)) * is a linear transformation that transforms the interval [0,1]
            // in the desired interval. The second generates the Halton sequence in [0,1]
            grid_points[i](j) = lowerBound(j) + (upperBound(j) - lowerBound(j)) * halton_sequences[j](i);
        }
    }

    return grid_points;
}

std::vector<Eigen::VectorXd> cmp::grid::gridMonteCarlo(Eigen::VectorXd const &lowerBound, Eigen::VectorXd const &upperBound, int n, std::default_random_engine &rng) {
    
    // Dimension
    size_t dim = lowerBound.size();
    std::uniform_real_distribution<double> dist_u(0,1);

    // Generate the grid points
    std::vector<Eigen::VectorXd> grid_points(n);
    for (int i = 0; i < n; i++) {
        grid_points[i] = Eigen::VectorXd::Zero(dim);
        for (int j = 0; j < dim; j++) {
            // The first part lowerBound(j) + (upperBound(j) - lowerBound(j)) * is a linear transformation that transforms the interval [0,1]
            // in the desired interval. The second generates the Halton sequence in [0,1]
            grid_points[i](j) = lowerBound(j) + (upperBound(j) - lowerBound(j)) * dist_u(rng);
        }
    }

    return grid_points;
}