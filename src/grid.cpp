#include "grid.h"
#include <boost/math/special_functions/prime.hpp>
#include <random>
using namespace cmp;

std::vector<vector_t> cmp::uniform_grid(vector_t const &lb, vector_t const &ub, int n) {

    //Get the number of parameters and the number of points of the grid
    int n_par = lb.size();
    int n_points = pow(n, n_par);

    //this contains the parameter vector to be added to the grid
    vector_t par(n_par);

    //Index will be updated to contain numbers from 0 to n-1
    std::vector<int> index(n_par);

    // Generate the vector containing the integration points and fill it
    std::vector<vector_t> grid_points(n_points);
    for (int i = 0; i < n_points; i++) {

        //Generate current indices for the parameters
        index = std_grid_element(i, n, n_par);
        
        for (int j = 0; j < n_par; j++) {

            // perform a linear mapping to make the point in the interval [lb, ub]
            par(j) = lb(j) + (index[j] + 0.5) * (ub(j) - lb(j)) / double(n);
        }

        grid_points[i] = par;
    }

    return grid_points;
};

std::vector<int> cmp::std_grid_element(int index, const int n_pts, const int dim){

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

std::vector<vector_t> cmp::lhs_grid(vector_t const &lb, vector_t const &ub, int n, std::default_random_engine &rng) {

    // Initialize the data
    std::uniform_real_distribution<double> u_dist(0, 1);
    size_t dim = lb.size();

    // For each dimension, pick a random permutation of [0,... n-1]
    std::vector<vector_t> perm_1n(dim);
    for (int i=0; i<dim; i++) {
        
        // Create an array containing [0,1,... n-1]
        vector_t array(n);
        for (size_t i=0; i<n; i++)  {
            array(i) = i;
        }
        
        //shuffle it
        std::shuffle(array.begin(),array.end(),rng);
        perm_1n[i] = array;
    }
    
    // Generate the grid points
    std::vector<vector_t> grid_points(n);
    for (int i = 0; i < n; i++) {
        grid_points[i] = vector_t::Zero(dim);
        for (int j = 0; j < dim; j++) {
            // The first part lb(j) + (ub(j) - lb(j)) * is just a linear transformation that transforms the interval [0,1]
            // in the desired interval. The second generates the LH in [0,1]
            grid_points[i](j) = lb(j) + (ub(j) - lb(j)) * perm_1n[j](i) / double(n);
        }
    }

    return grid_points;
}

vector_t cmp::halton_sequence_1d(int first_el, int length) {

    // Interval bounds 
    int lb = 0;
    int ub = 1;

    // Sequence elements
    vector_t sequence(length);
    int x = 0;
    int y = 0;
    for(int i=0; i<length; i++) {
        
        x = ub-lb;
        if (x==1) {
            lb = 1;
            ub *= first_el;
        } else {
            y = ub/first_el;
            while (x<=y){
                y = y/first_el;
            }
            lb = (first_el+1)*y-x;
        }

        sequence(i) = lb/double(ub);
    }
    return sequence;
}

std::vector<vector_t> cmp::qmc_halton_grid(vector_t const &lb, vector_t const &ub, int n) {
    
    // Dimension
    size_t dim = lb.size();

    // For each dimension, generate a Halton sequence
    std::vector<vector_t> halton_sequences(dim);
    for (int i=0; i<dim; i++) {
        halton_sequences[i] = halton_sequence_1d(boost::math::prime(i),n);
    }
    
    // Generate the grid points
    std::vector<vector_t> grid_points(n);
    for (int i = 0; i < n; i++) {
        grid_points[i] = vector_t::Zero(dim);
        for (int j = 0; j < dim; j++) {
            // The first part lb(j) + (ub(j) - lb(j)) * is a linear transformation that transforms the interval [0,1]
            // in the desired interval. The second generates the Halton sequence in [0,1]
            grid_points[i](j) = lb(j) + (ub(j) - lb(j)) * halton_sequences[j](i);
        }
    }

    return grid_points;
}

std::vector<vector_t> cmp::mc_uniform_grid(vector_t const &lb, vector_t const &ub, int n, std::default_random_engine &rng) {
    
    // Dimension
    size_t dim = lb.size();
    std::uniform_real_distribution<double> dist_u(0,1);

    // Generate the grid points
    std::vector<vector_t> grid_points(n);
    for (int i = 0; i < n; i++) {
        grid_points[i] = vector_t::Zero(dim);
        for (int j = 0; j < dim; j++) {
            // The first part lb(j) + (ub(j) - lb(j)) * is a linear transformation that transforms the interval [0,1]
            // in the desired interval. The second generates the Halton sequence in [0,1]
            grid_points[i](j) = lb(j) + (ub(j) - lb(j)) * dist_u(rng);
        }
    }

    return grid_points;
}