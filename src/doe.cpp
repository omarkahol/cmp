#include "doe.h"

using namespace cmp;

doe::doe(vector_t const &lb, vector_t const &ub, int n) : m_lb(lb), m_ub(ub) {

    //Get the number of parameters and the number of points of the grid
    int n_par = lb.size();
    int n_points = pow(n, n_par);

    //Theta contains the parameter vector to be added to the grid
    vector_t par(n_par);

    //Index will be updated to contain numbers from 0 to n-1
    vector_t index(n_par);

    for (int i = 0; i < n_points; i++) {

        //Generate curent indeces for the parameters
        index = multi_index(i, n, n_par);
        
        for (int j = 0; j < n_par; j++) {

            // perform a linear mapping to make the point in the interval [lb, ub]
            par(j) = m_lb(j) + (index(j) + 0.5) * (m_ub(j) - m_lb(j)) / double(n);
        }

        m_grid.push_back(par);
    }
};

vector_t doe::multi_index(int curr_point, const int n, int n_par) {

    //Create a vector for the indices of each parameter
    vector_t index(n_par);

    int index_i;           //The i-th index of point number curr_point
    
    /*
    Perform the index computation, view documentation for detail
    */
    for (int pp = n_par - 1; pp > -1; pp--) {

        index_i = curr_point % n; //Compute the remainder
        index(pp) = index_i;

        curr_point = (curr_point - index_i) / n; //Update current point variable
    }
    return index;
};