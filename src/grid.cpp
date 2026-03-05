#include "grid.h"
#include <random>


Eigen::VectorXs cmp::grid::LinspacedGrid::gridElement(const size_t &index, const size_t &n_pts, const size_t &dim) {

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
    Eigen::VectorXs element(dim);
    size_t index_copy = index;
    for(size_t i = 0; i < dim; i++) {

        element[dim - 1 - i] = index_copy % n_pts;          // Take the remainder and save it (step 1 and 2)
        index_copy = (index_copy - element[dim - 1 - i]) / n_pts; //Increase the dimension number and update (step 3)

    }
    return element;
}



