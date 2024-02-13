#ifndef DOE_H
#define DOE_H

#include "io.h"
#include "cmp_defines.h"

namespace cmp {

    /**
     * This class build a grid of points in the parameters space \f$\theta\f$. \n 
     * It is used by the KOH optimization method to retrive a non-normalized version of the integral over the parameter space.
    */
    class doe {

        friend class density;

    public:

        doe() = default;

        /**
        * Build a DoE consisting of a grid of n^d regularly spaced integers (d is the number of parameters).
        * @param lb lower bounds
        * @param ub ubber bounds
        * @param n number of points per parameter
        */
        doe(vector_t const &lb, vector_t const &ub, int n);                  

        /**
        * Return the grid of the parameters
        */
        std::vector<vector_t> get_grid() const { return m_grid; }

        /**
        In the case of only 2 parameters present the flatten grid consists of the following coordinates
        
        \n 
        (0,0), (0,1), (0,2), ... (0,n-1), \n 
        (1,0), (1,1), (1,2), ... (1, n-1), \n 
        ... \n 
        (n-1,0), (n-1,1), ...   (n-1, n-1) \n
        \n 

        The total number of elements is n ^ n_par. A linear transformation will be used to 
        transform the range from [0,n-1] to [lb,ub]. This function returns the element number 
        curr_point of the grid.

        In the case of 2D grid with 5 points per dimension, to compute the coordinate of element numer 17 we should first do 17%5=2 and this gives the 
        first index. To move on to the second idex we subtract the remainder and then divide by n.
        In this case we end up with 15/5 = 3. We perform the same operation as before 3%5 = 3 and this is the 
        second coordinate so the index will be (2,3). This procedure can be extended to any number of dimension.

        @param curr_point current coordinate to be constucted.
        @param n number of points per parameter.
        @param n_par number of parameters.
        */
        vector_t multi_index(int curr_point, const int n, const int n_par);


    protected:
        
        vector_t m_lb;                  ///< parameters lower bounds.
        vector_t m_ub;                  ///< parameters upper bounds.
        std::vector<vector_t> m_grid;   ///< grid of parameters points.
    };

}

#endif