#ifndef OPTMIZATION_H
#define OPTIMIZATION_H
#include "cmp_defines.h"

namespace cmp{
    /**
    Optimize a function using a non-gradient based method.
    @param opt_fun function to optimize in the type of double x, void *data \f$ \rightarrow \f$ double \f$ f(x) \f$ 
    @param data_ptr additional data
    @param x0 initial guess
    @param lb lower bounds
    @param ub upper bounds
    @param ftol_rel realtive tolerance
    @param algorithm the algorithm
    
    @return the value of the function at the maximum \n 

    Suggested algorithms : \n
        1. nlopt::LN_SBPLX for a non-gradient based method \n 
        2. nlopt::LD_TNEWTON_PRECOND_RESTART for a grandient based method \n 

    @note If a gradient based method is used, you should define the gradients
    */
    double opt_routine(nlopt::vfunc opt_fun, void *data_ptr, vector_t &x0, const vector_t &lb, const vector_t &ub, double ftol_rel, nlopt::algorithm alg);
}


#endif