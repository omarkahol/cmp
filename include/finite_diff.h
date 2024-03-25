#ifndef FINITE_DIFF
#define FINITE_DIFF

#include "cmp_defines.h"

namespace cmp {

    /**
     * @brief Accuracy order of the FD stencil
     * 
     * Note, the higher the accuracy the higher the number of function evaluations required
     */
    enum accuracy {SECOND, FOURTH, SIXTH, EIGHTH};

    /**
     * @brief Compte the \p i-th component of the gradient of the function \p fun evaluated at \p x_0 using a central difference scheme.
     * 
     * @param x_0 The point at which to evaluate the gradient  
     * @param fun The function of which to evaluate the gradient 
     * @param i The component of the gradient
     * @param order The accuracy order (default = 2)
     * @param h The step size (default = 1E-5)
     * @return The required component
     */
    double fd_gradient(const vector_t &x_0, const std::function<double(const vector_t&)> fun, const int &i, const accuracy order = SECOND, const double h=1E-5);


    /**
     * @brief Compte the \p i-th and \p j-th component of the hessian of the function \p fun evaluated at \p x_0 using a central difference scheme.
     * 
     * @param x_0 The point at which to evaluate the hessian  
     * @param fun The function of which to evaluate the hessian 
     * @param i Row of the hessian
     * @param j Colum of the hessian
     * @param order The accuracy order (default = 2)
     * @param h The step size (default = 1E-5)
     * @return The required component
     */
    double fd_hessian(const vector_t &x_0, const std::function<double(const vector_t&)> fun, const int &i, const int &j, const accuracy order = SECOND, const double h=1E-5);

}

#endif