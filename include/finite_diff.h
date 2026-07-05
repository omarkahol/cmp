#ifndef FINITE_DIFF
#define FINITE_DIFF

#include "cmp_defines.h"

/**
 * @addtogroup core
 * @{
 */
namespace cmp {

/**
 * @brief Accuracy order of the FD stencil
 *
 * Note, the higher the accuracy the higher the number of function evaluations required
 */
enum accuracy {SECOND, FOURTH, SIXTH, EIGHTH};

/**
 * @brief Computes the i-th component of the gradient of the function fun evaluated at x_0 using a central difference scheme.
 * 
 * @details Mathematical Formulation
 * Approximates the partial derivative of \f$f: \mathbb{R}^D \to \mathbb{R}\f$ at \f$\mathbf{x}_0\f$ with respect to the \f$i\f$-th variable using a central difference stencil:
 * \f[
 * \frac{\partial f(\mathbf{x}_0)}{\partial x_i} \approx \frac{f(\mathbf{x}_0 + h \mathbf{e}_i) - f(\mathbf{x}_0 - h \mathbf{e}_i)}{2h}
 * \f]
 * where \f$\mathbf{e}_i\f$ is the \f$i\f$-th standard basis vector, and \f$h > 0\f$ is the step size. For higher-order accuracy (e.g., 4th, 6th, or 8th order), wider stencils are employed.
 * 
 * @details Implementation Algorithm
 * 1. Creates coordinate perturbations \f$\mathbf{x}_0 \pm n h \mathbf{e}_i\f$ for stencil points \f$n\f$.
 * 2. Evaluates the function `fun` at each perturbed point.
 * 3. Takes a linear combination of the function values scaled by the central difference stencil coefficients.
 * 
 * @param x_0 The point at which to evaluate the gradient
 * @param fun The function of which to evaluate the gradient
 * @param i The component of the gradient
 * @param order The accuracy order (default = SECOND)
 * @param h The step size (default = 1E-5)
 * @return The required component of the gradient
 */
double fd_gradient(const Eigen::VectorXd &x_0, const std::function<double(const Eigen::VectorXd&)> fun, const int &i, const accuracy order = SECOND, const double h = 1E-5);


/**
 * @brief Computes the i-th and j-th component of the Hessian of the function fun evaluated at x_0 using a central difference scheme.
 * 
 * @details Mathematical Formulation
 * Approximates the second-order partial derivative \f$\frac{\partial^2 f(\mathbf{x}_0)}{\partial x_i \partial x_j}\f$ at \f$\mathbf{x}_0\f$.
 * For diagonal elements (\f$i = j\f$):
 * \f[
 * \frac{\partial^2 f(\mathbf{x}_0)}{\partial x_i^2} \approx \frac{f(\mathbf{x}_0 + h\mathbf{e}_i) - 2f(\mathbf{x}_0) + f(\mathbf{x}_0 - h\mathbf{e}_i)}{h^2}
 * \f]
 * For off-diagonal elements (\f$i \neq j\f$):
 * \f[
 * \frac{\partial^2 f(\mathbf{x}_0)}{\partial x_i \partial x_j} \approx \frac{f(\mathbf{x}_0 + h\mathbf{e}_i + h\mathbf{e}_j) - f(\mathbf{x}_0 + h\mathbf{e}_i - h\mathbf{e}_j) - f(\mathbf{x}_0 - h\mathbf{e}_i + h\mathbf{e}_j) + f(\mathbf{x}_0 - h\mathbf{e}_i - h\mathbf{e}_j)}{4h^2}
 * \f]
 * 
 * @details Implementation Algorithm
 * 1. Checks if \f$i = j\f$ or \f$i \neq j\f$.
 * 2. Prepares perturbations along standard directions \f$\mathbf{e}_i\f$ and \f$\mathbf{e}_j\f$.
 * 3. Evaluates the function at the perturbed points and computes the stencil difference quotient.
 * 
 * @param x_0 The point at which to evaluate the hessian
 * @param fun The function of which to evaluate the hessian
 * @param i Row of the hessian
 * @param j Column of the hessian
 * @param order The accuracy order (default = SECOND)
 * @param h The step size (default = 1E-5)
 * @return The required component of the Hessian
 */
double fd_hessian(const Eigen::VectorXd &x_0, const std::function<double(const Eigen::VectorXd&)> fun, const int &i, const int &j, const accuracy order = SECOND, const double h = 1E-5);

}

/** @} */

#endif