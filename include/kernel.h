/**
 * Contains the definition of the most commonly used kernel functions along with their derivatives with respect to parameters.
*/


#ifndef KERNEL_H
#define KERNEL_H

#include "cmp_defines.h"

namespace cmp {
    
    /**
     * @brief squared exponential kernel.
     * Evaluates the squared exponential kernel on two points \p x and \p y 
     * @param x first point.
     * @param y second point.
     * @param s kernel strength, \f$ \sigma \f$.
     * @param l the length scale.
     * 
     * @return the kernel evaluation \f$ \sigma^2 \exp(-\frac{1}{2}(\frac{x-y}{l})^2)\f$
    */
    double squared_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l);

    /**
     * @brief White noise kernel.
     * Evaluates the white noise kernel on two points \p x and \p y 
     * @param x first point.
     * @param y second point.
     * @param s kernel strength, \f$ \sigma \f$.
     * 
     * @return the kernel evaluation \f$ \sigma^2 \delta_{xy}\f$
    */
    double white_noise_kernel(const vector_t &x, const vector_t &y, const double &s);

   /**
     * @brief squared exponential kernel gradient.
     * Evaluates the gradient of the squared exponential kernel on two points \p x and \p y 
     * @param x first point.
     * @param y second point.
     * @param s kernel strength, \f$ \sigma \f$.
     * @param l the length scale.
     * @param i component of the gradient (0 for s and 1 for l)
     * 
     * @return the kernel derivative evaluated at x and y
    */
    double squared_kernel_grad(const vector_t &x, const vector_t &y, const double &s, const double &l, const int &i);


    /**
     * @brief White noise kernel gradient.
     * Evaluates the gradient of white noise kernel on two points \p x and \p y 
     * @param x first point.
     * @param y second point.
     * @param s kernel strength, \f$ \sigma \f$.
     * @param i the component of the gradient
     * 
     * @return the kernel derivative evaluated at x and y
    */
   double white_noise_kernel_grad(const vector_t &x, const vector_t &y, const double &s, const int &i);

   /**
     * @brief squared exponential kernel hessian.
     * Evaluates the gradient of the squared exponential kernel on two points \p x and \p y 
     * @param x first point.
     * @param y second point.
     * @param s kernel strength, \f$ \sigma \f$.
     * @param l the length scale.
     * @param i row of the hessian(0 for sigma and 1 for l)
     * @param j colum of the hessian (0 for sigma and 1 for l)
     * 
     * @return the kernel hessian evaluated at x and y
    */
    double squared_kernel_hess(const vector_t &x, const vector_t &y, const double &s, const double &l, const int &i, const int &j);


   /**
    * @brief White noise kernel gradient.
    * Evaluates the gradient of white noise kernel on two points \p x and \p y
    * @param x first point.
    * @param y second point.
    * @param s kernel strength, \f$ \sigma \f$.
    * @param i row of the hessian
    * @param j colum of the hessian
    *
    *
    * @return the kernel hessian evaluated at x and y
    */
    double white_noise_kernel_hess(const vector_t &x, const vector_t &y, const double &s, const int &i, const int &j);


    /**
     * @brief Rational Quadratic kernel
     * Evaluates the rational quadratic kernel at two points \p x and \p y
     * 
     * @param x first point
     * @param y second point
     * @param s kernel strength
     * @param l correlation length
     * @param a length scale variation
     * 
     * @return the rq kernel evaluation 
     */
    double rational_quadratic_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l, const double &a);

    /**
     * @brief Periodic kernel
     * Evaluates the periodic kernel at two points \p x and \p y
     * 
     * @param x first point
     * @param y second point
     * @param s kernel strength
     * @param l correlation length
     * @param p period
     * 
     * @return the periodic kernel evaluation 
     */
    double periodic_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l, const double &p);

    /**
     * @brief Locally periodic kernel
     * Evaluates the locally periodic kernel at two points \p x and \p y
     * 
     * @param x first point
     * @param y second point
     * @param s kernel strength
     * @param l correlation length
     * @param a length scale variation
     * 
     * @return the locally periodic kernel evaluation 
     */
    double locally_periodic_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l, const double &p);


    double matern_12_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l);
    double matern_32_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l);
    double matern_52_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l);
}

#endif