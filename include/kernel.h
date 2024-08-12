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
    double squared_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l);

    /**
     * @brief White noise kernel.
     * Evaluates the white noise kernel on two points \p x and \p y 
     * @param x first point.
     * @param y second point.
     * @param s kernel strength, \f$ \sigma \f$.
     * 
     * @return the kernel evaluation \f$ \sigma^2 \delta_{xy}\f$
    */
    double white_noise_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s);

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
    double squared_kernel_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l, const int &i);


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
   double white_noise_kernel_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const int &i);

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
    double squared_kernel_hess(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l, const int &i, const int &j);


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
    double white_noise_kernel_hess(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const int &i, const int &j);


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
    double rational_quadratic_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l, const double &a);

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
    double periodic_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l, const double &p);

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
    double locally_periodic_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l, const double &p);

    
    /**
     * @brief Calculates the squared exponential kernel correlation.
     *
     * This function calculates the squared exponential kernel correlation between two points.
     *
     * @param d The distance between the two points.
     * @param l The length scale parameter.
     * @return The kernel correlation value.
     */
    double se_kernel_corr(const double &d, const double &l);

    /**
     * Calculates the derivative correlation of the squared exponential kernel wrt `l`.
     *
     * This function takes two parameters, `d` and `l`, and calculates the derivative of the correlation
     * of the squared exponential kernel. The `d` parameter represents the distance
     * between two points, and the `l` parameter represents the length scale of the kernel.
     *
     * @param d The distance between two points.
     * @param l The length scale of the kernel.
     * @return The derivative correlation of the squared exponential kernel wrt `l`.
     */
    double d_se_kernel_corr(const double &d, const double &l);



    /**
     * Calculates the second derivative correlation of the squared exponential kernel wrt `l`.
     *
     * This function takes two parameters, `d` and `l`, and calculates the second derivative of the correlation
     * of the squared exponential kernel. The `d` parameter represents the distance
     * between two points, and the `l` parameter represents the length scale of the kernel.
     *
     * @param d The distance between two points.
     * @param l The length scale of the kernel.
     * @return The second derivative correlation of the squared exponential kernel wrt `l`.
     */
    double dd_se_kernel_corr(const double &d, const double &l);

    
    /**
     * Calculates the Matérn 1/2 kernel value between two vectors.
     *
     * @param x The first vector.
     * @param y The second vector.
     * @param s The scaling parameter.
     * @param l The length scale parameter.
     * @return The Matérn 1/2 kernel value between x and y.
     */
    double matern_12_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l);


    /**
     * Calculates the Matérn 3/2 kernel value between two vectors.
     *
     * @param x The first vector.
     * @param y The second vector.
     * @param s The scaling parameter.
     * @param l The length scale parameter.
     * @return The Matérn 3/2 kernel value between x and y.
     */
    double matern_32_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l);


    /**
     * Calculates the Matérn 5/2 kernel value between two vectors.
     *
     * @param x The first vector.
     * @param y The second vector.
     * @param s The kernel scale parameter.
     * @param l The kernel length scale parameter.
     * @return The Matérn 5/2 kernel value between x and y.
     */
    double matern_52_kernel(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double &s, const double &l);


    /**
     * Calculates the Matérn 1/2 correlation value for a given distance and length scale.
     *
     * @param d The distance between two points.
     * @param l The length scale parameter.
     * @return The Matérn 1/2 correlation value.
     */
    double matern_12_corr(const double &d, const double &l);


    /**
     * Calculates the Matérn 3/2 correlation value for a given distance and length scale.
     *
     * @param d The distance between two points.
     * @param l The length scale parameter.
     * @return The Matérn 3/2 correlation value.
     */
    double matern_32_corr(const double &d, const double &l);


    /**
     * Calculates the Matérn 5/2 correlation value for a given distance and length scale.
     *
     * @param d The distance between two points.
     * @param l The length scale parameter.
     * @return The Matérn 5/2 correlation value.
     */
    double matern_52_corr(const double &d, const double &l);
}

#endif