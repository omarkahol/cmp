/**
 * Contains the definition of the most commonly used pdf functions along with their derivatives with respect to parameters.
*/

#ifndef PDF_H
#define PDF_H 

#include "cmp_defines.h"

namespace cmp {
    
    /**
    * Evaluate the logarithm of the Inverse-Gamma PDF.
    * 
    * @param x A real number.
    * @param a The shape factor
    * @param b the scale factor
    * @return The logarithm of the inverse gamma distribution evaluated at \p x and using \p a and \p b parameters.
    */
    double log_inv_gamma_pdf(const double &x, const double &a, const double &b);

    /**
    * Evaluate the derivative (wrt x) of the logarithm of the Inverse-Gamma PDF.
    * 
    * @param x A real number.
    * @param a The shape factor
    * @param b the scale factor
    * @return The the derivative (wrt x) of the logarithm of the inverse gamma distribution evaluated at \p x and using \p a and \p b parameters.
    */
    double d_log_inv_gamma_pdf(const double &x, const double &a, const double &b);

    /**
    * Evaluate the II derivative (wrt x) of the logarithm of the Inverse-Gamma PDF.
    * 
    * @param x A real number.
    * @param a The shape factor
    * @param b the scale factor
    * @return The the II derivative (wrt x) of the logarithm of the inverse gamma distribution evaluated at \p x and using \p a and \p b parameters.
    */
    double dd_log_inv_gamma_pdf(const double &x, const double &a, const double &b);

    /**
    * Evaluate the logarithm of the normal pdf.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The logarithm of the normal distribution evaluated at \p x.
    */
    double log_norm_pdf(const double &x, const double &mean, const double &std);
}
#endif