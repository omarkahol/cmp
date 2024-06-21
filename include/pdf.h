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
    double d_log_inv_gamma_pdf(const double &x, const double &a, const double &b,const int& i=0);

    /**
    * Evaluate the II derivative (wrt x) of the logarithm of the Inverse-Gamma PDF.
    * 
    * @param x A real number.
    * @param a The shape factor
    * @param b the scale factor
    * @return The the II derivative (wrt x) of the logarithm of the inverse gamma distribution evaluated at \p x and using \p a and \p b parameters.
    */
    double dd_log_inv_gamma_pdf(const double &x, const double &a, const double &b,const int& i=0,const int& j=0);

    /**
    * Evaluate the logarithm of the normal pdf.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The logarithm of the normal distribution evaluated at \p x.
    */
    double log_normal_pdf(const double &x, const double &mean, const double &std);

    /**
    * Evaluate the derivative logarithm of the normal pdf wrt x.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The derivative of logarithm of the normal distribution evaluated at \p x.
    */
    double d_log_normal_pdf(const double &x, const double &mean, const double &std,const int& i=0);

    /**
    * Evaluate the II derivative logarithm of the normal pdf wrt \p x.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The II derivative of logarithm of the normal distribution evaluated at \p x wrt at \p x 
    */
    double dd_log_normal_pdf(const double &x, const double &mean, const double &std, const int& i=0, const int& j=0);

    /**
    * Evaluate the logarithm of the lognormal pdf.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The logarithm of the lognormal distribution evaluated at \p x.
    */
    double log_lognormal_pdf(const double &x, const double &mean, const double &std);

    /**
    * Evaluate the derivative of the logarithm of the lognormal pdf wrt x.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The derivative of logarithm of the lognormal distribution evaluated at \p x.
    */
    double d_log_lognormal_pdf(const double &x, const double &mean, const double &std, const int& i=0);

    /**
    * Evaluate the II derivative of the logarithm of the lognormal pdf wrt \p x.
    * 
    * @param x A real number.
    * @param mu The mean \f$\mu\f$ of the pdf.
    * @param std The standard deviation \f$\sigma\f$ of the pdf.
    * @return The II derivative of logarithm of the lognormal distribution evaluated at \p x wrt at \p x 
    */
    double dd_log_lognormal_pdf(const double &x, const double &mean, const double &std, const int& i=0, const int& j=0);

    /**
    * Evaluate the logarithm of a power law pdf, x^-n.
    * 
    * @param x A real number.
    * @param n The exponent.
    * @return The logarithm of the power law evaluated at \p x.
    */
    double log_power_pdf(const double &x, const double &n);

    /**
    * Evaluate the derivative of the logarithm of a power law pdf, x^-n, wrt x.
    * 
    * @param x A real number.
    * @param n The exponent.
    * @return The derivative of the logarithm of the power law evaluated at \p x.
    */
    double d_log_power_pdf(const double &x, const double &n, const int& i=0);

    /**
    * Evaluate the II derivative of the logarithm of a power law pdf, x^-n, wrt x.
    * 
    * @param x A real number.
    * @param n The exponent.
    * @return The II derivative of the logarithm of the power law evaluated at \p x.
    */
    double dd_log_power_pdf(const double &x, const double &n,const int& i=0,const int& j=0);
}
#endif