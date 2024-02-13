#include "pdf.h"

double cmp::log_inv_gamma_pdf(const double &x, const double &a, const double &b) {
    return - (a + 1) * log(x) - b / x;
}

double cmp::log_norm_pdf(const double &x, const double &mean, const double &std) {
    return -0.5 * pow((x - mean) / std, 2);
}


double cmp::d_log_inv_gamma_pdf(const double &x, const double &a, const double &b) {
    return (b-(a+1)*x)/pow(x,2);
}

    
double cmp::dd_log_inv_gamma_pdf(const double &x, const double &a, const double &b) {
    return (-2*b+x+x*a)/pow(x,3);
}