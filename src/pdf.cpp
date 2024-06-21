#include "pdf.h"

/*
Inverse Gamma
*/
double cmp::log_inv_gamma_pdf(const double &x, const double &a, const double &b) {
    return - (a + 1) * log(x) - b / x;
}
double cmp::d_log_inv_gamma_pdf(const double &x, const double &a, const double &b,const int &i) {
    if (i==0)
        return (b-(a+1)*x)/pow(x,2);
    else
        return 0;
}  
double cmp::dd_log_inv_gamma_pdf(const double &x, const double &a, const double &b,const int &i,const int &j) {
    if (i==0 && j==0)
        return (-2*b+x+x*a)/pow(x,3);
    else 
        return 0;
}

/*
Normal PDFs
*/
double cmp::log_normal_pdf(const double &x, const double &mean, const double &std) {
    return -0.5 * pow((x - mean) / std, 2);
}

double cmp::d_log_normal_pdf(const double &x, const double &mean, const double &std,const int& i){
    if (i==0)
        return -(x-mean)/pow(std,2);
    else 
        return 0;
}

double cmp::dd_log_normal_pdf(const double &x, const double &mean, const double &std,const int& i,const int& j){
    if (i==0 && j==0)
        return -1/pow(std,2);
    else 
        return 0;
}

/*
Log-normal PDFs
*/
double cmp::log_lognormal_pdf(const double &x, const double &mean, const double &std) {
    return -0.5 * pow((log(x) - mean) / std, 2) - log(x*std);
}

double cmp::d_log_lognormal_pdf(const double &x, const double &mean, const double &std,const int& i){
    if (i==0)
        return -(-mean+pow(std,2)+log(x))/(x*pow(std,2));
    else 
        return 0;
}

double cmp::dd_log_lognormal_pdf(const double &x, const double &mean, const double &std,const int& i,const int& j){
    if (i==0 && j==0)
        return (-1-mean+pow(std,2)+log(x))/pow(x*std,2);
    else 
        return 0;
}

/*
Log-power PDFs
*/
double cmp::log_power_pdf(const double &x, const double &n) {
    return -n*log(x);
}

double cmp::d_log_power_pdf(const double &x, const double &n,const int& i) {
    if (i==0)
        return -n/x;
    else 
        return 0;
}

double cmp::dd_log_power_pdf(const double &x, const double &n,const int& i,const int& j) {
    if (i==0 && j==0)
        return n/pow(x,2);
    else 
        return 0;
}