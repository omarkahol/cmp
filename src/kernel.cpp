#include "kernel.h"
const double TOL = 1e-5;
using namespace cmp;

/*
Squared kernel - factor
*/

double cmp::se_kernel_corr(const double &d, const double &l) {
    return exp(-0.5*pow(d/l,2));
}


double cmp::d_se_kernel_corr(const double &d, const double &l){
    return se_kernel_corr(d,l)*pow(d,2)/pow(l,3);
}


double cmp::dd_se_kernel_corr(const double &d, const double &l){
    return (se_kernel_corr(d,l)*pow(d,2)/pow(l,4))*(-3+pow(d,2)/pow(l,3));
}

double cmp::squared_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l) {
    double d = (x-y).norm();
    return pow(s,2)*se_kernel_corr(d,l);
}

/*
Squared kernel - one-dimensional
*/

double cmp::squared_kernel_grad(const vector_t &x, const vector_t &y, const double &s, const double &l, const int &i) {
    double d = (x-y).norm();
    if (i==0) {
        return 2*s*se_kernel_corr(d,l);
    } else if (i == 1) {
        return pow(s,2)*d_se_kernel_corr(d,l);
    } else {
        return 0;
    }
}

double cmp::squared_kernel_hess(const vector_t &x, const vector_t &y, const double &s, const double &l, const int &i, const int &j) {
    double d = (x-y).norm();
    if (i==0 && j==0) {
        return 2*se_kernel_corr(d,l);

    } else if (i==1 && j==1) {
        return pow(s,2)*dd_se_kernel_corr(d,l);
    
    } else if((i==1 && j==0) || (i==0 && j==1)) {
        return 2*s*d_se_kernel_corr(d,l);

    } else {
        return 0;
    }
}

/*
White Noise - kernel
*/

double cmp::white_noise_kernel(const vector_t &x, const vector_t &y, const double &s) {

    if ((x-y).norm() < TOL) {
        return s*s;
    } else {
        return 0;
    }
}

double cmp::white_noise_kernel_grad(const vector_t &x, const vector_t &y, const double &s, const int &i) {
    if (i==0) {
        if ((x-y).norm() < TOL) {
            return s*2;
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}

double cmp::white_noise_kernel_hess(const vector_t &x, const vector_t &y, const double &s, const int&i, const int &j) {
    
    if (i==0 && j==0) {
        if ((x-y).norm() < TOL) {
            return 2.;
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}

/*
 Kernel for which the gradients are not defined
*/

double cmp::rational_quadratic_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l, const double &a) {
    return s*s*pow(1+0.5*pow((x-y).norm()/l,2)/a,-a);
}


double periodic_corr(double d, double l, double p) {
    return exp(-2*pow(sin(M_PI*d/p)/l,2));
}

double cmp::periodic_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l, const double &p) {
    return s*s*periodic_corr((x-y).norm(),l,p);
}

double cmp::locally_periodic_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l, const double &p) {
    double d = (x-y).norm();
    return s*s*periodic_corr(d,l,p)*se_kernel_corr(d,l);
}


double cmp::matern_12_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l) {
    double d = (x-y).norm();
    return s*s*matern_12_corr(d,l);
}
double cmp::matern_32_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l) {
    double d = (x-y).norm();
    return s*s*matern_32_corr(d,l);
}
double cmp::matern_52_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l) {
    double d = (x-y).norm();
    return s*s*matern_52_corr(d,l);
}

double cmp::matern_12_corr(const double &d, const double &l) {
    return exp(-d/l);
}

double cmp::matern_32_corr(const double &d,const double &l) {
    double c_1 = sqrt(3)*d/l;
    return (1+c_1)*exp(-c_1);
}

double cmp::matern_52_corr(const double &d,const double &l) {
    double c_1 = sqrt(5)*d/l;
    double c_2 = (5/3)*pow(d/l,2);
    return (1+c_1+c_2)*exp(-c_1);
}