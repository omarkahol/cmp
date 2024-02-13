#include "kernel.h"
const double TOL = 1e-5;

double cmp::squared_kernel(const vector_t &x, const vector_t &y, const double &s, const double &l) {
    return pow(s,2)*exp(-0.5*pow((x-y).norm()/l,2));
}

double cmp::white_noise_kernel(const vector_t &x, const vector_t &y, const double &s) {

    if ((x-y).norm() < TOL) {
        return s*s;
    } else {
        return 0;
    }
}

double cmp::squared_kernel_grad(const vector_t &x, const vector_t &y, const double &s, const double &l, const int &i) {
    if (i==0) {
        return 2*squared_kernel(x,y,s,l)/s;
    } else if (i == 1) {
        return squared_kernel(x,y,s,l)*pow((x-y).norm(),2)/pow(l,3);
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

double cmp::squared_kernel_hess(const vector_t &x, const vector_t &y, const double &s, const double &l, const int &i, const int &j) {

    if (i==0 && j==0) {
        return 2*squared_kernel(x,y,s,l)/pow(s,2);

    } else if (i==1 && j==1) {
        return squared_kernel(x,y,s,l)*pow((x-y).norm(),2)* (  pow((x-y).norm(),2) -3*l*l   )/pow(l,6);
    
    } else if((i==1 && j==0) || (i==0 && j==1)) {
        return 2*squared_kernel(x,y,s,l)*pow((x-y).norm(),2)/(s*pow(l,3));

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
