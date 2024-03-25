#ifndef CMP_DEFINES_H
#define CMP_DEFINES_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <nlopt.h>
#include <nlopt.hpp>
#include "spdlog/fmt/ostr.h"

namespace cmp {
    typedef Eigen::VectorXd vector_t;
    typedef Eigen::MatrixXd matrix_t;

    typedef std::function<double(vector_t const &, vector_t const &)> model_t;
    typedef std::function<double(vector_t const &, vector_t const &, vector_t const &)> kernel_t;
    typedef std::function<double(vector_t const &)> prior_t;

    typedef std::function<double(const vector_t &par, const vector_t &hpar)> score_t;
    typedef std::function<vector_t(const vector_t &par, const vector_t &hpar)> get_hpar_t;
    typedef std::function<bool(const vector_t &par)> in_bounds_t;
}

#endif