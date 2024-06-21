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

    typedef std::function<double(const vector_t &par)> score_t;
}

#endif