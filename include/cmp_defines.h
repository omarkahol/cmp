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
    
    typedef std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &)> model_t;
    typedef std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)> kernel_t;
    typedef std::function<double(Eigen::VectorXd const &)> prior_t;
    typedef std::function<double(Eigen::VectorXd const &)> score_t;
}

#endif