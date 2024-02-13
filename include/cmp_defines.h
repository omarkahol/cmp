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
#include <gp++.h>
#include "spdlog/fmt/ostr.h"

namespace cmp {
    typedef Eigen::VectorXd vector_t;
    typedef Eigen::MatrixXd matrix_t;
}

#endif