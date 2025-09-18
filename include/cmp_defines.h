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
#include <nlopt.hpp>
#include <functional>

namespace cmp {
    /**
     * @brief The score_t type
     * 
     */
    using score_t = std::function<double(const Eigen::VectorXd &)>;

}

#endif // CMP_DEFINES_H