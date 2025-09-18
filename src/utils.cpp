#include "utils.h"
#include <optimization.h>
#include <finite_diff.h>

using namespace cmp;

double cmp::v_mean(const Eigen::VectorXd &v) {
    int n = v.size();
    double mean=0.0;
    for(int i=0; i<n; i++) {
        mean += v(i);
    }
    return mean/double(n);
}

double cmp::v_std(const Eigen::VectorXd &v) {
    int n = v.size();
    double mean=v_mean(v);
    double std = 0;
    for(int i=0; i<n; i++) {
        std += pow(v(i)-mean,2);
    }
    return sqrt(std/double(n));
}

std::vector<double> cmp::get_column(const std::vector<Eigen::VectorXd> &data, const int &index) {
    int rows = data.size();
    std::vector<double> my_col(rows);

    for (int i=0; i<rows; i++) {
        my_col[i] = data[i](index);
    }

    return my_col;
}

std::pair<Eigen::VectorXd, Eigen::LLT<Eigen::MatrixXd>> cmp::normalize(std::vector<Eigen::VectorXd> &grid) {
    
    int rows = grid.size();
    int cols = grid[0].size();

    // Compute the mean
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(cols);
    for (int i=0; i<rows; i++) {
        mean += grid[i];
    }
    mean = mean / double(rows);

    // Compute the covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(cols,cols);
    for (size_t i=0; i<rows; i++){
        cov += (grid[i]-mean)*(grid[i]-mean).transpose();
    }
    cov = cov / double(rows-1);

    // compute the Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> cov_llt(cov);
    auto L = cov_llt.matrixL();

    for (int i=0; i<rows; i++){
        grid[i] = L.solve(grid[i] - mean);
    }

    return std::make_pair(mean,cov_llt);
}

std::pair<double, double> cmp::normalize(std::vector<double> &grid) {
    
    
    int rows = grid.size();

    double mean = 0.0;
    for (int i=0; i<rows; i++) {
        mean += grid[i];
    }
    mean = mean / double(rows);

    double var = 0.0;
    for (int i=0; i<rows; i++){
        var += pow(grid[i]-mean,2);
    }
    var = var / double(rows-1);

    double std = sqrt(var);

    for (int i=0; i<rows; i++){
        grid[i] = (grid[i] - mean)/std;
    }

    return std::make_pair(mean,std);
}

void cmp::scale(Eigen::VectorXd &v, const std::pair<Eigen::VectorXd, Eigen::LLT<Eigen::MatrixXd>> &scale) {
    v = scale.second.matrixL().solve(v-scale.first);
}

void cmp::scale(double &v, const std::pair<double, double> &scale) {
    v = (v - scale.first)/scale.second;
}

void cmp::un_scale(Eigen::VectorXd &v, const std::pair<Eigen::VectorXd, Eigen::LLT<Eigen::MatrixXd>> &scale) {
    v = scale.second.matrixL()*v + scale.first;
}

void cmp::un_scale(double &v, const std::pair<double, double> &scale) {
    v = v*scale.second + scale.first;
}


std::vector<Eigen::VectorXd> cmp::matrix_to_vvxd(const Eigen::MatrixXd &data) {
    int n_rows = data.rows();

    std::vector<Eigen::VectorXd> data_v(n_rows);
    for (int i=0; i<n_rows; i++) {
        data_v[i] = data.row(i);
    }

    return data_v;
}

Eigen::VectorXd cmp::v_to_vxd(std::vector<double> const &v)
{

    Eigen::VectorXd x(v.size());
    for (int i = 0; i < v.size(); i++) {
        x(i) = v[i];
    }
    return x;
}

std::vector<double> cmp::vxd_to_v(Eigen::VectorXd const &x) {
  
    std::vector<double> v(x.size());
    for (int i = 0; i < v.size(); i++) {
        v[i] = x(i);
    }
    return v;

}

std::vector<Eigen::VectorXd> cmp::v_to_vvxd(const std::vector<double> &v) {
    
    std::vector<Eigen::VectorXd> x(v.size());
    for (int i = 0; i < v.size(); i++) {
        Eigen::VectorXd vv(1);
        vv << v[i];
        x[i] = vv;
    }
    return x;
}

void cmp::write_vector(const std::vector<Eigen::VectorXd> &data, std::ofstream &o_file) {

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            o_file << data[i](j) << " ";
        }
        o_file << std::endl;
    }
}

void cmp::write_data(const std::vector<Eigen::VectorXd> &x, const Eigen::MatrixXd &y, std::ofstream &o_file) {
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            o_file << x[i](j) << " ";
        }
        for (int j = 0; j < y.cols(); j++) {
            o_file << y(i,j) << " ";
        }
        o_file << std::endl;
    }
}

std::vector<Eigen::VectorXd> cmp::read_vector(std::ifstream &i_file, std::string delimiter, size_t header) {

    std::vector<Eigen::VectorXd> v;

    //Check if file exists
    if (!i_file) {
        spdlog::error("File is not open! returning an empty vector");
        return v;
    }
    
    std::string line;

    // Skip the header lines
    for (size_t i = 0; i < header; i++) {
        if (!getline(i_file, line)) {
            spdlog::error("Error reading header line {0:d}", i);
            return v;
        }
    }

    while (getline(i_file, line)) {

        // Check if the line is empty
        if (line.empty()) {
            continue;
        }
        // Check if the line is a comment
        if (line[0] == '#') {
            continue;
        }
        
        // Split the line into tokes using the delimiter
        std::vector<std::string> tokens;
        size_t pos = 0;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            tokens.push_back(line.substr(0, pos));
            line.erase(0, pos + delimiter.length());
        }
        tokens.push_back(line);

        // Convert the tokens to doubles and store them in a vector
        Eigen::VectorXd row(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            try {

                // Strip the token of any whitespace
                tokens[i].erase(std::remove_if(tokens[i].begin(), tokens[i].end(), ::isspace), tokens[i].end());

                row(i) = std::stod(tokens[i]);
            } catch (const std::invalid_argument &e) {
                spdlog::error("Invalid argument: {0}", e.what());
                spdlog::log(spdlog::level::err, "Error parsing token: {0}", tokens[i]);
                return v;
            }
        }
        // Add the row to the vector
        v.push_back(row);
    }

    spdlog::info("number of lines in the file : {0:d}", v.size());
    spdlog::info("number of data in a line : {0:d}", v[0].size());
    return v;
}
