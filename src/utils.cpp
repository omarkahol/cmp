#include "utils.h"
#include <optimization.h>
#include <finite_diff.h>

using namespace cmp;

double cmp::v_mean(const vector_t &v) {
    int n = v.size();
    double mean=0.0;
    for(int i=0; i<n; i++) {
        mean += v(i);
    }
    return mean/double(n);
}

double cmp::v_std(const vector_t &v) {
    int n = v.size();
    double mean=v_mean(v);
    double std = 0;
    for(int i=0; i<n; i++) {
        std += pow(v(i)-mean,2);
    }
    return sqrt(std/double(n));
}

std::vector<double> cmp::get_column(const std::vector<vector_t> &data, const int &index) {
    int rows = data.size();
    std::vector<double> my_col(rows);

    for (int i=0; i<rows; i++) {
        my_col[i] = data[i](index);
    }

    return my_col;
}

std::pair<vector_t, Eigen::LLT<matrix_t>> cmp::normalize(std::vector<vector_t> &grid) {
    
    int rows = grid.size();
    int cols = grid[0].size();

    // Compute the mean
    vector_t mean = vector_t::Zero(cols);
    for (int i=0; i<rows; i++) {
        mean += grid[i];
    }
    mean = mean / double(rows);

    // Compute the covariance
    matrix_t cov = matrix_t::Zero(cols,cols);
    for (size_t i=0; i<rows; i++){
        cov += (grid[i]-mean)*(grid[i]-mean).transpose();
    }
    cov = cov / double(rows-1);

    // compute the Cholesky decomposition
    Eigen::LLT<matrix_t> cov_llt(cov);
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

void cmp::scale(vector_t &v, const std::pair<vector_t, Eigen::LLT<matrix_t>> &scale) {
    v = scale.second.matrixL().solve(v-scale.first);
}

void cmp::scale(double &v, const std::pair<double, double> &scale) {
    v = (v - scale.first)/scale.second;
}

void cmp::un_scale(vector_t &v, const std::pair<vector_t, Eigen::LLT<matrix_t>> &scale) {
    v = scale.second.matrixL()*v + scale.first;
}

void cmp::un_scale(double &v, const std::pair<double, double> &scale) {
    v = v*scale.second + scale.first;
}


std::vector<vector_t> cmp::matrix_to_vvxd(const matrix_t &data) {
    int n_rows = data.rows();

    std::vector<vector_t> data_v(n_rows);
    for (int i=0; i<n_rows; i++) {
        data_v[i] = data.row(i);
    }

    return data_v;
}

vector_t cmp::v_to_vxd(std::vector<double> const &v)
{

    vector_t x(v.size());
    for (int i = 0; i < v.size(); i++) {
        x(i) = v[i];
    }
    return x;
}

std::vector<double> cmp::vxd_to_v(vector_t const &x) {
  
    std::vector<double> v(x.size());
    for (int i = 0; i < v.size(); i++) {
        v[i] = x(i);
    }
    return v;

}

std::vector<vector_t> cmp::v_to_vvxd(const std::vector<double> &v) {
    
    std::vector<vector_t> x(v.size());
    for (int i = 0; i < v.size(); i++) {
        vector_t vv(1);
        vv << v[i];
        x[i] = vv;
    }
    return x;
}

void cmp::write_vector(const std::vector<vector_t> &data, std::ofstream &o_file) {

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            o_file << data[i](j) << " ";
        }
        o_file << std::endl;
    }
}

void cmp::write_data(const std::vector<vector_t> &x, const matrix_t &y, std::ofstream &o_file) {
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

std::vector<vector_t> cmp::read_vector(std::ifstream &i_file) {

    std::vector<vector_t> v;

    //Check if file exists
    if (!i_file) {
        spdlog::error("File is not open! returning an empty vector");
        return v;
    }
    
    std::string line;

    while (getline(i_file, line)) {

        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        std::vector<double> values;

        for (int i = 0; i < words.size(); i++) {
            values.push_back(stod(words[i]));
        }
        v.push_back(cmp::v_to_vxd(values));
    }

    spdlog::info("number of lines in the file : {0:d}", v.size());
    spdlog::info("number of data in a line : {0:d}", v[0].size());
    return v;
}
