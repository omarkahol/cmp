#include "io.h"

using namespace cmp;

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
