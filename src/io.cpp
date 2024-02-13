#include "io.h"

using namespace cmp; 

vector_t cmp::v_to_vxd(std::vector<double> const &v) {

    vector_t X(v.size());
    for (int i = 0; i < v.size(); i++) {
        X(i) = v[i];
    }
    return X;

}

std::vector<double> cmp::vxd_to_v(vector_t const &X) {
  
    std::vector<double> v(X.size());
    for (int i = 0; i < v.size(); i++) {
        v[i] = X(i);
    }
    return v;

}

void cmp::write_vector(std::vector<vector_t> const &v, std::ofstream &o_file) {

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[0].size(); j++) {
            o_file << v[i](j) << " ";
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
