#include "io.h"

void cmp::write_vector(const std::vector<Eigen::VectorXd> &data, std::ofstream &o_file) {

    for(int i = 0; i < data.size(); i++) {
        for(int j = 0; j < data[0].size(); j++) {
            o_file << data[i](j) << " ";
        }
        o_file << std::endl;
    }
}

void cmp::write_data(const std::vector<Eigen::VectorXd> &x, const Eigen::MatrixXd &y, std::ofstream &o_file) {
    for(int i = 0; i < x.size(); i++) {
        for(int j = 0; j < x[0].size(); j++) {
            o_file << x[i](j) << " ";
        }
        for(int j = 0; j < y.cols(); j++) {
            o_file << y(i, j) << " ";
        }
        o_file << std::endl;
    }
}

std::vector<Eigen::VectorXd> cmp::read_vector(std::ifstream &i_file, std::string delimiter, size_t header) {

    std::vector<Eigen::VectorXd> v;

    //Check if file exists
    if(!i_file) {
        std::cout << "File does not exist. Returning an empty vector." << std::endl;
        return v;
    }

    std::string line;

    // Skip the header lines
    for(size_t i = 0; i < header; i++) {
        if(!getline(i_file, line)) {
            std::cout << "Error reading header line " << i << std::endl;
            return v;
        }
    }

    while(getline(i_file, line)) {

        // Check if the line is empty
        if(line.empty()) {
            continue;
        }
        // Check if the line is a comment
        if(line[0] == '#') {
            continue;
        }

        // Split the line into tokes using the delimiter
        std::vector<std::string> tokens;
        size_t pos = 0;
        while((pos = line.find(delimiter)) != std::string::npos) {
            tokens.push_back(line.substr(0, pos));
            line.erase(0, pos + delimiter.length());
        }
        tokens.push_back(line);

        // Convert the tokens to cmp::Reals and store them in a vector
        Eigen::VectorXd row(tokens.size());
        for(size_t i = 0; i < tokens.size(); i++) {
            try {

                // Strip the token of any whitespace
                tokens[i].erase(std::remove_if(tokens[i].begin(), tokens[i].end(), ::isspace), tokens[i].end());

                row(i) = std::stod(tokens[i]);
            } catch(const std::invalid_argument &e) {
                std::cout << "Invalid argument: " << e.what() << " for token: " << tokens[i] << std::endl;
                return v;
            }
        }
        // Add the row to the vector
        v.push_back(row);
    }

    std::cout << "number of lines in the file : " << v.size() << std::endl;
    std::cout << "number of data in a line : " << v[0].size() << std::endl;
    return v;
}
