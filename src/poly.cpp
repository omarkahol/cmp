#include <poly.h>


void cmp::MultiIndex::set(size_t dimension, size_t totalDegree) {

    // Check for valid inputs
    if(dimension == 0) {
        throw std::invalid_argument("Dimension must be greater than 0.");
    }

    Index current(dimension, 0);

    // Recursive generation of multi-indices
    for(int k = 0; k <= totalDegree; ++k) {
        generateRecursive(current, 0, k);
    }
}

void cmp::MultiIndex::print() const {
    for(const auto& idx : indices) {
        std::cout << "(";
        for(size_t j = 0; j < idx.size(); ++j) {
            std::cout << idx[j] << (j + 1 < idx.size() ? "," : "");
        }
        std::cout << ")\n";
    }
}

void cmp::MultiIndex::generateRecursive(Index& current, int pos, int remaining) {
    if(pos == (int)current.size() - 1) {
        current[pos] = remaining;
        indices.push_back(current);
        return;
    }
    for(int k = 0; k <= remaining; ++k) {
        current[pos] = k;
        generateRecursive(current, pos + 1, remaining - k);
    }
}

// Hermite Polynomial Evaluation using the probabilists' definition
double cmp::HermitePolynomial::evaluate(const size_t &deg, const double &x) const {
    if(deg == 0) return 1.0;
    if(deg == 1) return x;

    double Hn_2 = 1.0; // H_0(x)
    double Hn_1 = x;   // H_1(x)
    double Hn;

    for(size_t n = 2; n <= deg; ++n) {
        Hn = x * Hn_1 - (n - 1) * Hn_2;
        Hn_2 = Hn_1;
        Hn_1 = Hn;
    }
    return Hn;
}

// Legendre Polynomial Evaluation using the standard definition
double cmp::LegendrePolynomial::evaluate(const size_t &deg, const double &x) const {
    if(deg == 0) return 1.0;
    if(deg == 1) return x;
    double Pn_2 = 1.0; // P_0(x)
    double Pn_1 = x;   // P_1(x)
    double Pn;

    for(size_t n = 2; n <= deg; ++n) {
        Pn = ((2 * n - 1) * x * Pn_1 - (n - 1) * Pn_2) / n;
        Pn_2 = Pn_1;
        Pn_1 = Pn;
    }
    return Pn;
}

std::pair<double, double> cmp::PolynomialExpansion::predict(const Eigen::Ref<const Eigen::VectorXd> &x) const {
    if(x.size() != (int)multiIndex_.indices[0].size()) {
        throw std::invalid_argument("Input vector size does not match the polynomial dimension.");
    }

    Eigen::VectorXd X = Eigen::VectorXd::Zero(coefficients_.size());
    for(size_t i = 0; i < multiIndex_.size(); ++i) {
        double term = 1.0;
        for(size_t d = 0; d < x.size(); ++d) {
            term *= basis_->evaluate(multiIndex_.indices[i][d], x(d));
        }
        X(i) = term;
    }

    double mean = coefficients_.dot(X);
    double variance = 0.0;
    if(coefficientsCovariance_.size() > 0) {
        variance = X.transpose() * coefficientsCovariance_ * X;
    }
    return {mean, variance};
}

std::pair<double, double> cmp::PolynomialExpansion::predictLOO(const size_t& i) const {
    return predict(xObs_.row(i));
}

void cmp::PolynomialExpansion::fit(const Eigen::Ref<const Eigen::MatrixXd> &samples, const Eigen::Ref<const Eigen::VectorXd> &values) {
    if(samples.rows() != values.size()) {
        throw std::invalid_argument("Number of samples must match number of values.");
    }
    if(samples.rows() == 0) {
        throw std::invalid_argument("Samples cannot be empty.");
    }
    int numBasis = coefficients_.size();
    int numSamples = samples.rows();
    Eigen::MatrixXd A(numSamples, numBasis);
    xObs_ = samples;

    for(int i = 0; i < numSamples; ++i) {
        if(samples.row(i).size() != (int)multiIndex_.indices[0].size()) {
            throw std::invalid_argument("Sample vector size does not match the polynomial dimension.");
        }
        for(int j = 0; j < numBasis; ++j) {
            double term = 1.0;
            for(size_t d = 0; d < samples.row(i).size(); ++d) {
                term *= basis_->evaluate(multiIndex_.indices[j][d], samples(i, d));
            }
            A(i, j) = term;
        }
    }

    Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());


    // Solve the least squares problem A * coeffs = b
    coefficients_ = A.colPivHouseholderQr().solve(b);

    // Estimate covariance matrix of the coefficients
    Eigen::VectorXd residuals = b - A * coefficients_;
    double sigma2 = residuals.squaredNorm() / (numSamples - numBasis);
    Eigen::MatrixXd AtA_inv = (A.transpose() * A).inverse();
    coefficientsCovariance_ = sigma2 * AtA_inv;
}
