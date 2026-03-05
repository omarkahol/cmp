#ifndef CMP_DEFINES_H
#define CMP_DEFINES_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <nlopt.hpp>
#include <functional>

// Additional Eigen typedefs
namespace Eigen {
using VectorXs = Matrix<size_t, Eigen::Dynamic, 1>;
}


namespace cmp {

inline constexpr double TOL = 1e-12;

// Dummy Matrix and Vector
inline const Eigen::MatrixXd DummyMatrix(0, 0);
inline const Eigen::VectorXd DummyVector(0);


template<typename T>
concept VectorLike = requires(T a, size_t i) {
    typename T::value_type;
    {
        a.size()
    }
    -> std::convertible_to<size_t>;
    {
        a[i]
    }
    -> std::convertible_to<typename T::value_type>;
    {
        a.data()
    }
    -> std::convertible_to<const typename T::value_type*>;
    {
        a.begin()
    };
    {
        a.end()
    };
};

// Define a   Eigen::VectorXd of size 1 containing the value zero and one containing the value one
const Eigen::VectorXd ScalarZero = Eigen::VectorXd::Zero(1);
const Eigen::VectorXd ScalarOne =  Eigen::VectorXd::Ones(1);

/**
 * @brief The score_t type
 * A function that takes a const reference to an  Eigen::VectorXd and returns a double
 *
 */
using score_t = std::function<double(const Eigen::VectorXd &)>;


/**
 * Function for the LDLT decomposition of a covariance matrix
 */
inline Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition(const Eigen::Ref<const Eigen::MatrixXd> &cov) {
    return cov.ldlt();
}

/**
 * Functions to convert between vector types and  Eigen::VectorXd
// Non owning (read only) view of vector-like containers as Eigen::VectorXd\
    */
template <VectorLike V>
auto asEigen(const V &v) -> Eigen::Map<const Eigen::Matrix<typename V::value_type, Eigen::Dynamic, 1>> {
    using T = typename V::value_type;
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(), static_cast<Eigen::Index>(v.size()));
}

// Non owning (read and write) view of vector-like containers as  Eigen::VectorXd
template <VectorLike V>
auto asEigen(V &v) -> Eigen::Map<Eigen::Matrix<typename V::value_type, Eigen::Dynamic, 1>> {
    using T = typename V::value_type;
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(), static_cast<Eigen::Index>(v.size()));
}

// Copy version from vector-like containers to Eigen::VectorXd
template <VectorLike V>
auto toEigen(const V &v) -> Eigen::VectorXd {
    using T = typename V::value_type;
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(), static_cast<Eigen::Index>(v.size()));
}


// Copy version from Eigen::VectorXd to std::vector
template <VectorLike V>
auto toStdVector(const V &v) -> std::vector<typename V::value_type> {
    using T = typename V::value_type;
    return std::vector<T>(v.begin(), v.end());
}

// Function to slice a matrix given a vector of indices
template<typename Derived>
typename Derived::PlainObject slice(const Eigen::MatrixBase<Derived>& mat, const Eigen::VectorXs& indices) {
    typename Derived::PlainObject result(indices.size(), mat.cols());
    for(Eigen::Index i = 0; i < indices.size(); ++i) {
        result.row(i) = mat.row(indices(i));
    }
    return result;
}

/**
 * @brief Function to split training data into training and test sets.
 * @param xObs The observations matrix.
 * @param yObs The labels/values vector.
 * @param train_test_pair A pair of vectors containing the training and test indices.
 *
 * @return A tuple containing the training observations, training labels, test observations, and test labels.
 */
template<typename DerivedX, typename DerivedY>
auto trainTestSplit(const Eigen::MatrixBase<DerivedX>& xObs, const Eigen::MatrixBase<DerivedY>& yObs, const std::pair<Eigen::VectorXs, Eigen::VectorXs>& train_test_pair) {
    auto xTrain = slice(xObs, train_test_pair.first);
    auto yTrain = slice(yObs, train_test_pair.first);
    auto xTest = slice(xObs, train_test_pair.second);
    auto yTest = slice(yObs, train_test_pair.second);
    return std::make_tuple(xTrain, yTrain, xTest, yTest);
}

} // namespace cmp
#endif // CMP_DEFINES_H