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


/**
 * @addtogroup core
 * @{
 */
namespace cmp {

inline constexpr double TOL = 1e-12;

// Dummy Matrix and Vector
inline const Eigen::MatrixXd DummyMatrix(0, 0);
inline const Eigen::VectorXd DummyVector(0);


/**
 * @brief Concept for compile-time validation of vector-like containers.
 * 
 * @details Mathematical Formulation
 * Defines a set of constraints for type \f$T\f$ representing an element in a finite-dimensional vector space \f$\mathbb{R}^N\f$.
 * 
 * @details Implementation Algorithm
 * Ensures that \f$T\f$ supports:
 * - `.size()` returning an integer/size type.
 * - `operator[]` for coordinate access.
 * - `.data()` providing a raw pointer to contiguous memory.
 * - Iterators `.begin()` and `.end()`.
 */
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
 * And also its gradient counterpart, gradient_t, which takes a const reference to an  Eigen::Ref<const Eigen::VectorXd> and returns an  Eigen::VectorXd
 */
using score_t = std::function<double(const Eigen::VectorXd &)>;
using gradient_t = std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>&)>;


/**
 * @brief Computes the LDLT decomposition of a symmetric matrix.
 * 
 * @details Mathematical Formulation
 * For a symmetric positive semidefinite matrix \f$\mathbf{A} \in \mathbb{R}^{n \times n}\f$, LDLT decomposes \f$\mathbf{A}\f$ into:
 * \f[
 * \mathbf{A} = \mathbf{L} \mathbf{D} \mathbf{L}^T
 * \f]
 * where \f$\mathbf{L}\f$ is a lower unit triangular matrix and \f$\mathbf{D}\f$ is a diagonal matrix. This decomposition is more numerically stable than Cholesky decomposition for semi-definite matrices as it avoids taking square roots of diagonal elements.
 * 
 * @details Implementation Algorithm
 * Calls Eigen's `.ldlt()` method on the matrix block.
 */
inline Eigen::LDLT<Eigen::MatrixXd> ldltDecomposition(const Eigen::Ref<const Eigen::MatrixXd> &cov) {
    return cov.ldlt();
}

/**
 * @brief Creates a non-owning, read-only Eigen view (Eigen::Map) of a vector-like container.
 * 
 * @details Mathematical Formulation
 * Maps a vector \f$\mathbf{v} \in \mathbb{R}^N\f$ represented by a contiguous-memory container 
 * to an Eigen Vector expression without copying:
 * \f[
 * \mathbf{v}_{\text{Eigen}} = \text{Map}(\mathbf{v})
 * \f]
 * 
 * @tparam V The type of the vector-like container, must satisfy the VectorLike concept.
 * @param v The vector-like container.
 * @return A non-owning Eigen::Map view pointing to the underlying data.
 */
template <VectorLike V>
auto asEigen(const V &v) -> Eigen::Map<const Eigen::Matrix<typename V::value_type, Eigen::Dynamic, 1>> {
    using T = typename V::value_type;
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(), static_cast<Eigen::Index>(v.size()));
}

/**
 * @brief Creates a non-owning, read-write Eigen view (Eigen::Map) of a vector-like container.
 * 
 * @tparam V The type of the vector-like container, must satisfy the VectorLike concept.
 * @param v The vector-like container.
 * @return A mutable, non-owning Eigen::Map view pointing to the underlying data.
 */
template <VectorLike V>
auto asEigen(V &v) -> Eigen::Map<Eigen::Matrix<typename V::value_type, Eigen::Dynamic, 1>> {
    using T = typename V::value_type;
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(), static_cast<Eigen::Index>(v.size()));
}

/**
 * @brief Copies the elements of a vector-like container to a new Eigen::VectorXd.
 * 
 * @tparam V The type of the vector-like container, must satisfy the VectorLike concept.
 * @param v The vector-like container to copy.
 * @return A new Eigen::VectorXd containing a copy of the elements of v.
 */
template <VectorLike V>
auto toEigen(const V &v) -> Eigen::VectorXd {
    using T = typename V::value_type;
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(), static_cast<Eigen::Index>(v.size()));
}

/**
 * @brief Copies the elements of a vector-like container to a new std::vector.
 * 
 * @tparam V The type of the vector-like container, must satisfy the VectorLike concept.
 * @param v The vector-like container to copy.
 * @return A new std::vector containing a copy of the elements of v.
 */
template <VectorLike V>
auto toStdVector(const V &v) -> std::vector<typename V::value_type> {
    using T = typename V::value_type;
    return std::vector<T>(v.begin(), v.end());
}

/**
 * @brief Slices a matrix along its rows based on a set of indices.
 * 
 * @details Mathematical Formulation
 * For a matrix \f$\mathbf{X} \in \mathbb{R}^{M \times D}\f$ and a vector of indices \f$\mathcal{I} = [i_1, i_2, \dots, i_K]^T\f$,
 * computes the sliced matrix \f$\mathbf{X}_{\mathcal{I}} \in \mathbb{R}^{K \times D}\f$:
 * \f[
 * \left(\mathbf{X}_{\mathcal{I}}\right)_{k, \cdot} = \mathbf{X}_{i_k, \cdot}
 * \f]
 * 
 * @tparam Derived The derived Eigen matrix expression type.
 * @param mat The input matrix.
 * @param indices A vector of row indices to extract.
 * @return A new plain matrix containing only the specified rows of mat.
 */
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
/** @} */

#endif // CMP_DEFINES_H