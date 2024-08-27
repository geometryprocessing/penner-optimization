#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#include "spdlog/spdlog.h"

#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/globals.hh"

namespace Penner {
using namespace OverlayProblem;

#ifdef MULTIPRECISION
#include <unsupported/Eigen/MPRealSupport>
#include "mpreal.h"
typedef mpfr::mpreal Scalar;
#else
typedef double Scalar;
#endif

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<Scalar> MatrixX;

typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 2, 2> Matrix2x2;

typedef Eigen::Triplet<Scalar> T;


using std::max;
using std::min;
using std::isnan;

const Scalar INF = 1e10;

/// Swap two doubles.
///
/// @param[in, out] a: first double to swap
/// @param[in, out] b: second double to swap
inline void swap(double& a, double& b)
{
    std::swap<double>(a, b);
}

/// Get the max of two doubles.
///
/// @param[in] a: first double to max
/// @param[in] b: second double to max
/// @return max of a and b
//inline double max(const double& a, const double& b)
//{
//    return std::max(a, b);
//}

/// Check if two values are equal, up to a tolerance.
///
/// @param[in] a: first value to compare
/// @param[in] b: second value to compare
/// @param[in] eps: tolerance for equality
/// @return true iff |a - b| < eps
inline bool float_equal(Scalar a, Scalar b, Scalar eps = 1e-10)
{
    return (abs(a - b) < eps);
}

/// Create a vector with values 0,1,...,n-1
///
/// @param[in] n: size of the output vector
/// @param[out] vec: output arangement vector
inline void arange(size_t n, std::vector<int>& vec)
{
    vec.resize(n);
    std::iota(vec.begin(), vec.end(), 0);
}

} // namespace Penner