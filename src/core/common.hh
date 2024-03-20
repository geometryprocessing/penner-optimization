#pragma once

#include "globals.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <array>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/spdlog.h"

namespace CurvatureMetric {

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
inline double max(const double& a, const double& b)
{
    return std::max(a, b);
}

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

// Check if two vectors are component-wise equal, up to a tolerance.
//
/// @param[in] v: first vector to compare
/// @param[in] w: second vector to compare
/// @param[in] eps: tolerance for equality
/// @return true iff ||v - i||_inf < eps
bool vector_equal(VectorX v, VectorX w, Scalar eps = 1e-10);

/// Check if a matrix contains a nan
///
/// @param[in] mat: matrix to check
/// @return true iff mat contains a nan
bool matrix_contains_nan(const Eigen::MatrixXd& mat);

// Compute the sup norm of a vector.
//
/// @param[in] v: vector
/// @return sup norm of v
Scalar sup_norm(const VectorX& v);

// Compute the sup norm of a matrix.
//
/// @param[in] matrix: matrix
/// @return sup norm of the matrix
Scalar matrix_sup_norm(const MatrixX& matrix);

/// Create a vector with values 0,1,...,n-1
///
/// @param[in] n: size of the output vector
/// @param[out] vec: output arangement vector
inline void arange(size_t n, std::vector<int>& vec)
{
    vec.resize(n);
    std::iota(vec.begin(), vec.end(), 0);
}

} // namespace CurvatureMetric
