// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

/**
 * @brief Typedefs and basic utility functions.
 * 
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#include "spdlog/spdlog.h"
#include <spdlog/fmt/fmt.h>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace OverlayProblem {}

namespace Penner {
using namespace OverlayProblem;

#ifdef MULTIPRECISION
#include <unsupported/Eigen/MPRealSupport>
#include "mpreal.h"
typedef mpfr::mpreal Scalar;
const Scalar PI = Scalar(mpfr::const_pi());
#else
typedef double Scalar;
const Scalar PI = M_PI;
#endif

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<Scalar> MatrixX;

typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 2, 2> Matrix2x2;

typedef Eigen::Triplet<Scalar> T;

#ifdef ENABLE_VISUALIZATION
extern glm::vec3 BEIGE;
extern glm::vec3 BLACK;
extern glm::vec3 BLACK_BROWN;
extern glm::vec3 TAN;
extern glm::vec3 MUSTARD;
extern glm::vec3 FOREST_GREEN;
extern glm::vec3 TEAL;
extern glm::vec3 DARK_TEAL;
#endif

using std::max;
using std::min;
using std::isnan;

const Scalar INF = 1e10;

/**
 * @brief Compute the square of a scalar.
 *
 * @param x: value to square
 * @return squared value
 */
template <typename FloatScalar>
FloatScalar square(FloatScalar x)
{
    return x * x;
}

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
template <typename FloatScalar>
bool float_equal(FloatScalar a, FloatScalar b, FloatScalar eps = 1e-10)
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
inline std::vector<int> arange(size_t n)
{
    std::vector<int> vec;
    arange(n, vec);
    return vec;
}

/**
 * @brief Compute the real modulus of x mod y
 * 
 * @param x: positive number to mod
 * @param y: positive modulus
 * @return x (mod y)
 */
inline
Scalar pos_fmod(Scalar x, Scalar y) { return (0 == y) ? x : x - y * floor(x / y); }

} // namespace Penner