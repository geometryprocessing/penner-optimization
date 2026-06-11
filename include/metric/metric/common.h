// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

/**
 * @brief Assorted utility functions.
 * 
 */

#include "util/common.h"
#include "metric/globals.h"

#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <array>
#include <deque>

namespace Penner {

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


} // namespace Penner