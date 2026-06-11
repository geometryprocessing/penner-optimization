// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "feature/core/common.h"

/**
 * @brief Methods to compute relevant quad mesh statistics, including degree valences.
 * 
 */

namespace Penner {
namespace Feature {

/**
 * @brief Compute the vertex valences of a given (tri or quad) mesh
 * 
 * @param F: mesh faces
 * @return valences of the mesh vertices
 */
std::vector<int> compute_valences(const Eigen::MatrixXi& F);

} // namespace Feature
} // namespace Penner