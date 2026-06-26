// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "util/common.h"

/**
 * @brief Methods to generate field directions and cones from an nrosy field.
 * 
 */

namespace Penner {
namespace Field {

/**
 * @brief Generate a cross field for a mesh
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @return |F|x3 frame field of per-face field direction vectors
 * @return per-vertex cone angles corresponding to the frame field
 */
std::tuple<Eigen::MatrixXd, std::vector<Scalar>> generate_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);


} // namespace Field
} // namespace Penner