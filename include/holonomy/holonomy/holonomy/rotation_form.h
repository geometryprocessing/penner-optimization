// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "holonomy/core/common.h"
#include "field/intrinsic_field.h"

/**
 * @brief Methods to compute a rotation form, either from an existing field or generated
 * using a cross-field optimization method.
 * 
 * The rotation form is a minimal representation of a cross field (up to global phase),
 * encoding the rotation of the field across edges.
 * 
 */

namespace Penner {
namespace Holonomy {

/**
 * @brief Given a mesh with a per-face frame field, 
 * 
 * @param m: halfedge mesh
 * @param vtx_reindex: map from halfedge to VF vertex indices
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param frame_field: |F|x3 frame field of per-face field direction vectors 
 * @return per-halfedge rotation form
 */
VectorX generate_rotation_form_from_cross_field(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field);

VectorX generate_intrinsic_rotation_form(const Mesh<Scalar>& m, const Field::FieldParameters& field_params);

VectorX generate_intrinsic_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Field::FieldParameters& field_params);

} // namespace Holonomy
} // namespace Penner