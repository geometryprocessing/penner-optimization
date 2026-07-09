// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/cone_metric.h"

/**
 * @brief Methods to compute cones on a surface from field data, as well as various
 * queries and methods to modify invalid or challenging cone configurations.
 * 
 */

namespace Penner {
namespace Field {

/**
 * @brief Compute the cones from a rotation form on an intrinsic mesh.
 * 
 * @param m: mesh with metric
 * @param rotation_form: per-halfedge rotation form
 * @return per-vertex cones corresponding to the rotation form
 */
std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form);

/**
 * @brief Compute the cones from a rotation form on an extrinsic mesh with reindexed vertices.
 * 
 * @param m: mesh with metric
 * @param vtx_reindex: map from halfedge to VF vertex indices
 * @param rotation_form: per-halfedge rotation form
 * @param has_boundary: (optional) if true, treat mesh as a doubled mesh with boundary
 * @return per-vertex cones corresponding to the rotation form
 */
std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const VectorX& rotation_form,
    bool has_boundary=false);


/**
 * @brief Determine if a mesh cone prescription contains cones below a threshold
 * 
 * @param Th_hat: mesh with cone constraints
 * @param min_cone_index: minimum index for cones
 * @return true if there is a cone strictly below the threshold
 * @return false otherwise
 */
bool contains_small_cones(const std::vector<Scalar>& Th_hat, int min_cone_index=1);

/**
 * @brief Determine if a mesh cone prescription corresponds to a trivial torus
 * 
 * @param m: mesh with cone constraints
 * @return true if the cones correspond to a trivial torus
 * @return false otherwise
 */
bool is_trivial_torus(const Mesh<Scalar>& m);

/**
 * @brief Determine if a mesh cone prescription corresponds to a torus with a cone pair.
 * 
 * @param m: mesh with cone constraints
 * @return true if the mesh is a torus with a pair of cones
 * @return false otherwise
 */
bool is_torus_with_cone_pair(const Mesh<Scalar>& m);

/**
 * @brief Given target cone angles, fix any problems that prevent them from being valid
 * for seamless holonomy constraints.
 * 
 * @param m: mesh with cone constraints
 * @param min_cone_index: replace cones smaller index
 */
void fix_cones(Mesh<Scalar>& m, int min_cone_index=1);

/**
 * @brief Add positive and negative curvature cones to constraints.
 * 
 * WARNING: this method places cones arbitrarily and may produce difficult constraints
 * 
 * @param m: mesh with cone constraints
 * @param only_interior: only add cone constraints to interior if possible
 * @param offset: approximate vertex index to place cones near
 */
void add_random_cone_pair(Mesh<Scalar>& m, bool only_interior=true, int offset=0);

/**
 * @brief Count the number of positive and negative curvature cones.
 * 
 * @param m: mesh with cone metric
 * @return number of negative curvature cones (cone angle above 2 Pi)
 * @return number of postiive curvature cones (cone angle below 2 Pi)
 */
std::pair<int, int> count_cones(const Mesh<Scalar>& m);


} // namespace Field
} // namespace Penner