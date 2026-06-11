// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "holonomy/core/common.h"

/**
 * @brief Methods to compute the quality of triangles in a mesh.
 * 
 */

namespace Penner {
namespace Holonomy {

/**
 * @brief Compute the triangle quality as the ratio of outradius to inradius.
 * 
 * @param lij: first edge length
 * @param ljk: second edge length
 * @param lki: third edge length
 * @return triangle quality measure
 */
Scalar compute_triangle_quality(Scalar lij, Scalar ljk, Scalar lki);
 
 /**
  * @brief Compute the mesh per-face triangle qualities
  * 
  * @param cone_metric: mesh with metric
  * @return: per-face triangle quality measures
  */
VectorX compute_mesh_quality(const DifferentiableConeMetric& cone_metric);

/**
  * @brief Compute the minimum corner angle of the mesh
  * 
  * @param cone_metric: mesh with metric
  * @return: minimum corner angle
 */
Scalar compute_min_angle(const DifferentiableConeMetric& cone_metric);

} // namespace Holonomy
} // namespace Penner