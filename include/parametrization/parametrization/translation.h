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
 * @brief Method to compute hyperbolic translations determining a continuous map between
 * two intrinsic metrics.
 * 
 */

namespace Penner {

/// Generate the least squares solution to the halfedge translations in the
/// hyperbolic metric needed to satisfy the per halfedge shear change and face
/// zero sum condition.
///
/// The zero sum face condition is necessary to extend the edge translations
/// to a projective transformation on the entire face.
///
/// @param[in] m: mesh
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[in] he_metric_target: target metric coordinates for m
/// @param[out] he_translations: per halfedge translations
template <typename OverlayScalar>
bool compute_as_symmetric_as_possible_translations(
    const Mesh<OverlayScalar>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_translations);
// TODO: Add option to bypass and use zero translations or to solve in double precision

} // namespace Penner