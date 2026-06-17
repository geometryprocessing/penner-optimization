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
 * @brief Methods for changing edge and interior barycentric coordinates as determined
 * by hyperbolic edge translations.
 * 
 * Translations for mapping between surfaces can be computed using methods in translation.h.
 */

namespace Penner {

/// Reparametrize the barycentric coordinates for the equilateral triangle by
/// translating a constant hyperbolic distance along each halfedge. If the sum
/// of translations per triangle is 0, then this corresponds to a projective map
/// over the triangle.
///
/// @param[in, out] m_o: mesh to reparametrize
/// @param[in] tau: per halfeedge hyperbolic translation distances
template <typename OverlayScalar>
void bc_reparametrize_eq(OverlayMesh<OverlayScalar>& m_o, const VectorX& tau);

/// Reparametrize points contained in equilateral reference triangles by
/// translating a constant hyperbolic distance along each halfedge. If the sum
/// of translations per triangle is 0, then this corresponds to a projective map
/// over the triangle.
///
/// @param[in, out] pts: points to reparameterize
/// @param[in] n: next halfedge array for mesh
/// @param[in] h: face to halfedge array for mesh
/// @param[in] tau: per halfeedge hyperbolic translation distances
template <typename OverlayScalar>
void reparametrize_equilateral(
    std::vector<Pt<OverlayScalar>>& pts,
    const std::vector<int>& n,
    const std::vector<int>& h,
    const VectorX& tau);

#ifdef PYBIND
#endif

} // namespace Penner