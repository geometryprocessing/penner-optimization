// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/cone_metric.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "util/embedding.h"

/// \file energy_functor.h
///
/// Methods to scale per-element values by element weights

namespace Penner {
namespace Optimization {

/// @brief Given a vector of weights and a vector of values, compute the weighted 2 norm
/// as the sum of the product of the weights and squared values
///
/// @param[in] weights: per term weights
/// @param[in] values: value vector for the norm
/// @return weighted norm
Scalar compute_weighted_norm(const VectorX& weights, const VectorX& values);

/// @brief Compute per edge weights for a mesh with a given metric as 1/3 of the areas
/// of the two adjacent faces
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @param[out] edge_area_weights: weights per edge
VectorX compute_edge_area_weights(const DifferentiableConeMetric& cone_metric);

/// @brief Compute per vertex area weights for a mesh
///
/// @param[in] m: mesh
/// @return weights per vertex
VectorX compute_vertex_area_weights(const Mesh<Scalar>& m);

/// @brief Compute per independent vertex area weights for a mesh
///
/// The weights are half the sum of identified vertex weights.
///
/// @param[in] m: mesh
/// @return weights per independent vertex
VectorX compute_independent_vertex_area_weights(const Mesh<Scalar>& m);

/// @brief Compute per face area weights for a mesh
///
/// @param[in] m: mesh
/// @return weights per face
VectorX compute_face_area_weights(const Mesh<Scalar>& m);

/// Compute a vector of weights for faces adjacent to the boundary.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] reduction_maps: reduction maps
/// @param[in] bd_weight: weight to give boundary adjacent faces
/// @param[out] face_weights: weights for faces
void compute_boundary_face_weights(
    const Mesh<Scalar>& m,
    const ReductionMaps& reduction_maps,
    Scalar bd_weight,
    std::vector<Scalar>& face_weights);


} // namespace Optimization
} // namespace Penner