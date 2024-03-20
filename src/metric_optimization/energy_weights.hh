#pragma once

#include "common.hh"
#include "cone_metric.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "embedding.hh"

/// \file energy_functor.hh
///
/// Methods to weight energies, e.g., per element energies by element area weights

namespace CurvatureMetric {

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
/// @param[in] m: mesh
/// @param[in] log_edge_lengths: log edge length metric for the mesh
/// @param[out] edge_area_weights: weights per edge
VectorX compute_edge_area_weights(const DifferentiableConeMetric& cone_metric);

/// @brief Compute per face area weights for a mesh
///
/// @param[in] m: mesh
/// @param[in] log_edge_lengths: log edge length metric for the mesh
/// @param[out] face_area_weights: weights per face
VectorX compute_face_area_weights(const DifferentiableConeMetric& cone_metric);

/// Compute a vector of weights for faces adjacent to cones.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] reduction_maps: reduction maps
/// @param[in] cone_weight: weight to give cone adjacent faces
/// @param[out] face_weights: weights for faces
//[[deprecated]] void
// compute_cone_face_weights(
//    const Mesh<Scalar> &m,
//    const ReductionMaps &reduction_maps,
//    Scalar cone_weight,
//    std::vector<Scalar> &face_weights);

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


} // namespace CurvatureMetric
