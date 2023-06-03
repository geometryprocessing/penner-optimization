#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {


/// Generate the Lagrangian system Lx = b for the least squares solution to the halfedge
/// translations in the hyperbolic metric needed to satisfy the per halfedge shear
/// change and face zero sum condition.
///
/// The zero sum face condition is necessary to extend the edge translations
/// to a projective transformation on the entire face.
///
/// @param[in] m: mesh
/// @param[in] halfedge_shear_change: constraint for the change in shear per halfedge
/// @param[out] lagrangian_matrix: matrix L defining the lagrangian system
/// @param[out] right_hand_side: vector b defining the right hand side of the lagrangian system
void
generate_translation_lagrangian_system(const Mesh<Scalar>& m,
                                       const VectorX& halfedge_shear_change,
                                       MatrixX& lagrangian_matrix,
                                       VectorX& right_hand_side);

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
void
compute_as_symmetric_as_possible_translations(
  const Mesh<Scalar>& m,
  const VectorX& he_metric_coords,
  const VectorX& he_metric_target,
  VectorX& he_translations
);

#ifdef PYBIND
std::tuple<MatrixX, VectorX>
generate_translation_lagrangian_system_pybind(const Mesh<Scalar>& m,
                                              const VectorX& halfedge_shear_change);

VectorX
compute_as_symmetric_as_possible_translations_pybind(
  const Mesh<Scalar>& m,
  const VectorX& he_metric_coords,
  const VectorX& he_metric_target
);

#endif

}
