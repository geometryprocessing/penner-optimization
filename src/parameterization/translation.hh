#pragma once

#include "common.hh"

namespace CurvatureMetric {

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
void compute_as_symmetric_as_possible_translations(
    const Mesh<Scalar>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_translations);
// TODO: Add option to bypass and use zero translations or to solve in double precision

} // namespace CurvatureMetric
