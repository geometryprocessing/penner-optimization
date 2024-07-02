#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace PennerHolonomy {

/**
 * @brief Compute the holonomy angle with respect to a metric along a dual segment.
 *
 * @param m: Delaunay mesh
 * @param he2angle: map from halfedges to opposing angle in a given metric
 * @param dual_segment: dual segment in a face
 * @return holonomy of the dual segment
 */
Scalar compute_dual_segment_holonomy(
    const Mesh<Scalar>& m,
    const VectorX& he2angle,
    const DualSegment& dual_segment);

/**
 * @brief Compute the holonomy angle with respect to a metric along a dual loop.
 *
 * @param m: Delaunay mesh
 * @param he2angle: map from halfedges to opposing angle in a given metric
 * @param dual_loop: vector of dual segments specifying a dual loop
 * @return holonomy of the dual loop
 */
Scalar compute_dual_loop_holonomy(
    const Mesh<Scalar>& m,
    const VectorX& he2angle,
    const DualLoop& dual_loop);

/**
 * @brief Compute the rotation angle with respect to a rotation form along a dual loop.
 *
 * @param m: mesh
 * @param rotation_form: map from halfedges to rotation across that edge
 * @param dual_loop: vector of dual segments specifying a dual loop
 * @return rotation along dual loop
 */
Scalar compute_dual_loop_rotation(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    const DualLoop& dual_loop);

} // namespace PennerHolonomy