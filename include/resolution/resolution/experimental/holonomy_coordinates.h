#pragma once

#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

/**
 * @brief Update the per-corner holonomy coordinates to reflect holonomy after the
 * counter-clockwise flip of the given edge
 * 
 * @param m: halfedge mesh
 * @param alpha: holonomy coordinates before clip
 * @param halfedge_index: edge to be flipped
 */
void update_holonomy_coordinates_under_ccw_flip(
	const Mesh<Scalar>& m,
	VectorX& alpha,
	int halfedge_index);

/**
 * @brief Update the per-corner holonomy coordinates to reflect holonomy after the
 * clockwise flip of the given edge
 * 
 * @param m: halfedge mesh
 * @param alpha: holonomy coordinates before clip
 * @param halfedge_index: edge to be flipped clockwise
 */
void update_holonomy_coordinates_under_clockwise_flip(
	const Mesh<Scalar>& m,
	VectorX& alpha,
	int halfedge_index);

} // namespace Resolution
} // namespace Penner