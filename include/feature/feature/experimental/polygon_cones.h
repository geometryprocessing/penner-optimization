#pragma once

#include "feature/core/common.h"


namespace Penner {
namespace Feature {


/**
 * @brief Generate a cone angle prescription for a topological disk mapping it to a regular polygon.
 *
 * If more vertices are requested than possible, all boundary vertices are used. By default,
 * edges are equally spaced according to boundary index position; approximately equal edge
 * lengths can be used, but the number of vertices is not guaranteed to agree with the input.
 *  
 * @param m: underlying mesh (must be topological disk)
 * @param vtx_reindex: reindexing from the mesh to the VF vertex indices
 * @param num_vertices: number of vertices in the target polygon (or 0 for all)
 * @return cone prescription for m with target boundary shape
 */
std::vector<Scalar> generate_polygon_cones(
	const Mesh<Scalar>& m,
	const std::vector<int>& vtx_reindex,
	int num_vertices=0,
	bool use_length=false);

} // namespace Feature
} // namespace Penner