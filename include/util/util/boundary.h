#pragma once

#include "util/common.h"
#include "util/vector.h"
#include "util/map.h"

namespace Penner {

/**
 * @brief Circulate halfedge ccw to next halfedge on boundary
 * 
 * @param m: mesh
 * @param halfedge_index: starting halfedge
 * @return next halfedge ccw around the base vertex of the halfedge
 */

int circulate_ccw_to_boundary(const Mesh<Scalar>& m, int halfedge_index);

/**
 * @brief Generate a list of boundary halfedge indices in the primal copy of the mesh.
 * 
 * @param m: mesh
 * @return list of boundary halfedge indices
 */
std::vector<int> find_primal_boundary_halfedges(const Mesh<Scalar>& m);

/**
 * @brief Generate a list of boundary vertex indices in a mesh.
 * 
 * @param m: mesh
 * @return list of boundary vertex indices
 */
std::vector<int> find_boundary_vertices(const Mesh<Scalar>& m);

/**
 * @brief Generate a list of boundary vertex indices in the original VF mesh.
 * 
 * @param m: mesh
 * @param vtx_reindex: map from mesh to original vertex indices
 * @return list of boundary vertex indices in the original mesh
 */
std::vector<int> find_boundary_vertices(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex);

/**
 * @brief Compute the boundary vertices of a mesh.
 *
 * @param m: mesh
 * @return boolean mask of boundary vertices
 */
std::vector<bool> compute_boundary_vertices(const Mesh<Scalar>& m);

/**
 * @brief Generate a list of representative halfedges on each boundary component
 * 
 * @param m: mesh
 * @return list of primal halfedges on the boundaries
 */
std::vector<int> find_boundary_components(const Mesh<Scalar>& m);

/**
 * @brief Generate a list of halfedges on the boundary from a given index
 * 
 * @param m: mesh
 * @param halfedge_index: starting halfedge of the boundary component
 * @return list of primal halfedges on the boundary
 */
std::vector<int> build_boundary_component(const Mesh<Scalar>& m, int halfedge_index);

} // namespace Penner
