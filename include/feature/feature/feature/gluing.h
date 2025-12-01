
#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Given a mesh with vertex pairings arising from cuts, generate pairings of cut edges.
 * 
 * @param m: halfedge mesh
 * @param vtx_reindex: halfedge to VF vertex index map
 * @param V_map: vertex identifications in the VF mesh
 * @return list of paired halfedges across cuts
 * @return list of all boundary halfedges
 * @return map from all halfedges to boundary halfedge index (-1 if interior)
 */
std::tuple<
    std::vector<std::pair<int, int>>,
    std::vector<int>,
    std::vector<int>
>
generate_boundary_pairs(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);
		
/**
 * @brief Compute the target cone angles of the mesh after gluing feature edges.
 *
 * @param m: cut mesh
 * @param vtx_reindex: cut mesh to cut VF reindexing
 * @param V_map: map from cut VF vertices to original VF vertices
 * @return vector of glued angles
 */
std::vector<Scalar> compute_glued_angles(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

/**
 * @brief Compute the target cone angle defects of the mesh after gluing feature edges.
 *
 * @param m: cut mesh
 * @param vtx_reindex: cut mesh to cut VF reindexing
 * @param V_map: map from cut VF vertices to original VF vertices
 * @return vector of glued angle defects
 */
std::vector<Scalar> compute_glued_angle_defects(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

/**
 * @brief Determine how many cones are in the mesh after gluing feature edges
 *
 * @param m: cut mesh
 * @param vtx_reindex: cut mesh to cut VF reindexing
 * @param V_map: map from cut VF vertices to original VF vertices
 * @return number of negative and positive cones
 */
std::pair<int, int> count_glued_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

} // namespace Feature
} // namespace Penner