
#pragma once

#include "holonomy/core/common.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Load a per-facet frame field with edge matchings from file
 * 
 * @param fname: name of the file to load
 * @return field version number
 * @return per-face crossfield angles
 * @return face vertex ids
 * @return face edge matchings (matching i on edge with base vertex id(i))
 * @return face sharp values
 */
std::tuple<
    int, // version
    Eigen::VectorXd, // crossfield_angles
    Eigen::MatrixXi, // F_id
    Eigen::MatrixXi, // F_matching
    Eigen::MatrixXi> // F_sharp
load_facet_field(const std::string& fname);

/**
 * @brief Given matchings defined on face corners, generate the corresponding halfedge matchings.
 * 
 * WARNING: Assumes that halfedge hij corresponds to the corner of fijk at the tip vj, not the
 * opposite corner vk.
 * 
 * @param m: underlying mesh for the matching
 * @param vtx_reindex: map from halfedge mesh vertex indices to VF
 * @param crossfield_angles: per face angles (only needed for versions before 2)
 * @param F: VF face representation
 * @param F_matchings: per-corner matchings 
 * @param version: version of the facet field represenatation
 * @return list of reference halfedges for base tangent direction per face
 * @return matching for halfedge hij to opposite halfedge hji
 */
std::tuple<Eigen::VectorXi, Eigen::VectorXi> generate_halfedge_from_face_matchings(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXd& crossfield_angles,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_matchings,
    int version=0);

} // namespace Feature 
} // namespace Penner
