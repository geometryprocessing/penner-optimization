#pragma once

#include "common.hh"

namespace CurvatureMetric {

/// Compute the number of connected components of a mesh
///
/// @param[in] F: mesh faces
/// @return number of connected components
int count_components(const Eigen::MatrixXi& F);

/// Given a face index matrix, reindex the vertex indices to removed unreferenced
/// vertex indices in O(|F|) time.
///
/// Note that libigl has function with similar behavior, but it is a O(|V| + |F|)
/// algorithm due to their bookkeeping method
///
/// @param[in] F: initial mesh faces
/// @param[out] FN: reindexed mesh faces
/// @param[out] new_to_old_map: map from new to old vertex indices
/// @return number of connected components
void remove_unreferenced(
    const Eigen::MatrixXi& F,
    Eigen::MatrixXi& FN,
    std::vector<int>& new_to_old_map);

/// Given a mesh with a parametrization, cut the mesh along the parametrization seams to
/// create a vertex set corresponding to the faces of the uv domain.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: parametrization vertices
/// @param[in] FT: parametrization faces
/// @param[in] V: cut mesh vertices
void cut_mesh_along_parametrization_seams(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    Eigen::MatrixXd& V_cut);

} // namespace CurvatureMetric
