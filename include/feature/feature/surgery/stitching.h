
#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Stitch a refined cut mesh with overlay data and feature corner mask into a
 * closed refined mesh with the corresponding overlay and feature data.
 * 
 * @param V: refined cut mesh vertices
 * @param F: refined cut mesh faces
 * @param uv: refined cut mesh uv vertices
 * @param F_uv: refined cut mesh uv faces
 * @param Fn_to_F: map from refined faces to original faces
 * @param endpoints: map from refined vertices to original edge endpoints
 * @param is_feature_corner: mask for refined mesh corners opposite feature edges
 * @param V_map: gluing map from original cut mesh vertices to closed mesh vertices
 * @param use_uniform_bc: (optional) use uniform interpolation instead of uv metric
 * @return stitched mesh with overlay data and feature corner mask
 */
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>,
    Eigen::MatrixXi>
stitch_cut_overlay(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::MatrixXi& is_feature_corner,
    const Eigen::VectorXi& V_map,
    bool use_uniform_bc=false);

} // namespace Feature
} // namespace Penner