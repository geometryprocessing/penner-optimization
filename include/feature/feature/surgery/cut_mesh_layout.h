#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"

namespace Penner {
namespace Feature {

/**
 * @brief Parametrize a cut mesh with optimized metric.
 * 
 * @tparam Scalar type to use for layout (can use higher precision to improve robustness)
 * @param embedding_metric: halfedge Euclidean metric for original embedding
 * @param original_metric: initial Penner metric before optimization
 * @param marked_metric: final Penner optimized halfedge metric
 * @param V_cut: cut mesh vertices
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param face_reindex: face reindexing from halfedge to VF mesh
 * @param use_uniform_bc: (optional) use uniform barycentric coordinates instead of metric based
 * @param output_dir: (optional) output directory for intermediate layouts
 * @return refined parameterized mesh vertices
 * @return refined parameterized mesh faces
 * @return refined parameterized mesh uv vertices
 * @return refined parameterized mesh uv faces
 * @return map from refined mesh faces to original
 * @return endpoints of refined edge vertices in original mesh
 */
template <typename OverlayScalar>
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
parameterize_cut_mesh(
    const MarkedPennerConeMetric& embedding_metric,
    const MarkedPennerConeMetric& original_metric,
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::MatrixXd& V_cut,
    const std::vector<int>& vtx_reindex,
    const std::vector<int>& face_reindex,
    bool use_uniform_bc = false,
    std::string output_dir="");

/**
 * @brief Rotate parameterizations to align hard features to coordinate axes.
 * 
 * @param uv: uv vertices
 * @param FT: uv faces
 * @param F_is_hard_feature: mask for hard feature edges
 * @return aligned mesh uv vertices
 */
Eigen::MatrixXd align_to_hard_features(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXi& F_is_hard_feature);

/**
 * @brief Combine a parameterization with multiple components into a single patch.
 * 
 * @tparam Scalar type to use for layout (can use higher precision to improve robustness)
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: mesh uv vertices
 * @param F_uv: mesh uv faces
 * @return connected mesh uv vertices
 * @return connected mesh uv faces
 */
template <typename OverlayScalar>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_connected_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

} // namespace Feature
} // namespace Penner