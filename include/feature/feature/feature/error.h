
#pragma once

#include "feature/core/common.h"
#include "util/vf_corners.h"

namespace Penner {
namespace Feature {

/**
 * @brief Compute how aligned an edge is with a coordinate axis in normalized
 * angle coordinates (between 0 for fully aligned and 1 for unaligned)
 *
 * @param d: edge direction
 * @return alignment measure
 */
double compute_edge_alignment(const Eigen::Vector2d& d);

/**
 * @brief Compute the alignment for a halfedge indexed by the opposite corner
 *
 * @param uv: parametrization vertices
 * @param F_uv: parametrization faces
 * @param corner: face index and local corner index of a corner opposite a halfedge
 * @return masked alignment error for halfedge opposite corner
 */
double compute_corner_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::pair<int, int>& corner);

/**
 * @brief Compute the alignment for all halfedges, indexed by corners (with a mask).
 *
 * @param uv: parametrization vertices
 * @param F_uv: parametrization faces
 * @param F_is_aligned: mask for corners; set alignment error to 0 if corner (f, i) is 0
 * @return masked alignment error for halfedges opposite corners
 */
Eigen::MatrixXd compute_mask_uv_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_aligned);

/**
 * @brief Compute the alignment for a list of corners opposite halfedges
 *
 * @param uv: parametrization vertices
 * @param F_uv: parametrization faces
 * @param aligned_corners: list of corners opposite halfedges
 * @return alignment error for halfedges opposite given corners
 */
std::vector<double> compute_uv_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::pair<int, int>>& aligned_halfedges);

/**
 * @brief Compute the alignment for edges in a mesh as the maximum of both cut edge
 * alignments in the uv connectivity. 
 * 
 * @param F: mesh faces
 * @param uv: mesh uv vertices
 * @param F_uv: mesh uv faces
 * @param E: feature edges (represented by vertex endpoints in F)
 * @return alignment of all feature edges
 */
std::vector<double> compute_feature_alignment(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::array<int, 2>>& E);

/**
 * @brief Prune a list of corners opposite misaligned halfedges.
 * 
 * @param uv: mesh uv vertices
 * @param corners: list of corners to prune
 * @return pruned list of aligned corners
 * @return complementary list of removed misaligned corners
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>
prune_misaligned_corners(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::pair<int, int>>& corners);

/**
 * @brief Prune a list of edges represented as the edge endpoints.
 * 
 * @param uv: mesh uv vertices
 * @param edges: list of edges (represented by endpoints)
 * @return pruned list of aligned edges
 * @return complementary list of removed misaligned edges
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>
prune_misaligned_edges(const Eigen::MatrixXd& uv, const std::vector<std::pair<int, int>>& edges);

/**
 * @brief Prune a list of edges represented as face edge data.
 * 
 * @param uv: mesh uv vertices
 * @param F_uv: mesh uv faces
 * @param edges: list of edges (represented by the face edge data structure)
 * @return pruned list of aligned edges
 * @return complementary list of removed misaligned edges
 */
std::tuple<std::vector<FaceEdge>, std::vector<FaceEdge>> prune_misaligned_face_edges(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<FaceEdge>& edges,
    Scalar feature_threshold = 1e-10);

/**
 * @brief Check whether all cones in a refined mesh are consistent with the target values.
 * 
 * WARNING: assumes any refined vertices are at the end of V
 * 
 * @param Th_hat: vertex angle targets in the original mesh
 * @param V: refined mesh vertices
 * @param F: refined mesh faces
 * @param uv: refined mesh uv vertices
 * @param FT: refined mesh uv faces
 * @return true iff the original vertex angles are consistent and inserted angles are flat.
 */
bool check_seamless_cones(
    const std::vector<Scalar>& Th_hat,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT);

/**
 * @brief Compute the maximum constraint error per connected component.
 *
 * @param marked_metric: marked mesh
 * @param vertex_component: map from vertices to component id
 * @return per-component function of maximum error
 */
VectorX compute_component_error(
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::VectorXi& vertex_component);

/**
 * @brief Compute the number of flipped faces with exact predicates.
 * 
 * @param uv: mesh uv vertices
 * @param FT: mesh uv faces
 * @return number of flipped faces
 */
int check_flip(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& FT);

/**
 * @brief Compute the height of each triangle in the mesh, relative to each possible base.
 * 
 * @param uv: mesh uv vertices
 * @param FT: mesh uv faces
 * @return |F|x3 matrix of triangle heights relative to the base opposite the local corner
 */
Eigen::MatrixXd compute_height(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& FT);

} // namespace Feature
} // namespace Penner
