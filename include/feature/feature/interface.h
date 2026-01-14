#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "feature/feature/features.h"
#include "holonomy/field/intrinsic_field.h"
#include "feature/core/vf_corners.h"
#include "holonomy/interface.h"

namespace Penner {
namespace Feature {

/**
 * @brief Find features on a mesh with dihedral angle heuristics, and generate a refined
 * mesh suitable for seamless parameterization.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param use_minimal_forest: (optional) if true, use minimal feature forest instead of spanning tree
 * @param target_angle: (optional) dihedral angle for features
 * @return refined vertex positions
 * @return refined face indices
 * @return refined feature edges
 * @return refined feature forest edges (spanning or minimal)
 * @return map from refined faces to parent faces
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<VertexEdge>, std::vector<VertexEdge>, std::vector<int>> generate_refined_feature_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool use_minimal_forest=false,
    Scalar target_angle=60.);

/**
 * @brief Generate a parameterization aligned to features and the given cross field.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param feature_edges: feature edges to softly align to coordinate axes
 * @param hard_feature_edges: feature edges to hard align to coordinate axes
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @param kappa: per-corner rotation angle of the reference direction field across the opposite edge
 * @param period_jump: per-corner period jump of the cross field across the opposite edge
 * @param alg_params: parameters for Newton method
 * @return aligned mesh vertices
 * @return aligned mesh faces
 * @return aligned mesh uv vertices
 * @return aligned mesh uv faces
 * @return map from refined aligned mesh faces to original faces
 * @return map from refined aligned vertices to original edge endpoints
 */
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
generate_feature_aligned_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<VertexEdge>& feature_edges,
    const std::vector<VertexEdge>& hard_feature_edges,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    const NewtonParameters& alg_params);

/**
 * @brief Generate an intrinsic metric aligned to features and the given cross field.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param feature_edges: feature edges to softly align to coordinate axes
 * @param hard_feature_edges: feature edges to hard align to coordinate axes
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @param kappa: per-corner rotation angle of the reference direction field across the opposite edge
 * @param period_jump: per-corner period jump of the cross field across the opposite edge
 * @param alg_params: parameters for Newton method
 * @return per-corner aligned metric coordinates corresponding to opposite halfedges
 */
Eigen::MatrixXd generate_feature_aligned_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<VertexEdge>& feature_edges,
    const std::vector<VertexEdge>& hard_feature_edges,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    const NewtonParameters& alg_params);

/**
 * @brief Generate a dirichlet metric from a VF mesh, cones, and rotation form with
 * fixed boundary length constraints
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: mesh metric vertices
 * @param F_uv: mesh metric faces
 * @param Th_hat: per-vertex cone angles
 * @param rotation_form: per-halfedge rotation values
 * @param free_cones: list of cones to leave free
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return dirichlet metric
 * @return vertex reindexing from the halfedge to VF vertices
 */
std::tuple<DirichletPennerConeMetric, std::vector<int>> generate_dirichlet_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    const VectorX& rotation_form,
    std::vector<int> free_cones,
    MarkedMetricParameters marked_mesh_params);

/**
 * @brief Generate a dirichlet metric from a halfedge mesh and rotation form.
 *
 * @param m: mesh with metric and cones
 * @param rotation_form: per-halfedge rotation values
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return dirichlet metric
 */
DirichletPennerConeMetric generate_dirichlet_metric_from_mesh(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_mesh_params);

/**
 * @brief Class to handle aligned metric generation with 
 * 
 */
class AlignedMetricGenerator
{
public:
    AlignedMetricGenerator(
        const Eigen::MatrixXd& V_,
        const Eigen::MatrixXi& F_,
        const std::vector<VertexEdge>& feature_edges_,
        const std::vector<VertexEdge>& hard_feature_edges_,
        const Eigen::MatrixXd& reference_field_,
        const Eigen::VectorXd& theta_,
        const Eigen::MatrixXd& kappa_,
        const Eigen::MatrixXi& period_jump_,
        MarkedMetricParameters marked_metric_params,
        Scalar regularization_factor=1.,
        bool use_minimal_forest=false);

    void make_minimal_forest();

    void optimize_full(const NewtonParameters& alg_params);

    void optimize_relaxed(const NewtonParameters& alg_params);

    Scalar compute_error() const;

    // get the output metric
    Eigen::MatrixXd get_metric() const;

    void parameterize(bool use_high_precision=false);

    // get the output parameterization
    std::tuple<
        Eigen::MatrixXd,
        Eigen::MatrixXi,
        Eigen::MatrixXd,
        Eigen::MatrixXi,
        std::vector<int>,
        std::vector<std::pair<int, int>>>
    get_parameterization();

    // get the refined field on the output refined mesh with parameterization
    std::tuple<
        Eigen::MatrixXd,
        Eigen::VectorXd,
        Eigen::MatrixXd,
        Eigen::MatrixXi>
    get_refined_field();

    std::tuple<std::vector<FaceEdge>, std::vector<FaceEdge>> get_refined_features();

private:
    bool parameterized;

    // input
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<VertexEdge> feature_edges;
    std::vector<VertexEdge> hard_feature_edges;
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;

    // cut mesh
    Eigen::MatrixXd V_cut;
    Eigen::MatrixXi F_cut;
    Eigen::VectorXi V_map;
    Eigen::MatrixXi F_is_hard_feature, F_is_feature;
    Eigen::MatrixXi F_is_soft_feature;

    DirichletPennerConeMetric embedding_metric;
    DirichletPennerConeMetric dirichlet_metric;
    std::vector<int> vtx_reindex;
    std::vector<int> face_reindex;
    VectorX rotation_form;
    std::vector<Scalar> Th_hat;

    // optimized metric
    DirichletPennerConeMetric opt_dirichlet_metric;
    VectorX opt_metric_coords;

    // refined parameterization
    Eigen::MatrixXd V_r;
    Eigen::MatrixXi F_r;
    Eigen::MatrixXd uv_r;
    Eigen::MatrixXi FT_r;
    std::vector<int> fn_to_f_r;
    std::vector<std::pair<int, int>> endpoints_r;

    // refined feature edges
    std::vector<FaceEdge> feature_edges_r, misaligned_edges_r;

    // refined cross field
    Eigen::MatrixXd reference_field_r;
    Eigen::VectorXd theta_r;
    Eigen::MatrixXd kappa_r;
    Eigen::MatrixXi period_jump_r;
};


} // namespace Feature
} // namespace Penner
