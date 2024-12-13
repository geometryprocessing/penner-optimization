#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/field.h"
#include "holonomy/holonomy/rotation_form.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Parameters for marked metric construction
 *
 */
struct MarkedMetricParameters
{
    typedef HomotopyBasisGenerator::Weighting Weighting;

    bool use_initial_zero = false; // use initial zero Penner coordinates
    bool use_log_length = false; // use initial log length coordinates instead of Penner
    bool remove_loop_constraints = false; // don't set dual loop holonomy constraints if true
    int max_loop_constraints = -1; // set maximum number of loop constraints if positive
    int max_boundary_constraints = -1; // set maximum number of boundary constraints if positive
    Weighting weighting = Weighting::minimal_homotopy; // weighting for tree-cotree
    bool remove_symmetry = false; // remove symmetry structure from doubled mesh
    bool free_interior = false; // remove interior cone constraints
    bool remove_trivial_torus = true; // remove loop constraints from trivial torus to make independent
};

/**
 * @brief Generate a mesh with metric from a VF mesh and cones.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: mesh metric vertices
 * @param F_uv: mesh metric faces
 * @param Th_hat: per-vertex cone angles
 * @param free_cones: (optional) list of cones to leave free
 * @return mesh with metric
 * @return vertex reindexing from the halfedge to VF vertices
 */
std::tuple<Mesh<Scalar>, std::vector<int>>
generate_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    std::vector<int> free_cones={});

/**
 * @brief Generate a marked metric from a VF mesh, cones, and rotation form.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: mesh metric vertices
 * @param F_uv: mesh metric faces
 * @param Th_hat: per-vertex cone angles
 * @param rotation_form: per-halfedge rotation values
 * @param free_cones: list of cones to leave free
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return marked cone metric
 * @return vertex reindexing from the halfedge to VF vertices
 */
std::tuple<MarkedPennerConeMetric, std::vector<int>> generate_marked_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    const VectorX& rotation_form,
    std::vector<int> free_cones,
    MarkedMetricParameters marked_mesh_params = MarkedMetricParameters());

/**
 * @brief Generate a marked metric from a VF mesh using the embedding metric and holonomy
 * constraints inferred from a fit cross-field.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param use_intrinsic: (optional) if true, use intrinsic instead of extrinsic frame field
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return marked cone metric
 * @return vertex reindexing from the halfedge to VF vertices
 * @return per-halfedge rotation form
 * @return per-vertex cone angles
 */
std::tuple<MarkedPennerConeMetric, std::vector<int>, VectorX, std::vector<Scalar>>
infer_marked_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool use_intrinsic=false,
    MarkedMetricParameters marked_mesh_params = MarkedMetricParameters());

/**
 * @brief Generate an intrinsic refined marked metric from a VF mesh using the embedding metric and
 * holonomy constraints inferred from a fit cross-field on the refined mesh
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param min_angle: (optional) minimum angle allowed for the intrinsic refinement (may not converge
 * above 30)
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return refined marked cone metric
 * @return per-halfedge rotation form
 * @return per-vertex cone angles
 */
std::tuple<MarkedPennerConeMetric, VectorX, std::vector<Scalar>> generate_refined_marked_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double min_angle = 25.,
    MarkedMetricParameters marked_mesh_params = MarkedMetricParameters());

/**
 * @brief Generate a marked metric from a halfedge mesh and rotation form.
 *
 * @param m: mesh with metric and cones
 * @param rotation_form: per-halfedge rotation values
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return marked cone metric
 */
MarkedPennerConeMetric generate_marked_metric_from_mesh(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_mesh_params = MarkedMetricParameters(),
    std::vector<int> marked_halfedges = {});

/**
 * @brief Generate a similarity metric from a VF mesh, cones, and rotation form.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: mesh metric vertices
 * @param F_uv: mesh metric faces
 * @param Th_hat: per-vertex cone angles
 * @param rotation_form: per-halfedge rotation values
 * @param free_cones: list of cones to leave free
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return similarity metric
 * @return vertex reindexing from the halfedge to VF vertices
 */
std::tuple<SimilarityPennerConeMetric, std::vector<int>> generate_similarity_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    const VectorX& rotation_form,
    std::vector<int> free_cones,
    MarkedMetricParameters marked_mesh_params = MarkedMetricParameters());

/**
 * @brief Generate a similarity metric from a halfedge mesh and rotation form.
 *
 * @param m: mesh with metric and cones
 * @param rotation_form: per-halfedge rotation values
 * @param marked_mesh_params: (optional) parameters for the marked mesh construction
 * @return similarity metric
 */
SimilarityPennerConeMetric generate_similarity_metric_from_mesh(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_mesh_params = MarkedMetricParameters());

/**
 * @brief Regularize the metric to have bounded triangle quality.
 *
 * @param marked_metric: metric to regularize
 * @param max_triangle_quality: (optional) maximum allowed triangle quality measure
 */
void regularize_metric(MarkedPennerConeMetric& marked_metric, double max_triangle_quality = 50);

/**
 * @brief Regularize the metric to have bounded triangle quality by using gradient descent.
 * 
 * WARNING: Works poorly and distorts metric substantially.
 *
 * @param marked_metric: metric to regularize
 * @param max_triangle_quality: (optional) maximum allowed triangle quality measure
 */
void optimize_triangle_quality(MarkedPennerConeMetric& marked_metric, double max_triangle_quality = 50);

VectorX generate_penner_coordinates(const Mesh<Scalar>& m);

void generate_basis_loops(
    const Mesh<Scalar>& m,
    std::vector<std::unique_ptr<DualLoop>>& basis_loops,
    MarkedMetricParameters marked_metric_params);

std::vector<int> extend_vtx_reindex(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex
);

std::tuple<VectorX, std::vector<Scalar>> generate_intrinsic_rotation_form(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const FieldParameters& field_params);

std::vector<Scalar> compute_kappa(
    const Mesh<Scalar>& discrete_metric,
    const VectorX& rotation_form,
    const std::vector<std::unique_ptr<DualLoop>>& basis_loops);

} // namespace Holonomy
} // namespace Penner