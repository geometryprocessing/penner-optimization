// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "holonomy/core/common.h"
#include "field/vector_field.h"
#include "field/rotation_form.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"
#include "metric/interface.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Parameters for marked metric construction
 *
 */
struct MarkedMetricParameters : ConeMetricParameters
{
    typedef HomotopyBasisGenerator::Weighting Weighting;

    bool remove_loop_constraints = false; // don't set dual loop holonomy constraints if true
    int max_loop_constraints = -1; // set maximum number of loop constraints if positive
    int max_boundary_constraints = -1; // set maximum number of boundary constraints if positive
    Weighting weighting = Weighting::minimal_homotopy; // weighting for tree-cotree
    bool remove_trivial_torus = true; // remove loop constraints from trivial torus to make independent
    bool use_connectivity = true; // use connectivity structure for markings
};

/**
 * @brief Generate a parametrization satisfying seamless constraints inferred from
 * a smooth cross-field, optimized with a MIQ method 
 * 
 * The parametrization algorithm is based on a modified newton's method.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param field_params: (optional) parameters for field generation
 * @param alg_params: (optional) parameters for modified Newton method
 * @return parametrized VF mesh with uv coordinates
 * @return cross field direction for u coordinate gradient
 * @return cross field direction for v coordinate gradient
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXd>
parametrize_seamless(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Field::FieldParameters field_params=Field::FieldParameters(),
    NewtonParameters alg_params=NewtonParameters());

/**
 * @brief Generate a parametrization for the metric with given seamless constraints
 * 
 * The parametrization algorithm is based on a modified newton's method.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param marked_metric: metric with differentiable seamless constraints
 * @param alg_params: (optional) parameters for modified Newton method
 * @return parametrized VF mesh with uv coordinates
 * @return map from refined faces to original faces
 * @return map from refined edge vertices to edge endpoints 
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi, std::vector<int>, std::vector<std::pair<int, int>>>
parametrize_seamless_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const MarkedPennerConeMetric& marked_metric,
    NewtonParameters alg_params=NewtonParameters());

/**
 * @brief Generate a parametrization satisfying seamless constraints inferred from the given cross field.
 * 
 * Since the mesh is refined by the parametrization, the cross field also needs to be refined.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param theta: field angles relative to the reference directions
 * @param kappa: angles across edges between reference directions
 * @param period_jump: jump in period across edges
 * @return parametrized VF mesh with uv coordinates
 * @return refined cross field
 * 
 */
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
generate_seamless_parametrization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    NewtonParameters alg_params=NewtonParameters());

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
 * @brief Generate a marked metric object from a mesh with holonomy constraints inferred from a cross field.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param theta: field angles relative to the reference directions
 * @param kappa: angles across edges between reference directions
 * @param period_jump: jump in period across edges
 * @param marked_metric_params: parameters for the metric object
 * @return marked metric for the mesh with field holonomy constraints
 * @return vertex reindexing from the halfedge to VF mesh
 * @return field rotations across halfedges
 * @return inferred cones from the field on the VF mesh
 * 
 */
std::tuple<MarkedPennerConeMetric, std::vector<int>, VectorX, std::vector<Scalar>>
generate_metric_from_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    MarkedMetricParameters marked_metric_params = MarkedMetricParameters());

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

void generate_basis_loops(
    const Mesh<Scalar>& m,
    std::vector<std::unique_ptr<DualLoop>>& basis_loops,
    MarkedMetricParameters marked_metric_params,
    std::vector<int> marked_halfedges={});

std::vector<int> extend_vtx_reindex(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex
);

std::tuple<VectorX, std::vector<Scalar>> generate_intrinsic_rotation_form(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Field::FieldParameters& field_params);

std::vector<Scalar> compute_kappa(
    const Mesh<Scalar>& discrete_metric,
    const VectorX& rotation_form,
    const std::vector<std::unique_ptr<DualLoop>>& basis_loops);

std::tuple<int, int> get_constraint_outliers(
    MarkedPennerConeMetric& marked_metric,
    bool use_interior_vertices=true,
    bool use_flat_vertices=true);
std::tuple<int, int> add_optimal_cone_pair(MarkedPennerConeMetric& marked_metric);

} // namespace Holonomy
} // namespace Penner