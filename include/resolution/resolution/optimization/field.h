#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/voronoi.h"
#include "field/intrinsic_field.h"
#include "field/frame_field.h"

namespace Penner {
namespace Resolution {

/**
 * @brief Generate an overlay mesh with a uniformly sampled uniform triangulation
 * 
 * @param V: original mesh vertices
 * @param F: original mesh faces
 * @param ratio: approximate ratio of simplified to original mesh vertices
 * @return overlay mesh with both original and simplified embedded meshes
 */
VoronoiOverlayMesh generate_uniform_triangulation(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Scalar ratio = 1.);

/**
 * @brief Given an overlay mesh with a simplified mesh, construct an intrinsic
 * cross field on the simplified geometry
 * 
 * @param voronoi_mesh: overlay mesh
 * @param V_simp: vertices of the embedded simplified mesh
 * @param F_simp: faces of the embedded simplified mesh
 * @param field_params: parameters for the field generation
 * @return intrinsic cross field on the simplified mesh
 */
Field::IntrinsicNRosyField compute_simplified_field(
    const VoronoiOverlayMesh& voronoi_mesh,
    const Eigen::MatrixXd& V_simp,
    const Eigen::MatrixXi& F_simp,
    const Field::FieldParameters& field_params);

/**
 * @brief Given an overlay mesh and a rotation form on the simplified embedded mesh,
 * generate a topologyically equivalent intrinsic field on the original mesh
 * 
 * WARNING: the per-face directions are not yet initialized (since they are irrelevant for topology)
 * 
 * @param voronoi_mesh: overlay mesh
 * @param rotation_form_simp: rotation form on the simplified geometry
 * @param field_generator: output field on the original geometry
 */
void lift_simplified_field_to_original(
    const VoronoiOverlayMesh& voronoi_mesh,
    const VectorX& rotation_form_simp,
    Field::IntrinsicNRosyField& field_generator);

/**
 * @brief Given an overlay mesh and an intrinsic cross field on the simplified mesh,
 * generate sampled field directions on the faces of the overlay mesh
 * 
 * @param voronoi_mesh: overlay mesh
 * @param field_generator: field on the simplified embedded mesh
 * @param V_simp: vertices of the embedded simplified mesh
 * @param F_simp: faces of the embedded simplified mesh
 * @return sampled field directions on the overlay geometry
 */
Eigen::MatrixXd lift_simplified_directions_to_overlay(
    const VoronoiOverlayMesh& voronoi_mesh,
    const Field::IntrinsicNRosyField& field_generator,
    const Eigen::MatrixXd& V_simp,
    const Eigen::MatrixXi& F_simp);

/**
 * @brief Given an overlay mesh and a field on the simplified embedded mesh,
 * generate a topologically equivalent intrinsic field on the original mesh
 * with approximately equal field directions.
 * 
 * @param voronoi_mesh: overlay mesh
 * @param vtx_reindex: map from halfedge to VF vertices
 * @param V: original mesh vertices
 * @param F: original mesh faces
 * @param rotation_form: rotation form on the simplified geometry
 * @param reference_field_simp: per-face referecne vector field on the simplified mesh
 * @param theta_simp: per-face tangent rotation angles for the field relative to the reference
 * @param kappa_simp: per-corner rotation of the reference frame field across edges opposite the corner
 * @param period_jump_simp: per-corner period jumps of the cross field across edges opposite the corner
 * @param smooth_iters: (optional) number of iterations of field direction smoothing
 * @return frame field on the original mesh
 */
std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi
>
lift_simplified_field_to_original(
    const VoronoiOverlayMesh& voronoi_mesh,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& rotation_form,
    const Eigen::MatrixXd& reference_field_simp,
    const Eigen::VectorXd& theta_simp,
    const Eigen::MatrixXd& kappa_simp,
    const Eigen::MatrixXi& period_jump_simp,
    int smooth_iters=10);

void comb_field_to_cut(
    const Mesh<Scalar>& m,
    const std::vector<bool> is_cut,
    Eigen::VectorXd& thetas,
    Eigen::VectorXi& period_jumps);

void lift_basis_loops(
    const VoronoiOverlayMesh& voronoi_mesh,    
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& rotation_form,
    std::vector<std::unique_ptr<DualLoop>>& basis_loops,
    std::vector<Scalar>& kappa);

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
refine_overlay_field(
    const VoronoiOverlayMesh& voronoi_mesh,    
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F,
    const std::vector<int>& fn_to_f_o,
    const Eigen::MatrixXi& hn_to_h_o,
    const Eigen::MatrixXi& is_edge_o,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump);

} // namespace Resolution
} // namespace Penner
