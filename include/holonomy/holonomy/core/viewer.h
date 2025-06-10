
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "optimization/core/viewer.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

using Optimization::generate_doubled_mesh;
using Optimization::generate_FV_halfedge_data;

void view_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle="");

/**
 * @brief View a cross field on a mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @param kappa: per-corner rotation angle of the reference direction field across the opposite edge
 * @param period_jump: per-corner period jump of the cross field across the opposite edge
 * @param mesh_handle: (optional) name for edge mesh
 */
void view_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    std::string mesh_handle = "intrinsic_field_mesh");


void view_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& rotation_form,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle="",
    bool show=true);

void view_constraint_error(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View a parametrization with seamless error colormaps.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: uv vertices
 * @param F_uv: uv faces
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_seamless_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle="",
    bool show=true);


void view_homology_basis(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    int num_homology_basis_loops=-1,
    std::string mesh_handle="",
    bool show=true);

std::tuple<VectorX, VectorX, VectorX, VectorX> compute_seamless_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

void view_parameterization_quality(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle="",
    bool show=true);

VectorX compute_angle_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

Scalar compute_uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

Scalar uv_cos_angle(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

Scalar uv_angle(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

} // namespace Holonomy
} // namespace Penner
