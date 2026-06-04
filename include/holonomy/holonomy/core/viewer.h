
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "optimization/core/viewer.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

/**
 * @brief Collection of viewers.
 * 
 */

namespace Penner {
namespace Holonomy {

using Optimization::generate_doubled_mesh;
using Optimization::generate_FV_halfedge_data;

/**
 * @brief View a frame field on a mesh.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param frame_field: per-face frame directions
 * @param Th_hat: target cone angles at vertices
 * @param mesh_handle: (optional) name for the surface mesh in the viewer
 * @param scale: (optional) glyph scale for displayed frame vectors
 */
void view_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle="",
    Scalar scale=0.005);


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

/**
 * @brief View a vector field on a mesh.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param vector_field: per-face tangent vectors to display
 * @param mesh_handle: (optional) name for the surface mesh in the viewer
 */
void view_vector_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& vector_field,
    std::string mesh_handle="");

/**
 * @brief View a rotation form on a mesh as a per-edge colormap.
 *
 * @param m: underlying mesh
 * @param vtx_reindex: map from mesh vertex indices to original vertex indices
 * @param V: mesh vertices
 * @param rotation_form: per-halfedge rotation angles
 * @param Th_hat: target cone angles at vertices
 * @param mesh_handle: (optional) name for the surface mesh in the viewer
 * @param show: (optional) if true, open the viewer
 */
void view_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& rotation_form,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View the holonomy constraint error of a cone metric as a per-vertex colormap.
 *
 * @param marked_metric: marked Penner cone metric with holonomy constraints
 * @param vtx_reindex: map from mesh vertex indices to original vertex indices
 * @param V: mesh vertices
 * @param mesh_handle: (optional) name for the surface mesh in the viewer
 * @param show: (optional) if true, open the viewer
 */
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


/**
 * @brief View the homology basis loops of a marked cone metric on a mesh.
 *
 * @param marked_metric: marked Penner cone metric containing the homology basis
 * @param vtx_reindex: map from mesh vertex indices to original vertex indices
 * @param V: mesh vertices
 * @param num_homology_basis_loops: (optional) number of basis loops to display; -1 shows all
 * @param mesh_handle: (optional) name for the surface mesh in the viewer
 * @param show: (optional) if true, open the viewer
 */
void view_homology_basis(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    int num_homology_basis_loops=-1,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief Compute per-edge seamless error metrics for a UV parameterization.
 *
 * Returns four per-edge vectors measuring how far the parameterization deviates
 * from seamlessness across each interior edge.
 *
 * @param F: mesh faces (vertex indices)
 * @param uv: UV vertex positions
 * @param F_uv: UV faces (uv vertex indices, parallel to F)
 * @return tuple of (translation error, rotation error, period jump error, total error)
 */
std::tuple<VectorX, VectorX, VectorX, VectorX> compute_seamless_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/**
 * @brief View parameterization quality metrics as per-face colormaps.
 *
 * Displays distortion measures (e.g., conformal and isometric distortion) to
 * assess the quality of a UV parameterization.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: uv vertex positions
 * @param FT: uv faces
 * @param mesh_handle: (optional) name for the surface mesh in the viewer
 * @param show: (optional) if true, open the viewer
 */
void view_parameterization_quality(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief Compute the per-corner angle error of a UV parameterization.
 *
 * Measures how much the UV angles deviate from being quantized to multiples
 * of Pi/2.
 *
 * @param F: mesh faces (vertex indices)
 * @param uv: UV vertex positions
 * @param F_uv: UV faces (uv vertex indices, parallel to F)
 * @return per-corner angle errors
 */
VectorX compute_angle_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/**
 * @brief Compute the Euclidean length of a UV edge.
 *
 * @param uv_0: first UV coordinate
 * @param uv_1: second UV coordinate
 * @return length of the segment from uv_0 to uv_1
 */
Scalar compute_uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

/**
 * @brief Compute the cosine of the angle at uv_0 in the triangle (uv_0, uv_1, origin).
 *
 * @param uv_0: vertex at which the angle is measured
 * @param uv_1: opposite vertex
 * @return cosine of the angle at uv_0
 */
Scalar uv_cos_angle(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

/**
 * @brief Compute the angle at uv_0 in the triangle (uv_0, uv_1, origin).
 *
 * @param uv_0: vertex at which the angle is measured
 * @param uv_1: opposite vertex
 * @return angle in radians at uv_0
 */
Scalar uv_angle(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

} // namespace Holonomy
} // namespace Penner
