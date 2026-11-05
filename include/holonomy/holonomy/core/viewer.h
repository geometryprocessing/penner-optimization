// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "metric/viewer.h"
#include "field/viewer.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

/**
 * @brief Collection of viewers.
 * 
 */

namespace Penner {
namespace Holonomy {

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