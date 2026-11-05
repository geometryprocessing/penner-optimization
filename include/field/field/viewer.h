// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "metric/viewer.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

/**
 * @brief Collection of viewers.
 * 
 */

namespace Penner {
namespace Field {

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

} // namespace Field
} // namespace Penner