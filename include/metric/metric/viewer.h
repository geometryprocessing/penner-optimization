// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "metric/common.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

/**
 * @brief Assorted viewers.
 * 
 */

namespace Penner {

#ifdef ENABLE_VISUALIZATION
extern glm::vec3 BEIGE;
extern glm::vec3 BLACK_BROWN;
extern glm::vec3 TAN;
extern glm::vec3 MUSTARD;
extern glm::vec3 FOREST_GREEN;
extern glm::vec3 TEAL;
extern glm::vec3 DARK_TEAL;
#endif


// TODO Refactor and add some more convenient viewers

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> generate_halfedge_faces(
    const std::vector<int>& next,
    const std::vector<int>& f2h,
    const std::vector<int>& to,
    const std::vector<int>& vtx_reindex);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& m);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& Th_hat);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_closed_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& Th_hat);

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> generate_mesh_faces(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> generate_doubled_mesh(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex);

VectorX generate_FV_halfedge_data(const Eigen::MatrixXi& F_halfedge, const VectorX& halfedge_data);

Eigen::MatrixXd generate_subset_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vertex_indices);

void view_mesh_quality(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle="",
    bool show=true);

void view_mesh_topology(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle="",
    bool show=true);

void view_mesh_uv_topology(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    std::string mesh_handle="",
    bool show=true);

void view_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle="",
    bool show=true);

void view_layout(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle="",
    bool show=true);

void view_dual_graph(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const std::vector<bool> is_edge);

void view_primal_graph(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const std::vector<bool> is_edge,
    std::string handle="",
    bool show=true);

void view_triangulation(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& fn_to_f,
    const std::vector<std::pair<int, int>>& endpoints,
    std::string mesh_handle="",
    bool show=true);

void view_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& vertex_function,
    std::string mesh_handle="",
    bool show=true);
void view_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& vertex_function,
    std::string mesh_handle="",
    bool show=true);

void view_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle="cone mesh",
    bool show=true);

void view_independent_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& vertex_function,
    std::string mesh_handle="",
    bool show=true);

void view_halfedge_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& halfedge_function,
    std::string mesh_handle="",
    bool show=true);

VectorX compute_corner_angles(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
);

/// View the triangles in the mesh with inverted elements.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv vertices
/// @param[in] F_uv: mesh uv faces
void
view_flipped_triangles(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
);

/// View a layout for a mesh with uv coordinates assigned per halfedge.
///
/// @param[in] m: mesh
/// @param[in] u_vec: per-halfedge u coordinates
/// @param[in] v_vec: per-halfedge v coordinates
void
view_halfedge_mesh_layout(
  const Mesh<Scalar>& m,
  const std::vector<Scalar>& u_vec,
  const std::vector<Scalar>& v_vec
);
void view_halfedge_mesh_layout(
    const std::vector<int>& next,
    const std::vector<int>& f2h,
    const std::vector<Scalar>& u_vec,
    const std::vector<Scalar>& v_vec);

} // namespace Penner