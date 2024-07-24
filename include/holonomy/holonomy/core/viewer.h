
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

#ifdef ENABLE_VISUALIZATION
extern glm::vec3 BEIGE;
extern glm::vec3 BLACK_BROWN;
extern glm::vec3 TAN;
extern glm::vec3 MUSTARD;
extern glm::vec3 FOREST_GREEN;
#endif

// TODO Refactor and add some more convenient viewers

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& m);

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

void view_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle="");

void view_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& rotation_form,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle="",
    bool show=true);

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
    const std::vector<bool> is_edge);

void view_triangulation(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& fn_to_f,
    std::string mesh_handle="",
    bool show=true);

void view_constraint_error(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle="",
    bool show=true);

void view_quad_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
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

void view_independent_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& vertex_function,
    std::string mesh_handle="",
    bool show=true);

} // namespace Holonomy
} // namespace Penner