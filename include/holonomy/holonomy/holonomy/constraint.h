
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Compute dual loop holonomy angles for a mesh with given angles
 * 
 * @param[in] m: mesh topology
 * @param[in] homology_basis_loops: dual loops for holonomy angle computation
 * @param[in] alpha: per-halfedge angles for the mesh
 * @return vector of dual loop holonomy angles
 */
VectorX
Kappa(const Mesh<Scalar>& m, const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops, const VectorX& alpha);
VectorX Kappa(const MarkedPennerConeMetric& marked_metric, const VectorX& alpha);

/**
 * @brief Compute vertex cone holonomy constraints
 * 
 * @param[in] marked_metric: marked mesh with metric
 * @param[in] angles: per-corner angles of the metric
 * @return vector of vertex constraint errors
 */
VectorX compute_vertex_constraint(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles);

/**
 * @brief Compute vertex and dual loop holonomy constraints
 * 
 * @param[in] marked_metric: marked mesh with metric
 * @param[in] angles: per-corner angles of the metric
 * @param[in] only_free_vertices: (optional) only add constraints for free vertices if true
 * @return vector of holonomy constraint errors
 */
VectorX compute_metric_constraint(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles,
    bool only_free_vertices=true);

/**
 * @brief Compute the jacobian of vertex and dual loop holonomy constraints with respect
 * to the metric coordinates
 * 
 * @param[in] marked_metric: marked mesh with metric
 * @param[in] cotangents: per-corner cotangent angles of the metric
 * @param[in] only_free_vertices: (optional) only add constraints for free vertices if true
 * @return holonomy constraint error jacobian matrix
 */
MatrixX compute_metric_constraint_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& cotangents,
    bool only_free_vertices=true);

MatrixX compute_holonomy_matrix(
    const Mesh<Scalar>& m,
    const MatrixX& angle_constraint_system,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops);

MatrixX compute_triangle_corner_angle_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& cotangents);

MatrixX compute_metric_holonomy_matrix(
    const MarkedPennerConeMetric& marked_metric,
    bool only_free_vertices=true);

MatrixX compute_loop_holonomy_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops);
    
MatrixX build_symmetric_matrix_system(const MatrixX& A, int offset, int size);
std::tuple<MatrixX, VectorX> build_reduced_matrix_system(const MatrixX& A, int cols);
//VectorX build_reduced_matrix_rhs(const MatrixX& A);
VectorX build_reduced_matrix_rhs(const Eigen::MatrixXd& A);
MatrixX build_metric_matrix(const Mesh<Scalar>& m);

/**
 * @brief Compute the holonomy constraints and the jacobian with respect to metric coordinates.
 * 
 * @param[in] marked_metric: marked mesh with metric
 * @param[out] constraint: vector of one form constraint errors
 * @param[out] J_constraint: one form constraint error jacobian matrix
 * @param[in] need_jacobian: (optional) only build jacobian if true
 * @param[in] only_free_vertices: (optional) only add constraints for free vertices if true
 */
void compute_metric_constraint_with_jacobian(
    const MarkedPennerConeMetric& similarity_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian=true,
    bool only_free_vertices=true);


void add_vertex_constraints(
    const MarkedPennerConeMetric& marked_metric,
    const MatrixX& angle_constraint_system,
    const VectorX& angles,
    VectorX& constraint,
    int offset = 0);

MatrixX build_free_vertex_system(const Mesh<Scalar>& m);

void add_basis_loop_constraints(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles,
    VectorX& constraint,
    int offset = 0);

MatrixX compute_metric_corner_angle_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& cotangents);

std::tuple<VectorX, MatrixX> compute_metric_constraint_with_jacobian_pybind(
    const MarkedPennerConeMetric& marked_metric);
    
} // namespace Holonomy
} // namespace Penner