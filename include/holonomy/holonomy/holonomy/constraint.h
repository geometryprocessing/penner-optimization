
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace PennerHolonomy {

/**
 * @brief Compute vertex holonomy angles for a mesh with given angles
 *
 * @param[in] m: mesh topology
 * @param[in] alpha: per-halfedge angles for the mesh
 * @return vector of vertex holonomy angles
 */
VectorX Theta(const Mesh<Scalar>& m, const VectorX& alpha);

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
    const std::vector<int>& v_map,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops,
    int num_vertex_forms);
    
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
    const std::vector<int> v_map,
    const VectorX& angles,
    VectorX& constraint,
    int offset = 0);

void add_basis_loop_constraints(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles,
    VectorX& constraint,
    int offset = 0);
    
} // namespace PennerHolonomy