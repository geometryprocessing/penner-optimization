
#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"

namespace Penner {
namespace Feature {

/**
 * @brief Compute the boundary length constraint function for a Dirichlet metric
 * 
 * @param dirichlet_metric: mesh with dirichlet metric structure
 * @return vector of boundary path length errors
 */
VectorX compute_boundary_constraint(DirichletPennerConeMetric& dirichlet_metric);

/**
 * @brief Compute the Jacobian of the length constraint function for a Dirichlet metric
 * with respect to the metric coordinates.
 * 
 * @param dirichlet_metric: mesh with dirichlet metric structure
 * @return constraint jacobian matrix
 */
MatrixX compute_boundary_constraint_jacobian(DirichletPennerConeMetric& dirichlet_metric);

/**
 * @brief Compute the holonomy and boundary length constraint function for a Dirichlet metric
 * 
 * @param dirichlet_metric: mesh with dirichlet metric structure
 * @param angles: per-corner angles of the metric
 * @return vector of all dirichlet constraint errors
 */
VectorX compute_dirichlet_constraint(
    DirichletPennerConeMetric& dirichlet_metric,
    const VectorX& angles);

/**
 * @brief Compute the Jacobian of the holonomy and boundary length constraint function
 * for a Dirichlet metric with respect to the metric coordinates.
 * 
 * @param dirichlet_metric: mesh with dirichlet metric structure
 * @param cotangents: per-corner cotangent angles of the metric
 * @return constraint jacobian matrix
 */
MatrixX compute_dirichlet_constraint_jacobian(
    DirichletPennerConeMetric& dirichlet_metric,
    const VectorX& cotangents);

/**
 * @brief Compute the dirichlet constraints and the jacobian with respect to metric coordinates.
 * 
 * @param[in] dirichlet_metric: marked mesh with dirichlet constraints
 * @param[out] constraint: vector of one form constraint errors
 * @param[out] J_constraint: one form constraint error jacobian matrix
 * @param[in] need_jacobian: (optional) only build jacobian if true
 */
void compute_dirichlet_constraint_with_jacobian(
    const DirichletPennerConeMetric& dirichlet_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian);

} // namespace Feature
} // namespace Penner