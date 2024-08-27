
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Compute vector of one form constraints.
 * 
 * The constraints are vertex holonomy constraints, dual loop holonomy constraints, and
 * closed form constraints.
 * 
 * @param[in] similarity_metric: mesh with similarity metric
 * @param[in] angles: per-corner angles of the metric
 * @return vector of one form constraint errors
 */
VectorX compute_similarity_constraint(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& angles);

/**
 * @brief Compute jacobian of the similarity one form constraints with respect to one form
 * edge values.
 * 
 * @param[in] similarity_metric: mesh with similarity metric
 * @param[in] cotangents: per-corner cotangent angles of the metric
 * @return one form constraint error jacobian matrix
 */
MatrixX compute_similarity_constraint_jacobian(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& cotangents);

/**
 * @brief Compute the similarity one form constraints and the jacobian with respect to one form
 * edge values.
 * 
 * @param[in] similarity_metric: mesh with similarity metric
 * @param[out] constraint: vector of one form constraint errors
 * @param[out] J_constraint: one form constraint error jacobian matrix
 * @param[in] need_jacobian: (optional) only build jacobian if true
 */
void compute_similarity_constraint_with_jacobian(
    const SimilarityPennerConeMetric& similarity_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian=true);

} // namespace Holonomy
} // namespace Penner