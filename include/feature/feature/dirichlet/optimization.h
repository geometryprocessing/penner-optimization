#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"

namespace Penner {
namespace Feature {

/**
 * @brief Solve for a metric with angle and feature alignment constraints with
 * relaxed aligned edges that are only sofly enforced.
 * 
 * This method first ensures the relaxed constraints are satisfied to machine precision,
 * and then attempts to reduce for the full constraint error as much as possible without
 * degenerating the metric.
 * 
 * @param initial_dirichlet_metric: metric with full alignment constraints
 * @param alg_params: parameters for the optimization
 * @return output aligned metric
 */
DirichletPennerConeMetric optimize_relaxed_angles(
		const DirichletPennerConeMetric& initial_dirichlet_metric,
		const Holonomy::NewtonParameters& alg_params);

// TODO: experimental method to reduce the number of relaxed edges
std::vector<std::pair<int, int>> reduce_relaxed_edges(
    DirichletPennerConeMetric& relaxed_dirichlet_metric,
    const std::vector<std::pair<int, int>>& initial_relaxed_edges,
    Holonomy::NewtonParameters alg_params,
		int num_reductions=10);

} // namespace Feature
} // namespace Penner