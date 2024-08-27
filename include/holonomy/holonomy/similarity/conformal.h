#pragma once

#include "holonomy/core/common.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Compute a conformally equivalent similarity metric satisfying holonomy constraints.
 *
 * In order to satisfy dual loop constraints, this method computes a scaling one form rather than
 * a scaling zero form that must be integrated first, potentially resulting in jumps.
 *
 * @param similarity_metric: similarity metric structure with holonomy constraints
 * @param alg_params: global parameters for the algorithm
 * @param ls_params: parameters for the line search
 */
void compute_conformal_similarity_metric(
    SimilarityPennerConeMetric& similarity_metric,
    const AlgorithmParameters& alg_params,
    const LineSearchParameters& ls_params);

} // namespace Holonomy
} // namespace Penner