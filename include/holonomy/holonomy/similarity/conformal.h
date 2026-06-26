// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

/**
 * @brief Method to find a conformally equivalent similarity metric satisfying
 * holonomy constraints.
 * 
 */

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