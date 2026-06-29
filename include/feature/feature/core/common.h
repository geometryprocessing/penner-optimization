// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "holonomy/core/common.h"

// get holonomy classes to put in feature namespace
#include "holonomy/core/dual_loop.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"
#include "holonomy/interface.h"

/**
 * @brief Assorted utility functions.
 * 
 */

namespace Penner {
namespace Feature {

using Holonomy::DualLoop;
using Holonomy::MarkedPennerConeMetric;
using Holonomy::NewtonParameters;
using Holonomy::MarkedMetricParameters;

/**
 * @brief Compute the one ring of halfedges emenating from a mesh vertex.
 *
 * @param m: mesh
 * @param vertex_index: index of a vertex in the mesh
 * @return halfedges emenating from a given vertex
 */
std::vector<int> generate_vertex_one_ring(const Mesh<Scalar>& m, int vertex_index);


/**
 * @brief Reindex list of edge endpoints under vertex reindexing.
 * 
 * @param endpoints: list of vertex endpoints before reindexing
 * @param vtx_reindex: map from old to new vertex indices
 * @return list of reindexed vertex endpoints
 */
std::vector<std::pair<int, int>> reindex_endpoints(
    const std::vector<std::pair<int, int>>& endpoints,
    const std::vector<int>& vtx_reindex);


} // namespace Feature
} // namespace Penner