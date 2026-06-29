// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "feature/core/common.h"

namespace Penner {
namespace Feature {


std::vector<int> generate_vertex_one_ring(const Mesh<Scalar>& m, int vertex_index)
{
    // circulate over one ring
    int h_start = m.out[vertex_index];
    int hij = h_start;
    std::vector<int> one_ring = {};
    do 
    {
        one_ring.push_back(hij);
        hij = m.n[m.opp[hij]];
    }
    while (hij != h_start);

    return one_ring;
}


std::vector<std::pair<int, int>> reindex_endpoints(
    const std::vector<std::pair<int, int>>& endpoints,
    const std::vector<int>& vtx_reindex)
{
    std::vector<int> vtx_reindex_inverse = invert_map(vtx_reindex);

    int num_orig_vertices = vtx_reindex.size();
    std::vector<std::pair<int, int>> endpoints_reindex = endpoints;
    for (int vi = 0; vi < num_orig_vertices; ++vi) {
        endpoints_reindex[vtx_reindex[vi]] = endpoints[vi];

        // check if need to reindex endpoints
        int e0 = endpoints[vi].first;
        int e1 = endpoints[vi].second;
        if ((e0 >= 0) || (e1 >= 0)) {
            spdlog::error("{} should be original vertex", vi);
        }
    }

    int num_vertices = endpoints.size();
    for (int vi = num_orig_vertices; vi < num_vertices; ++vi) {
        endpoints_reindex[vi] = endpoints[vi];

        // check if need to reindex endpoints
        int e0 = endpoints[vi].first;
        int e1 = endpoints[vi].second;
        if (e0 >= 0) {
            endpoints_reindex[vi].first = vtx_reindex[e0];
        }
        if (e1 >= 0) {
            endpoints_reindex[vi].second = vtx_reindex[e1];
        }
    }

    return endpoints_reindex;
}


} // namespace Feature
} // namespace Penner