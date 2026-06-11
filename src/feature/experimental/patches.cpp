// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "feature/experimental/patches.h"

namespace Penner {
namespace Feature {

/*
ConvexPatchDecomposition::ConvexPatchDecomposition(const Mesh<Scalar>& m)
	: m_mesh(m)
	, m_cuts({})
	, m_is_cut(m.n_halfedges(), false)
{
		initialize_boundary_edges();
}

void ConvexPatchDecomposition::initialize_boundary_edges()
{
		// get boundary edges 
		m_boundary_edges = find_primal_boundary_halfedges(m);

		// build map from all halfedges to boundary halfedges
		int num_bd_edges = m_boundary_edges.size();
		m_h2bd = std::vector<int>(num_bd_edges, -1);
		for (int i = 0; i < num_bd_edges; ++i)
		{
				m_h2bd[m_boundary_edges[i]] = i;
		}

		// get components 
		m_components = find_boundary_components(m);

		// build next and prev maps from components
		for (int h_start : m_components) {
				std::vector<int> component = build_boundary_component(m, h_start);
				int component_size = component.size();
				for (int i = 0; i < component.size(); ++i)
				{
						int j = (i + 1) % component_size;
						int curr = m_h2bd[component[i]];
						int next = m_h2bd[component[j]];
						m_next[curr] = next;
						m_prev[next] = curr;
				}
		}

		// 

}
*/

} // namespace Feature
} // namespace Penner