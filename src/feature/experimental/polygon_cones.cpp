#include "feature/experimental/polygon_cones.h"

#include "util/boundary.h"


namespace Penner {
namespace Feature {

std::vector<Scalar> generate_polygon_cones(
	const Mesh<Scalar>& m,
	const std::vector<int>& vtx_reindex,
	int num_vertices,
	bool use_length)
{
	if (use_length)
	{
		// FIXME
		spdlog::info("Length based polygon cone generation not implemented");
		return {};
	}

	// Check if topological disk
	int genus = Holonomy::compute_genus(m);
	std::vector<int> boundary_components = find_boundary_components(m);
	if ((genus != 0) || (boundary_components.size() != 1)) {
		spdlog::error("Polygon cones only supported for disk topology");
		return {};
	}

	// Build unique boundary component
	std::vector<int> boundary_component = build_boundary_component(m, boundary_components.front());
	int num_bd_edges = boundary_component.size();

	// Find delta for indexing
	if ((num_vertices <= 0) || (num_vertices > num_bd_edges))
	{
		num_vertices = num_bd_edges;
	}
	int delta = num_bd_edges / num_vertices;

	// Build list of vertices to make cones
	int num_ind_vertices = m.n_ind_vertices();
	std::vector<Scalar> Th_hat(num_ind_vertices, 2. * M_PI);
	for (int hij : boundary_component)
	{
		int vj = m.v_rep[m.to[hij]];
		Th_hat[vtx_reindex[vj]] = M_PI;
	}

	// Add defect for corners
	Scalar cone_defect = (2 * M_PI) / num_vertices;
	for (int i = 0; i < num_vertices; ++i)
	{
		int hij = boundary_component[i * delta];
		int vj = m.v_rep[m.to[hij]];
		Th_hat[vtx_reindex[vj]] -= cone_defect;
	}

	return Th_hat;

}

} // namespace Feature
} // namespace Penner