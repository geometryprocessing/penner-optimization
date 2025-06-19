#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

// TODO Refactor this and cone metric for more minimal inheritance

/**
 * @brief Class to generate primal edge cuts to convex patches for a mesh.
 */
class ConvexPatchDecomposition
{
public:

ConvexPatchDecomposition(const Mesh<Scalar>& m);

private:

	Mesh<Scalar> m_mesh;

	std::vector<int> m_boundary_edges;
	std::vector<int> m_rotation_index;

	std::vector<std::vector<int>> m_cuts;
	std::vector<bool> m_is_cut;

	void initialize_boundary_edges();

	int find_concave_vertex() const;
	std::vector<int> find_path(int vertex_index) const;

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_mesh(
			const Eigen::MatrixXd& V,
			const std::vector<int>& vtx_reindex) const;
	void view_patch_decomposition() const;

};

} // namespace Feature
} // namespace Penner