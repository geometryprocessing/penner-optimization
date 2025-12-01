#include "feature/core/quads.h"

#include <igl/boundary_loop.h>

namespace Penner {
namespace Feature {

std::vector<int> compute_valences(const Eigen::MatrixXi& F)
{
	// initialize trivial valences
	int num_vertices = F.maxCoeff() + 1;
	std::vector<int> valences(num_vertices, 0);

	// increment valence for each corner of the triangle
	int num_faces = F.rows();
	for (int fijk = 0; fijk < num_faces; ++fijk)
	{
		for (int i = 0; i < F.cols(); ++i)
		{
			int vi = F(fijk, i);
			valences[vi] += 1;
		}
	}

	// TODO: open meshes
	// add additional valence for boundary vertices
	//std::vector<std::vector<int>> loops;
	//igl::boundary_loop(F, loops);
	//for (const auto& loop : loops)
	//{
	//	for (int vi : loop)
	//	{
	//		valences[vi] += 1;
	//	}
	//}

	return valences;
}

} // namespace Feature
} // namespace Penner