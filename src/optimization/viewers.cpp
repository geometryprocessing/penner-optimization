#include "viewers.hh"
#include <igl/flipped_triangles.h>
#if ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif // ENABLE_VISUALIZATION

namespace CurvatureMetric {

void
view_flipped_triangles(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
) {
  // Cut the mesh along the parametrization seams
	Eigen::MatrixXd V_cut;
	cut_mesh_along_parametrization_seams(V, F, uv, F_uv, V_cut);

  // Get the flipped elements
  Eigen::VectorXi flipped_f;
  igl::flipped_triangles(uv, F_uv, flipped_f);
  spdlog::info("{} flipped elements", flipped_f.size());

  // Convert flipped element list to a scalar function
  int num_faces = F_uv.rows();
  Eigen::VectorXd is_flipped;
  is_flipped.setZero(num_faces);
  for (int i = 0; i < flipped_f.size(); ++i)
  {
    int fi = flipped_f[i];
    is_flipped[fi] = 1.0;
    spdlog::info("Face {}: {} is flipped", fi, F_uv.row(fi));
  }

#if ENABLE_VISUALIZATION
  // Add the mesh and uv mesh with flipped element maps
  polyscope::init();
	polyscope::registerSurfaceMesh("mesh", V_cut, F_uv)
    ->addFaceScalarQuantity("is_flipped", is_flipped);
  polyscope::getSurfaceMesh("mesh")
		->addVertexParameterizationQuantity("uv", uv);
	polyscope::registerSurfaceMesh2D("uv mesh", uv, F_uv)
    ->addFaceScalarQuantity("is_flipped", is_flipped);
  polyscope::show();
#endif // ENABLE_VISUALIZATION
}

}
