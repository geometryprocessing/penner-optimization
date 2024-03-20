#include "viewers.hh"

#include "vector.hh"
#include "vf_mesh.hh"

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

void
view_halfedge_mesh_layout(
  const Mesh<Scalar>& m,
  const std::vector<Scalar>& u_vec,
  const std::vector<Scalar>& v_vec
) {
    Eigen::VectorXd u, v;
    convert_std_to_eigen_vector(u_vec, u);
    convert_std_to_eigen_vector(v_vec, v);
    Eigen::MatrixXd uv(u.size(), 3);
    uv.col(0) = u;
    uv.col(1) = v;
    Eigen::MatrixXi F(m.n_faces(), 3);
    for (int f = 0; f < m.n_faces(); ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];
        F(f, 0) = hij;
        F(f, 1) = hjk;
        F(f, 2) = hki;
    }
#if ENABLE_VISUALIZATION
    spdlog::info("Viewing layout");
    polyscope::init();
    polyscope::registerSurfaceMesh2D("layout", uv, F);
    polyscope::show();
#endif
}

}