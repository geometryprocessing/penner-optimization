#include "shapes.hh"

// Some good simple tests are simplex embeddings that are natural (one vertex
// at the origin and others at unit vectors) with metrics that are uniform 
// (length 1 for every edge). The uniform lengths are fully symmetric, and the
// embedded case has three symmetric edges adjacent to the origin and three
// symmetric edges not adjacent to the origin.

namespace CurvatureMetric {
void
map_to_sphere(size_t num_vertices, std::vector<Scalar>& Th_hat)
{
  Scalar global_curvature = 4 * M_PI;
  Scalar average_curvature = global_curvature / num_vertices;
  fill_vector(num_vertices, 2 * M_PI - average_curvature, Th_hat);
}

void
generate_double_triangle(Eigen::MatrixXd& V,
                         Eigen::MatrixXi& F,
                         std::vector<Scalar>& Th_hat)
{
  V.resize(3, 3);
  F.resize(2, 3);
  V << 0, 0, 0, 1, 0, 0, 0, 1, 0;
  F << 0, 1, 2, 0, 2, 1;
  map_to_sphere(V.rows(), Th_hat);
}

void
generate_triangle(Eigen::MatrixXd& V,
                         Eigen::MatrixXi& F,
                         std::vector<Scalar>& Th_hat)
{
  V.resize(3, 3);
  F.resize(1, 3);
  V << 0, 0, 0, 1, 0, 0, 0, 1, 0;
  F << 0, 1, 2;
  Th_hat = std::vector<Scalar>(3, M_PI / 3.0);
}

void
generate_double_triangle_mesh(Mesh<Scalar>& m, std::vector<int>& vtx_reindex)
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<Scalar> Th_hat;
  generate_double_triangle(V, F, Th_hat);

  std::vector<int> indep_vtx, dep_vtx, v_rep, bnd_loops;
  m = FV_to_double(
    V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, std::vector<int>(), false);
}

void
generate_tetrahedron(Eigen::MatrixXd& V,
                     Eigen::MatrixXi& F,
                     std::vector<Scalar>& Th_hat)
{
  V.resize(4, 3);
  F.resize(4, 3);
  V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
  F << 0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2;
  map_to_sphere(V.rows(), Th_hat);
}

void
generate_tetrahedron_mesh(Mesh<Scalar>& m, std::vector<int>& vtx_reindex)
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<Scalar> Th_hat;
  generate_tetrahedron(V, F, Th_hat);

  std::vector<int> indep_vtx, dep_vtx, v_rep, bnd_loops;
  m = FV_to_double(
    V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, std::vector<int>(), false);
}

std::tuple<Eigen::MatrixXd,    // V
           Eigen::MatrixXi,    // F
           std::vector<Scalar> // Th_hat
           >
generate_tetrahedron_pybind()
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<Scalar> Th_hat;
  generate_tetrahedron(V, F, Th_hat);

  return std::make_tuple(V, F, Th_hat);
}

std::tuple<Mesh<Scalar>,    // m
           std::vector<int> // vtx_reindex
           >
generate_tetrahedron_mesh_pybind()
{
  Mesh<Scalar> m;
  std::vector<int> vtx_reindex;
  generate_tetrahedron_mesh(m, vtx_reindex);
  return std::make_tuple(m, vtx_reindex);
}

std::tuple<Eigen::MatrixXd,    // V
           Eigen::MatrixXi,    // F
           std::vector<Scalar> // Th_hat
           >
generate_double_triangle_pybind()
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<Scalar> Th_hat;
  generate_double_triangle(V, F, Th_hat);

  return std::make_tuple(V, F, Th_hat);
}

std::tuple<Mesh<Scalar>,    // m
           std::vector<int> // vtx_reindex
           >
generate_double_triangle_mesh_pybind()
{
  Mesh<Scalar> m;
  std::vector<int> vtx_reindex;
  generate_double_triangle_mesh(m, vtx_reindex);
  return std::make_tuple(m, vtx_reindex);
}
}
