#include "optimization_layout.hh"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "embedding.hh"
#include "targets.hh"
#include "interpolation.hh"
#include "transitions.hh"
#include "translation.hh"
#include "projection.hh"
#include "energies.hh"
#include <igl/doublearea.h>
#include "igl/edge_flaps.h"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

OverlayMesh<Scalar>
add_overlay(const Mesh<Scalar>& m, const VectorX& reduced_metric_coords)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Build overlay mesh from mesh m
  Mesh<Scalar> m_l = m;

  // Convert mesh Penner coordinates to a halfedge length array l for m
  int num_halfedges = he2e.size();
  for (int h = 0; h < num_halfedges; ++h) {
    m_l.l[h] = exp(reduced_metric_coords[proj[he2e[h]]] / 2.0);
  }

  OverlayMesh<Scalar> mo(m_l);

  return mo;
}

void
make_tufted_overlay(OverlayMesh<Scalar>& mo,
                    const Eigen::MatrixXd& V,
                    const Eigen::MatrixXi& F,
                    const std::vector<Scalar>& Theta_hat)
{
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  FV_to_double(
    V, F, V, F, Theta_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

  if (bnd_loops.size() != 0) {
    int n_v = V.rows();
    auto mc = mo.cmesh();
    create_tufted_cover(
      mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep, mo._m.out, mo._m.to);
    mo._m.v_rep = range(0, n_v);
  }
}

bool check_areas(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F
) {
  Eigen::VectorXd areas;
  igl::doublearea(V, F, areas);
  Scalar min_area = areas.minCoeff() / 2.0;
  Scalar max_area = areas.maxCoeff() / 2.0;
  spdlog::info("Minimum face area: {}", min_area);
  spdlog::info("Maximum face area: {}", max_area);

  return (!float_equal(min_area, 0.0));
}

Scalar uv_length_squared(
  const Eigen::Vector2d& uv_0,
  const Eigen::Vector2d& uv_1
) {
  Eigen::Vector2d difference_vector = uv_1 - uv_0;
  Scalar length_sq = difference_vector.dot(difference_vector);
  return length_sq;
}

Scalar uv_length(
  const Eigen::Vector2d& uv_0,
  const Eigen::Vector2d& uv_1
) {
  return sqrt(uv_length_squared(uv_0, uv_1));
}

Scalar compute_uv_length_error(
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
) {
  // Get the edge topology for the original uncut mesh
  Eigen::MatrixXi uE, EF, EI;
  Eigen::VectorXi EMAP;
  igl::edge_flaps(F, uE, EMAP, EF, EI);

  // Iterate over edges to check the length inconsistencies
  Scalar max_uv_length_error = 0.0;
  for (Eigen::Index e = 0; e < EF.rows(); ++e)
  {
    // Get face corners corresponding to the current edge
    int f0 = EF(e, 0);
    if (f0 < 0) continue;
    int i0 = EI(e, 0); // corner vertex face index
    int v0n = F_uv(f0, (i0 + 1) % 3); // next vertex
    int v0p = F_uv(f0, (i0 + 2) % 3); // previous vertex
    int f1 = EF(e, 1);
    if (f1 < 0) continue;
    int i1 = EI(e, 1); // corner vertex face index
    int v1n = F_uv(f1, (i1 + 1) % 3); // next vertex
    int v1p = F_uv(f1, (i1 + 2) % 3); // next vertex

    // Compute the length of each halfedge corresponding to the corner in the cut mesh
    Scalar l0 = uv_length(uv.row(v0n), uv.row(v0p));
    Scalar l1 = uv_length(uv.row(v1n), uv.row(v1p));

    // Determine if the max length inconsistency has increased
    max_uv_length_error = max(max_uv_length_error, abs(l0 - l1));
  }

  // Return the max uv length error
  return max_uv_length_error;
}

bool check_uv(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
) {
  int n_vertices = V.rows();
  int n_faces = F.rows();
  bool is_valid = true;

  // Check faces agree in number
  if (F_uv.rows() != n_faces)
  {
    spdlog::error("Mesh and uv faces are not in one to one correspondence");
    is_valid = false;
  }

  // Build halfedge mesh
  const std::vector<Scalar> Theta_hat(n_vertices, 0.0);
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, free_cones;
  bool fix_boundary=false;
  Mesh<Scalar> m = FV_to_double<Scalar>(
    V,
    F,
    uv,
    F_uv,
    Theta_hat,
    vtx_reindex,
    indep_vtx,
    dep_vtx,
    v_rep,
    bnd_loops,
    free_cones,
    fix_boundary
  );

  // Check output
  if (m.n_ind_vertices() != V.rows())
  {
    spdlog::error("Failed to build an overlay mesh");
    is_valid = false;
  }

  // Check length consistency
  if (!check_length_consistency(m))
  {
    spdlog::error("Inconsistent uv lengths across edges");
  }

  // Check mesh face areas
  if (!check_areas(V, F)){
    spdlog::warn("Mesh face area is zero");
  }

  // Check uv face areas
  //Eigen::MatrixXd uv_embed(uv.rows(), 3);
  //uv_embed.setZero();
  //uv_embed.col(0) = uv.col(0);
  //uv_embed.col(1) = uv.col(1);
  if (!check_areas(uv, F_uv)){
    spdlog::warn("Mesh layout face area is zero");
  }

  // Return true if no issues found
  return is_valid;
}

std::tuple<std::vector<std::vector<Scalar>>, // V_out
           std::vector<std::vector<int>>,    // F_out
           std::vector<Scalar>,              // layout u (per vertex)
           std::vector<Scalar>,              // layout v (per vertex)
           std::vector<std::vector<int>>,    // FT_out
           std::vector<bool>,                // is_cut
           std::vector<bool>,                // is_cut_o
           std::vector<int>,                 // Fn_to_F
           std::vector<std::pair<int, int>>  // endpoints
           >
parametrize_mesh(const Eigen::MatrixXd& V,
                 const Eigen::MatrixXi& F,
                 const std::vector<Scalar>& Theta_hat,
                 const Mesh<Scalar> &m,
                 const std::vector<int>& vtx_reindex,
                 const VectorX reduced_metric_coords)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Get initial penner coordinates for the mesh
  VectorX reduced_metric_init;
  std::vector<int> flip_sequence;
  compute_penner_coordinates(V, F, Theta_hat, reduced_metric_init, flip_sequence);

  // Expand metric coordinates to the full mesh
	int num_edges = e2he.size();
  VectorX metric_coords(num_edges);
  VectorX metric_init(num_edges);
  for (int e = 0; e < num_edges; ++e)
  {
    metric_coords[e] = reduced_metric_coords[proj[e]];
    metric_init[e] = reduced_metric_init[proj[e]];
  }

  // Expand metric coordinates to halfedges
	int num_halfedges = he2e.size();
  VectorX he_metric_coords(num_halfedges);
  VectorX he_metric_init(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h)
  {
    he_metric_coords[h] = metric_coords[he2e[h]];
    he_metric_init[h] = metric_init[he2e[h]];
  }

  // Remove fit scale factors to reduce numerical instability
  VectorX scale_factors = best_fit_conformal(m, metric_init, metric_coords);
  MatrixX B = conformal_scaling_matrix(m);
  VectorX metric_scaled = metric_coords - B * scale_factors;
  VectorX he_metric_scaled(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h)
  {
    he_metric_scaled[h] = metric_scaled[he2e[h]];
  }

  // Interpolate coordinates
  InterpolationMesh interpolation_mesh, reverse_interpolation_mesh;
  interpolate_penner_coordinates(
    m,
    he_metric_scaled,
    scale_factors,
    interpolation_mesh,
    reverse_interpolation_mesh
  );

  // Get interpolated vertex positions
  Eigen::MatrixXd V_overlay;
  interpolate_vertex_positions(
    V,
    vtx_reindex,
    interpolation_mesh,
    reverse_interpolation_mesh,
    V_overlay
  );

  // Convert overlay vertices to transposed array form
  std::vector<std::vector<Scalar>> V_overlay_vec(V_overlay.cols());
  for (Eigen::Index i = 0; i < V_overlay.cols(); ++i)
  {
    V_overlay_vec[i].resize(V_overlay.rows());
    for (Eigen::Index j = 0; j < V_overlay.cols(); ++j)
    {
      V_overlay_vec[i][j] = V_overlay(j, i);
    }
  }

  // Get tufted overlay mesh
  OverlayMesh<Scalar> mo = interpolation_mesh.get_overlay_mesh();
  make_tufted_overlay(mo, V, F, Theta_hat);

  // Get endpoints
  std::vector<std::pair<int, int>> endpoints;
  find_origin_endpoints(mo, endpoints);

  // Convert scale factors to vector
  std::vector<Scalar> u;
  convert_eigen_to_std_vector(scale_factors, u);

  // Convert overlay to VL
  std::vector<int> vtx_reindex_mutable = vtx_reindex;
  return overlay_mesh_to_VL<Scalar>(
    V,
    F,
    Theta_hat,
    mo,
    u,
    V_overlay_vec,
    vtx_reindex_mutable,
    endpoints,
    -1
  );
}

void
extract_embedded_mesh(
  const Mesh<Scalar>& m,
  const std::vector<int>& vtx_reindex,
  Eigen::MatrixXi& F,
  Eigen::MatrixXi& corner_to_halfedge
) {
  // Get number of vertices and faces for the embedded mesh
  int num_faces = m.n_faces();
  int num_embedded_faces = 0;
  for (int f = 0; f < num_faces; ++f)
  {
    // Skip face if it is in the doubled mesh
    int hij = m.h[f];
    if (m.type[hij] == 2) continue;

    // Count embedded faces
    num_embedded_faces++;
  }

  // Build faces and halfedge lengths
  F.resize(num_embedded_faces, 3);
  corner_to_halfedge.resize(num_embedded_faces, 3);
  int face_count = 0;
  for (int f = 0; f < num_faces; ++f)
  {
    // Get halfedges of face
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Skip face if it is in the doubled mesh
    if (m.type[hij] == 2) continue;

    // Get vertices of the face
    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];

    // Build vertex embedding for the face
    F(face_count, 0) = vtx_reindex[vi];
    F(face_count, 1) = vtx_reindex[vj];
    F(face_count, 2) = vtx_reindex[vk];

    // Build halfedge index map for the face
    corner_to_halfedge(face_count, 0) = hjk;
    corner_to_halfedge(face_count, 1) = hki;
    corner_to_halfedge(face_count, 2) = hij;

    // Increment face count for unique indexing
    face_count++;
  }
}

#ifdef PYBIND
std::tuple<
  Eigen::MatrixXi, // F
  Eigen::MatrixXi // corner_to_halfedge
>
extract_embedded_mesh_pybind(
  const Mesh<Scalar>& m,
  const std::vector<int>& vtx_reindex
) {
  Eigen::MatrixXi F;
  Eigen::MatrixXi corner_to_halfedge;
  extract_embedded_mesh(m, vtx_reindex, F, corner_to_halfedge);
  return std::make_tuple(F, corner_to_halfedge);
}

#endif

}
