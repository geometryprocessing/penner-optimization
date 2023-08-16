#include "optimization_layout.hh"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/Layout.hh"
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

// Helper function to triangulate a polygon mesh
// WARNING: This only updates the topology and will generally invalidate the lengths
void triangulate_mesh(Mesh<Scalar>& m) {
    int n_f0 = m.n_faces();
    for(int f = 0; f < n_f0; f++){
        int n_f = m.n_faces();
        int h0 = m.h[f];
        int hc = h0;
        std::vector<int> hf;
        do{
            hf.push_back(hc);
            hc = m.n[hc];
        }while(h0 != hc);
        int face_size = hf.size();
        if(face_size == 3) continue;
        spdlog::debug("triangulate face {}, #e {}", f, face_size);
        int N = hf.size();
        int n_new_f = N-3;
        int n_new_he = 2*(N-3);
        int n_he = m.n.size();
        m.n.resize(n_he   + n_new_he);
        m.to.resize(n_he  + n_new_he);
        m.opp.resize(n_he + n_new_he);
        m.l.resize(n_he + n_new_he);
        m.f.resize(n_he + n_new_he);
        m.h.resize(n_f + n_new_f);
        m.n[n_he] = hf[0];
        m.n[hf[1]] = n_he;
        m.opp[n_he] = n_he+1;
        m.to[n_he] = m.to[hf.back()];
        m.h[f] = n_he;
        m.f[n_he] = f;
        assert(hf.back() < m.to.size() && hf[0] < m.to.size());
        m.l[n_he] = 0.0; // Set invalid 0 length
        for(int k = 1; k < 2*(N-3); k++){
            if(k%2 == 0){
                m.n[n_he+k] = n_he+k-1;
                m.opp[n_he+k] = n_he+k+1;
                m.to[n_he+k] = m.to[hf.back()];
                m.l[n_he+k] = 0.0; // Set invalid 0 length
                m.f[n_he+k] = n_f+k/2-1;
                m.h[n_f+k/2-1] = n_he+k;
            }else{
                m.n[n_he+k] = hf[(k-1)/2+2];
                if((k-1)/2+2 != face_size-2)
                    m.n[hf[(k-1)/2+2]] = n_he+k+1;
                m.opp[n_he+k] = n_he+k-1;
                m.to[n_he+k] = m.to[m.n[m.n[m.opp[n_he+k]]]];
                m.l[n_he+k] = 0.0; // Set invalid 0 length
                m.f[n_he+k] = n_f+(k+1)/2-1;
                m.h[n_f+(k+1)/2-1] = n_he+k;
                m.f[m.n[n_he+k]] = n_f+(k+1)/2-1;
            }                
        }
        m.n[hf.back()] = n_he + n_new_he - 1;
        m.f[hf.back()] = n_f + n_new_f - 1;
    }
}

std::vector<bool>
pullback_cut_to_overlay(
  OverlayMesh<Scalar> &m_o,
  const std::vector<bool>& is_cut_h
) {
  std::vector<bool> is_cut_o(m_o.n_halfedges(), false);
  for (int hi = 0; hi < m_o.n_halfedges(); ++hi)
  {
    // Don't cut edges not in the original mesh
    if (m_o.edge_type[hi] == CURRENT_EDGE)
    {
      continue;
    }
    else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE)
    {
      is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
    }
    else if (m_o.edge_type[hi] == ORIGINAL_EDGE)
    {
      is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
    }

  }

  return is_cut_o;
}

/**
 * @brief Given overlay mesh with associated flat metric compute the layout
 * 
 * @tparam Scalar double/mpfr::mpreal
 * @param m_o, overlay mesh
 * @param u_vec, per-vertex scale factor
 * @param bd, list of boundary vertex ids
 * @param singularities, list of singularity vertex ids
 * @return _u_o, _v_o, is_cut_h (per-corner u/v assignment of overlay mesh and marked cut edges)
 */
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> get_consistent_layout(
             OverlayMesh<Scalar> &m_o,
             const std::vector<Scalar> &u_vec,
             const std::vector<int>& bd,
             std::vector<int> singularities,
             const std::vector<bool>& is_cut)
{
  // Get original overlay face labels 
  auto f_labels = get_overlay_face_labels(m_o);
  
  // Compute layout of the underlying flipped mesh
  std::vector<bool> _is_cut_place_holder;
  auto mc = m_o.cmesh();
  m_o.garbage_collection();
  mc.type = std::vector<char>(mc.n_halfedges(), 0);
  auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder, 0);
  auto _u_c = std::get<0>(layout_res);
  auto _v_c = std::get<1>(layout_res);
  auto is_cut_c = std::get<2>(layout_res);

  // Interpolate layout to the overlay mesh
  Eigen::Matrix<Scalar, -1, 1> u_eig;
  u_eig.resize(u_vec.size());
  for (size_t i = 0; i < u_vec.size(); i++)
  {
    u_eig(i) = u_vec[i];
  }
  m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);
  auto u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _u_c);
  auto v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _v_c);
  spdlog::info("Interpolate on overlay mesh done.");

  // Build a new mesh directly from the triangulated overlay mesh
  Mesh<Scalar> m;
  m.n = m_o.n;
  m.opp = m_o.opp;
  m.f = m_o.f;
  m.h = m_o.h;
  m.out = m_o.out;
  m.to = m_o.to;
  m.l = std::vector<Scalar>(m.n.size(), 0.0);
  for(int i = 0; i < m.n_halfedges(); i++){
      int h0 = i; 
      int h1 = h0;
      do{
          if(m.n[h1] == h0)
              break;
          h1 = m.n[h1];
      }while(h0 != h1);
      if(m.to[m.opp[h0]] != m.to[h1]){
          spdlog::error("h0 h1 picked wrong.");
          exit(0);
      }
      m.l[h0] = sqrt((u_o[h0]-u_o[h1])*(u_o[h0]-u_o[h1]) + (v_o[h0]-v_o[h1])*(v_o[h0]-v_o[h1]));
  }
  triangulate_polygon_mesh(m, u_o, v_o, f_labels);
  m.type = std::vector<char>(m.n.size(), 0);
  m.type_input = m.type;
  m.R = std::vector<int>(m.n.size(), 0);
  m.v_rep = range(0, m.out.size());
  m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);
  OverlayMesh<Scalar> m_o_tri(m);
  for(int i = m_o.n_halfedges(); i < m_o_tri.n_halfedges(); i++){
      m_o_tri.edge_type[i] = ORIGINAL_EDGE; // make sure do not use the new diagonal
  }

  // Pullback cut on the original mesh to the overlay
  std::vector<bool> is_cut_poly = pullback_cut_to_overlay(m_o, is_cut);

  // Extend the overlay cut to the triangulated mesh
  // WARNING: Assumes triangulation halfedges added to the end
  std::vector<bool> is_cut_o = std::vector<bool>(m.n_halfedges(), false);
  for (int h = 0; h < m_o.n_halfedges(); ++h)
  {
    is_cut_o[h] = is_cut_poly[h];
  }

  if (bd.empty()) {spdlog::info("No boundary");} //FIXME
  // now directly do layout on triangulated overlay mesh
  // TODO Make sure don't need to change edge type or other data fields
  std::vector<Scalar> _u_o, _v_o;
  std::vector<Scalar> phi(m.n_vertices(), 0.0);
  auto overlay_layout_res = compute_layout(m, phi, is_cut_o);
  _u_o = std::get<0>(overlay_layout_res);
  _v_o = std::get<1>(overlay_layout_res);
  is_cut_o = std::get<2>(overlay_layout_res);

  // Restrict back to original overlay
  // WARNING: Assumes triangulation halfedges added to the end
  _u_o.resize(m_o.n.size());
  _v_o.resize(m_o.n.size());
  is_cut_o.resize(m_o.n.size()); 
  trim_open_branch(m_o, f_labels, singularities, is_cut_o);

  return std::make_tuple(_u_o, _v_o, is_cut_o);
  
}


bool
is_valid_layout(
  const OverlayMesh<Scalar>& mo,
  const std::vector<bool> &is_cut_o,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
) {
  // Check consistency of uv lengths across cuts
  check_uv(V, F, uv, F_uv);
  // TODO

  // Check for inverted elements
  // TODO

  // Triangulate the overlay mesh
  Mesh<Scalar> m(mo);
  triangulate_mesh(m);
  
  // Get cut edges for the triangulated mesh (new edges are not cut)
  std::vector<bool> is_cut_tri(m.n_halfedges(), false);
  for (int h = 0; h < mo.n_halfedges(); ++h)
  {
    is_cut_tri[h] = is_cut_o[h];
  }

  // Check if the cuts agree with the is_cut mask
  // TODO May not be feasible to do reasonably; the face correspondence is lost
  return false;

}

void
check_if_flipped(
  Mesh<Scalar> &m,
  const std::vector<Scalar>& u,
  const std::vector<Scalar>& v
) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> uv(u.size(), 2);
  Eigen::MatrixXi F_uv(m.n_faces(), 3);

  for (size_t i = 0; i < u.size(); ++i)
  {
    uv(i, 0) = static_cast<double>(u[i]);
    uv(i, 1) = static_cast<double>(v[i]);
  }

  for (int fi = 0; fi < m.n_faces(); ++fi)
  {
    int h = m.h[fi];
    F_uv(fi, 0) = h;
    F_uv(fi, 1) = m.n[h];
    F_uv(fi, 2) = m.n[m.n[h]];
  }

  Eigen::VectorXi flipped_f;
  igl::flipped_triangles(uv, F_uv, flipped_f);
  spdlog::info("{} flipped elements in mesh", flipped_f.size());
  for (int i = 0; i < flipped_f.size(); ++i)
  {
    int fi = flipped_f[i];
    spdlog::info("Face {} is flipped", F_uv.row(fi));
    spdlog::info(
      "Vertices {}, {}, {}",
      uv.row(F_uv(fi, 0)),
      uv.row(F_uv(fi, 1)),
      uv.row(F_uv(fi, 2))
    );
  }
}

std::vector<bool>
compute_layout_topology(const Mesh<Scalar> &m, const std::vector<bool>& is_cut_h, int start_h)
{
  auto _u = std::vector<Scalar>(m.n_halfedges(), 0.0);
  auto _v = std::vector<Scalar>(m.n_halfedges(), 0.0);

  bool cut_given = !is_cut_h.empty();
  auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);

  // set starting point - use a boundary edge
  int h = 0;
  if (start_h == -1)
  {
    for (int i = 0; i < m.n_halfedges(); i++)
    {
      if (m.type[i] == 1 && m.type[m.opp[i]] == 2)
      {
        h = i;
        is_cut_h_gen[i] = true;
      }
    }
  }

  auto done = std::vector<bool>(m.n_faces(), false);

  // discard part 2
  for (size_t i = 0; i < done.size(); i++)
  {
    int hh = m.h[i];
    if (m.type[hh] == 2 && m.type[m.n[hh]] == 2 && m.type[m.n[m.n[hh]]] == 2)
    {
      done[i] = true;
    }
  }
  // set edge type 2 as cut
  for (size_t i = 0; i < is_cut_h.size(); i++)
  {
    if (m.type[i] == 2)
    {
      is_cut_h_gen[i] = true;
    }
  }

  // Initialize queue and record of faces to process
  std::queue<int> Q;
  Q.push(h);
  done[m.f[h]] = true;

  while (!Q.empty())
  {
    // Get next halfedge to process
    h = Q.front();
    Q.pop();
    
    // Get other triangle edges
    int hn = m.n[h];
    int hp = m.n[hn];
    int hno = m.opp[hn];
    int hpo = m.opp[hp];
    int ho = m.opp[h];

    // Check if next edge triangle should be laid out
    if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn]))
    {
      done[m.f[hno]] = true;
      Q.push(hno);
    }
    else
    {
      is_cut_h_gen[hn] = true;
      is_cut_h_gen[m.opp[hn]] = true;
    }

    // Check if previous edge triangle should be laid out
    if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp]))
    {
      done[m.f[hpo]] = true;
      Q.push(hpo);
    }
    else
    {
      is_cut_h_gen[hp] = true;
      is_cut_h_gen[m.opp[hp]] = true;
    }

    // Check if current edge triangle should be laid out
    // WARNING: Should only be used once for original edge
    if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho]))
    {
      done[m.f[ho]] = true;
      Q.push(ho);
    }
  }

  return is_cut_h_gen;
  
};

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
