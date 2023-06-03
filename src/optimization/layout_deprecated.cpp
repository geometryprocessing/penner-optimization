#include "layout_deprecated.hh"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "embedding.hh"
#include "interpolation.hh"
#include "transitions.hh"
#include <igl/doublearea.h>
#include "igl/edge_flaps.h"

namespace CurvatureMetric {

Mesh<double>
remove_overlay(const OverlayMesh<double>& mo)
{
  return mo._m;
}

void
flip_edges_overlay(OverlayMesh<double>& mo, const std::vector<int>& flip_seq)
{
  for (int i = 0; i < flip_seq.size(); ++i) {
    int h = flip_seq[i];

    // Euclidean flip
    if (h < 0) {
      mo.flip_ccw(-h - 1, false);
    }

    // Ptolemy flip
    else {
      mo.flip_ccw(h, true);
    }
  }

  mo.garbage_collection();
}

std::vector<int>
make_delaunay_overlay(OverlayMesh<double>& mo, bool Ptolemy)
{

  // Make overlay mesh Delaunay
  Mesh<double>& mc = mo.cmesh();
  VectorX u0;
  u0.setZero(mc.n_ind_vertices());
  DelaunayStats del_stats;
  SolveStats<double> solve_stats;
  ConformalIdealDelaunay<double>::MakeDelaunay(
    mo, u0, del_stats, solve_stats, Ptolemy);

  // Output the angle error FIXME Remove
  //VectorX g;
  //VectorX angles, cot_angles;
  //ConformalIdealDelaunay<double>::ComputeAngles(mc, u0, angles, cot_angles);
  //ConformalIdealDelaunay<double>::Gradient(mc, angles, g, solve_stats);
  //std::cout << "Angle error: " << g.cwiseAbs().maxCoeff() << std::endl;

  // Check for NonDelaunay edges FIXME Remove
  //for (int h = 0; h < mc.n.size(); ++h) {
  //  if (ConformalIdealDelaunay<double>::NonDelaunay(mc, u0, h, solve_stats))
  //    std::cout << "NonDelaunay" << std::endl;
  //}

  mo.garbage_collection();

  return del_stats.flip_seq;
}

void
change_lengths_overlay(OverlayMesh<double>& mo, const VectorX& l)
{
  auto mc = mo.cmesh();
  for (int i = 0; i < mc.l.size(); ++i) {
    mo._m.l[i] = l[i];
  }
}

void
bc_original_to_eq_overlay(OverlayMesh<double>& mo)
{
  Mesh<double>& mc = mo.cmesh();
  {
    mo.garbage_collection();
    mo.bc_original_to_eq(mc.n, mc.to, mc.l);
  }
}

std::vector<std::vector<int>>
reindex_F(const std::vector<std::vector<int>>& F,
          const std::vector<int> vtx_reindex)
{
  std::vector<std::vector<int>> F_out(F);
  for (int i = 0; i < F.size(); i++) {
    for (int j = 0; j < 3; j++) {
      if (F_out[i][j] < vtx_reindex.size()) {
        F_out[i][j] = vtx_reindex[F[i][j]];
      }
    }
  }

  return F_out;
}

std::tuple<std::vector<int>, std::vector<int>>
get_cones_and_bd(const Eigen::MatrixXd& V,
                 const Eigen::MatrixXi& F,
                 const std::vector<double>& Theta_hat,
                 const std::vector<int>& vtx_reindex)
{
  // get cones and bd
  std::vector<int> cones, bd;
  std::vector<bool> is_bd = igl::is_border_vertex(F);
  for (int i = 0; i < is_bd.size(); i++) {
    if (is_bd[i]) {
      bd.push_back(i);
    }
  }
  bool do_trim = false;
  auto gb = count_genus_and_boundary(V, F);
  int n_genus = gb.first, n_bd = gb.second;
  if ((n_genus >= 1 && n_bd != 0) || n_bd > 1) {
    do_trim = true;
  }
  for (int i = 0; i < Theta_hat.size(); i++) {
    if ((!is_bd[i]) && abs(Theta_hat[i] - 2 * M_PI) > 1e-15) {
      cones.push_back(i);
    }
  }

  // reindex cones and bd
  std::vector<int> vtx_reindex_rev(vtx_reindex.size());
  for (int i = 0; i < vtx_reindex.size(); i++) {
    vtx_reindex_rev[vtx_reindex[i]] = i;
  }
  for (int i = 0; i < cones.size(); i++) {
    cones[i] = vtx_reindex_rev[cones[i]];
  }
  for (int i = 0; i < bd.size(); i++) {
    bd[i] = vtx_reindex_rev[bd[i]];
  }

  return std::make_tuple(cones, bd);
}

void
change_shear_overlay(OverlayMesh<double>& mo,
                     const VectorX& lambdas_del_he,
                     const VectorX& tau)
{
  Mesh<double>& mc = mo.cmesh();

  // Change lengths to target values
  for (int h = 0; h < lambdas_del_he.size(); ++h) {
    mc.l[h] = exp(lambdas_del_he[h] / 2.0);
  }

  // Translate barycentric coordinates on edges, e.g. to rectify shear
  bc_reparametrize_eq(mo, tau);
  reparametrize_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, tau);
}

std::vector<std::vector<double>>
Interpolate_3d_reparametrized(OverlayMesh<double>& m_o,
                              const VectorX& lambdas_rev_he,
                              const VectorX& tau_rev,
                              const std::vector<int>& flip_seq_init,
                              const std::vector<int>& flip_seq,
                              const std::vector<std::vector<double>>& x)
{
  std::vector<std::vector<double>> z(3);
  auto mc = m_o.cmesh();
  OverlayMesh<double> m_o_rev(mc);
  std::vector<int> v_map(m_o.out.size());

  // Generate reverse map for Ptolemy flips (representing a hyperbolic surface)
  ConformalIdealDelaunay<double>::ReverseFlips(m_o_rev, flip_seq);

  // Reparametrize the mesh
  change_shear_overlay(m_o_rev, lambdas_rev_he, tau_rev);

  // Generate reverse map for Euclidean flips
  ConformalIdealDelaunay<double>::ReverseFlips(m_o_rev, flip_seq_init);
  v_map = ConformalIdealDelaunay<double>::GetVertexMap(m_o, m_o_rev);

  // Check final overlay mesh
  auto mc_rev = m_o_rev.cmesh();
	for (int h = 0; h < mc_rev.n_halfedges(); ++h)
	{
		m_o_rev.check_bc_alignment(&mc_rev, h);
	}

  // Map barycentric coordinates to scaled triangles
  Eigen::Matrix<double, -1, 1> u_0(m_o_rev.cmesh().out.size());
  u_0.setZero();
  m_o_rev.bc_eq_to_scaled(
    m_o_rev.cmesh().n, m_o_rev.cmesh().to, m_o_rev.cmesh().l, u_0);

  // Interpolate vertex positions along the original edges
  std::vector<std::vector<double>> z_rev(3);
  for (int j = 0; j < 3; j++) {
    z_rev[j] = m_o_rev.interpolate_along_o_bc(
      m_o_rev.cmesh().opp, m_o_rev.cmesh().to, x[j]);
  }

  // Reindex vertices
  for (int j = 0; j < 3; j++) {
    z[j].resize(z_rev[j].size());
    for (int i = 0; i < z[j].size(); i++) {
      z[j][i] = z_rev[j][v_map[i]];
    }
  }

  return z;
}

std::tuple<OverlayMesh<double>,
           std::vector<int>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>,
           std::vector<int>,
           std::vector<std::pair<int, int>>>
generate_optimized_overlay(
  const Eigen::MatrixXd& v,
  const Eigen::MatrixXi& f,
  const std::vector<double>& Theta_hat,
  const VectorX& lambdas,
  const VectorX& tau_init,
  const VectorX& tau,
  const VectorX& tau_post,
  const std::vector<int>& pt_fids_in,
  const std::vector<Eigen::Matrix<double, 3, 1>>& pt_bcs_in,
  bool initial_ptolemy,
  bool flip_in_original_metric)

{
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  Mesh<double> m = FV_to_double(
    v, f, v, f, Theta_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

  // Initialize points
  std::vector<int> pt_fids(pt_fids_in);
  std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs(pt_bcs_in);

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Get overlay mesh and copy
  OverlayMesh<double> mo(m);
  Mesh<double>& mc = mo.cmesh();
  mc.init_pts(pt_fids, pt_bcs);

  // Make original mesh Delaunay and duplicate flips for the final coordinates
  // with Ptolemy flips
  VectorX lambdas_full_flip;
  std::vector<int> flip_seq_init;
  VectorX lambdas_full;
  expand_reduced_function(proj, lambdas, lambdas_full);
  if (!initial_ptolemy) {
    Mesh<double> m_flip;
    flip_seq_init = make_delaunay_overlay(mo, initial_ptolemy);
    mo.garbage_collection();
    flip_edges(m, lambdas_full, flip_seq_init, m_flip, lambdas_full_flip);
    std::cout << "Flips to make input Delaunay: " << flip_seq_init.size()
              << std::endl;

    // Remap barycentric coordinates if Euclidean flips were used
    bc_original_to_eq_overlay(mo);
  } else {
    lambdas_full_flip = lambdas_full;
  }
  original_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.l);

  // Optionally change lengths in initial triangulation
  VectorX lambdas_flip_he;
  expand_edge_func(m, lambdas_full_flip, lambdas_flip_he);
  VectorX lambdas_init_flip_he;
  lambdas_init_flip_he.setZero(mc.l.size());
  for (int i = 0; i < mc.l.size(); ++i) {
    lambdas_init_flip_he[i] = 2.0 * log(mc.l[i]);
  }
  if (!flip_in_original_metric) {
    spdlog::info("Checking overlay mesh alignment before changing shear");
    for (int h = 0; h < mc.n_halfedges(); ++h)
    {
      mo.check_bc_alignment(&mc, h);
    }
    change_shear_overlay(mo, lambdas_flip_he, tau);
    spdlog::info("Checking overlay mesh alignment after changing shear");
    for (int h = 0; h < mc.n_halfedges(); ++h)
    {
      mo.check_bc_alignment(&mc, h);
    }
  }

  // Make final mesh Delaunay
  VectorX lambdas_full_del;
  MatrixX J_del;
  std::vector<int> flip_seq;
  make_delaunay_with_jacobian_in_place(
    mo, lambdas_full_flip, lambdas_full_del, J_del, flip_seq, false);
  VectorX lambdas_del_he;
  expand_edge_func(m, lambdas_full_del, lambdas_del_he);
  std::cout << "Flips to make output Delaunay: " << flip_seq.size()
            << std::endl;
  spdlog::info("Checking overlay mesh alignment after flipping to Delaunay");
  for (int h = 0; h < mc.n_halfedges(); ++h)
  {
    mo.check_bc_alignment(&mc, h);
  }

  // Interpolate points in the original mesh
  mo.garbage_collection();
  std::vector<std::vector<double>> v_reindex(3);
  for (int i = 0; i < 3; i++) {
    v_reindex[i].resize(mc.out.size(), 0);
    for (int j = 0; j < v.rows(); j++) {
      v_reindex[i][j] = v(vtx_reindex[j], i);
    }
  }
  std::vector<std::vector<double>> v_overlay;
  // std::vector<int> flip_seq_full(flip_seq_init);
  // flip_seq_full.insert(flip_seq_full.end(), flip_seq.begin(),
  // flip_seq.end());

  // Change edge lengths and translate barycentric coordinates
  VectorX tau_rev;
  tau_rev.setZero(tau_init.size());
  if (flip_in_original_metric) {
    for (int i = 0; i < tau_init.size(); ++i) {
      tau_rev[i] = -tau_init[i];
    }
    v_overlay = Interpolate_3d_reparametrized(
      mo, lambdas_init_flip_he, tau_rev, flip_seq_init, flip_seq, v_reindex);
    spdlog::info("Checking overlay mesh alignment before changing shear in the original metric");
    for (int h = 0; h < mc.n_halfedges(); ++h)
    {
      mo.check_bc_alignment(&mc, h);
    }
    change_shear_overlay(mo, lambdas_del_he, tau);
    spdlog::info("Checking overlay mesh alignment after changing shear in the original metric");
    for (int h = 0; h < mc.n_halfedges(); ++h)
    {
      mo.check_bc_alignment(&mc, h);
    }
  } else {
    for (int i = 0; i < tau_init.size(); ++i) {
      tau_rev[i] = -tau_init[i] - tau[i];
    }
    v_overlay = Interpolate_3d_reparametrized(
      mo, lambdas_init_flip_he, tau_rev, flip_seq_init, flip_seq, v_reindex);
  }

  // Optionally do additional reparametrization
  bc_reparametrize_eq(mo, tau_post);
  reparametrize_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, tau_post);

  // Check final overlay mesh
  spdlog::info("Checking final overlay mesh alignment");
	for (int h = 0; h < mc.n_halfedges(); ++h)
	{
		mo.check_bc_alignment(&mc, h);
	}

  // Map barycentric coordinates from equilateral to scaled triangle
  VectorX u0;
  u0.setZero(mc.n_ind_vertices());
  equilateral_to_scaled(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, u0);

  // Overwrite points with output
  int cnt = 0;
  for (auto pt : mc.pts) {
    pt_fids[cnt] = pt.f_id;
    pt_bcs[cnt] = pt.bc;
    cnt++;
  }

  // Remove tufted cover
  if (bnd_loops.size() != 0) {
    int n_v = v.rows();
    // auto mc = mo.cmesh();
    create_tufted_cover(
      mc.type, mc.R, indep_vtx, dep_vtx, v_rep, mc.out, mc.to);
    mc.v_rep = range(0, n_v);
  }

  // Eigen::Vector to std::vector for pybind
  std::vector<std::vector<double>> pt_bcs_out(pt_bcs.size());
  for (int i = 0; i < pt_bcs.size(); i++) {
    for (int j = 0; j < 3; j++) {
      pt_bcs_out[i].push_back(pt_bcs[i](j));
    }
  }
  std::vector<std::pair<int, int>> endpoints;
  find_origin_endpoints<double>(mo, endpoints);

  spdlog::info("Done with generating optimized overlay");
  return std::make_tuple(
    mo, pt_fids, pt_bcs_out, v_overlay, vtx_reindex, endpoints);
}

#ifdef PYBIND

#endif

}
