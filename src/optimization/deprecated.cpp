Deprecated Code 

std::tuple<std::vector<std::vector<double>>, // V_out
           std::vector<std::vector<int>>,    // F_out
           std::vector<double>,              // layout u (per vertex)
           std::vector<double>,              // layout v (per vertex)
           std::vector<std::vector<int>>,    // FT_out
           std::vector<bool>,                // is_cut_h
           std::vector<bool>,                // is_cut_o
           std::vector<int>,                 // Fn_to_F
           std::vector<std::pair<int, int>>, // endpoints
           OverlayMesh<double>>              // endpoints
_parametrize_mesh(const Eigen::MatrixXd& V,
                  const Eigen::MatrixXi& F,
                  const std::vector<double>& Theta_hat,
                  const std::vector<double>& lambdas,
                  const std::vector<double>& tau_init,
                  const std::vector<double>& tau,
                  const std::vector<double>& tau_post,
                  std::vector<int>& pt_fids,
                  std::vector<Eigen::Matrix<double, 3, 1>>& pt_bcs,
                  bool initial_ptolemy = false,
                  bool flip_in_original_metric = true)
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

  // do conformal_metric
  auto overlay_out = generate_optimized_overlay(V,
                                                F,
                                                Theta_hat,
                                                lambdas,
                                                tau_init,
                                                tau,
                                                tau_post,
                                                pt_fids,
                                                pt_bcs,
                                                initial_ptolemy,
                                                flip_in_original_metric);
  OverlayMesh<double> mo = std::get<0>(overlay_out);
  auto V_overlay = std::get<3>(overlay_out);
  std::vector<int> vtx_reindex = std::get<4>(overlay_out);
  auto endpoints = std::get<5>(overlay_out);
  std::vector<double> u(mo._m.Th_hat.size(), 0.0);

  std::vector<int> f_labels = get_overlay_face_labels(mo);

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

  // get layout
  // FIXME Should be layout
  auto layout_res = get_layout_x(mo, u, bd, cones, do_trim);
  auto is_cut = std::get<2>(layout_res);
  auto u_o = std::get<3>(layout_res);
  auto v_o = std::get<4>(layout_res);
  auto is_cut_o = std::get<5>(layout_res);

  // get output VF and metric
  auto FVFT_res = get_FV_FTVT(mo, endpoints, is_cut_o, V_overlay, u_o, v_o);
  auto v3d = std::get<0>(FVFT_res);
  auto u_o_out = std::get<1>(FVFT_res);
  auto v_o_out = std::get<2>(FVFT_res);
  auto F_out = std::get<3>(FVFT_res);
  auto FT_out = std::get<4>(FVFT_res);
  auto Fn_to_F = std::get<5>(FVFT_res);
  auto remapped_endpoints = std::get<6>(FVFT_res);

  // v3d_out = v3d^T
  std::vector<std::vector<double>> v3d_out(v3d[0].size());
  for (int i = 0; i < v3d[0].size(); i++) {
    v3d_out[i].resize(3);
    for (int j = 0; j < 3; j++) {
      v3d_out[i][j] = v3d[j][i];
    }
  }

  // reindex back
  auto u_o_out_copy = u_o_out;
  auto v_o_out_copy = v_o_out;
  auto v3d_out_copy = v3d_out;
  auto endpoints_out = remapped_endpoints;
  for (int i = 0; i < F_out.size(); i++) {
    for (int j = 0; j < 3; j++) {
      if (F_out[i][j] < vtx_reindex.size()) {
        F_out[i][j] = vtx_reindex[F_out[i][j]];
      }
      if (FT_out[i][j] < vtx_reindex.size()) {
        FT_out[i][j] = vtx_reindex[FT_out[i][j]];
      }
    }
  }
  for (int i = 0; i < vtx_reindex.size(); i++) {
    u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
    v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
    v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
  }
  for (int i = vtx_reindex.size(); i < endpoints_out.size(); i++) {
    int a = vtx_reindex[endpoints_out[i].first];
    int b = vtx_reindex[endpoints_out[i].second];
    endpoints_out[i] = std::make_pair(a, b);
  }

  return std::make_tuple(v3d_out,
                         F_out,
                         u_o_out,
                         v_o_out,
                         FT_out,
                         is_cut,
                         is_cut_o,
                         Fn_to_F,
                         endpoints_out,
                         mo);
}

// FIXME Implement interpolation of colormaps
// std::vector<int> active_faces;
// for (int i = 0; i < F.rows(); i++)
//{
//    if (F.row(i).sum())
//    {
//        active_faces.push_back(i);
//    }
//}
// int index_f = 0;
// for (int f: active_faces)
//{
//    F.row(index_f) << F.row(f);
//    index_f++;
//}
//
// F.conservativeResize(index_f, 3);
//
//// remove unreferenced vertices
// MatrixX<double> V_unref;
// Eigen::MatrixXi F_unref;
// igl::remove_unreferenced(V, F, V_unref, F_unref, V_to_Vn, Vn_to_V); // I size
// V, J size Vn std::vector<std::pair<int,int>> remapped_endpoints(Vn.rows(),
// std::make_pair(-1, -1)); for (int i = 0; i < endpoints.size(); i++)
//{
//     int j = V_to_Vn[i];
//     int a = endpoints[i].first, b = endpoints[i].second;
//     if(a == -1 || b == -1) continue;
//     remapped_endpoints[V_to_Vn[i]] = std::make_pair(V_to_Vn[a], V_to_Vn[b]);
// }
//
//  FIXME Below here lies spaghetti

// FIXME REMOVE
std::tuple<OverlayMesh<double>,
           std::vector<int>,
           std::vector<std::vector<double>>,
           std::vector<int>,
           std::vector<std::vector<double>>>
_optimize_metric(const Eigen::MatrixXd& V_init,
                 const Eigen::MatrixXi& F_init,
                 const std::vector<double>& Th_hat_init,
                 std::vector<int>& pt_fids,
                 std::vector<Eigen::Matrix<double, 3, 1>>& pt_bcs,
                 const Mesh<double>& m,
                 const std::vector<double>& lambdas,
                 const std::vector<int>& flip_seq_init)
{
  // Get boundary list, reindex map, and independent vertices
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  FV_to_double(V_init,
               F_init,
               Th_hat_init,
               vtx_reindex,
               indep_vtx,
               dep_vtx,
               v_rep,
               bnd_loops,
               false);

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Convert embedded mesh log edge lengths to a full mesh log length array
  std::vector<double> lambdas_full(he2e.size());
  for (int e = 0; e < proj.size(); ++e) {
    lambdas_full[e] = lambdas[proj[e]];
  }

  // Make mesh Delaunay with flip sequence
  Mesh<double> m_del;
  std::vector<double> lambdas_full_del;
  MatrixX J_del;
  std::vector<int> flip_seq(flip_seq_init);
  // std::vector<int> flip_seq;
  // flip_seq.clear();
  make_delaunay_with_jacobian(
    m, lambdas_full, m_del, lambdas_full_del, J_del, flip_seq, false);

  // Convert mesh log edge lengths to a halfedge length array l for m
  // FIXME Remove std::vector<double> l(he2e.size());
  Mesh<double> m_l = m;
  for (int h = 0; h < he2e.size(); ++h) {
    m_l.l[h] = exp(lambdas_full[he2e[h]] / 2.0);
  }

  // Build overlay mesh from mesh m with initial flip sequence
  // FIXME Replace m with internal construction
  VectorX angles, cot_angles;
  VectorX g, u0;
  u0.setZero(m.n_ind_vertices());
  OverlayMesh<double> mo(m_l);
  Mesh<double>& mc = mo.cmesh();

  mc.init_pts(pt_fids, pt_bcs);
  original_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.l);

  // Perform flip sequence to make mesh Delaunay
  // for (int i = flip_seq_init.size(); i < flip_seq.size(); ++i)
  // for (int i = 0; i < flip_seq.size(); ++i)
  // {
  //    int h = flip_seq[i];
  //    if (h < 0)
  //    {
  //        mo.flip_ccw(-h-1, false);
  //    }
  //    else
  //    {
  //       mo.flip_ccw(h, true);
  //    }
  //}
  DelaunayStats del_stats;
  SolveStats<double> solve_stats;
  ConformalIdealDelaunay<double>::MakeDelaunay(mo, u0, del_stats, solve_stats);
  ConformalIdealDelaunay<double>::MakeDelaunay(mo, u0, del_stats, solve_stats);
  equilateral_to_scaled(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, u0);

  int cnt = 0;
  for (auto pt : mc.pts) {
    pt_fids[cnt] = pt.f_id;
    pt_bcs[cnt] = pt.bc;
    cnt++;
  }

  for (int i = 0; i < mc.n.size(); ++i) {
    if (mc.n[i] != m_del.n[i])
      std::cout << "ERROR";
  }

  ConformalIdealDelaunay<double>::ComputeAngles(mc, u0, angles, cot_angles);
  ConformalIdealDelaunay<double>::Gradient(mc, angles, g, solve_stats);
  std::cout << "Angle error: " << g.cwiseAbs().maxCoeff() << std::endl;

  for (int h = 0; h < he2e.size(); ++h) {
    if (ConformalIdealDelaunay<double>::NonDelaunay(mc, u0, h, solve_stats))
      std::cout << "NonDelaunay" << std::endl;
  }

  // Get overlay vertices and face labels
  mo.garbage_collection();
  std::vector<std::vector<double>> V_reindex(3);
  for (int i = 0; i < 3; i++) {
    V_reindex[i].resize(mo._m.out.size(), 0);
    for (int j = 0; j < V_init.rows(); j++) {
      V_reindex[i][j] = double(V_init(vtx_reindex[j], i));
    }
  }
  std::vector<std::vector<double>> V_overlay =
    ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq, V_reindex);
  if (bnd_loops.size() != 0) {
    int n_v = V_init.rows();
    auto mc = mo.cmesh();
    create_tufted_cover(
      mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep, mo._m.out, mo._m.to);
    mo._m.v_rep = range(0, n_v);
  }

  // Eigen::Vector to std::vector for pybind
  std::vector<std::vector<double>> pt_bcs_out(pt_bcs.size());
  for (int i = 0; i < pt_bcs.size(); i++) {
    for (int j = 0; j < 3; j++) {
      pt_bcs_out[i].push_back(pt_bcs[i](j));
    }
  }
  // std::vector<std::vector<double>> pt_bcs_out(pt_bcs.size());
  // std::vector<std::vector<double>> V_overlay;
  return std::make_tuple(mo, pt_fids, pt_bcs_out, vtx_reindex, V_overlay);
}

template<typename Scalar>
std::tuple<std::vector<Scalar>,
           std::vector<Scalar>,
           std::vector<bool>,
           std::vector<Scalar>,
           std::vector<Scalar>,
           std::vector<bool>>
get_layout_overlay(OverlayMesh<Scalar>& m_o,
                   const std::vector<Scalar>& u_vec,
                   std::vector<int> bd,
                   std::vector<int> singularities,
                   bool do_trim = false,
                   int root = -1)
{

  auto f_type = get_overlay_face_labels(m_o);

  // layout the current mesh with arbitrary cut
  std::vector<bool> is_cut;
  auto mc = m_o.cmesh();
  mc.type = std::vector<char>(mc.n_halfedges(), 0);
  auto layout_res = compute_layout(mc, u_vec, is_cut, 0);
  auto u_scalar = std::get<0>(layout_res);
  auto v_scalar = std::get<1>(layout_res);

  Eigen::Matrix<Scalar, -1, 1> u_eig;
  u_eig.resize(u_vec.size());
  for (int i = 0; i < u_vec.size(); i++) {
    u_eig(i) = u_vec[i];
  }

  m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);

  auto u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, u_scalar);
  auto v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, v_scalar);

  spdlog::info("Interpolate on overlay mesh done.");

  // compute edge lengths of overlay mesh and triangulate it
  Mesh<Scalar> m;
  m.n = m_o.n;
  m.opp = m_o.opp;
  m.f = m_o.f;
  m.h = m_o.h;
  m.out = m_o.out;
  m.to = m_o.to;
  m.l = std::vector<Scalar>(m.n.size(), 0.0);
  for (int i = 0; i < m.n.size(); i++) {
    int h0 = i;
    int h1 = h0;
    do {
      if (m.n[h1] == h0)
        break;
      h1 = m.n[h1];
    } while (h0 != h1);
    if (m.to[m.opp[h0]] != m.to[h1]) {
      spdlog::error("h0 h1 picked wrong.");
      exit(0);
    }
    m.l[h0] = sqrt((u_o[h0] - u_o[h1]) * (u_o[h0] - u_o[h1]) +
                   (v_o[h0] - v_o[h1]) * (v_o[h0] - v_o[h1]));
  }
  triangulate_polygon_mesh(m, u_o, v_o, f_type);

  m.type = std::vector<char>(m.n.size(), 0);
  m.type_input = m.type;
  m.R = std::vector<int>(m.n.size(), 0);
  m.v_rep = range(0, m.out.size());
  m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);

  // try to connect to singularties again with overlay mesh edges
  spdlog::info("try to connect to singularities using a tree rooted at root");
  std::vector<bool> is_cut_h;
  OverlayMesh<Scalar> mo(m);
  for (int i = m_o.n.size(); i < mo.n.size(); i++) {
    mo.edge_type[i] = ORIGINAL_EDGE; // make sure do not use the new diagonal
  }
  connect_to_singularities(mo, f_type, bd, singularities, is_cut_h, root);

  int start_h = 0;
  for (int i = 0; i < mo.n.size(); i++) {
    if (f_type[mo.f[i]] == 1 && f_type[mo.f[mo.opp[i]]] == 2) {
      start_h = i;
      break;
    }
  }
  spdlog::info("selected start h: {}, left: {}, right: {}",
               start_h,
               f_type[mo.f[start_h]],
               f_type[mo.f[mo.opp[start_h]]]);

  // sanity check for the input of compute layout
  // - opposite halfedges should have same edge lenghts (up to numerical error)
  // - all halfedges that belongs to a face with type 1 should have non-zero
  // edge lengths
  for (int i = 0; i < mo.n.size(); i++) {
    int h0 = i, h1 = mo.opp[h0];
    int ft0 = f_type[mo.f[h0]];
    int ft1 = f_type[mo.f[h1]];
    if (std::abs<Scalar>(mo._m.l[h0] - mo._m.l[h1]) > 1e-12 && ft0 == ft1 &&
        ft0 == 1) {
      spdlog::error("halfedge lengths mismatch, {}: {}, {}: {}; {}/{}",
                    h0,
                    mo._m.l[h0],
                    h1,
                    mo._m.l[h1],
                    ft0,
                    ft1);
    }
    int f0 = mo.f[h0];
    int f1 = mo.f[h1];
    if (f_type[f0] == 1 && mo._m.l[h0] == 0)
      spdlog::error("copy 1 has zero edge at {}, f{}", h0, f0);
    if (f_type[f1] == 1 && mo._m.l[h1] == 0)
      spdlog::error("copy 1 has zero edge at {}, f{}", h1, f1);
  }
  spdlog::info("sanity check done.");

  // mark boundary as cut
  for (int i = 0; i < is_cut_h.size(); i++) {
    if (f_type[mo.f[i]] != f_type[mo.f[mo.opp[i]]]) {
      is_cut_h[i] = true;
    }
  }
  // now directly do layout on overlay mesh
  for (int f = 0; f < f_type.size(); f++) {
    int h0 = mo.h[f];
    int h1 = mo.n[h0];
    int h2 = mo.n[h1];
    mo._m.type[h0] = f_type[f];
    mo._m.type[h1] = f_type[f];
    mo._m.type[h2] = f_type[f];
  }

  // get output connectivity and metric
  std::vector<Scalar> phi(mo.out.size(), 0.0);
  auto overlay_layout_res = compute_layout(mo._m, phi, is_cut_h, start_h);
  auto _u_o = std::get<0>(overlay_layout_res);
  auto _v_o = std::get<1>(overlay_layout_res);

  is_cut_h.resize(m_o.n.size()); // ignoring the newly added diagonal
  _u_o.resize(m_o.n.size());
  _v_o.resize(m_o.n.size());
  return std::make_tuple(u_scalar, v_scalar, is_cut_h, _u_o, _v_o, is_cut_h);
}
std::tuple<OverlayMesh<double>,              // m
           std::vector<double>,              // u
           std::vector<int>,                 // pt_fids
           std::vector<std::vector<double>>, // pt_bcs
           std::vector<int>,                 // vtx_reindex
           std::vector<std::vector<double>>, // V_overlay
           std::vector<int>,                 // flip_seq
           std::vector<std::pair<int, int>>>
mesh_metric(const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const std::vector<double>& Theta_hat,
            const std::vector<double>& l,
            std::vector<int>& pt_fids,
            std::vector<Eigen::Matrix<double, 3, 1>>& pt_bcs,
            std::shared_ptr<AlgorithmParameters> alg_params = nullptr,
            std::shared_ptr<LineSearchParameters> ls_params = nullptr,
            std::shared_ptr<StatsParameters> stats_params = nullptr,
            bool initial_ptolemy = true,
            bool make_tufted = true)
{

  if (alg_params == nullptr)
    alg_params = std::make_shared<AlgorithmParameters>();
  if (ls_params == nullptr)
    ls_params = std::make_shared<LineSearchParameters>();
  if (stats_params == nullptr)
    stats_params = std::make_shared<StatsParameters>();
  alg_params->max_itr = 1;
  alg_params->initial_ptolemy = initial_ptolemy;

#ifdef WITH_MPFR
  mpfr::mpreal::set_default_prec(alg_params->MPFR_PREC);
  mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
  mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

  std::vector<double> u;
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  Mesh<double> m = FV_to_double(
    V, F, Theta_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);
  for (int i = 0; i < l.size(); ++i) {
    m.l[i] = l[i];
  }

  OverlayMesh<double> mo(m);

  Eigen::Matrix<double, Eigen::Dynamic, 1> u0;
  u0.setZero(m.n_ind_vertices());
  std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs_scalar(pt_bcs.size());
  for (int i = 0; i < pt_bcs.size(); i++) {
    pt_bcs_scalar[i] = pt_bcs[i].template cast<double>();
  }
  auto conformal_out = ConformalIdealDelaunay<double>::FindConformalMetric(
    mo, u0, pt_fids, pt_bcs_scalar, *alg_params, *ls_params, *stats_params);
  auto u_o = std::get<0>(conformal_out);
  auto flip_seq = std::get<1>(conformal_out);
  u.resize(u_o.rows());
  for (int i = 0; i < u_o.rows(); i++)
    u[i] = u_o[i];
  mo.garbage_collection();
  std::vector<std::vector<double>> V_reindex(3);
  std::cout << "Sizes: " << V.rows() << " " << mo._m.out.size() << std::endl;
  for (int i = 0; i < 3; i++) {
    V_reindex[i].resize(mo._m.out.size(), 0);
    for (int j = 0; j < V.rows(); j++) {
      V_reindex[i][j] = double(V(vtx_reindex[j], i));
    }
  }
  std::vector<std::vector<double>> V_overlay;
  if (!mo.bypass_overlay)
    V_overlay =
      ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq, V_reindex);
  if (mo.bypass_overlay)
    spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
  if ((bnd_loops.size() != 0) && (make_tufted)) {
    int n_v = V.rows();
    auto mc = mo.cmesh();
    create_tufted_cover(
      mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep, mo._m.out, mo._m.to);
    mo._m.v_rep = range(0, n_v);
  }

  // Eigen::Vector to std::vector for pybind
  std::vector<std::vector<double>> pt_bcs_out(pt_bcs_scalar.size());
  for (int i = 0; i < pt_bcs_scalar.size(); i++) {
    for (int j = 0; j < 3; j++) {
      pt_bcs_out[i].push_back(pt_bcs_scalar[i](j));
    }
  }
  std::vector<std::pair<int, int>> endpoints;
  find_origin_endpoints<double>(mo, endpoints);

  return std::make_tuple(
    mo, u, pt_fids, pt_bcs_out, vtx_reindex, V_overlay, flip_seq, endpoints);
}

std::tuple<std::vector<std::vector<double>>, // V_out
           std::vector<std::vector<int>>,    // F_out
           std::vector<double>,              // layout u (per vertex)
           std::vector<double>,              // layout v (per vertex)
           std::vector<double>,              // layout u (per vertex)
           std::vector<double>,              // layout v (per vertex)
           std::vector<std::vector<int>>,    // FT_out
           OverlayMesh<double>,
           std::vector<std::vector<double>>> // V_overlay
mesh_parametrization_VL(const Eigen::MatrixXd& V,
                        const Eigen::MatrixXi& F,
                        const std::vector<double>& Theta_hat,
                        const Mesh<double>& m,
                        const std::vector<double>& lambdas)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Convert embedded mesh log edge lengths to a full mesh log length array
  std::vector<double> l(he2e.size());
  for (int h = 0; h < he2e.size(); ++h) {
    l[h] = exp(lambdas[proj[he2e[h]]] / 2.0);
  }

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

  // do conformal_metric
  std::vector<int> pt_fids_placeholder;
  std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs_placeholder;
  auto conformal_out =
    mesh_metric(V, F, Theta_hat, l, pt_fids_placeholder, pt_bcs_placeholder);
  OverlayMesh<double> mo = std::get<0>(conformal_out);
  std::vector<double> u = std::get<1>(conformal_out);
  std::vector<int> vtx_reindex = std::get<4>(conformal_out);
  auto V_overlay = std::get<5>(conformal_out);
  auto endpoints = std::get<7>(conformal_out);

  if (mo.bypass_overlay) {
    spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
    return std::make_tuple(std::vector<std::vector<double>>(),
                           std::vector<std::vector<int>>(),
                           std::vector<double>(),
                           std::vector<double>(),
                           std::vector<double>(),
                           std::vector<double>(),
                           std::vector<std::vector<int>>(),
                           mo,
                           V_overlay);
  }

  std::vector<int> f_labels = get_overlay_face_labels(mo);

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
  spdlog::info("#bd_vt: {}", bd.size());
  spdlog::info("#cones: {}", cones.size());
  spdlog::info("vtx reindex size: {}", vtx_reindex.size());
  spdlog::info("mc.out size: {}", mo.cmesh().out.size());

  // get layout
  auto mo_copy = mo;
  auto layout_res = get_layout_overlay(mo_copy, u, bd, cones, do_trim);
  auto u_o = std::get<3>(layout_res);
  auto v_o = std::get<4>(layout_res);
  auto is_cut_o = std::get<5>(layout_res);

  // get output VF and metric
  auto FVFT_res =
    get_FV_FTVT(mo_copy, endpoints, is_cut_o, V_overlay, u_o, v_o);
  auto v3d = std::get<0>(FVFT_res);
  auto u_o_out = std::get<1>(FVFT_res);
  auto v_o_out = std::get<2>(FVFT_res);
  auto F_out = std::get<3>(FVFT_res);
  auto FT_out = std::get<4>(FVFT_res);

  // v3d_out = v3d^T
  std::vector<std::vector<double>> v3d_out(v3d[0].size());
  for (int i = 0; i < v3d[0].size(); i++) {
    v3d_out[i].resize(3);
    for (int j = 0; j < 3; j++) {
      v3d_out[i][j] = v3d[j][i];
    }
  }

  // reindex back
  auto u_o_out_copy = u_o_out;
  auto v_o_out_copy = v_o_out;
  auto v3d_out_copy = v3d_out;
  for (int i = 0; i < F_out.size(); i++) {
    for (int j = 0; j < 3; j++) {
      if (F_out[i][j] < vtx_reindex.size()) {
        F_out[i][j] = vtx_reindex[F_out[i][j]];
      }
      if (FT_out[i][j] < vtx_reindex.size()) {
        FT_out[i][j] = vtx_reindex[FT_out[i][j]];
      }
    }
  }
  for (int i = 0; i < vtx_reindex.size(); i++) {
    u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
    v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
    v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
  }

  return std::make_tuple(
    v3d_out, F_out, u_o, v_o, u_o_out, v_o_out, FT_out, mo, V_overlay);
}

/**
 * Convert triangle mesh in V, F format to halfedge structure.
 *
 * @param V dim #v*3 matrix, each row corresponds to mesh vertex coordinates
 * @param F dim #f*3 matrix, each row corresponds to three vertex ids of each
 * facet
 * @param Theta_hat dim #v vector, each element is the prescribed angle sum at
 * each vertex
 * @param vtx_reindex, dim #v int-vector, stores the correspondence of new
 * vertex id in mesh m to old index in V
 * @param indep_vtx, int-vector, stores index of identified independent vertices
 * in the original copy
 * @param dep_vtx, int-vector, stores index of new added vertex copies of the
 * double cover
 * @param v_rep, dim #v int-vector, map independent vertices to unique indices
 * and dependent vertices to their reflection's index
 * @return m, Mesh data structure, for details check OverlayMesh.hh
 */
/*
template <typename Scalar>
Mesh<Scalar>
NOB_to_double(const std::vector<int>& next_he, const std::vector<int>opp, const
std::vector<int>& bnd_loops, const std::vector<Scalar> &Theta_hat,
std::vector<int>& indep_vtx, std::vector<int>& dep_vtx, std::vector<int>& v_rep,
bool fix_boundary=false){ Mesh<Scalar> m;

    // Build the connectivity arrays from the NOB arrays
    Connectivity C;
    NOB_to_connectivity(next_he, opp, bnd_loops, C);

    // Build the edge length array with initial unit lengths
    std::vector<Scalar> l(next_he.size(), Scalar(1));

    // Permute the target angles to match the new vertex indices of the halfedge
mesh std::vector<Scalar> Theta_hat_perm(Theta_hat.size()); for (int i = 0; i <
Theta_hat_perm.size(); ++i)
    {
        Theta_hat_perm[i] = Theta_hat[i];
    }

    // If there is no boundary, create a mesh with trivial reflection
information if (bnd_loops.size() == 0)
    {
        int n_v = C.out.size();
        int n_he = C.n.size();

        // Create trivial reflection information
        std::vector<char> type(n_he, 0);
        std::vector<int> R(n_he, 0);

        // Create a halfedge structure for the mesh
        m.n = C.n;
        m.to = C.to;
        m.f = C.f;
        m.h = C.h;
        m.out = C.out;
        m.opp = C.opp;
        m.type = type;
        m.type_input = type;
        m.R = R;
        m.l = l;
        m.Th_hat = Theta_hat_perm;
        m.v_rep = range(0, n_v);
        m.fixed_dof = std::vector<bool>(n_v, false);
        m.fixed_dof[0] = true;
    }
    // If there is boundary, create a double tufted cover with a reflection map
    else
    {
        // Create the doubled mesh connectivity information
        Connectivity C_double;
        std::vector<char> type;
        std::vector<int> R;
        NOB_to_double(next_he, opp, bnd_loops, C_double, type, R);
        find_indep_vertices(C_double.out, C_double.to, type, R, indep_vtx,
dep_vtx, v_rep); int n_v = C.out.size(); int n_v_double = C_double.out.size();

        // Double the target angle array
        std::vector<Scalar> Theta_hat_double(n_v);
        for (int i = 0; i < n_v; ++i)
        {
            Theta_hat_double[i] =
2*Theta_hat_perm[C.to[C.opp[C_double.out[indep_vtx[i]]]]];
        }

        // Double the length array FIXME Only works for double tufted cover
        std::vector<Scalar> l_double(C_double.n.size(), Scalar(1));

        // Create the halfedge structure for the doubled mesh
        m.n = C_double.n;
        m.to = C_double.to;
        m.f = C_double.f;
        m.h = C_double.h;
        m.out = C_double.out;
        m.opp = C_double.opp;
        m.type = type;
        m.type_input = type;
        m.R = R;
        m.l = l_double;
        m.Th_hat = Theta_hat_double;
        m.v_rep = v_rep;

        // Set the fixed_dof to the first boundary halfedge for symmetric meshes
        // If fixing boundary, fix all boundary halfedges
        m.fixed_dof = std::vector<bool>(n_v, false);
        for (int i = 0; i < n_v_double; ++i)
        {
            if (m.to[m.R[m.out[i]]] == i)
            {
                m.fixed_dof[m.v_rep[i]] = true;
                if (!fix_boundary) break;
            }
        }

    }
    return m;
}




OverlayMesh<double> build_overlay_mesh(const Mesh<double> &m,
                                       const std::vector<double> &lambdas,
                                       std::vector<int> &flip_seq,
                                       bool use_log_lengths=false)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Build refl projection and embedding
    std::vector<int> proj;
    std::vector<int> embed;
    build_refl_proj(m, proj, embed);

    // Convert embedded mesh log edge lengths to a full mesh log length array
    std::vector<double> lambdas_full(he2e.size());
    for (int e = 0; e < proj.size(); ++e)
    {
        lambdas_full[e] = lambdas[proj[e]];
    }

    // Make mesh Delaunay with flip sequence
    Mesh<double> m_del;
    std::vector<double> lambdas_full_del;
    MatrixX J_del;
    flip_seq.clear();
    make_delaunay_with_jacobian(m, lambdas_full, m_del, lambdas_full_del, J_del,
flip_seq, false);

    // Build overlay mesh from mesh m with initial flip sequence
    OverlayMesh<double> mo(m);
    Mesh<double>& mc = mo.cmesh();

    // Convert mesh log edge lengths to a halfedge length array l for m
    for (int h = 0; h < he2e.size(); ++h)
    {
       mc.l[h] = exp(lambdas_full[he2e[h]] / 2.0);
    }

    // Perform flip sequence to make mesh Delaunay
    VectorX u0;
    u0.setZero(mc.n_ind_vertices());
    DelaunayStats del_stats;
    SolveStats<double> solve_stats;
    if (use_log_lengths)
    {
        // Use same flip sequence as for MakeDelauany with log lengths
        for (int i = 0; i < flip_seq.size(); ++i)
        {
            int h = flip_seq[i];
            if (h < 0)
            {
                mo.flip_ccw(-h-1, false);
            }
            else
            {
                mo.flip_ccw(h, true);
            }
        }
    }
    else
    {
        // Make Delaunay with actual lengths
        ConformalIdealDelaunay<double>::MakeDelaunay(mo, u0, del_stats,
solve_stats);
    }

    // Check for discrepencies with the connectivity of the log lambdas Delaunay
output for (int i = 0; i < mc.n.size(); ++i)
    {
        if (mc.n[i] != m_del.n[i]) std::cout << "ERROR";
    }

    // Output the angle error
    VectorX g;
    VectorX angles, cot_angles;
    ConformalIdealDelaunay<double>::ComputeAngles(mc, u0, angles, cot_angles);
    ConformalIdealDelaunay<double>::Gradient(mc, angles, g, solve_stats);
    std::cout << "Angle error: " << g.cwiseAbs().maxCoeff() << std::endl;

    // Check for NonDelaunay edges
    for (int h = 0; h < he2e.size(); ++h)
    {
        if (ConformalIdealDelaunay<double>::NonDelaunay(mc, u0, h, solve_stats))
            std::cout << "NonDelaunay" << std::endl;
    }
    mo.garbage_collection();

    return mo;
}






std::tuple<
        std::vector<double>,                    // layout u (per vertex)
        std::vector<double>,                    // layout v (per vertex)
        std::vector<double>,                    // layout u (per vertex)
        std::vector<double>,                    // layout v (per vertex)
        std::vector<std::vector<int>>>          // FT_out
layout_lambdas_NOB(const std::vector<std::vector<double>> &V,
                   const std::vector<int>& next_he,
                   const std::vector<int>opp,
                   const std::vector<int>& bnd_loops,
                   const std::vector<double> &Th_hat_init,
                   const Mesh<double> &m,
                   const std::vector<double> &lambdas)
{
    std::vector<int> indep_vtx, dep_vtx, v_rep;
    NOB_to_double(next_he, opp, bnd_loops, Th_hat_init, indep_vtx, dep_vtx,
v_rep, false);

    // Get overlay mesh for m with the embedded log lengths lambdas
    std::vector<int> flip_seq;
    OverlayMesh<double> mo = build_overlay_mesh(m, lambdas, flip_seq, false);

    // Get trival cones and boundary
    std::vector<int> cones, bd;

    // Get overlay vertices and face labels
    mo.garbage_collection();

    // Get layout
    std::vector<double> u(m.n_ind_vertices(), 0.0);
    auto layout_res = get_layout_overlay(mo, u, bd, cones);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);
    auto is_cut_o = std::get<5>(layout_res);

    // Get output VF and metric
    auto FVFT_res = get_FV_FTVT(mo, is_cut_o, V, u_o, v_o);
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);

    return std::make_tuple(u_o, v_o, u_o_out, v_o_out, FT_out);
}

std::tuple<
        std::vector<double>,                    // layout u (per vertex)
        std::vector<double>,                    // layout v (per vertex)
        std::vector<bool>,
        std::vector<double>,                    // layout u (per vertex)
        std::vector<double>,                    // layout v (per vertex)
        std::vector<std::vector<int>>>          // FT_out
layout_lambdas_FV(const Eigen::MatrixXd &V_init,
                  const Eigen::MatrixXi &F_init,
                  const std::vector<double> &Th_hat_init,
                  const Mesh<double> &m,
                  const std::vector<double> &lambdas)
{
    // Get boundary list, reindex map, and independent vertices
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    FV_to_double(V_init, F_init, Th_hat_init, vtx_reindex, indep_vtx, dep_vtx,
v_rep, bnd_loops, false);

    // Get overlay mesh for m with the embedded log lengths lambdas
    std::vector<int> flip_seq;
    OverlayMesh<double> mo = build_overlay_mesh(m, lambdas, flip_seq, false);

    // Get cones and boundary
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F_init);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    for (int i = 0; i < Th_hat_init.size(); i++)
    {
        if ((!is_bd[i]) && abs(Th_hat_init[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // Get overlay vertices and face labels
    mo.garbage_collection();
    std::vector<std::vector<double>> V_reindex(3);
    for (int i = 0; i < 3; i++)
    {
        V_reindex[i].resize(mo._m.out.size(), 0);
        for (int j = 0; j < V_init.rows(); j++)
        {
            V_reindex[i][j] = double(V_init(vtx_reindex[j], i));
        }
    }
    std::vector<std::vector<double>> V_overlay =
ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq, V_reindex); if
(bnd_loops.size() != 0)
    {
        int n_v = V_init.rows();
        auto mc = mo.cmesh();
        create_tufted_cover(mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep,
mo._m.out, mo._m.to); mo._m.v_rep = range(0, n_v);
    }
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // Reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    // Get layout
    std::vector<double> u(m.n_ind_vertices(), 0.0);
    auto layout_res = get_layout_overlay(mo, u, bd, cones);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);
    auto is_cut_o = std::get<5>(layout_res);

    auto FVFT_res = get_FV_FTVT(mo, is_cut_o, V_overlay, u_o, v_o);
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);

    return std::make_tuple(u_o, v_o, is_cut_o, u_o_out, v_o_out, FT_out);
}

std::vector<std::vector<int>> get_faces(OverlayMesh<double> mo)
{
    std::vector<std::vector<int>> f;
    for (int i = 0; i < mo.h.size(); ++i)
    {
        int h = mo.h[i];
        int h0 = h;
        do
        {
            f.push_back(std::vector<int>({h, mo.n[h], mo.n[mo.n[h]]}));
            h = mo.n[h];
        }
        while (mo.n[h] != h0);
    }

    return f;
}

std::vector<std::vector<double>>       // V_out
Interpolate_3d(const Eigen::MatrixXd &V_init,
               const Eigen::MatrixXi &F_init,
               const std::vector<double> &Th_hat_init,
               const Mesh<double> &m,
               const std::vector<double> &lambdas)
{
    // Get boundary list, reindex map, and independent vertices
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    FV_to_double(V_init, F_init, Th_hat_init, vtx_reindex, indep_vtx, dep_vtx,
v_rep, bnd_loops, false);

    // Get overlay mesh for m with the embedded log lengths lambdas
    std::vector<int> flip_seq;
    OverlayMesh<double> mo = build_overlay_mesh(m, lambdas, flip_seq, false);

    // Get overlay vertices and face labels
    mo.garbage_collection();
    std::vector<std::vector<double>> V_reindex(3);
    for (int i = 0; i < 3; i++)
    {
        V_reindex[i].resize(mo._m.out.size(), 0);
        for (int j = 0; j < V_init.rows(); j++)
        {
            V_reindex[i][j] = double(V_init(vtx_reindex[j], i));
        }
    }

    return ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq,
V_reindex);
}


std::tuple<
        std::vector<std::vector<double>>,       // V_out
        std::vector<std::vector<int>>,          // F_out
        std::vector<double>,                    // layout u (per vertex)
        std::vector<double>,                    // layout v (per vertex)
        std::vector<double>,                    // layout u (per vertex)
        std::vector<double>,                    // layout v (per vertex)
        std::vector<std::vector<int>>>          // FT_out
layout_lambdas(const Eigen::MatrixXd &V_init,
               const Eigen::MatrixXi &F_init,
               const std::vector<double> &Th_hat_init,
               const Mesh<double> &m,
               const std::vector<double> &lambdas)
{
    // Get boundary list, reindex map, and independent vertices
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    FV_to_double(V_init, F_init, Th_hat_init, vtx_reindex, indep_vtx, dep_vtx,
v_rep, bnd_loops, false);

    // Get overlay mesh for m with the embedded log lengths lambdas
    std::vector<int> flip_seq;
    OverlayMesh<double> mo = build_overlay_mesh(m, lambdas, flip_seq, false);

    // Get cones and boundary
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F_init);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    for (int i = 0; i < Th_hat_init.size(); i++)
    {
        if ((!is_bd[i]) && abs(Th_hat_init[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // Get overlay vertices and face labels
    mo.garbage_collection();
    std::vector<std::vector<double>> V_reindex(3);
    for (int i = 0; i < 3; i++)
    {
        V_reindex[i].resize(mo._m.out.size(), 0);
        for (int j = 0; j < V_init.rows(); j++)
        {
            V_reindex[i][j] = double(V_init(vtx_reindex[j], i));
        }
    }
    std::vector<std::vector<double>> V_overlay =
ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq, V_reindex); if
(bnd_loops.size() != 0)
    {
        int n_v = V_init.rows();
        auto mc = mo.cmesh();
        create_tufted_cover(mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep,
mo._m.out, mo._m.to); mo._m.v_rep = range(0, n_v);
    }
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // Reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    // Get layout
    std::vector<double> u(m.n_ind_vertices(), 0.0);
    auto layout_res = get_layout_overlay(mo, u, bd, cones);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);
    auto is_cut_o = std::get<5>(layout_res);

    // Get output VF and metric
    auto FVFT_res = get_FV_FTVT(mo, is_cut_o, V_overlay, u_o, v_o);
    auto v3d = std::get<0>(FVFT_res);
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);

    // v3d_out = v3d^T
    std::vector<std::vector<double>> v3d_out(v3d[0].size());
    for (int i = 0; i < v3d[0].size(); i++)
    {
        v3d_out[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            v3d_out[i][j] = v3d[j][i];
        }
    }

    // reindex back
    auto u_o_out_copy = u_o_out;
    auto v_o_out_copy = v_o_out;
    auto v3d_out_copy = v3d_out;
    for (int i = 0; i < F_out.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (F_out[i][j] < vtx_reindex.size())
            {
                F_out[i][j] = vtx_reindex[F_out[i][j]];
            }
            if (FT_out[i][j] < vtx_reindex.size())
            {
                FT_out[i][j] = vtx_reindex[FT_out[i][j]];
            }
        }
    }

    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
        v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
        v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
    }

    return std::make_tuple(v3d_out, F_out, u_o, v_o, u_o_out, v_o_out, FT_out);
}

std::vector<int> get_to_map(OverlayMesh<double> &mo,
                            const std::vector<bool> &is_cut_o)
{
    // get h_group and to_map
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    int origin_size = mo.cmesh().out.size();
    std::vector<int> h_group(mo.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    std::vector<int> to_group(origin_size, -1);
    std::vector<int> which_to_group(mo.out.size(), -1);
    for (int i = 0; i < to_group.size(); i++)
    {
        to_group[i] = i;
        which_to_group[i] = i;
    }
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (h_group[i] != -1 || f_labels[mo.f[i]] == 2) continue;
        if (which_to_group[mo.to[i]] == -1)
        {
            which_to_group[mo.to[i]] = to_group.size();
            to_group.push_back(mo.to[i]);
        }
        if (mo.to[i] < origin_size && to_map[mo.to[i]] == -1)
        {
            h_group[i] = mo.to[i];
            to_map[mo.to[i]] = i;
        }
        else
        {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = mo.n[i];
        while (is_cut_o[cur] == false && mo.opp[cur] != i)
        {
            cur = mo.opp[cur];
            h_group[cur] = h_group[i];
            cur = mo.n[cur];
        }
        cur = mo.opp[i];
        while (is_cut_o[cur] == false && mo.prev[cur] != i)
        {
            cur = mo.prev[cur];
            h_group[cur] = h_group[i];
            cur = mo.opp[cur];
        }
    }

    return to_map;
}

std::vector<std::vector<int>>
get_he2v(OverlayMesh<double> &mo,
         std::vector<bool> &is_cut_o)
{
    // get h_group and to_map
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    int origin_size = mo.cmesh().out.size();
    std::vector<int> h_group(mo.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    std::vector<int> to_group(origin_size, -1);
    std::vector<int> which_to_group(mo.out.size(), -1);
    for (int i = 0; i < to_group.size(); i++)
    {
        to_group[i] = i;
        which_to_group[i] = i;
    }
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (h_group[i] != -1 || f_labels[mo.f[i]] == 2) continue;
        if (which_to_group[mo.to[i]] == -1)
        {
            which_to_group[mo.to[i]] = to_group.size();
            to_group.push_back(mo.to[i]);
        }
        if (mo.to[i] < origin_size && to_map[mo.to[i]] == -1)
        {
            h_group[i] = mo.to[i];
            to_map[mo.to[i]] = i;
        }
        else
        {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = mo.n[i];
        while (is_cut_o[cur] == false && mo.opp[cur] != i)
        {
            cur = mo.opp[cur];
            h_group[cur] = h_group[i];
            cur = mo.n[cur];
        }
        cur = mo.opp[i];
        while (is_cut_o[cur] == false && mo.prev[cur] != i)
        {
            cur = mo.prev[cur];
            h_group[cur] = h_group[i];
            cur = mo.opp[cur];
        }
    }

    std::vector<std::vector<int>> he2v;
    for (int i = 0; i < mo.n.size(); i++)
    {
        if (f_labels[mo.f[i]] == 2) continue;
        if (mo.edge_type[i] == 1) continue;
        he2v.push_back(std::vector<int>{h_group[i], h_group[mo.opp[i]]});
    }

    return he2v;
}

// FIXME Need to refactor this
std::vector<std::vector<int>>
get_edges(const Eigen::MatrixXd &V_init,
          const Eigen::MatrixXi &F_init,
          const std::vector<double> &Th_hat_init,
          const Mesh<double> &m,
          const std::vector<double> &lambdas)
{
    // Get boundary list, reindex map, and independent vertices
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    FV_to_double(V_init, F_init, Th_hat_init, vtx_reindex, indep_vtx, dep_vtx,
v_rep, bnd_loops, false);

    // Get overlay mesh for m with the embedded log lengths lambdas
    std::vector<int> flip_seq;
    OverlayMesh<double> mo = build_overlay_mesh(m, lambdas, flip_seq, false);

    // Get cones and boundary
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F_init);
    for (int i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    for (int i = 0; i < Th_hat_init.size(); i++)
    {
        if ((!is_bd[i]) && abs(Th_hat_init[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    // Get overlay vertices and face labels
    mo.garbage_collection();
    std::vector<std::vector<double>> V_reindex(3);
    for (int i = 0; i < 3; i++)
    {
        V_reindex[i].resize(mo._m.out.size(), 0);
        for (int j = 0; j < V_init.rows(); j++)
        {
            V_reindex[i][j] = double(V_init(vtx_reindex[j], i));
        }
    }
    std::vector<std::vector<double>> V_overlay =
ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq, V_reindex); if
(bnd_loops.size() != 0)
    {
        int n_v = V_init.rows();
        auto mc = mo.cmesh();
        create_tufted_cover(mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep,
mo._m.out, mo._m.to); mo._m.v_rep = range(0, n_v);
    }
    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // Reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (int i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (int i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    for (int i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    // Get layout
    std::vector<double> u(m.n_ind_vertices(), 0.0);
    auto layout_res = get_layout_overlay(mo, u, bd, cones);
    auto u_o = std::get<3>(layout_res);
    auto v_o = std::get<4>(layout_res);
    auto is_cut_o = std::get<5>(layout_res);

    // Get to map
    std::vector<std::vector<int>> he2v = get_he2v(mo, is_cut_o);

    // reindex back
    for (int i = 0; i < he2v.size(); i++)
    {
        for (int j = 0; j < 2; j++)
        {
            if (he2v[i][j] < vtx_reindex.size())
            {
                he2v[i][j] = vtx_reindex[he2v[i][j]];
            }
        }
    }

    return he2v;
}

//std::vector<std::vector<double>>
//get_V_overlay(const Eigen::MatrixXd &V_init,
//              const Eigen::MatrixXi &F_init,
//              const std::vector<double> &Th_hat_init,
//              const Mesh<double> &m,
//              const std::vector<double> &lambdas)
//{
//    // Get boundary list, reindex map, and independent vertices
//    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
//    FV_to_double(V_init, F_init, Th_hat_init, vtx_reindex, indep_vtx, dep_vtx,
v_rep, bnd_loops, false);
//
//    // Get overlay mesh for m with the embedded log lengths lambdas
//    std::vector<int> flip_seq;
//    OverlayMesh<double> mo = build_overlay_mesh(m, lambdas, flip_seq, false);
//
//    // Get cones and boundary
//    std::vector<int> cones, bd;
//    std::vector<bool> is_bd = igl::is_border_vertex(F_init);
//    for (int i = 0; i < is_bd.size(); i++)
//    {
//        if (is_bd[i])
//        {
//            bd.push_back(i);
//        }
//    }
//    for (int i = 0; i < Th_hat_init.size(); i++)
//    {
//        if ((!is_bd[i]) && abs(Th_hat_init[i] -  2 * M_PI) > 1e-15)
//        {
//            cones.push_back(i);
//        }
//    }
//
//    // Get overlay vertices and face labels
//    mo.garbage_collection();
//    std::vector<std::vector<double>> V_reindex(3);
//    for (int i = 0; i < 3; i++)
//    {
//        V_reindex[i].resize(mo._m.out.size(), 0);
//        for (int j = 0; j < V_init.rows(); j++)
//        {
//            V_reindex[i][j] = double(V_init(vtx_reindex[j], i));
//        }
//    }
//    return ConformalIdealDelaunay<double>::Interpolate_3d(mo, flip_seq,
V_reindex);
//}












void print_overlay_info(OverlayMesh<double> &mo)
{
    std::cout << "SEG BCS" << std::endl;
    for (int i = 0; i < mo.seg_bcs.size(); ++i)
    {
        std::cout << i << ": ";
        for (int j = 0; j < mo.seg_bcs[i].size(); ++j)
        {
            std::cout << mo.seg_bcs[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Follow the flip sequence in flip_seq, where nonnegative indices correspond to
Ptolemy flips
// and negative indices correspond to Euclidean flips of (-h-1), to generate a
flipped mesh
// m_flip with log lengths lambdas_full_flip from m with log lengths
lambdas_full.
//
// param[in] m: (possibly symmetric) mesh to flip
// param[in] lambdas_full: log edge lengths for m
// param[in] flip_seq: sequence of halfedges to flip
// param[out] m_flip: mesh after flips
// param[out] lambdas_full_flip: log edge lengths for m_flip
void flip_edges_deprecated(const Mesh<double> &m,
                const std::vector<double> &lambdas_full,
                const std::vector<int> &flip_seq,
                Mesh<double> &m_flip,
                std::vector<double> &lambdas_full_flip)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Initialize m_flip with m and lambdas_full
    m_flip = m;
    for (int h = 0; h < he2e.size(); ++h)
    {
        m_flip.l[h] = exp(lambdas_full[he2e[h]] / 2.0);
    }

    // Follow flip sequence with either Euclidean or Ptolemy flips
    for (int i = 0; i < flip_seq.size(); ++i)
    {
        int h = flip_seq[i];
        if (h < 0)
        {
            m_flip.flip_ccw(-h-1, false);
        }
        else
        {
            m_flip.flip_ccw(h, true);
        }
    }

    // Get lambdas of flipped edges
    lambdas_full_flip.resize(e2he.size(), 0);
    for (int e = 0; e < e2he.size(); ++e)
    {
        lambdas_full_flip[e] = 2.0 * log(m_flip.l[e2he[e]]);
    }
}
*/
}
#endif

// optimization_layout

//std::tuple<std::vector<std::vector<double>>, // V_out
//           std::vector<std::vector<int>>,    // F_out
//           std::vector<double>,              // layout u (per vertex)
//           std::vector<double>,              // layout v (per vertex)
//           std::vector<std::vector<int>>,    // FT_out
//           std::vector<bool>,                // is_cut_h
//           std::vector<bool>,                // is_cut_o
//           std::vector<int>,                 // Fn_to_F
//           std::vector<std::pair<int, int>>  // endpoints
//           >
//parametrize_mesh(const Eigen::MatrixXd& V,
//                 const Eigen::MatrixXi& F,
//                 const std::vector<double>& Theta_hat,
//                 OverlayMesh<double>& mo,
//                 std::vector<std::vector<double>>& V_overlay,
//                 std::vector<int>& vtx_reindex,
//                 std::vector<std::pair<int, int>>& endpoints)
//{
//  // get cones and bd
//  std::vector<int> cones, bd;
//  std::vector<bool> is_bd = igl::is_border_vertex(F);
//  for (int i = 0; i < is_bd.size(); i++) {
//    if (is_bd[i]) {
//      bd.push_back(i);
//    }
//  }
//  bool do_trim = false;
//  auto gb = count_genus_and_boundary(V, F);
//  int n_genus = gb.first, n_bd = gb.second;
//  if ((n_genus >= 1 && n_bd != 0) || n_bd > 1) {
//    do_trim = true;
//  }
//  for (int i = 0; i < Theta_hat.size(); i++) {
//    if ((!is_bd[i]) && abs(Theta_hat[i] - 2 * M_PI) > 1e-15) {
//      cones.push_back(i);
//    }
//  }
//
//  // reindex cones and bd
//  std::vector<int> vtx_reindex_rev(vtx_reindex.size());
//  for (int i = 0; i < vtx_reindex.size(); i++) {
//    vtx_reindex_rev[vtx_reindex[i]] = i;
//  }
//  for (int i = 0; i < cones.size(); i++) {
//    cones[i] = vtx_reindex_rev[cones[i]];
//  }
//  for (int i = 0; i < bd.size(); i++) {
//    bd[i] = vtx_reindex_rev[bd[i]];
//  }
//
//  // get layout
//  // FIXME Should be layout
//  std::vector<double> u(mo._m.Th_hat.size(), 0.0);
//  auto layout_res = get_layout_x(mo, u, bd, cones, do_trim);
//  auto is_cut = std::get<2>(layout_res);
//  auto u_o = std::get<3>(layout_res);
//  auto v_o = std::get<4>(layout_res);
//  auto is_cut_o = std::get<5>(layout_res);
//
//  // get output VF and metric
//  auto FVFT_res = get_FV_FTVT(mo, endpoints, is_cut_o, V_overlay, u_o, v_o);
//  auto v3d = std::get<0>(FVFT_res);
//  auto u_o_out = std::get<1>(FVFT_res);
//  auto v_o_out = std::get<2>(FVFT_res);
//  auto F_out = std::get<3>(FVFT_res);
//  auto FT_out = std::get<4>(FVFT_res);
//  auto Fn_to_F = std::get<5>(FVFT_res);
//  auto remapped_endpoints = std::get<6>(FVFT_res);
//
//  // v3d_out = v3d^T
//  std::vector<std::vector<double>> v3d_out(v3d[0].size());
//  for (int i = 0; i < v3d[0].size(); i++) {
//    v3d_out[i].resize(3);
//    for (int j = 0; j < 3; j++) {
//      v3d_out[i][j] = v3d[j][i];
//    }
//  }
//
//  // reindex back
//  auto u_o_out_copy = u_o_out;
//  auto v_o_out_copy = v_o_out;
//  auto v3d_out_copy = v3d_out;
//  auto endpoints_out = remapped_endpoints;
//  for (int i = 0; i < F_out.size(); i++) {
//    for (int j = 0; j < 3; j++) {
//      if (F_out[i][j] < vtx_reindex.size()) {
//        F_out[i][j] = vtx_reindex[F_out[i][j]];
//      }
//      if (FT_out[i][j] < vtx_reindex.size()) {
//        FT_out[i][j] = vtx_reindex[FT_out[i][j]];
//      }
//    }
//  }
//  for (int i = 0; i < vtx_reindex.size(); i++) {
//    u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
//    v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
//    v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
//  }
//  for (int i = vtx_reindex.size(); i < endpoints_out.size(); i++) {
//    int a = vtx_reindex[endpoints_out[i].first];
//    int b = vtx_reindex[endpoints_out[i].second];
//    endpoints_out[i] = std::make_pair(a, b);
//  }
//
//  return std::make_tuple(v3d_out,
//                         F_out,
//                         u_o_out,
//                         v_o_out,
//                         FT_out,
//                         is_cut,
//                         is_cut_o,
//                         Fn_to_F,
//                         endpoints_out);
//}
//

// NOTE: Minimal changes made. Just exposing the return parameters for repeated
// calls and separating into two components
void
GetReverseMap(OverlayMesh<double>& m_o,
              const std::vector<int>& flip_seq,
              OverlayMesh<double>& m_o_rev)
{
  auto mc = m_o.cmesh();
  bool do_Ptolemy = true;
  // do reverse flips
  for (int ii = flip_seq.size() - 1; ii >= 0; ii--) {
    if (do_Ptolemy && flip_seq[ii] < 0) {
      do_Ptolemy = false;
      m_o_rev.garbage_collection();
      Eigen::Matrix<double, -1, 1> u_0(m_o_rev.cmesh().out.size());
      u_0.setZero();
      m_o_rev.bc_eq_to_scaled(
        m_o_rev.cmesh().n, m_o_rev.cmesh().to, m_o_rev.cmesh().l, u_0);
    }
    if (do_Ptolemy) {
      m_o_rev.flip_ccw(flip_seq[ii], true);
      m_o_rev.flip_ccw(flip_seq[ii], true);
      m_o_rev.flip_ccw(flip_seq[ii], true);
    } else {
      m_o_rev.flip_ccw(-flip_seq[ii] - 1, false);
      m_o_rev.flip_ccw(-flip_seq[ii] - 1, false);
      m_o_rev.flip_ccw(-flip_seq[ii] - 1, false);
    }
  }

  m_o.garbage_collection();
  m_o_rev.garbage_collection();

  if (do_Ptolemy == false) {
    m_o_rev.bc_original_to_eq(
      m_o_rev.cmesh().n, m_o_rev.cmesh().to, m_o_rev.cmesh().l);
  }
  spdlog::debug(
    "#m_o.out: {}, #m_o_rev.out: {}", m_o.out.size(), m_o_rev.out.size());
  spdlog::debug("#m_o.n: {}, #m_o_rev.n: {}", m_o.n.size(), m_o_rev.n.size());
}

void
GetVertexMap(OverlayMesh<double>& m_o,
             OverlayMesh<double>& m_o_rev,
             std::vector<int>& v_map)
{
  auto mc = m_o.cmesh();

  // init the original vertices part with Id
  for (int i = 0; i < mc.out.size(); i++) {
    v_map[i] = i;
  }
  // init the segment vertices part with -1
  for (int i = mc.out.size(); i < v_map.size(); i++) {
    v_map[i] = -1;
  }

  for (int v_start = 0; v_start < mc.out.size(); v_start++) {
    int h_out0 = m_o.out[v_start];
    int h_out0_copy = h_out0;
    int v_end = m_o.find_end_origin(h_out0);

    int h_out0_rev = m_o_rev.out[v_start];
    bool flag = false;
    int while_cnt = 0;
    int caseid = 0;

    while (true) {
      if (m_o_rev.find_end_origin(h_out0_rev) == v_end &&
          m_o.dist_to_next_origin(h_out0) ==
            m_o_rev.dist_to_next_origin(h_out0_rev)) {
        // test first segment vertex
        // case 1, no segment vertex
        if (m_o_rev.to[h_out0_rev] == v_end) {
          caseid = 0;
          if (m_o.next_out(h_out0) != h_out0_copy) {
            h_out0 = m_o.next_out(h_out0);
            v_end = m_o.find_end_origin(h_out0);
          } else {
            flag = true;
          }

        } else {
          int h_first = m_o.n[h_out0];
          int h_first_rev = m_o_rev.n[h_out0_rev];

          if (m_o.find_end_origin(h_first) ==
                m_o_rev.find_end_origin(h_first_rev) &&
              m_o.find_end_origin(m_o.opp[h_first]) ==
                m_o_rev.find_end_origin(m_o_rev.opp[h_first_rev]) &&
              m_o.dist_to_next_origin(h_first) ==
                m_o_rev.dist_to_next_origin(h_first_rev)) {
            caseid = 1;
            flag = true;
          }
        }
      }

      if (flag)
        break;

      h_out0_rev = m_o_rev.next_out(h_out0_rev);
      while_cnt++;

      if (while_cnt > 99999) {
        spdlog::error("infi loop in finding first match");
        break;
      }
    }

    int h_out = h_out0;
    int h_out_rev = h_out0_rev;

    do {
      int h_current = h_out;
      int h_current_rev = h_out_rev;

      while (m_o.vertex_type[m_o.to[h_current]] != ORIGINAL_VERTEX) {

        if (m_o_rev.vertex_type[m_o_rev.to[h_current_rev]] == ORIGINAL_VERTEX) {
          spdlog::error("out path not matching, case: {}", caseid);
          break;
        }
        int v_current = m_o.to[h_current];
        int v_current_rev = m_o_rev.to[h_current_rev];
        if (v_map[v_current] == -1) {
          v_map[v_current] = v_current_rev;
        } else if (v_map[v_current] != v_current_rev) {
          spdlog::error("the mapping is wrong, case: {}", caseid);
        }
        h_current = m_o.n[m_o.opp[m_o.n[h_current]]];
        h_current_rev = m_o_rev.n[m_o_rev.opp[m_o_rev.n[h_current_rev]]];
      }
      h_out = m_o.next_out(h_out);
      h_out_rev = m_o_rev.next_out(h_out_rev);
    } while (h_out != h_out0);
  }
}

// pybind

  m.def("generate_optimized_overlay",
        &generate_optimized_overlay,
        "generate overlay mesh for optimized lambdas",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

// Script convert to vf

        else:
            logger.info("Getting as symmetric as possible translations")
            tau = interpolation.as_symmetric_as_possible_translations(
                C,
                lambdas_init,
                lambdas,
                not args['flip_in_original_metric']
            )
            logger.info("Range of translations is [{}, {}]".format(np.min(tau), np.max(tau)))

            overlay_res = opt.generate_optimized_overlay(
                v3d,
                f,
                Th_hat,
                lambdas,
                tau_init,
                tau,
                tau_post,
                [],
                [],
                False,
                args['flip_in_original_metric']
            )
            C_o, _, _, v_overlay, vtx_reindex, endpoints = overlay_res

// translation

// FIXME THIS SHOULD BE MOVED INTO THE LAYOUT CODE

//template<typename Scalar>
//void
//triangulate_polygon_mesh(Mesh<Scalar>& m,
//                         const std::vector<Scalar>& u,
//                         const std::vector<Scalar>& v,
//                         std::vector<int>& f_type)
//{
//  int n_f0 = m.n_faces();
//  spdlog::info("initial f size: {}", n_f0);
//  spdlog::info("initial he size: {}", m.n.size());
//  for (int f = 0; f < n_f0; f++) {
//    int n_f = m.n_faces();
//    int h0 = m.h[f];
//    int hc = h0;
//    std::vector<int> hf;
//    do {
//      hf.push_back(hc);
//      hc = m.n[hc];
//    } while (h0 != hc);
//    if (hf.size() == 3)
//      continue;
//    spdlog::debug("triangulate face {}, #e {}", f, hf.size());
//    int N = hf.size();
//    // new faces: N-3
//    // new halfedges 2*(N-3)
//    int n_new_f = N - 3;
//    int n_new_he = 2 * (N - 3);
//    int n_he = m.n.size();
//    m.n.resize(n_he + n_new_he);
//    m.to.resize(n_he + n_new_he);
//    m.opp.resize(n_he + n_new_he);
//    m.l.resize(n_he + n_new_he);
//    m.f.resize(n_he + n_new_he);
//    m.h.resize(n_f + n_new_f);
//    f_type.resize(n_f + n_new_f);
//    for (int k = 0; k < n_new_f; k++)
//      f_type[n_f + k] = f_type[f];
//    m.n[n_he] = hf[0];
//    m.n[hf[1]] = n_he;
//    m.opp[n_he] = n_he + 1;
//    m.to[n_he] = m.to[hf.back()];
//    m.h[f] = n_he;
//    m.f[n_he] = f;
//    assert(hf.back() < m.to.size() && hf[0] < m.to.size());
//    m.l[n_he] = sqrt((u[hf.back()] - u[hf[1]]) * (u[hf.back()] - u[hf[1]]) +
//                     (v[hf.back()] - v[hf[1]]) * (v[hf.back()] - v[hf[1]]));
//    for (int k = 1; k < 2 * (N - 3); k++) {
//      if (k % 2 == 0) {
//        m.n[n_he + k] = n_he + k - 1;
//        m.opp[n_he + k] = n_he + k + 1;
//        m.to[n_he + k] = m.to[hf.back()];
//        m.l[n_he + k] = sqrt((u[m.n[m.n[n_he + k]]] - u[hf.back()]) *
//                               (u[m.n[m.n[n_he + k]]] - u[hf.back()]) +
//                             (v[m.n[m.n[n_he + k]]] - v[hf.back()]) *
//                               (v[m.n[m.n[n_he + k]]] - v[hf.back()]));
//        m.f[n_he + k] = n_f + k / 2 - 1;
//        m.h[n_f + k / 2 - 1] = n_he + k;
//      } else {
//        m.n[n_he + k] = hf[(k - 1) / 2 + 2];
//        if ((k - 1) / 2 + 2 != hf.size() - 2)
//          m.n[hf[(k - 1) / 2 + 2]] = n_he + k + 1;
//        m.opp[n_he + k] = n_he + k - 1;
//        m.to[n_he + k] = m.to[m.n[m.n[m.opp[n_he + k]]]];
//        m.l[n_he + k] = sqrt((u[m.n[m.n[m.opp[n_he + k]]]] - u[hf.back()]) *
//                               (u[m.n[m.n[m.opp[n_he + k]]]] - u[hf.back()]) +
//                             (v[m.n[m.n[m.opp[n_he + k]]]] - v[hf.back()]) *
//                               (v[m.n[m.n[m.opp[n_he + k]]]] - v[hf.back()]));
//        m.f[n_he + k] = n_f + (k + 1) / 2 - 1;
//        m.h[n_f + (k + 1) / 2 - 1] = n_he + k;
//        m.f[m.n[n_he + k]] = n_f + (k + 1) / 2 - 1;
//      }
//    }
//    m.n[hf.back()] = n_he + n_new_he - 1;
//    m.f[hf.back()] = n_f + n_new_f - 1;
//// #define TRIANGULATE_POLYGON_MESH_DEBUG
//#ifdef TRIANGULATE_POLYGON_MESH_DEBUG
//    std::cout << "hf: ";
//    for (int x : hf)
//      std::cout << x << " ";
//    std::cout << std::endl;
//    for (int k = 0; k < n_new_he; k++) {
//      int hi = n_he + k;
//      std::cout << "--- check halfedge " << hi << ", " << k << " ---"
//                << std::endl;
//      std::cout << "opp[" << hi << "] = " << m.opp[hi] << std::endl;
//      std::cout << "n[" << hi << "] = " << m.n[hi] << std::endl;
//      std::cout << "l[" << hi << "] = " << m.l[hi] << std::endl;
//      std::cout << "f[" << hi << "] = " << m.f[hi] << std::endl;
//    }
//    for (int hi : hf) {
//      std::cout << "--- check original halfedge " << hi << " ---" << std::endl;
//      std::cout << "opp[" << hi << "] = " << m.opp[hi] << std::endl;
//      std::cout << "n[" << hi << "] = " << m.n[hi] << std::endl;
//      std::cout << "l[" << hi << "] = " << m.l[hi] << std::endl;
//      std::cout << "f[" << hi << "] = " << m.f[hi] << std::endl;
//    }
//    exit(0);
//#endif
//  }
//
//  for (int f = 0; f < m.h.size(); f++) {
//    int n_f = m.n_faces();
//    int h0 = m.h[f];
//    int hc = h0;
//    std::vector<int> hf;
//    do {
//      hf.push_back(hc);
//      hc = m.n[hc];
//      if (hf.size() > 3) {
//        spdlog::error("face {} has {} he!!!", f, hf.size());
//        for (auto x : hf)
//          std::cout << x << " ";
//        std::cout << std::endl;
//        exit(0);
//      }
//    } while (h0 != hc);
//  }
//}
//
//template<typename Scalar>
//void
//connect_to_singularities(OverlayMesh<Scalar>& m_o,
//                         const std::vector<int>& f_type,
//                         const std::vector<int>& bd,
//                         const std::vector<int>& singularities,
//                         std::vector<bool>& is_cut_o,
//                         int root = -1)
//{
//
//  int n_v = m_o.out.size();
//  int n_e = m_o.n.size();
//
//  std::vector<std::vector<int>> v2e(m_o.out.size(), std::vector<int>());
//  for (int i = 0; i < m_o.n.size(); i++) {
//    if (f_type[m_o.f[i]] == 1) {
//      v2e[m_o.v0(i)].push_back(i);
//    }
//  }
//
//  std::vector<int> min_distance(n_v, n_e);
//  std::vector<int> T(n_v, -1);
//  std::set<std::pair<int, int>> vertex_queue;
//  std::vector<bool> is_cone(n_v, false);
//  std::vector<bool> is_border(n_v, false);
//  for (int v : bd)
//    is_border[v] = true;
//  for (int v : singularities)
//    is_cone[v] = true;
//
//  // put boundary to queue
//  if (root != -1) {
//    assert(std::find(bd.begin(), bd.end(), root) != bd.end() &&
//           "selected root not on boundary");
//    spdlog::info("select root {} for layout", root);
//    vertex_queue.insert(std::make_pair(0, root));
//    min_distance[root] = 0;
//  } else {
//    for (int i = 0; i < bd.size(); i++) {
//      vertex_queue.insert(std::make_pair(0, bd[i]));
//      min_distance[bd[i]] = 0;
//    }
//  }
//
//  // do dijkstra
//  int n_visited = 0;
//  while (!vertex_queue.empty()) {
//    int dist_u = vertex_queue.begin()->first;
//    int u = vertex_queue.begin()->second;
//    // end earlier if all targets(singularities) are visited
//    if (is_cone[u]) // (std::find(singularities.begin(), singularities.end(), u)
//                    // != singularities.end())
//    {
//      n_visited++;
//      spdlog::debug("path to cone {}: len({})", u, dist_u);
//    }
//    if (n_visited == singularities.size())
//      break;
//    vertex_queue.erase(vertex_queue.begin());
//    if (root != -1 && u != root && is_border[u])
//      continue;
//    int h0 = m_o.out[u];
//    if (f_type[m_o.f[h0]] == 2) {
//      if (!v2e[u].empty()) // pick a type 1 edge if exist
//        h0 = v2e[u][0];
//    }
//
//    int h = h0;
//    do {
//      if (m_o.edge_type[h] != ORIGINAL_EDGE) {
//        int v = m_o.to[h];
//        int dist_v = dist_u + 1;
//        // update queue
//        if (min_distance[v] > dist_v) {
//          vertex_queue.erase(std::make_pair(min_distance[v], v));
//          min_distance[v] = dist_v;
//          T[v] = h;
//          vertex_queue.insert(std::make_pair(min_distance[v], v));
//        }
//      }
//      h = m_o.next_out(h);
//    } while (h != h0);
//  }
//
//  spdlog::info(
//    "dijsktra done, connected cones: {}/{}", n_visited, singularities.size());
//
//  // get cut_to_sin
//  is_cut_o = std::vector<bool>(n_e, false);
//  for (int s : singularities) {
//    int h = T[s];
//    std::set<int> seq_v;
//    while (h != -1) {
//      is_cut_o[h] = true;
//      is_cut_o[m_o.opp[h]] = true;
//      h = T[m_o.v0(h)];
//    }
//  }
//}
//
//// Modified from compute_layout
//template<typename Scalar>
//std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>>
//generate_layout_lambdas(Mesh<Scalar>& m,
//                        const VectorX& lambdas_he,
//                        std::vector<bool>& is_cut_h,
//                        int start_h = -1)
//{
//  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
//
//  auto _u = std::vector<Scalar>(m.n_halfedges(), 0.0);
//  auto _v = std::vector<Scalar>(m.n_halfedges(), 0.0);
//
//  bool cut_given = !is_cut_h.empty();
//  auto cut_final = std::vector<bool>(m.n_halfedges(), false);
//  auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);
//
//  // set starting point - use a boundary edge
//  int h = 0;
//  if (start_h == -1) {
//    for (int i = 0; i < m.n_halfedges(); i++) {
//      if (m.f[i] != -1 && m.f[m.opp[i]] == -1)
//        h = m.n[m.n[i]];
//    }
//  } else {
//    assert(m.f[start_h] != -1);
//    h = m.n[m.n[start_h]];
//  }
//
//  _u[h] = 0.0;
//  _v[h] = 0.0;
//  h = m.n[h];
//  assert(m.f[h] != -1);
//
//  _u[h] = exp(lambdas_he[m.e(h)] / 2.0);
//  _v[h] = 0.0;
//  auto done = std::vector<bool>(m.n_faces(), false);
//
//  // discard part 2
//  for (int i = 0; i < done.size(); i++) {
//    int hh = m.h[i];
//    if (m.type[hh] == 2 && m.type[m.n[hh]] == 2 && m.type[m.n[m.n[hh]]] == 2) {
//      done[i] = true;
//    }
//  }
//  // set edge type 2 as cut
//  for (int i = 0; i < is_cut_h.size(); i++) {
//    if (m.type[i] == 2) {
//      is_cut_h[i] = true;
//    }
//  }
//
//  std::queue<int> Q;
//  Q.push(h);
//  done[m.f[h]] = true;
//
//  auto perp = [](Eigen::Matrix<Scalar, 1, 2> a) {
//    Eigen::Matrix<Scalar, 1, 2> b;
//    b[0] = -a[1];
//    b[1] = a[0];
//    return b;
//  };
//
//  auto area_from_len = [](Scalar l1, Scalar l2, Scalar l3) {
//    auto s = 0.5 * (l1 + l2 + l3);
//    return sqrt(s * (s - l1) * (s - l2) * (s - l3));
//  };
//
//  auto square = [](Scalar x) { return x * x; };
//
//  while (!Q.empty()) {
//    h = Q.front();
//    Q.pop();
//    int hn = m.n[h];
//    int hp = m.n[hn];
//    Eigen::Matrix<Scalar, 1, 2> p1;
//    p1[0] = _u[hp];
//    p1[1] = _v[hp];
//    Eigen::Matrix<Scalar, 1, 2> p2;
//    p2[0] = _u[h];
//    p2[1] = _v[h];
//    Scalar l0 = Scalar(1.0);
//    Scalar l1 = exp((lambdas_he[m.e(hn)] - lambdas_he[m.e(h)]) / 2.0);
//    Scalar l2 = exp((lambdas_he[m.e(hp)] - lambdas_he[m.e(h)]) / 2.0);
//    Eigen::Matrix<Scalar, 1, 2> pn =
//      p1 + (p2 - p1) * (1 + square(l2 / l0) - square(l1 / l0)) / 2 +
//      perp(p2 - p1) * 2 * area_from_len(1.0, l1 / l0, l2 / l0);
//    _u[hn] = pn[0];
//    _v[hn] = pn[1];
//    int hno = m.opp[hn];
//    int hpo = m.opp[hp];
//    int ho = m.opp[h];
//
//    if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn])) {
//      done[m.f[hno]] = true;
//      _u[hno] = _u[h];
//      _v[hno] = _v[h];
//      _u[m.n[m.n[hno]]] = _u[hn];
//      _v[m.n[m.n[hno]]] = _v[hn];
//      Q.push(hno);
//    } else {
//      is_cut_h_gen[hn] = true;
//      is_cut_h_gen[m.opp[hn]] = true;
//    }
//
//    if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp])) {
//      done[m.f[hpo]] = true;
//      _u[hpo] = _u[hn];
//      _v[hpo] = _v[hn];
//      _u[m.n[m.n[hpo]]] = _u[hp];
//      _v[m.n[m.n[hpo]]] = _v[hp];
//      Q.push(hpo);
//    } else {
//      is_cut_h_gen[hp] = true;
//      is_cut_h_gen[m.opp[hp]] = true;
//    }
//
//    if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho])) {
//      done[m.f[ho]] = true;
//      _u[ho] = _u[hp];
//      _v[ho] = _v[hp];
//      _u[m.n[m.n[ho]]] = _u[h];
//      _v[m.n[m.n[ho]]] = _v[h];
//      Q.push(ho);
//    }
//  }
//
//  return std::make_tuple(_u, _v, is_cut_h_gen);
//};
//
//// Modified from get_layout_overlay
//template<typename Scalar>
//std::tuple<std::vector<Scalar>,
//           std::vector<Scalar>,
//           std::vector<Scalar>,
//           std::vector<Scalar>,
//           std::vector<Scalar>,
//           std::vector<Scalar>,
//           std::vector<bool>>
//generate_layout_overlay_lambdas(OverlayMesh<Scalar>& m_o,
//                                const VectorX& lambdas_he,
//                                std::vector<int> bd,
//                                std::vector<int> singularities,
//                                bool do_trim = false,
//                                int root = -1)
//{
//  auto f_type = get_overlay_face_labels(m_o);
//
//  // layout the current mesh with arbitrary cut
//  std::vector<bool> is_cut;
//  auto mc = m_o.cmesh();
//  mc.type = std::vector<char>(mc.n_halfedges(), 0);
//  // FIXME Changed code here
//  auto layout_res = generate_layout_lambdas(mc, lambdas_he, is_cut, 0);
//  // FIXME End change
//  auto u_scalar = std::get<0>(layout_res);
//  auto v_scalar = std::get<1>(layout_res);
//
//  // FIXME This is second portion of the only changed code
//  // Moving rescaling out of the layout and into the main code for easier
//  // manipulation
//  // bc_eq_to_two_triangle_chart(m_o, mc);
//  // bc_two_triangle_chart_to_scaled(m_o, mc, lambdas_he);
//  // Eigen::Matrix<Scalar, -1, 1> u_eig;
//  // u_eig.resize(u_vec.size());
//  // for (int i = 0; i < u_vec.size(); i++)
//  //{
//  //  u_eig(i) = u_vec[i];
//  //}
//  // m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);
//  // FIXME End changed code
//
//  auto u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, u_scalar);
//  auto v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, v_scalar);
//
//  spdlog::info("Interpolate on overlay mesh done.");
//
//  // compute edge lengths of overlay mesh and triangulate it
//  Mesh<Scalar> m;
//  m.n = m_o.n;
//  m.opp = m_o.opp;
//  m.f = m_o.f;
//  m.h = m_o.h;
//  m.out = m_o.out;
//  m.to = m_o.to;
//  m.l = std::vector<Scalar>(m.n.size(), 0.0);
//  for (int i = 0; i < m.n.size(); i++) {
//    int h0 = i;
//    int h1 = h0;
//    do {
//      if (m.n[h1] == h0)
//        break;
//      h1 = m.n[h1];
//    } while (h0 != h1);
//    if (m.to[m.opp[h0]] != m.to[h1]) {
//      spdlog::error("h0 h1 picked wrong.");
//      exit(0);
//    }
//    m.l[h0] = sqrt((u_o[h0] - u_o[h1]) * (u_o[h0] - u_o[h1]) +
//                   (v_o[h0] - v_o[h1]) * (v_o[h0] - v_o[h1]));
//  }
//  triangulate_polygon_mesh(m, u_o, v_o, f_type);
//
//  m.type = std::vector<char>(m.n.size(), 0);
//  m.type_input = m.type;
//  m.R = std::vector<int>(m.n.size(), 0);
//  m.v_rep = range(0, m.out.size());
//  m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);
//
//  // try to connect to singularties again with overlay mesh edges
//  spdlog::info("try to connect to singularities using a tree rooted at root");
//  std::vector<bool> is_cut_h;
//  OverlayMesh<Scalar> mo(m);
//  for (int i = m_o.n.size(); i < mo.n.size(); i++) {
//    mo.edge_type[i] = ORIGINAL_EDGE; // make sure do not use the new diagonal
//  }
//  connect_to_singularities(mo, f_type, bd, singularities, is_cut_h, root);
//
//  int start_h = 0;
//  for (int i = 0; i < mo.n.size(); i++) {
//    if (f_type[mo.f[i]] == 1 && f_type[mo.f[mo.opp[i]]] == 2) {
//      start_h = i;
//      break;
//    }
//  }
//  spdlog::info("selected start h: {}, left: {}, right: {}",
//               start_h,
//               f_type[mo.f[start_h]],
//               f_type[mo.f[mo.opp[start_h]]]);
//
//  // sanity check for the input of compute layout
//  // - opposite halfedges should have same edge lenghts (up to numerical error)
//  // - all halfedges that belongs to a face with type 1 should have non-zero
//  // edge lengths
//  for (int i = 0; i < mo.n.size(); i++) {
//    int h0 = i, h1 = mo.opp[h0];
//    int ft0 = f_type[mo.f[h0]];
//    int ft1 = f_type[mo.f[h1]];
//    if (std::abs<Scalar>(mo._m.l[h0] - mo._m.l[h1]) > 1e-12 && ft0 == ft1 &&
//        ft0 == 1) {
//      std::cout << "Halfedge lengths mismatch" << std::endl;
//      spdlog::error("halfedge lengths mismatch, {}: {}, {}: {}; {}/{}",
//                    h0,
//                    mo._m.l[h0],
//                    h1,
//                    mo._m.l[h1],
//                    ft0,
//                    ft1);
//    }
//    int f0 = mo.f[h0];
//    int f1 = mo.f[h1];
//    if (f_type[f0] == 1 && mo._m.l[h0] == 0) {
//      std::cout << "Halfedge length 0" << std::endl;
//      spdlog::error("copy 1 has zero edge at {}, f{}", h0, f0);
//    }
//    if (f_type[f1] == 1 && mo._m.l[h1] == 0) {
//      std::cout << "Halfedge length 0" << std::endl;
//      spdlog::error("copy 1 has zero edge at {}, f{}", h1, f1);
//    }
//  }
//  spdlog::info("sanity check done.");
//
//  // mark boundary as cut
//  for (int i = 0; i < is_cut_h.size(); i++) {
//    if (f_type[mo.f[i]] != f_type[mo.f[mo.opp[i]]]) {
//      is_cut_h[i] = true;
//    }
//  }
//  // now directly do layout on overlay mesh
//  for (int f = 0; f < f_type.size(); f++) {
//    int h0 = mo.h[f];
//    int h1 = mo.n[h0];
//    int h2 = mo.n[h1];
//    mo._m.type[h0] = f_type[f];
//    mo._m.type[h1] = f_type[f];
//    mo._m.type[h2] = f_type[f];
//  }
//
//  // get output connectivity and metric
//  std::vector<Scalar> phi(mo.out.size(), 0.0);
//  auto overlay_layout_res = compute_layout(mo._m, phi, is_cut_h, start_h);
//  auto _u_o = std::get<0>(overlay_layout_res);
//  auto _v_o = std::get<1>(overlay_layout_res);
//
//  is_cut_h.resize(m_o.n.size()); // ignoring the newly added diagonal
//  _u_o.resize(m_o.n.size());
//  _v_o.resize(m_o.n.size());
//  return std::make_tuple(u_scalar, v_scalar, u_o, v_o, _u_o, _v_o, is_cut_h);
//  // return std::make_tuple(u_scalar, v_scalar, is_cut_h);
//  // return std::make_tuple(u_o, v_o, is_cut_h);
//}

/*
Eigen::MatrixXd get_per_corner_uv(const std::vector<std::vector<int>> &F,
                                  const std::vector<std::vector<int>> &Fuv,
                                  const Eigen::MatrixXd &uv)

{
    std::vector<int> next_he;
    std::vector<int> opp;
    std::vector<int> bnd_loops;
    std::vector<int> vtx_reindex;
    Connectivity C;
    FV_to_NOB(F, next_he, opp, bnd_loops, vtx_reindex);
    NOB_to_connectivity(next_he, opp, bnd_loops, C);

        std::vector<int> u(next_he.size(), 0);
        std::vector<int> v(next_he.size(), 0);
        for (int i = 0; i < Fuv.size(); ++i)
    {
                int hh = C.f2he[i];
                u[hh] = uv[Fuv[i,1],0];
                v[hh] = uv[Fuv[i,1],1];
                u[n[hh]] = uv[Fuv[i,2],0];
                v[n[hh]] = uv[Fuv[i,2],1];
                u[n[next_he[hh]]] = uv[Fuv[i,0],0];
    }

    return std::make_tuple()
*/

// bc_reparametrize_eq

    // Compute the coordinate for the image of the midpoint
    // double Dij = exp(2*dij);
    // double cij = (Dij - 1) / (Dij + 1);

    // Compute new barycentric coordinates
    // m_o.seg_bcs[h][0] *= (1 + cij);
    // m_o.seg_bcs[h][1] *= (1 - cij);
    // double sum = m_o.seg_bcs[h][0] + m_o.seg_bcs[h][1];
    // m_o.seg_bcs[h][0] /= sum;
    // m_o.seg_bcs[h][1] /= sum;

// refinement

void
triangulate_self_overlapping_subpolygon(
	const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	const std::vector<std::vector<int>>& splitting_vertices,
	int start_vertex,
	int end_vertex,
	std::vector<std::array<int, 3>>& faces
) {
	int face_size = is_self_overlapping_subpolygon.size();
	spdlog::info(
		"Triangulating subpolygon ({}, {}) of polygon of size {}",
		start_vertex,
		end_vertex,
		face_size
	);

	// Get starting edge (j,i)
	int j = start_vertex;
	int i = (j + 1) % face_size; // j = i - 1

	// Do nothing in base case of a single vertex or edge
	if ((i == end_vertex) || (j == end_vertex))
	{
		return;
	}

	// Iterate over the subpolygon vertices to find a splitting vertex
	while (j != end_vertex)
	{
		if (is_self_overlapping_subpolygon[i][j])
		{
			// Add splitting face Tikj
			int k = splitting_vertices[i][j];
			faces.push_back( {i, k, j} );
			spdlog::info("Adding triangle ({}, {}, {})", i, k, j);

			// Triangulate subpolygon (i, k) unless the current subpolygon is (k, i)
			// We skip the (k, i) case as the full polygon is (i, k) union (k, i), and 
			// we assume the 
			spdlog::info("Recursing to subpolygon ({}, {})", i, k);
			triangulate_self_overlapping_subpolygon(
				is_self_overlapping_subpolygon,
				splitting_vertices,
				i,
				k,
				faces
			);

			// Triangulate subpolygon (k, j) = (k, i-1) unless the current subpolygon is (j, k)
			spdlog::info("Recursing to subpolygon ({}, {})", k, j);
			if ((k != ))
			triangulate_self_overlapping_subpolygon(
				is_self_overlapping_subpolygon,
				splitting_vertices,
				k,
				j,
				faces
			);

			return;
		}

		// Iterate edge
		j = (j + 1) % face_size;
		i = (i + 1) % face_size;
	}
}

void
triangulate_self_overlapping_polygon(
	const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	const std::vector<std::vector<int>>& splitting_vertices,
	std::vector<std::array<int, 3>>& faces
) {
	faces.clear();
	int face_size = is_self_overlapping_subpolygon.size();
	if (face_size < 3)
	{
		spdlog::warn("Triangulated trivial face");
		return;
	}

	// Call recursive subroutine on the whole polygon
	triangulate_self_overlapping_subpolygon(
		is_self_overlapping_subpolygon,
		splitting_vertices,
		0,
		face_size - 1,
		faces
	);
}

// Layout


template <typename Scalar>
void
add_edge_to_mask(
  OverlayMesh<Scalar> &mo,
  std::vector<bool> &mask,
  int h,
  bool value
) {
  int seg = h;
  while (true) {
    // Add segment to the spanning tree
    mask[seg] = value;
    mask[mo.opp[seg]] = value;

    // Break if last segment or continue to next segment
    if (mo.vertex_type[mo.to[seg]] == ORIGINAL_VERTEX) break;
    seg = mo.n[mo.opp[mo.n[seg]]];
  }
}

// Compute a cut for the overlay mesh that only uses edges in the original mesh
template <typename Scalar>
void
compute_overlay_cut(
  OverlayMesh<Scalar> &mo,
  std::vector<bool> &is_cut_h
) {
  // Initialize an array to keep track of vertices
  int num_vertices = mo.n_vertices();
  if (num_vertices == 0)
  {
    spdlog::error("Cannot cut a trivial mesh");
    return;
  }

  // Initialize spanning tree
  int num_halfedges = mo.n_halfedges();
  std::vector<bool> is_spanning_tree_h(num_halfedges, false);
  std::vector<bool> is_found_vertex(num_vertices, false);

  // Initialize the stack of vertices to process with an arbitrary on the original
  // connectivity
  std::deque<int> vertices_to_process;
  for (int vi = 0; vi < num_vertices; ++vi)
  {
    if (mo.vertex_type[vi] == ORIGINAL_VERTEX)
    {
      vertices_to_process.push_back(vi);
      is_found_vertex[vi] = true;
      break;
    }
  }

  // Check that some vertex was found
  if (vertices_to_process.empty())
  {
    spdlog::error("Could not find starting vertex for overlay cut");
    return;
  }

  // Perform breadth first search
  while (!vertices_to_process.empty())
  {
    // Get the next vertex to process
    int current_vertex = vertices_to_process.front();
    vertices_to_process.pop_front();

    // Iterate over the vertex circulator via halfedges
    int h_start = mo.out[current_vertex];
    int h = h_start;
    do
    {
      // Get the vertex in the one ring at the tip of the halfedge
      int one_ring_vertex = mo.find_end_origin(h);

      // Check if the edge is in the original mesh and the tip vertex hasn't been processed yet
      if ((mo.edge_type[h] != CURRENT_EDGE) && (!is_found_vertex[one_ring_vertex]))
      {
        // Add segments for the entire edge to the spanning tree
        add_edge_to_mask(mo, is_spanning_tree_h, h, true);
        add_edge_to_mask(mo, is_spanning_tree_h, mo.opp[h], true);

        // Mark the vertex as found and add it to the vertices to process
        vertices_to_process.push_back(one_ring_vertex);
        is_found_vertex[one_ring_vertex] = true;
      }

      // Progress to the next halfedge in the vertex circulator
      h = mo.n[mo.opp[h]];
    }
    while (h != h_start);
  }

  // Now, perform breadth first search over faces to build a cotree of edges not to cut
  int num_faces = mo.n_faces();
  is_cut_h = std::vector<bool>(num_halfedges, true);
  std::vector<bool> is_found_face(num_faces, false);
  std::deque<int> faces_to_process = {0};
  is_found_face[0] = true;
  while (!faces_to_process.empty())
  {
    // Get the next face to process
    int current_face = faces_to_process.front();
    faces_to_process.pop_front();

    // Iterate over the face via halfedges
    int h_start = mo.h[current_face];
    int h = h_start;
    do
    {
      // Get the face adjacent to the given edge
      int adjacent_face = mo.f[mo.opp[h]];

      // If the edge is a current edge, circulate around the vertex at the tip
      if (mo.edge_type[h] == CURRENT_EDGE)
      {
        // Do not cut edges not in the original connectivity
        add_edge_to_mask(mo, is_cut_h, h, false);
        add_edge_to_mask(mo, is_cut_h, mo.opp[h], false);

        // Mark the adjacent face as found
        is_found_face[adjacent_face] = true;

        // Circulate and continue
        h = mo.n[mo.opp[h]];
        continue;
      }
      
      // Check if the edge is not in the spanning tree and the adjacent face is not processed yet
      if ((!is_spanning_tree_h[h]) && (!is_found_face[adjacent_face]))
      {
        // Add the edge to the spanning cotree
        add_edge_to_mask(mo, is_cut_h, h, false);
        add_edge_to_mask(mo, is_cut_h, mo.opp[h], false);

        // Mark the face as found and add it to the face to process
        faces_to_process.push_back(adjacent_face);
        is_found_face[adjacent_face] = true;
      }

      // Progress to the next halfedge in the face circulator
      h = mo.n[h];
    }
    while (h != h_start);
  }


}

/**
 * @brief Given overlay mesh with associated flat metric compute the layout
 * 
 * @tparam Scalar double/mpfr::mpreal
 * @param m_o, overlay mesh
 * @param u_vec, per-vertex scale factor
 * @param bd, list of boundary vertex ids
 * @param singularities, list of singularity vertex ids
 * @param root (optional) index of a boundary vertex, when not -1, this will be the only intersection of the cut to singularity edges with boundary
 * @param use_original_edge_cut (optional) cut to singularities using only original mesh edges if true
 * @return _u_c, _v_c, is_cut_c (per-corner u/v assignment of current mesh and marked cut edges) 
 *         _u_o, _v_o, is_cut_h (per-corner u/v assignment of overlay mesh and marked cut edges)
 */
template <typename Scalar>
static
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>,
           std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> get_layout(OverlayMesh<Scalar> &m_o, const std::vector<Scalar> &u_vec, std::vector<int> bd, std::vector<int> singularities, bool do_trim = false, int root=-1)
{
    
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
  check_if_flipped(mc, _u_c, _v_c);
  std::vector<Scalar> _u_o, _v_o;
  std::vector<bool> is_cut_o;

  // Interpolate layout to the overlay mesh
  Eigen::Matrix<Scalar, -1, 1> u_eig;
  u_eig.resize(u_vec.size());
  for (int i = 0; i < u_vec.size(); i++)
  {
    u_eig(i) = u_vec[i];
  }
  m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);
  auto u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _u_c);
  auto v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _v_c);
  spdlog::info("Interpolate on overlay mesh done.");
  check_if_flipped(m_o, u_o, v_o);

  if(!bd.empty()){ 
    // FIXME TODO
    // compute edge lengths of overlay mesh and triangulate it
    Mesh<Scalar> m;
    m.n = m_o.n;
    m.opp = m_o.opp;
    m.f = m_o.f;
    m.h = m_o.h;
    m.out = m_o.out;
    m.to = m_o.to;
    m.l = std::vector<Scalar>(m.n.size(), 0.0);
    for(int i = 0; i < m.n.size(); i++){
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
    check_if_flipped(m, u_o, v_o);

    m.type = std::vector<char>(m.n.size(), 0);
    m.type_input = m.type;
    m.R = std::vector<int>(m.n.size(), 0);
    m.v_rep = range(0, m.out.size());
    m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);
    
    // try to connect to singularties again with overlay mesh edges
    spdlog::info("try to connect to singularities using a tree rooted at root");
  
    OverlayMesh<Scalar> m_o_tri(m);
    for(int i = m_o.n.size(); i < m_o_tri.n.size(); i++){
        m_o_tri.edge_type[i] = ORIGINAL_EDGE; // make sure do not use the new diagonal
    }
    spdlog::info("root = {}", root);
    //connect_to_singularities(m_o_tri, f_labels, bd, singularities, is_cut_o, root);
    // FIXME Check on this; we have replaced the cut to singularity with an overlay cut

    // Get overlay cut using only original edges
    std::vector<bool> is_cut_poly;
    //compute_overlay_cut(m_o, is_cut_poly);
    connect_to_singularities(m_o, f_labels, bd, singularities, is_cut_poly, root);

    // Optionally do trim
    // FIXME Check if should be optional
    //trim_open_branch(m_o, f_labels, singularities, is_cut_poly);

    // Extend the overlay cut to the triangulated mesh
    // WARNING: Assumes new halfedges added to the end
    is_cut_o = std::vector<bool>(m.n_halfedges(), false);
    for (int h = 0; h < m_o.n_halfedges(); ++h)
    {
      is_cut_o[h] = is_cut_poly[h];
    }
    
    int start_h = 0;
    for(int i = 0; i < m_o_tri.n.size(); i++){
        if(f_labels[m_o_tri.f[i]] == 1 && f_labels[m_o_tri.f[m_o_tri.opp[i]]] == 2){
            start_h = i; break;
        }
    }
    spdlog::info("selected start h: {}, left: {}, right: {}", start_h, f_labels[m_o_tri.f[start_h]], f_labels[m_o_tri.f[m_o_tri.opp[start_h]]]);

    // sanity check for the input of compute layout
    // - opposite halfedges should have same edge lenghts (up to numerical error)
    // - all halfedges that belongs to a face with type 1 should have non-zero edge lengths
    for(int i = 0; i < m_o_tri.n.size(); i++){
        int h0 = i, h1 = m_o_tri.opp[h0];
        int ft0 = f_labels[m_o_tri.f[h0]];
        int ft1 = f_labels[m_o_tri.f[h1]];
        if(std::abs<Scalar>(m_o_tri._m.l[h0]-m_o_tri._m.l[h1]) > 1e-8 && ft0 == ft1 && ft0 == 1){
            spdlog::error("halfedge lengths mismatch, {}: {}, {}: {}; {}/{}", h0, m_o_tri._m.l[h0], h1, m_o_tri._m.l[h1], ft0, ft1);
        }
        int f0 = m_o_tri.f[h0];
        int f1 = m_o_tri.f[h1];
        if(f_labels[f0] == 1 && m_o_tri._m.l[h0] == 0)
            spdlog::error("copy 1 has zero edge at {}, f{}", h0, f0);
        if(f_labels[f1] == 1 && m_o_tri._m.l[h1] == 0)
            spdlog::error("copy 1 has zero edge at {}, f{}", h1, f1);
    }
    spdlog::info("sanity check done.");

    // mark boundary as cut
    for (int i = 0; i < is_cut_o.size(); i++)
    {
        if (f_labels[m_o_tri.f[i]] != f_labels[m_o_tri.f[m_o_tri.opp[i]]])
        {
            is_cut_o[i] = true;
        }
    }

    // now directly do layout on overlay mesh
    for(int f = 0; f < f_labels.size(); f++){
        int h0 = m_o_tri.h[f];
        int h1 = m_o_tri.n[h0];
        int h2 = m_o_tri.n[h1];
        m_o_tri._m.type[h0] = f_labels[f];
        m_o_tri._m.type[h1] = f_labels[f];
        m_o_tri._m.type[h2] = f_labels[f];
    }

    // get output connectivity and metric
    std::vector<Scalar> phi(m_o_tri.out.size(), 0.0);
    auto overlay_layout_res = compute_layout(m_o_tri._m, phi, is_cut_o, start_h);
    _u_o = std::get<0>(overlay_layout_res);
    _v_o = std::get<1>(overlay_layout_res);
    check_if_flipped(m, _u_o, _v_o);

    // recursively remove degree-1 edges unless it's connected to a singularity
    if (do_trim){
      spdlog::info("Trimming cut");
      is_cut_o = std::get<2>(overlay_layout_res);
      //trim_open_branch(m_o_tri, f_labels, singularities, is_cut_o);
    }

    _u_o.resize(m_o.n.size());
    _v_o.resize(m_o.n.size());
    is_cut_o.resize(m_o.n.size()); 

  }else{
    // Compute overlay mesh with lengths obtained from the uv layout
    Mesh<Scalar> m;
    m.n = m_o.n;
    m.opp = m_o.opp;
    m.f = m_o.f;
    m.h = m_o.h;
    m.out = m_o.out;
    m.to = m_o.to;
    m.l = std::vector<Scalar>(m.n.size(), 0.0);
    for(int i = 0; i < m.n.size(); i++){
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

    // Get overlay cut using only original edges
    std::vector<bool> is_cut_poly;
    compute_overlay_cut(m_o, is_cut_poly);

    // Optionally do trim
    // FIXME Check if should be optional
    trim_open_branch(m_o, f_labels, singularities, is_cut_poly);

    // Extend the overlay cut to the triangulated mesh
    // WARNING: Assumes new halfedges added to the end
    is_cut_o = std::vector<bool>(m.n_halfedges(), false);
    for (int h = 0; h < m_o.n_halfedges(); ++h)
    {
      is_cut_o[h] = is_cut_poly[h];
    }
    
    // now directly do layout on overlay mesh
    // TODO Make sure don't need to change edge type or other data fields
    std::vector<Scalar> phi(m.n_vertices(), 0.0);
    auto overlay_layout_res = compute_layout(m, phi, is_cut_o);
    _u_o = std::get<0>(overlay_layout_res);
    _v_o = std::get<1>(overlay_layout_res);
  }

  return std::make_tuple(_u_c, _v_c, is_cut_c, _u_o, _v_o, is_cut_o);
  
}

// optimization_layout

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