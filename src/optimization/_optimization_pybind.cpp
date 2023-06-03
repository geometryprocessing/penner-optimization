/*********************************************************************************
 *  This file is part of reference implementation of SIGGRAPH Asia 2021 Paper *
 *  `Efficient and Robust Discrete Conformal Equivalence with Boundary` * v1.0 *
 *                                                                                *
 *  The MIT License *
 *                                                                                *
 *  Permission is hereby granted, free of charge, to any person obtaining a *
 *  copy of this software and associated documentation files (the "Software"), *
 *  to deal in the Software without restriction, including without limitation *
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense, *
 *  and/or sell copies of the Software, and to permit persons to whom the *
 *  Software is furnished to do so, subject to the following conditions: *
 *                                                                                *
 *  The above copyright notice and this permission notice shall be included in *
 *  all copies or substantial portions of the Software. *
 *                                                                                *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, *
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
 ** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING *
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 ** IN THE SOFTWARE. *
 *                                                                                *
 *  Author(s): * Marcel Campen, Institute for Computer Science, Osnabrück
 *University, Germany. * Ryan Capouellez, Hanxiao Shen, Leyi Zhu, Daniele
 *Panozzo, Denis Zorin,        * Courant Institute of Mathematical Sciences, New
 *York University, USA          *
 *                                          * *
 *********************************************************************************/
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ConformalInterface.hh"
#include "Sampling.hh"
#include "area.hh"
#include "constraint.hh"
#include "embedding.hh"
#include "energies.hh"
#include "interpolation.hh"
#include "optimization.hh"
#include "optimization_layout.hh"
#include "projection.hh"
#include "shapes.hh"
#include "transitions.hh"
#include "validation.hh"
#include "visualization.hh"

namespace CurvatureMetric {

std::tuple<Mesh<double>,
           std::vector<double>,
           Eigen::SparseMatrix<double, Eigen::RowMajor>,
           std::vector<int>>
make_delaunay_with_jacobian_pybind(const Mesh<double>& C,
                                   const std::vector<double>& lambdas,
                                   bool need_jacobian = true)
{
  Mesh<double> C_del(C);
  std::vector<double> lambdas_del;
  Eigen::SparseMatrix<double, Eigen::RowMajor> J_del;
  std::vector<int> flip_seq;
  make_delaunay_with_jacobian(
    C, lambdas, C_del, lambdas_del, J_del, flip_seq, need_jacobian);

  return std::make_tuple(C_del, lambdas_del, J_del, flip_seq);
}

std::tuple<OverlayMesh<double>,
           std::vector<double>,
           Eigen::SparseMatrix<double, Eigen::RowMajor>,
           std::vector<int>>
make_delaunay_with_jacobian_overlay(const OverlayMesh<double>& C,
                                    const std::vector<double>& lambdas,
                                    bool need_jacobian = true)
{
  OverlayMesh<double> C_del(C);
  std::vector<double> lambdas_del;
  Eigen::SparseMatrix<double, Eigen::RowMajor> J_del;
  std::vector<int> flip_seq;
  make_delaunay_with_jacobian(
    C, lambdas, C_del, lambdas_del, J_del, flip_seq, need_jacobian);

  return std::make_tuple(C_del, lambdas_del, J_del, flip_seq);
}

std::vector<double>
flip_translations_pybind(const Mesh<double>& m,
                         const std::vector<int>& flip_seq,
                         const std::vector<double>& tau)
{
  std::vector<double> tau_out = tau;
  flip_translations(m, flip_seq, tau_out);
  return tau_out;
}

std::tuple<Mesh<double>, std::vector<double>>
flip_ccw_log_pybind(const Mesh<double>& C,
                    const std::vector<double>& lambdas,
                    int h)
{
  Mesh<double> C_flip(C);
  std::vector<double> lambdas_flip(lambdas);
  flip_ccw_log(C_flip, lambdas_flip, h);

  return std::make_tuple(C_flip, lambdas_flip);
}

std::vector<int>
make_delaunay_pybind(Mesh<double>& C, bool ptolemy = true)
{
  DelaunayStats delaunay_stats;
  SolveStats<double> solve_stats;
  VectorX u;
  u.setZero(C.n_ind_vertices());
  ConformalIdealDelaunay<double>::MakeDelaunay(
    C, u, delaunay_stats, solve_stats, ptolemy);
  return delaunay_stats.flip_seq;
}

std::tuple<Mesh<double>, std::vector<int>>
FV_to_double_pybind(const Eigen::MatrixXd& V,
                    const Eigen::MatrixXi& F,
                    const std::vector<double>& Theta_hat,
                    bool fix_boundary)
{
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  Mesh<double> m = FV_to_double(V,
                                F,
                                Theta_hat,
                                vtx_reindex,
                                indep_vtx,
                                dep_vtx,
                                v_rep,
                                bnd_loops,
                                fix_boundary);
  return std::make_tuple(m, vtx_reindex);
}

// FIXME
// Mesh<double>
// NOB_to_double_pybind(const std::vector<int> &next_he,
//                     const std::vector<int> &opp,
//                     const std::vector<int> &bnd_loops,
//                     const std::vector<double> &Theta_hat,
//                     bool fix_boundary)
//{
//    std::vector<int> indep_vtx, dep_vtx, v_rep;
//    Mesh<double> m = NOB_to_double(next_he, opp, bnd_loops, Theta_hat,
//    indep_vtx, dep_vtx, v_rep, fix_boundary); return m;
//}

std::tuple<std::vector<double>, std::vector<double>>
project_to_constraint_py(
  const Mesh<double>& m,
  const std::vector<double>& lambdas,
  std::shared_ptr<ProjectionParameters> proj_params = nullptr)
{
  VectorX u0;
  u0.setZero(m.n_ind_vertices());
  std::vector<double> lambdas_proj =
    project_to_constraint(m, lambdas, u0, proj_params);
  std::vector<double> u(u0.size());
  for (int i = 0; i < u0.size(); ++i) {
    u[i] = u0[i];
  }

  return std::make_tuple(lambdas_proj, u);
}

std::tuple<std::vector<int>, std::vector<int>>
build_edge_maps_py(const Mesh<double>& m)
{
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  return std::make_tuple(he2e, e2he);
}

std::tuple<std::vector<int>, std::vector<int>>
build_refl_proj_py(const Mesh<double>& m)
{
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  return std::make_tuple(proj, embed);
}

std::tuple<std::vector<int>, std::vector<int>>
build_refl_he_proj_py(const Mesh<double>& m)
{
  std::vector<int> he_proj;
  std::vector<int> he_embed;
  build_refl_he_proj(m, he_proj, he_embed);

  return std::make_tuple(he_proj, he_embed);
}

std::tuple<std::vector<double>, std::vector<double>>
angles_from_lambdas_py(const Mesh<double>& m,
                       const std::vector<double>& lambdas_full)
{
  std::vector<double> he2angle;
  std::vector<double> he2cot;
  angles_from_lambdas(m, lambdas_full, he2angle, he2cot);

  return std::make_tuple(he2angle, he2cot);
}

std::tuple<
  std::vector<std::vector<double>>, // V_out
  std::vector<std::vector<int>>,    // F_out
  std::vector<double>,              // layout u (per vertex)
  std::vector<double>,              // layout v (per vertex)
  std::vector<std::vector<int>>,    // FT_out
  std::vector<int>,                 // Fn_to_F
  std::vector<std::pair<int, int>>> // map from new vertices to original
                                    // endpoints
conformal_parametrization_VL_pybind(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<double>& Theta_hat,
  std::shared_ptr<ProjectionParameters> proj_params = nullptr)
{
  std::shared_ptr<AlgorithmParameters> alg_params;
  std::shared_ptr<LineSearchParameters> ls_params;
  std::shared_ptr<StatsParameters> stats_params;
  alg_params->initial_ptolemy = true;
  alg_params->max_itr = proj_params->max_itr;
  alg_params->error_eps = proj_params->error_eps;
  ls_params->bound_norm_thres = proj_params->bound_norm_thres;
  ls_params->do_reduction = proj_params->do_reduction;
  return conformal_parametrization_VL<double>(
    V, F, Theta_hat, alg_params, ls_params, stats_params);
}

std::tuple<Mesh<double>, std::vector<double>>
flip_edges_pybind(const Mesh<double>& m,
                  const std::vector<double>& lambdas_full,
                  const std::vector<int>& flip_seq)
{
  Mesh<double> m_flip;
  std::vector<double> lambdas_full_flip;
  flip_edges(m, lambdas_full, flip_seq, m_flip, lambdas_full_flip);

  return std::make_tuple(m_flip, lambdas_full_flip);
}

std::tuple<VectorX, VectorX, VectorX>
solve_translation_lagrangian_system_pybind(const Mesh<double>& m,
                                           const VectorX& delta_sigmas)
{
  VectorX tau;
  VectorX mu;
  VectorX nu;
  solve_translation_lagrangian_system(m, delta_sigmas, tau, mu, nu);

  return std::make_tuple(tau, mu, nu);
}

std::tuple<MatrixX, VectorX>
generate_translation_lagrangian_system_pybind(const Mesh<double>& m,
                                              const VectorX& delta_sigmas)
{
  MatrixX L;
  VectorX b;
  generate_translation_lagrangian_system(m, delta_sigmas, L, b);

  return std::make_tuple(L, b);
}

std::tuple<MatrixX, VectorX, std::vector<int>, std::vector<int>>
generate_interior_translation_lagrangian_system_pybind(
  const Mesh<double>& m,
  const VectorX& delta_sigmas)
{
  MatrixX L;
  VectorX b;
  std::vector<int> interior_he;
  std::vector<int> interior_he_map;
  generate_interior_translation_lagrangian_system(
    m, delta_sigmas, L, b, interior_he, interior_he_map);

  return std::make_tuple(L, b, interior_he, interior_he_map);
}
// using namespace CurvatureMetric;
//  wrap as Python module
PYBIND11_MODULE(optimization_py, m)
{
  m.doc() = "pybind for optimization module";

  // Classes
  pybind11::class_<ProjectionParameters, std::shared_ptr<ProjectionParameters>>(
    m, "ProjectionParameters")
    .def(pybind11::init<>())
    .def_readwrite("max_itr", &ProjectionParameters::max_itr)
    .def_readwrite("bound_norm_thres", &ProjectionParameters::bound_norm_thres)
    .def_readwrite("error_eps", &ProjectionParameters::error_eps)
    .def_readwrite("do_reduction", &ProjectionParameters::do_reduction)
    .def_readwrite("use_edge_flips", &ProjectionParameters::use_edge_flips)
    .def_readwrite("initial_ptolemy", &ProjectionParameters::initial_ptolemy);
  pybind11::class_<OptimizationParameters,
                   std::shared_ptr<OptimizationParameters>>(
    m, "OptimizationParameters")
    .def(pybind11::init<>())
    .def_readwrite("num_iter", &OptimizationParameters::num_iter)
    .def_readwrite("beta_0", &OptimizationParameters::beta_0)
    .def_readwrite("min_ratio", &OptimizationParameters::min_ratio)
    .def_readwrite("require_energy_decr",
                   &OptimizationParameters::require_energy_decr)
    .def_readwrite("max_angle_incr", &OptimizationParameters::max_angle_incr)
    .def_readwrite("max_angle", &OptimizationParameters::max_angle)
    .def_readwrite("p", &OptimizationParameters::p)
    .def_readwrite("fix_bd_lengths", &OptimizationParameters::fix_bd_lengths)
    .def_readwrite("energy_choice", &OptimizationParameters::energy_choice)
    .def_readwrite("bd_weight", &OptimizationParameters::bd_weight)
    .def_readwrite("max_grad_range", &OptimizationParameters::max_grad_range)
    .def_readwrite("max_beta", &OptimizationParameters::max_beta)
    .def_readwrite("direction_choice",
                   &OptimizationParameters::direction_choice)
    .def_readwrite("verbosity", &OptimizationParameters::verbosity)
    .def_readwrite("reg_factor", &OptimizationParameters::reg_factor)
    .def_readwrite("use_edge_lengths",
                   &OptimizationParameters::use_edge_lengths)
    .def_readwrite("use_log", &OptimizationParameters::use_log);
  pybind11::class_<Viewer>(m, "Viewer").def(pybind11::init<>());
  pybind11::class_<Connectivity>(m, "Connectivity")
    .def(pybind11::init<>())
    .def_readwrite("n", &Connectivity::n)
    .def_readwrite("prev", &Connectivity::prev)
    .def_readwrite("to", &Connectivity::to)
    .def_readwrite("f", &Connectivity::f)
    .def_readwrite("h", &Connectivity::h)
    .def_readwrite("out", &Connectivity::out)
    .def_readwrite("opp", &Connectivity::opp);
  pybind11::class_<AlgorithmParameters, std::shared_ptr<AlgorithmParameters>>(
    m, "AlgorithmParameters")
    .def(pybind11::init<>())
    .def_readwrite("MPFR_PREC", &AlgorithmParameters::MPFR_PREC)
    .def_readwrite("initial_ptolemy", &AlgorithmParameters::initial_ptolemy)
    .def_readwrite("error_eps", &AlgorithmParameters::error_eps)
    .def_readwrite("min_lambda", &AlgorithmParameters::min_lambda)
    .def_readwrite("newton_decr_thres", &AlgorithmParameters::newton_decr_thres)
    .def_readwrite("max_itr", &AlgorithmParameters::max_itr);
  pybind11::class_<StatsParameters, std::shared_ptr<StatsParameters>>(
    m, "StatsParameters")
    .def(pybind11::init<>())
    .def_readwrite("flip_count", &StatsParameters::flip_count)
    .def_readwrite("name", &StatsParameters::name)
    .def_readwrite("output_dir", &StatsParameters::output_dir)
    .def_readwrite("error_log", &StatsParameters::error_log)
    .def_readwrite("print_summary", &StatsParameters::print_summary)
    .def_readwrite("log_level", &StatsParameters::log_level);
  pybind11::class_<LineSearchParameters, std::shared_ptr<LineSearchParameters>>(
    m, "LineSearchParameters")
    .def(pybind11::init<>())
    .def_readwrite("energy_cond", &LineSearchParameters::energy_cond)
    .def_readwrite("energy_samples", &LineSearchParameters::energy_samples)
    .def_readwrite("do_reduction", &LineSearchParameters::do_reduction)
    .def_readwrite("descent_dir_max_variation",
                   &LineSearchParameters::descent_dir_max_variation)
    .def_readwrite("do_grad_norm_decrease",
                   &LineSearchParameters::do_grad_norm_decrease)
    .def_readwrite("bound_norm_thres", &LineSearchParameters::bound_norm_thres)
    .def_readwrite("lambda0", &LineSearchParameters::lambda0)
    .def_readwrite("reset_lambda", &LineSearchParameters::reset_lambda);
  pybind11::class_<OverlayProblem::Mesh<double>>(m, "Mesh")
    .def(pybind11::init<>())
    .def_readwrite("n", &OverlayProblem::Mesh<double>::n)
    .def_readwrite("to", &OverlayProblem::Mesh<double>::to)
    .def_readwrite("f", &OverlayProblem::Mesh<double>::f)
    .def_readwrite("h", &OverlayProblem::Mesh<double>::h)
    .def_readwrite("out", &OverlayProblem::Mesh<double>::out)
    .def_readwrite("opp", &OverlayProblem::Mesh<double>::opp)
    .def_readwrite("R", &OverlayProblem::Mesh<double>::R)
    .def_readwrite("type", &OverlayProblem::Mesh<double>::type)
    .def_readwrite("Th_hat", &OverlayProblem::Mesh<double>::Th_hat)
    .def_readwrite("l", &OverlayProblem::Mesh<double>::l)
    .def_readwrite("v_rep", &OverlayProblem::Mesh<double>::v_rep)
    .def_readwrite("fixed_dof", &OverlayProblem::Mesh<double>::fixed_dof);
  pybind11::class_<OverlayProblem::OverlayMesh<double>>(m, "OverlayMesh")
    .def_readwrite("n", &OverlayProblem::OverlayMesh<double>::n)
    .def_readwrite("to", &OverlayProblem::OverlayMesh<double>::to)
    .def_readwrite("f", &OverlayProblem::OverlayMesh<double>::f)
    .def_readwrite("h", &OverlayProblem::OverlayMesh<double>::h)
    .def_readwrite("out", &OverlayProblem::OverlayMesh<double>::out)
    .def_readwrite("opp", &OverlayProblem::OverlayMesh<double>::opp)
    .def_readwrite("R", &OverlayProblem::OverlayMesh<double>::R)
    .def_readwrite("type", &OverlayProblem::OverlayMesh<double>::type)
    .def_readwrite("prev", &OverlayProblem::OverlayMesh<double>::prev)
    .def_readwrite("first_segment",
                   &OverlayProblem::OverlayMesh<double>::first_segment)
    .def_readwrite("origin", &OverlayProblem::OverlayMesh<double>::origin)
    .def_readwrite("origin_of_origin",
                   &OverlayProblem::OverlayMesh<double>::origin_of_origin)
    .def_readwrite("vertex_type",
                   &OverlayProblem::OverlayMesh<double>::vertex_type)
    .def_readwrite("edge_type", &OverlayProblem::OverlayMesh<double>::edge_type)
    .def_readwrite("seg_bcs", &OverlayProblem::OverlayMesh<double>::seg_bcs)
    .def_readwrite("_m", &OverlayProblem::OverlayMesh<double>::_m);

  // Conformal
  m.def("fv_to_double",
        &FV_to_double_pybind,
        "Create double mesh from FV",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("find_conformal_metric",
        &ConformalIdealDelaunay<double>::FindConformalMetric,
        "Run conformal method",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("make_delaunay",
        &make_delaunay_pybind,
        "Make mesh Delauany with optional Euclidean flips",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("cpp_viewer", &cpp_viewer, "viewer mesh in libigl gui");
  m.def("get_pt_mat", &get_pt_mat, "get pt_mat");
  m.def("get_edges", &get_edges, "get edges mesh");
  m.def("compute_layout",
        &compute_layout<double>,
        "layout function",
        pybind11::arg("mesh"),
        pybind11::arg("u"),
        pybind11::arg("is_cut_h"),
        pybind11::arg("start_h"),
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  // Transitions
  m.def("make_delaunay_with_jacobian",
        &make_delaunay_with_jacobian_pybind,
        "Make mesh Delauany with Jacobian",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("make_delaunay_with_jacobian_overlay",
        &make_delaunay_with_jacobian_overlay,
        "Make mesh Delauany with Jacobian with overlay",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  init_area_pybind(m);
  init_constraint_pybind(m);
  init_energies_pybind(m);
  init_optimization_pybind(m);
  init_shapes_pybind(m);

  // Projection
  m.def("line_search_direction",
        &line_search_direction,
        "Get line search direction",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("project_to_constraint",
        &project_to_constraint_py,
        "Project to constraint manifold",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  // Embedding
  m.def("build_edge_maps",
        &build_edge_maps_py,
        "Build maps from edges to halfedges and vise versa",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("build_refl_he_proj",
        &build_refl_he_proj_py,
        "Build halfedge projection and embedding for reflection",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("build_refl_proj",
        &build_refl_proj_py,
        "Build projection and embedding for reflection",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("refl_matrix",
        &refl_matrix,
        "Build matrix for projection to the embedded mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  m.def("scale_distortion_direction",
        &scale_distortion_direction,
        "Get the scale distortion descent direction",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("id_matrix",
        &id_matrix,
        "Build identity matrix",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("Delaunay_inds_lambdas",
        &Delaunay_inds_lambdas,
        "Get Delaunay indices of all halfedges",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("angles_from_lambdas",
        &angles_from_lambdas_py,
        "Get triangle angles and cotangents",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("conformal_parametrization_VL",
        &conformal_parametrization_VL_pybind,
        "Get conformal parametrization",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("flip_ccw_log",
        &flip_ccw_log_pybind,
        "Flip a halfedge with log edge lengths",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  //  m.def("optimize_metric", &optimize_metric, "Get output for rendering
  //  optimized metric",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  m.def("project_descent_direction",
        &project_descent_direction,
        "Project general descent direction",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("halfedge_function_to_vertices",
        &halfedge_function_to_vertices,
        "Get max of halfedges",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("flip_edges",
        &flip_edges_pybind,
        "Flip edges to generate new mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("interpolate_3d",
        &ConformalIdealDelaunay<double>::Interpolate_3d,
        "Get interpolated vertices for the overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_optimized_overlay",
        &generate_optimized_overlay,
        "generate overlay mesh for optimized lambdas",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("_parametrize_mesh",
        &_parametrize_mesh,
        "generate parametrization for mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("parametrize_mesh",
        &parametrize_mesh,
        "generate parametrization for mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("add_overlay",
        &add_overlay,
        "Make mesh into overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("remove_overlay",
        &remove_overlay,
        "Get mesh from overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("make_delaunay_overlay",
        &make_delaunay_overlay,
        "Make overlay mesh Delaunay",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("flip_edges_overlay",
        &flip_edges_overlay,
        "Flip edges in overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("get_cones_and_bd",
        &get_cones_and_bd,
        "Get cones and boundary vertices for mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("reindex_F",
        &reindex_F,
        "Reindex vertices in an array of faces",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("get_layout",
        &get_layout<double>,
        "Get layout for a mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("get_FV_FTVT",
        &get_FV_FTVT<double>,
        "Get face indices for an overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("make_tufted_overlay",
        &make_tufted_overlay,
        "Make overlay mesh a tufted cover",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("change_lengths_overlay",
        &change_lengths_overlay,
        "Change lengths of an overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("bc_original_to_eq_overlay",
        &bc_original_to_eq_overlay,
        "Fix boundary points",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  // m.def("conformal_parametrization_vf_debug",
  // &conformal_parametrization_VL_debug<double>, "get conformal
  // parametrization,output: (V, F, u, v)", pybind11::arg("V"),
  // pybind11::arg("F"), pybind11::arg("Theta_hat"),
  //  pybind11::arg("alg_params") = nullptr,
  // pybind11::arg("ls_params") = nullptr,
  // pybind11::arg("stats_params") =
  // nullptr,pybind11::call_guard<pybind11::scoped_ostream_redirect,
  // pybind11::scoped_estream_redirect>());
  m.def("generate_layout_overlay_lambdas",
        &generate_layout_overlay_lambdas<double>,
        "layout overlay mesh with given log edge lengths",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("determine_intersection_translations",
        &determine_intersection_translations,
        "get translations for intersection points",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("view_mesh",
        &view_mesh,
        "open libigl mesh viewer",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_mesh_viewer",
        &generate_mesh_viewer,
        "generate viewer for mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("add_shading_to_mesh",
        &add_shading_to_mesh,
        "add shading to mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("add_normals_to_mesh",
        &add_normals_to_mesh,
        "add normals to mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("save_mesh_screen_capture",
        &save_mesh_screen_capture,
        "save viewer image to png",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("bc_reparametrize_eq",
        &bc_reparametrize_eq,
        "translate halfedges by a hyperbolic distance",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("is_valid_symmetry",
        &is_valid_symmetry,
        "Check mesh is valid double mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  // Interpolation
  m.def("compute_shear",
        &compute_shear,
        "compute the per halfedge shear",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_four_point_projection",
        &generate_four_point_projection,
        "map standard basis to four projective points",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_four_point_projection_from_shear",
        &generate_four_point_projection_from_shear,
        "map standard basis to four projective points for two triangle chart "
        "with given shear",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_four_point_projection_from_lengths",
        &generate_four_point_projection_from_lengths,
        "map standard basis to four projective points determined by layout of "
        "lengths",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_circumcirle_preserving_projection",
        &generate_circumcirle_preserving_projection,
        "map equilateral reference triangle to another triangle with preserved "
        "circumcircle",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("change_edge_shear",
        &change_edge_shear,
        "map between diagonals of two triangle charts",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("bc_eq_to_two_triangle_chart",
        &bc_eq_to_two_triangle_chart,
        "map from eq triangle to two triangle charts",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("bc_eq_to_scaled",
        &bc_eq_to_scaled,
        "map from eq triangle to scaled triangles",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("bc_two_triangle_chart_to_scaled",
        &bc_two_triangle_chart_to_scaled,
        "map from two triangle charts to scaled triangles",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_pack_halfedge_form_matrix",
        &generate_pack_halfedge_form_matrix,
        "Generate matrix to pack halfedge forms",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_unpack_halfedge_form_matrix",
        &generate_unpack_halfedge_form_matrix,
        "Generate matrix to unpack halfedge forms",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("determine_linear_translations",
        &determine_linear_translations,
        "Get the translation values for linear maps",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("flip_translations",
        &flip_translations_pybind,
        "Update translation values after edge flips",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_halfedge_sum_matrix",
        &generate_halfedge_sum_matrix,
        "Build matrix for summing halfedge values",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_face_sum_matrix",
        &generate_face_sum_matrix,
        "Build matrix for summing face values",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_translation_lagrangian_system",
        &generate_translation_lagrangian_system_pybind,
        "Generate least squares translation Lagrangian system",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("generate_interior_translation_lagrangian_system",
        &generate_interior_translation_lagrangian_system_pybind,
        "Generate least squares translation Lagrangian system",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def(
    "generate_symmetric_translation_lagrangian_system",
    &generate_symmetric_translation_lagrangian_system,
    "Generate least squares translation Lagrangian system for a symmetric mesh",
    pybind11::call_guard<pybind11::scoped_ostream_redirect,
                         pybind11::scoped_estream_redirect>());
  m.def("solve_translation_lagrangian_system",
        &solve_translation_lagrangian_system_pybind,
        "Solve for least squares translations",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  // Validation
  m.def("validate_hencky_strain_face",
        &validate_hencky_strain_face,
        "Validate surface Hencky strain energy with Maple generated code",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());

  //  m.def("nob_to_double", &NOB_to_double_pybind, "Create double mesh from
  //  NOB",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("layout_lambdas", &layout_lambdas, "get a layout from a mesh with
  //  log edge lengths",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("layout_lambdas_nob", &layout_lambdas_NOB, "get a layout from a mesh
  //  with log edge lengths",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("layout_lambdas_fv", &layout_lambdas_FV, "get a layout from a mesh
  //  with log edge lengths",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("build_overlay_mesh", &build_overlay_mesh, "Build overlay mesh for
  //  mesh with log lengths",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("get_he2v", &get_he2v, "Get map from halfedges to endpoints",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("get_edges", &get_edges, "Get map from halfedges to endpoints",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("get_faces", &get_faces, "Get triangulated faces of an overlay
  //  mesh",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("get_to_map", &get_to_map, "Get triangulated faces of an overlay
  //  mesh",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  m.def("mesh_parametrization_VL",
        &mesh_parametrization_VL,
        "Get triangulated faces of an overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  //  m.def("get_layout_overlay", &get_layout_overlay<double>, "Get layout for
  //  an overlay mesh",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("print_overlay_info", &print_overlay_info, "Print information about
  //  overlay mesh",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("mesh_metric", &mesh_metric, "Get Delaunay overlay mesh for a mesh
  //  with lengths",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
  //  m.def("generate_layout_lambdas", &generate_layout_lambdas<double>, "layout
  //  mesh with given log edge lengths",
  //    pybind11::call_guard<pybind11::scoped_ostream_redirect,
  //    pybind11::scoped_estream_redirect>());
}
}
