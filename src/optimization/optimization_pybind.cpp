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

#include "common.hh"

#include "area.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Sampling.hh"
#include "constraint.hh"
#include "convergence.hh"
#include "delaunay.hh"
#include "embedding.hh"
#include "energies.hh"
#include "energy_functor.hh"
#include "interpolation.hh"
#include "implicit_optimization.hh"
#include "explicit_optimization.hh"
#include "layout.hh"
#include "optimization_interface.hh"
#include "projection.hh"
#include "reparametrization.hh"
#include "refinement.hh"
#include "shapes.hh"
#include "shear.hh"
#include "targets.hh"
#include "translation.hh"
#include "visualization.hh"
#include <highfive/H5Easy.hpp>

namespace CurvatureMetric {

#ifdef PYBIND
#ifndef MULTIPRECISION

std::tuple<Mesh<Scalar>, std::vector<int>>
FV_to_double_pybind(const Eigen::MatrixXd& V,
                    const Eigen::MatrixXi& F,
                    const Eigen::MatrixXd& uv,
                    const Eigen::MatrixXi& F_uv,
                    const std::vector<Scalar>& Theta_hat,
                    const std::vector<int> &free_cones,
                    bool fix_boundary)
{
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  Mesh<Scalar> m = FV_to_double(V,
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
                                fix_boundary);
  return std::make_tuple(m, vtx_reindex);
}

void
init_classes_pybind(pybind11::module& m)
{
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
    .def_readwrite("require_gradient_proj_negative",
                   &OptimizationParameters::require_gradient_proj_negative)
    .def_readwrite("max_angle_incr", &OptimizationParameters::max_angle_incr)
    .def_readwrite("use_optimal_projection", &OptimizationParameters::use_optimal_projection)
    .def_readwrite("max_angle", &OptimizationParameters::max_angle)
    .def_readwrite("p", &OptimizationParameters::p)
    .def_readwrite("fix_bd_lengths", &OptimizationParameters::fix_bd_lengths)
    .def_readwrite("energy_choice", &OptimizationParameters::energy_choice)
    .def_readwrite("bd_weight", &OptimizationParameters::bd_weight)
    .def_readwrite("cone_weight", &OptimizationParameters::cone_weight)
    .def_readwrite("max_grad_range", &OptimizationParameters::max_grad_range)
    .def_readwrite("max_energy_incr", &OptimizationParameters::max_energy_incr)
    .def_readwrite("max_ratio_incr", &OptimizationParameters::max_ratio_incr)
    .def_readwrite("max_beta", &OptimizationParameters::max_beta)
    .def_readwrite("direction_choice",
                   &OptimizationParameters::direction_choice)
    .def_readwrite("output_dir", &OptimizationParameters::output_dir)
    .def_readwrite("reg_factor", &OptimizationParameters::reg_factor)
    .def_readwrite("use_edge_lengths",
                   &OptimizationParameters::use_edge_lengths)
    .def_readwrite("use_checkpoints",
                   &OptimizationParameters::use_checkpoints)
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

  pybind11::class_<OverlayProblem::Mesh<Scalar>>(m, "Mesh")
    .def(pybind11::init<>())
    .def_readwrite("n", &OverlayProblem::Mesh<Scalar>::n)
    .def_readwrite("to", &OverlayProblem::Mesh<Scalar>::to)
    .def_readwrite("f", &OverlayProblem::Mesh<Scalar>::f)
    .def_readwrite("h", &OverlayProblem::Mesh<Scalar>::h)
    .def_readwrite("out", &OverlayProblem::Mesh<Scalar>::out)
    .def_readwrite("opp", &OverlayProblem::Mesh<Scalar>::opp)
    .def_readwrite("R", &OverlayProblem::Mesh<Scalar>::R)
    .def_readwrite("type", &OverlayProblem::Mesh<Scalar>::type)
    .def_readwrite("Th_hat", &OverlayProblem::Mesh<Scalar>::Th_hat)
    .def_readwrite("l", &OverlayProblem::Mesh<Scalar>::l)
    .def_readwrite("v_rep", &OverlayProblem::Mesh<Scalar>::v_rep)
    .def_readwrite("fixed_dof", &OverlayProblem::Mesh<Scalar>::fixed_dof);

  pybind11::class_<OverlayProblem::OverlayMesh<Scalar>>(m, "OverlayMesh")
    .def_readwrite("n", &OverlayProblem::OverlayMesh<Scalar>::n)
    .def_readwrite("to", &OverlayProblem::OverlayMesh<Scalar>::to)
    .def_readwrite("f", &OverlayProblem::OverlayMesh<Scalar>::f)
    .def_readwrite("h", &OverlayProblem::OverlayMesh<Scalar>::h)
    .def_readwrite("out", &OverlayProblem::OverlayMesh<Scalar>::out)
    .def_readwrite("opp", &OverlayProblem::OverlayMesh<Scalar>::opp)
    .def_readwrite("R", &OverlayProblem::OverlayMesh<Scalar>::R)
    .def_readwrite("type", &OverlayProblem::OverlayMesh<Scalar>::type)
    .def_readwrite("prev", &OverlayProblem::OverlayMesh<Scalar>::prev)
    .def_readwrite("first_segment",
                   &OverlayProblem::OverlayMesh<Scalar>::first_segment)
    .def_readwrite("origin", &OverlayProblem::OverlayMesh<Scalar>::origin)
    .def_readwrite("origin_of_origin",
                   &OverlayProblem::OverlayMesh<Scalar>::origin_of_origin)
    .def_readwrite("vertex_type",
                   &OverlayProblem::OverlayMesh<Scalar>::vertex_type)
    .def_readwrite("edge_type", &OverlayProblem::OverlayMesh<Scalar>::edge_type)
    .def_readwrite("seg_bcs", &OverlayProblem::OverlayMesh<Scalar>::seg_bcs)
    .def_readwrite("_m", &OverlayProblem::OverlayMesh<Scalar>::_m);

  pybind11::class_<ReductionMaps>(m, "ReductionMaps")
    .def(pybind11::init<const Mesh<Scalar> &>())
    .def_readwrite("he2e", &ReductionMaps::he2e)
    .def_readwrite("e2he", &ReductionMaps::e2he)
    .def_readwrite("proj", &ReductionMaps::proj)
    .def_readwrite("embed", &ReductionMaps::embed);

  pybind11::class_<EnergyFunctor>(m, "EnergyFunctor");

  pybind11::class_<InterpolationMesh>(m, "InterpolationMesh")
    .def(pybind11::init<const Eigen::MatrixXd &, // V
                        const Eigen::MatrixXi &, // F
                        const std::vector<Scalar>&  // Theta_hat
                        >())
    .def("get_overlay_mesh",
      &InterpolationMesh::get_overlay_mesh,
      pybind11::return_value_policy::copy);

  pybind11::class_<RefinementMesh>(m, "RefinementMesh")
    .def(pybind11::init<const Eigen::MatrixXd &, // V
                        const Eigen::MatrixXi &, // F
                        const Eigen::MatrixXd &, // uv
                        const Eigen::MatrixXi &, // F_uv
                        const std::vector<int>&, // Fn_to_F,
                        const std::vector<std::pair<int, int>>& // endpoints
                        >())
    .def("get_VF_mesh",
      static_cast<
            std::tuple<
                  Eigen::MatrixXd, // V
                  Eigen::MatrixXi, // F
                  Eigen::MatrixXd, // uv
                  Eigen::MatrixXi, // F_uv
                  std::vector<int>, // Fn_to_F
                  std::vector<std::pair<int, int>> // endpoints
            >(RefinementMesh::*)() const
      >(&RefinementMesh::get_VF_mesh),
      pybind11::return_value_policy::copy);
}

void
init_conformal_pybind(pybind11::module& m)
{
  m.def("fv_to_double",
        &FV_to_double_pybind,
        "Create double mesh from FV",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("overlay_mesh_to_VL",
        &overlay_mesh_to_VL<Scalar>,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("get_pt_mat", &get_pt_mat, "get pt_mat");
  m.def("get_edges", &get_edges, "get edges mesh");
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
  m.def("save_mesh_screen_capture",
        &save_mesh_screen_capture,
        "save viewer image to png",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
}

void
init_energies_pybind(pybind11::module& m)
{
  m.def("first_invariant_vf",
        &first_invariant_vf_pybind,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("second_invariant_vf",
        &second_invariant_vf_pybind,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("best_fit_conformal",
        &best_fit_conformal,
        "Get the best fit conformal map for a metric map",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
}

void
init_optimization_pybind(pybind11::module& m)
{
  m.def("generate_VF_mesh_from_metric",
        &generate_VF_mesh_from_metric,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("correct_cone_angles",
        &correct_cone_angles_pybind,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("consistent_overlay_mesh_to_VL",
        &consistent_overlay_mesh_to_VL,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("compute_log_edge_lengths",
        &compute_log_edge_lengths_pybind,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("compute_penner_coordinates",
        &compute_penner_coordinates_pybind,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("optimize_metric",
        &optimize_metric,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
}

void
init_parameterization_pybind(pybind11::module& m)
{
  m.def("add_overlay",
        &add_overlay,
        "Make mesh into overlay mesh",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("make_tufted_overlay",
        &make_tufted_overlay,
        "Make overlay mesh a tufted cover",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("compute_uv_length_error",
        &compute_uv_length_error,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
}

// FIXME Move
std::vector<int>
make_delaunay_pybind(Mesh<Scalar>& C, bool ptolemy = true)
{
  DelaunayStats delaunay_stats;
  SolveStats<Scalar> solve_stats;
  VectorX u;
  u.setZero(C.n_ind_vertices());
  ConformalIdealDelaunay<Scalar>::MakeDelaunay(
    C, u, delaunay_stats, solve_stats, ptolemy);
  return delaunay_stats.flip_seq;
}


void save_simplify_overlay_input(std::string fname,
                                 std::vector<std::pair<int,int>> endpoints,
                                 Eigen::MatrixXd V,
                                 Eigen::MatrixXi F,
                                 Eigen::MatrixXd uv,
                                 Eigen::MatrixXi Fuv,
                                 Eigen::MatrixXi cut_type)
{
    H5Easy::File hd_file(fname, H5Easy::File::Overwrite);

    // Save endpoints
    Eigen::MatrixXi endPT(endpoints.size(), 2);
    for(int i = 0; i < endPT.rows(); i++)
        endPT.row(i) << endpoints[i].first, endpoints[i].second;
    H5Easy::dump(hd_file, "endPT", endPT);

    // Save mesh
    assert(V.cols() == 3 && uv.cols() == 2);
    H5Easy::dump(hd_file, "V", V);
    H5Easy::dump(hd_file, "uv", uv);
    H5Easy::dump(hd_file, "F", F);
    H5Easy::dump(hd_file, "Fuv", Fuv);
    H5Easy::dump(hd_file, "cut_type", cut_type);
}


std::tuple<std::vector<std::pair<int,int>>, // endpoints
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, // V
           Eigen::MatrixXi, // F
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, // uv
           Eigen::MatrixXi, // Fuv
           Eigen::MatrixXi, // cut_type
           Eigen::MatrixXi> // Vn_to_V
load_simplify_overlay_output(
  std::string fname
){
  H5Easy::File hd_file(fname, H5Easy::File::ReadOnly);

  // Get endpoints and convert to vector format
  auto endPT = H5Easy::load<Eigen::MatrixXi>(hd_file, "endPT");
  std::vector<std::pair<int,int>> endpoints;
  endpoints.reserve(endPT.rows());
  for(int i = 0; i < endPT.rows(); i++)
  {
    endpoints.push_back(std::make_pair(endPT(i, 0), endPT(i, 1)));
  }

  // Load mesh
  Eigen::MatrixXd V = H5Easy::load<Eigen::MatrixXd>(hd_file, "V");
  Eigen::MatrixXi F = H5Easy::load<Eigen::MatrixXi>(hd_file, "F");
  Eigen::MatrixXd uv = H5Easy::load<Eigen::MatrixXd>(hd_file, "uv");
  Eigen::MatrixXi Fuv = H5Easy::load<Eigen::MatrixXi>(hd_file, "Fuv");
  Eigen::MatrixXi cut_type = H5Easy::load<Eigen::MatrixXi>(hd_file, "cut_type");
  Eigen::MatrixXi Vn_to_V = H5Easy::load<Eigen::MatrixXi>(hd_file, "Vn_to_V");

  return std::make_tuple(endpoints, V, F, uv, Fuv, cut_type, Vn_to_V);
}
  


// wrap as Python module
PYBIND11_MODULE(optimization_py, m)
{
  m.doc() = "pybind for optimization module";
  spdlog::set_level(spdlog::level::info);

  init_classes_pybind(m);
  init_conformal_pybind(m);

  init_energies_pybind(m);
  init_optimization_pybind(m);
  init_parameterization_pybind(m);

  m.def("make_delaunay",
        &make_delaunay_pybind,
        "Make mesh Delauany with optional Euclidean flips",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
  m.def("save_simplify_overlay_input", 
        &save_simplify_overlay_input,
        "Save simplify overlay mesh input to file",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
        pybind11::scoped_estream_redirect>());
  m.def("load_simplify_overlay_output", 
        &load_simplify_overlay_output,
        "Load simplify overlay mesh output from file",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
        pybind11::scoped_estream_redirect>());
  m.def("write_obj_with_uv",
        &write_obj_with_uv,
        "Write obj file with uv coordinates",
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
        pybind11::scoped_estream_redirect>());

}
#endif
#endif
}
