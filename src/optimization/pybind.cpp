// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "optimization/pybind.h"


#include "metric/area.h"
#include "metric/constraint.h"
#include "util/io.h"
#include "optimization/metric_optimization/energies.h"
#include "optimization/metric_optimization/energy_functor.h"
#include "optimization/metric_optimization/explicit_optimization.h"
#include "optimization/metric_optimization/implicit_optimization.h"
#include "optimization/metric_optimization/uv_optimization.h"
#include "parametrization/interpolation.h"
#include "parametrization/layout.h"
#include "optimization/interface.h"
#include "metric/projection.h"
#include "parametrization/refinement.h"
#include "parametrization/parametrize.h"
#include "optimization/util/shapes.h"
#include "metric/shear.h"
#include "parametrization/translation.h"
#include "parametrization/error.h"

#ifdef USE_HIGHFIVE
#include <highfive/H5Easy.hpp>
#endif

#ifdef RENDER_TEXTURE
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Sampling.hh"
#endif

namespace Penner {
namespace Optimization {

#ifdef PYBIND
#ifndef MULTIPRECISION

std::tuple<Mesh<Scalar>, std::vector<int>> FV_to_double_pybind(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Theta_hat,
    const std::vector<int>& free_cones,
    bool fix_boundary)
{
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double(
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
        fix_boundary);
    return std::make_tuple(m, vtx_reindex);
}

void init_classes_pybind(pybind11::module& m)
{
    pybind11::class_<OptimizationParameters, std::shared_ptr<OptimizationParameters>>(
        m,
        "OptimizationParameters")
        .def(pybind11::init<>())
        .def_readwrite("output_dir", &OptimizationParameters::output_dir)
        .def_readwrite("use_checkpoints", &OptimizationParameters::use_checkpoints)
        .def_readwrite("min_ratio", &OptimizationParameters::min_ratio)
        .def_readwrite("num_iter", &OptimizationParameters::num_iter)
        .def_readwrite("require_energy_decr", &OptimizationParameters::require_energy_decr)
        .def_readwrite(
            "require_gradient_proj_negative",
            &OptimizationParameters::require_gradient_proj_negative)
        .def_readwrite("max_angle_incr", &OptimizationParameters::max_angle_incr)
        .def_readwrite("max_energy_incr", &OptimizationParameters::max_energy_incr)
        .def_readwrite("direction_choice", &OptimizationParameters::direction_choice)
        .def_readwrite("beta_0", &OptimizationParameters::beta_0)
        .def_readwrite("max_beta", &OptimizationParameters::max_beta)
        .def_readwrite("max_grad_range", &OptimizationParameters::max_grad_range)
        .def_readwrite("max_angle", &OptimizationParameters::max_angle);

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
        m,
        "AlgorithmParameters")
        .def(pybind11::init<>())
        .def_readwrite("MPFR_PREC", &AlgorithmParameters::MPFR_PREC)
        .def_readwrite("initial_ptolemy", &AlgorithmParameters::initial_ptolemy)
        .def_readwrite("error_eps", &AlgorithmParameters::error_eps)
        .def_readwrite("min_lambda", &AlgorithmParameters::min_lambda)
        .def_readwrite("newton_decr_thres", &AlgorithmParameters::newton_decr_thres)
        .def_readwrite("max_itr", &AlgorithmParameters::max_itr);

    pybind11::class_<StatsParameters, std::shared_ptr<StatsParameters>>(m, "StatsParameters")
        .def(pybind11::init<>())
        .def_readwrite("flip_count", &StatsParameters::flip_count)
        .def_readwrite("name", &StatsParameters::name)
        .def_readwrite("output_dir", &StatsParameters::output_dir)
        .def_readwrite("error_log", &StatsParameters::error_log)
        .def_readwrite("print_summary", &StatsParameters::print_summary)
        .def_readwrite("log_level", &StatsParameters::log_level);

    pybind11::class_<LineSearchParameters, std::shared_ptr<LineSearchParameters>>(
        m,
        "LineSearchParameters")
        .def(pybind11::init<>())
        .def_readwrite("energy_cond", &LineSearchParameters::energy_cond)
        .def_readwrite("energy_samples", &LineSearchParameters::energy_samples)
        .def_readwrite("do_reduction", &LineSearchParameters::do_reduction)
        .def_readwrite(
            "descent_dir_max_variation",
            &LineSearchParameters::descent_dir_max_variation)
        .def_readwrite("do_grad_norm_decrease", &LineSearchParameters::do_grad_norm_decrease)
        .def_readwrite("bound_norm_thres", &LineSearchParameters::bound_norm_thres)
        .def_readwrite("lambda0", &LineSearchParameters::lambda0)
        .def_readwrite("reset_lambda", &LineSearchParameters::reset_lambda);

    pybind11::class_<ReductionMaps>(m, "ReductionMaps")
        .def(pybind11::init<const Mesh<Scalar>&>())
        .def_readwrite("he2e", &ReductionMaps::he2e)
        .def_readwrite("e2he", &ReductionMaps::e2he)
        .def_readwrite("proj", &ReductionMaps::proj)
        .def_readwrite("embed", &ReductionMaps::embed);

    pybind11::class_<EnergyFunctor>(m, "EnergyFunctor")
        .def("energy", static_cast<Scalar (EnergyFunctor::*)(const DifferentiableConeMetric&) const>(&EnergyFunctor::energy));

    pybind11::class_<LogLengthEnergy, EnergyFunctor>(m, "LogLengthEnergy")
        .def(pybind11::init<const DifferentiableConeMetric&, int>());

    pybind11::class_<QuadraticSymmetricDirichletEnergy, EnergyFunctor>(
        m,
        "QuadraticSymmetricDirichletEnergy")
        .def(pybind11::init<const DifferentiableConeMetric&, const DiscreteMetric&>());

    pybind11::class_<SymmetricDirichletEnergy, EnergyFunctor>(m, "SymmetricDirichletEnergy")
        .def(pybind11::init<const DifferentiableConeMetric&, const DiscreteMetric&>());

    pybind11::class_<LogScaleEnergy, EnergyFunctor>(m, "LogScaleEnergy")
        .def(pybind11::init<const DifferentiableConeMetric&>());

    pybind11::class_<InterpolationMesh<Scalar>>(m, "InterpolationMesh")
        .def(pybind11::init<
             const Eigen::MatrixXd&, // V
             const Eigen::MatrixXi&, // F
             const Eigen::MatrixXd&, // uv
             const Eigen::MatrixXi&, // F_uv
             const std::vector<Scalar>& // Theta_hat
             >())
        .def(
            "get_overlay_mesh",
            &InterpolationMesh<Scalar>::get_overlay_mesh,
            pybind11::return_value_policy::copy);

    pybind11::class_<RefinementMesh>(m, "RefinementMesh")
        .def(pybind11::init<
             const Eigen::MatrixXd&, // V
             const Eigen::MatrixXi&, // F
             const Eigen::MatrixXd&, // uv
             const Eigen::MatrixXi&, // F_uv
             const std::vector<int>&, // Fn_to_F,
             const std::vector<std::pair<int, int>>& // endpoints
             >())
        .def(
            "get_VF_mesh",
            static_cast<std::tuple<
                Eigen::MatrixXd, // V
                Eigen::MatrixXi, // F
                Eigen::MatrixXd, // uv
                Eigen::MatrixXi, // F_uv
                std::vector<int>, // Fn_to_F
                std::vector<std::pair<int, int>> // endpoints
                > (RefinementMesh::*)() const>(&RefinementMesh::get_VF_mesh),
            pybind11::return_value_policy::copy);
}

void init_conformal_pybind(pybind11::module& m)
{
    m.def(
        "fv_to_double",
        &FV_to_double_pybind,
        "Create double mesh from FV",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
// TODO Replace with internal method
  m.def("overlay_mesh_to_VL",
        &overlay_mesh_to_VL<Scalar>,
        pybind11::call_guard<pybind11::scoped_ostream_redirect,
                             pybind11::scoped_estream_redirect>());
#ifdef RENDER_TEXTURE
    m.def("cpp_viewer", &cpp_viewer, "viewer mesh in libigl gui");
    m.def("get_pt_mat", &get_pt_mat, "get pt_mat");
    m.def("get_edges", &get_edges, "get edges mesh");
#endif
}

void init_energies_pybind(pybind11::module& m)
{
    m.def(
        "first_invariant",
        &first_invariant_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "second_invariant_squared",
        &second_invariant_squared_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "metric_distortion_energy",
        &metric_distortion_energy_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "area_distortion_energy",
        &area_distortion_energy_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "symmetric_dirichlet_energy",
        &symmetric_dirichlet_energy_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "first_invariant_vf",
        &first_invariant_vf_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "second_invariant_vf",
        &second_invariant_vf_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "surface_hencky_strain_energy_vf",
        &surface_hencky_strain_energy_vf,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "conformal_scaling_matrix",
        &conformal_scaling_matrix,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "best_fit_conformal",
        &best_fit_conformal,
        "Get the best fit conformal map for a metric map",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "root_mean_square_relative_error",
        &root_mean_square_relative_error,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
}

void init_opt_pybind(pybind11::module& m)
{
    m.def(
        "correct_cone_angles",
        &correct_cone_angles,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "generate_initial_mesh",
        &generate_initial_mesh,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "project_metric_to_constraint",
        &project_metric_to_constraint,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "compute_max_constraint",
        &compute_max_constraint,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "optimize_metric",
        &optimize_metric,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "compute_shear_dual_basis",
        &compute_shear_dual_basis_pybind,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "optimize_domain_coordinates",
        &optimize_domain_coordinates,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "optimize_shear_basis_coordinates",
        &optimize_shear_basis_coordinates,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "generate_VF_mesh_from_metric",
        &generate_VF_mesh_from_metric,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "generate_VF_mesh_from_discrete_metric",
        &generate_VF_mesh_from_discrete_metric,
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
}

void init_parameterization_pybind(pybind11::module& m)
{
    m.def(
        "add_overlay",
        &add_overlay<Scalar>,
        "Make mesh into overlay mesh",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "make_tufted_overlay",
        &make_tufted_overlay<Scalar>,
        "Make overlay mesh a tufted cover",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def("compute_layout_VF", &compute_layout_VF<double>, 
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
}

#ifdef USE_HIGHFIVE
void save_simplify_overlay_input(
    std::string fname,
    std::vector<std::pair<int, int>> endpoints,
    Eigen::MatrixXd V,
    Eigen::MatrixXi F,
    Eigen::MatrixXd uv,
    Eigen::MatrixXi Fuv,
    Eigen::MatrixXi cut_type)
{
    H5Easy::File hd_file(fname, H5Easy::File::Overwrite);

    // Save endpoints
    Eigen::MatrixXi endPT(endpoints.size(), 2);
    for (int i = 0; i < endPT.rows(); i++) endPT.row(i) << endpoints[i].first, endpoints[i].second;
    H5Easy::dump(hd_file, "endPT", endPT);

    // Save mesh
    assert(V.cols() == 3 && uv.cols() == 2);
    H5Easy::dump(hd_file, "V", V);
    H5Easy::dump(hd_file, "uv", uv);
    H5Easy::dump(hd_file, "F", F);
    H5Easy::dump(hd_file, "Fuv", Fuv);
    H5Easy::dump(hd_file, "cut_type", cut_type);
}


std::tuple<
    std::vector<std::pair<int, int>>, // endpoints
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, // V
    Eigen::MatrixXi, // F
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, // uv
    Eigen::MatrixXi, // Fuv
    Eigen::MatrixXi, // cut_type
    Eigen::MatrixXi> // Vn_to_V
load_simplify_overlay_output(std::string fname)
{
    H5Easy::File hd_file(fname, H5Easy::File::ReadOnly);

    // Get endpoints and convert to vector format
    auto endPT = H5Easy::load<Eigen::MatrixXi>(hd_file, "endPT");
    std::vector<std::pair<int, int>> endpoints;
    endpoints.reserve(endPT.rows());
    for (int i = 0; i < endPT.rows(); i++) {
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
#endif

void init_uv_optimization_pybind(pybind11::module& m)
{
#ifdef USE_UV_OPTIMIZATION
    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    pybind11::class_<SymDir::Parameters>(m, "SymDirParameters")
        .def(pybind11::init<>());

    m.def("load_parameters", &load_parameters, default_call_guard);
    m.def("optimize_seamless_parameterization", &optimize_seamless_parameterization, default_call_guard);
    m.def("optimize_aligned_parameterization", &optimize_aligned_parameterization, default_call_guard);
#endif
}

void init_optimization_pybind(pybind11::module& m)
{
    init_classes_pybind(m);
    init_conformal_pybind(m);

    init_energies_pybind(m);
    init_opt_pybind(m);
    init_parameterization_pybind(m);
    init_uv_optimization_pybind(m);


#ifdef USE_HIGHFIVE
    m.def(
        "save_simplify_overlay_input",
        &save_simplify_overlay_input,
        "Save simplify overlay mesh input to file",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
    m.def(
        "load_simplify_overlay_output",
        &load_simplify_overlay_output,
        "Load simplify overlay mesh output from file",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
#endif

    m.def(
        "write_obj_with_uv",
        &write_obj_with_uv,
        "Write obj file with uv coordinates",
        pybind11::
            call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>());
}

#endif
#endif

} // namespace Optimization
} // namespace Penner