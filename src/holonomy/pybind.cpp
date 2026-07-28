// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "holonomy/pybind.h"

#include "holonomy/interface.h"
#include "metric/quality.h"
#include "field/cones.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"
#include "holonomy/holonomy/constraint.h"
#include "holonomy/core/viewer.h"
#include "field/rotation_form.h"
#include "field/intrinsic_field.h"
#include "field/frame_field.h"
#include "holonomy/similarity/conformal.h"
#include "holonomy/similarity/energy.h"
#include "holonomy/similarity/layout.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"
#include "util/boundary.h"
#include "igl/writeOBJ.h"

namespace Penner {
namespace Holonomy {

#ifdef PYBIND
#ifndef MULTIPRECISION

void writeOBJ(
    const std::string str,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& CN,
    const Eigen::MatrixXi& FN,
    const Eigen::MatrixXd& TC,
    const Eigen::MatrixXi& FTC)
{
    igl::writeOBJ(str, V, F, CN, FN, TC, FTC);
}

void init_holonomy_pybind(pybind11::module& m)
{
    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    pybind11::class_<NewtonParameters, std::shared_ptr<NewtonParameters>>(m, "NewtonParameters")
        .def(pybind11::init<>())
        .def_readwrite("output_dir", &NewtonParameters::output_dir)
        .def_readwrite("error_log", &NewtonParameters::error_log)
        .def_readwrite("reset_lambda", &NewtonParameters::reset_lambda)
        .def_readwrite("do_reduction", &NewtonParameters::do_reduction)
        .def_readwrite("lambda0", &NewtonParameters::lambda0)
        .def_readwrite("error_eps", &NewtonParameters::error_eps)
        .def_readwrite("bound_norm_thres", &NewtonParameters::bound_norm_thres)
        .def_readwrite("max_itr", &NewtonParameters::max_itr)
        .def_readwrite("max_time", &NewtonParameters::max_time)
        .def_readwrite("min_lambda", &NewtonParameters::min_lambda)
        .def_readwrite("solver", &NewtonParameters::solver)
        .def_readwrite("log_level", &NewtonParameters::log_level);

    pybind11::class_<MarkedMetricParameters, ConeMetricParameters>(
        m,
        "MarkedMetricParameters")
        .def(pybind11::init<>())
        .def_readwrite("remove_loop_constraints", &MarkedMetricParameters::remove_loop_constraints)
        .def_readwrite("remove_trivial_torus", &MarkedMetricParameters::remove_trivial_torus)
        .def_readwrite("weighting", &MarkedMetricParameters::weighting);

    pybind11::class_<Field::FieldParameters, std::shared_ptr<Field::FieldParameters>>(m, "FieldParameters")
        .def(pybind11::init<>())
        .def_readwrite("min_cone", &Field::FieldParameters::min_cone)
        .def_readwrite("fix_cone_pair", &Field::FieldParameters::fix_cone_pair)
        .def_readwrite("use_principal_directions", &Field::FieldParameters::use_principal_directions);


    pybind11::class_<MarkedPennerConeMetric, PennerConeMetric>(m, "MarkedPennerConeMetric")
        .def(pybind11::init<
            const Mesh<Scalar>&,
            const VectorX&,
            const std::vector<std::unique_ptr<DualLoop>>&,
            const std::vector<Scalar>&
        >())
        .def_readwrite("kappa_hat", &MarkedPennerConeMetric::kappa_hat)
        .def("change_metric", &MarkedPennerConeMetric::change_metric)
        .def("reset_marked_metric", &MarkedPennerConeMetric::reset_marked_metric)
        .def("n_homology_basis_loops", &MarkedPennerConeMetric::n_homology_basis_loops);

    m.def("generate_marked_metric", &generate_marked_metric, default_call_guard);
    m.def("generate_metric_from_field", &generate_metric_from_field, default_call_guard);
    m.def("regularize_metric", &regularize_metric, default_call_guard);
    m.def("optimize_metric_angles", &optimize_metric_angles, default_call_guard);

    m.def("parametrize_seamless", &parametrize_seamless, default_call_guard);
    m.def("parametrize_seamless_metric", &parametrize_seamless_metric, default_call_guard);
    m.def("generate_seamless_parametrization", &generate_seamless_parametrization, default_call_guard);

    m.def("view_cross_field", &view_cross_field, default_call_guard);
    m.def("view_seamless_parameterization",  &view_seamless_parameterization, default_call_guard);
}

void init_additional_holonomy_pybind(pybind11::module& m)
{
    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    pybind11::class_<HomotopyBasisGenerator> homotopy_basis_generator(m, "HomotopyBasisGenerator");
    pybind11::enum_<HomotopyBasisGenerator::Weighting>(homotopy_basis_generator, "Weighting")
        .value("minimal_homotopy", HomotopyBasisGenerator::Weighting::minimal_homotopy)
        .value("maximal_homotopy", HomotopyBasisGenerator::Weighting::maximal_homotopy)
        .value("dual_min_primal_max", HomotopyBasisGenerator::Weighting::dual_min_primal_max)
        .value("primal_min_dual_max", HomotopyBasisGenerator::Weighting::primal_min_dual_max)
        .export_values();

    pybind11::class_<Field::IntrinsicNRosyField>(m, "IntrinsicNRosyField")
        .def(pybind11::init<>())
        .def("initialize", &Field::IntrinsicNRosyField::initialize)
        .def("set_field", pybind11::overload_cast<
            const Mesh<Scalar>&,
            const std::vector<int>&,
            const Eigen::MatrixXi&,
            const std::vector<int>&,
            const Eigen::VectorXd&,
            const Eigen::MatrixXd&,
            const Eigen::MatrixXi&>(&Field::IntrinsicNRosyField::set_field))
        .def("set_field", pybind11::overload_cast<
            const Mesh<Scalar>&,
            const std::vector<int>&,
            const Eigen::MatrixXi&,
            const Eigen::VectorXd&,
            const Eigen::MatrixXd&,
            const Eigen::MatrixXi&>(&Field::IntrinsicNRosyField::set_field))
        //.def("set_field", &Field::IntrinsicNRosyField::set_field)
        .def("compute_principal_matchings", &Field::IntrinsicNRosyField::compute_principal_matchings)
        .def("fix_inconsistent_matchings", &Field::IntrinsicNRosyField::fix_inconsistent_matchings)
        .def("compute_rotation_form", &Field::IntrinsicNRosyField::compute_rotation_form);

    pybind11::class_<DualLoop>(m, "DualLoop");

    pybind11::class_<CoordinateEnergy, Optimization::EnergyFunctor>(m, "CoordinateEnergy")
        .def(pybind11::init<const DifferentiableConeMetric&, std::vector<int>>());
    pybind11::class_<IntegratedEnergy, Optimization::EnergyFunctor>(m, "IntegratedEnergy")
        .def(pybind11::init<const SimilarityPennerConeMetric&>());

    m.def("compute_mesh_quality", &compute_mesh_quality, default_call_guard);
    m.def("compute_min_angle", &compute_min_angle, default_call_guard);
    m.def("fix_cones", &Field::fix_cones, default_call_guard);
    m.def("add_random_cone_pair", &Field::add_random_cone_pair, default_call_guard);
    m.def("add_optimal_cone_pair", &add_optimal_cone_pair, default_call_guard);
    m.def(
        "find_boundary_vertices",
        pybind11::overload_cast<const Mesh<Scalar>&, const std::vector<int>&>(
            &find_boundary_vertices),
        default_call_guard);

    m.def("generate_mesh", &generate_mesh, default_call_guard);
    m.def("compute_metric_holonomy_matrix", &compute_metric_holonomy_matrix, default_call_guard);
    m.def("build_reduced_matrix_system", &build_reduced_matrix_system, default_call_guard);
    m.def("build_reduced_matrix_rhs", &build_reduced_matrix_rhs, default_call_guard);
    m.def("compute_triangle_corner_angle_jacobian", &compute_triangle_corner_angle_jacobian, default_call_guard);
    m.def("FE_to_double", &FE_to_double<Scalar>, default_call_guard);
    m.def("generate_marked_metric_from_mesh", &generate_marked_metric_from_mesh, default_call_guard);
    m.def("generate_refined_marked_metric", &generate_refined_marked_metric, default_call_guard);
    m.def("build_symmetric_matrix_system", &build_symmetric_matrix_system, default_call_guard);
    m.def("build_metric_matrix", &build_metric_matrix, default_call_guard);
    m.def("compute_metric_corner_angle_jacobian", &compute_metric_corner_angle_jacobian, default_call_guard);
    m.def("compute_metric_constraint_with_jacobian", &compute_metric_constraint_with_jacobian_pybind, default_call_guard);
    m.def("generate_similarity_metric", &generate_similarity_metric, default_call_guard);
    m.def(
        "compute_conformal_similarity_metric",
        &compute_conformal_similarity_metric,
        default_call_guard);

    m.def("optimize_subspace_metric_angles", &optimize_subspace_metric_angles, default_call_guard);
    m.def(
        "generate_intrinsic_rotation_form",
        pybind11::
            overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXi&, const Field::FieldParameters&>(
                &generate_intrinsic_rotation_form),
        default_call_guard);

    m.def("make_interior_free", &make_interior_free, default_call_guard);
    m.def(
        "generate_cones_from_rotation_form",
        pybind11::overload_cast<
            const Mesh<Scalar>&,
            const std::vector<int>&,
            const VectorX&,
            bool>(&Field::generate_cones_from_rotation_form), default_call_guard);

    m.def(
        "generate_VF_mesh_from_similarity_metric",
        &generate_VF_mesh_from_similarity_metric,
        default_call_guard);
    m.def(
        "generate_penner_coordinates",
        &generate_penner_coordinates,
        default_call_guard);

    m.def("compute_loop_holonomy_matrix", &compute_loop_holonomy_matrix, default_call_guard);
    m.def("write_obj", writeOBJ, default_call_guard);

    m.def("compute_field_direction", &Field::compute_field_direction, default_call_guard);

}

#endif
#endif

} // namespace Holonomy
} // namespace Penner