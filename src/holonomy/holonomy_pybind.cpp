#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "holonomy/core/common.h"
#include "holonomy/core/quality.h"
#include "holonomy/holonomy/cones.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"
#include "holonomy/holonomy/rotation_form.h"
#include "holonomy/holonomy_interface.h"
#include "holonomy/similarity/conformal.h"
#include "holonomy/similarity/energy.h"
#include "holonomy/similarity/layout.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"
#include "util/boundary.h"

namespace Penner {
namespace Holonomy {

#ifdef PYBIND
#ifndef MULTIPRECISION

// wrap as Python module
PYBIND11_MODULE(holonomy_py, m)
{
    m.doc() = "pybind for optimization module";
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

    pybind11::class_<NewtonParameters, std::shared_ptr<NewtonParameters>>(m, "NewtonParameters")
        .def(pybind11::init<>())
        .def_readwrite("output_dir", &NewtonParameters::output_dir)
        .def_readwrite("error_log", &NewtonParameters::error_log)
        .def_readwrite("reset_lambda", &NewtonParameters::reset_lambda)
        .def_readwrite("do_reduction", &NewtonParameters::do_reduction)
        .def_readwrite("lambda0", &NewtonParameters::lambda0)
        .def_readwrite("error_eps", &NewtonParameters::error_eps)
        .def_readwrite("max_itr", &NewtonParameters::max_itr)
        .def_readwrite("max_time", &NewtonParameters::max_time)
        .def_readwrite("min_lambda", &NewtonParameters::min_lambda)
        .def_readwrite("solver", &NewtonParameters::solver)
        .def_readwrite("log_level", &NewtonParameters::log_level);

    pybind11::class_<MarkedMetricParameters, std::shared_ptr<MarkedMetricParameters>>(
        m,
        "MarkedMetricParameters")
        .def(pybind11::init<>())
        .def_readwrite("use_initial_zero", &MarkedMetricParameters::use_initial_zero)
        .def_readwrite("remove_loop_constraints", &MarkedMetricParameters::remove_loop_constraints)
        .def_readwrite("free_interior", &MarkedMetricParameters::free_interior)
        .def_readwrite("weighting", &MarkedMetricParameters::weighting);

    pybind11::class_<FieldParameters, std::shared_ptr<FieldParameters>>(m, "FieldParameters")
        .def(pybind11::init<>())
        .def_readwrite("min_angle", &FieldParameters::min_angle);

    pybind11::class_<MarkedPennerConeMetric, DifferentiableConeMetric>(m, "MarkedPennerConeMetric")
        .def_readwrite("kappa_hat", &MarkedPennerConeMetric::kappa_hat)
        .def("flip_ccw", &MarkedPennerConeMetric::flip_ccw)
        .def("undo_flips", &MarkedPennerConeMetric::undo_flips)
        .def("clone_cone_metric", &MarkedPennerConeMetric::clone_cone_metric)
        .def("make_discrete_metric", &MarkedPennerConeMetric::make_discrete_metric)
        .def("get_flip_sequence", &MarkedPennerConeMetric::get_flip_sequence)
        .def("max_constraint_error", &MarkedPennerConeMetric::max_constraint_error)
        .def("n_vertices", &MarkedPennerConeMetric::n_vertices)
        .def("n_edges", &MarkedPennerConeMetric::n_edges)
        .def("n_faces", &MarkedPennerConeMetric::n_faces)
        .def("n_homology_basis_loops", &MarkedPennerConeMetric::n_homology_basis_loops);

    pybind11::class_<DualLoop>(m, "DualLoop");

    pybind11::class_<CoordinateEnergy, Optimization::EnergyFunctor>(m, "CoordinateEnergy")
        .def(pybind11::init<const DifferentiableConeMetric&, std::vector<int>>());
    pybind11::class_<IntegratedEnergy, Optimization::EnergyFunctor>(m, "IntegratedEnergy")
        .def(pybind11::init<const SimilarityPennerConeMetric&>());

    m.def("compute_mesh_quality", &compute_mesh_quality, default_call_guard);
    m.def("compute_min_angle", &compute_min_angle, default_call_guard);
    m.def("fix_cones", &fix_cones, default_call_guard);
    m.def("add_optimal_cone_pair", &add_optimal_cone_pair, default_call_guard);
    m.def(
        "find_boundary_vertices",
        pybind11::overload_cast<const Mesh<Scalar>&, const std::vector<int>&>(
            &find_boundary_vertices),
        default_call_guard);

    m.def("generate_mesh", &generate_mesh, default_call_guard);
    m.def("generate_marked_metric", &generate_marked_metric, default_call_guard);
    m.def("generate_refined_marked_metric", &generate_refined_marked_metric, default_call_guard);
    m.def("generate_similarity_metric", &generate_similarity_metric, default_call_guard);
    m.def(
        "compute_conformal_similarity_metric",
        &compute_conformal_similarity_metric,
        default_call_guard);

    m.def("optimize_subspace_metric_angles", &optimize_subspace_metric_angles, default_call_guard);
    m.def("optimize_metric_angles", &optimize_metric_angles, default_call_guard);
    m.def(
        "generate_intrinsic_rotation_form",
        pybind11::
            overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXi&, const FieldParameters&>(
                &generate_intrinsic_rotation_form),
        default_call_guard);

    m.def("make_interior_free", &make_interior_free, default_call_guard);

    m.def(
        "generate_VF_mesh_from_similarity_metric",
        &generate_VF_mesh_from_similarity_metric,
        default_call_guard);
}

#endif
#endif

} // namespace Holonomy
} // namespace Penner
