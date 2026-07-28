// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "metric/cone_metric.h"
#include "metric/interface.h"
#include "metric/viewer.h"
#include "parametrization/parametrize.h"

#include "optimization/pybind.h"
#include "holonomy/pybind.h"
#include "holonomy/core/viewer.h"
#include "feature/pybind.h"

#include "field/frame_field.h"
#include "field/cross_field.h"

using namespace Penner;
using namespace Optimization;
using namespace Holonomy;
using namespace Feature;
using namespace Field;

#ifdef PYBIND
#ifndef MULTIPRECISION

void init_metric_pybind(pybind11::module& m)
{
    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    pybind11::class_<ProjectionParameters, std::shared_ptr<ProjectionParameters>>(
        m,
        "ProjectionParameters")
        .def(pybind11::init<>())
        .def_readwrite("max_itr", &ProjectionParameters::max_itr)
        .def_readwrite("bound_norm_thres", &ProjectionParameters::bound_norm_thres)
        .def_readwrite("error_eps", &ProjectionParameters::error_eps)
        .def_readwrite("do_reduction", &ProjectionParameters::do_reduction)
        .def_readwrite("initial_ptolemy", &ProjectionParameters::initial_ptolemy)
        .def_readwrite("use_edge_flips", &ProjectionParameters::use_edge_flips)
        .def_readwrite("output_dir", &ProjectionParameters::output_dir);

    pybind11::class_<ConeMetricParameters>(
        m,
        "ConeMetricParameters")
        .def(pybind11::init<>())
        .def_readwrite("use_log_length", &ConeMetricParameters::use_log_length)
        .def_readwrite("use_initial_zero", &ConeMetricParameters::use_initial_zero)
        .def_readwrite("free_interior", &ConeMetricParameters::free_interior);

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
        .def_readwrite("fixed_dof", &OverlayProblem::Mesh<Scalar>::fixed_dof)
        .def("n_vertices", &OverlayProblem::Mesh<Scalar>::n_vertices)
        .def("n_edges", &OverlayProblem::Mesh<Scalar>::n_edges)
        .def("n_faces", &OverlayProblem::Mesh<Scalar>::n_faces);

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
        .def_readwrite("first_segment", &OverlayProblem::OverlayMesh<Scalar>::first_segment)
        .def_readwrite("origin", &OverlayProblem::OverlayMesh<Scalar>::origin)
        .def_readwrite("origin_of_origin", &OverlayProblem::OverlayMesh<Scalar>::origin_of_origin)
        .def_readwrite("vertex_type", &OverlayProblem::OverlayMesh<Scalar>::vertex_type)
        .def_readwrite("edge_type", &OverlayProblem::OverlayMesh<Scalar>::edge_type)
        .def_readwrite("seg_bcs", &OverlayProblem::OverlayMesh<Scalar>::seg_bcs)
        .def_readwrite("_m", &OverlayProblem::OverlayMesh<Scalar>::_m);

    pybind11::
        class_<DifferentiableConeMetric, std::unique_ptr<DifferentiableConeMetric>, Mesh<Scalar>>(
            m,
            "DifferentiableConeMetric")
            .def("get_metric_coordinates", &DifferentiableConeMetric::get_metric_coordinates)
            //.def(
            //    "get_reduced_metric_coordinates",
            //    &DifferentiableConeMetric::get_reduced_metric_coordinates)
            .def(
                "get_corner_angles",
                static_cast<
                std::tuple<
                    VectorX,
                    VectorX
                > (DifferentiableConeMetric::*)() const>(&DifferentiableConeMetric::get_corner_angles))
            .def("max_constraint_error", &DifferentiableConeMetric::max_constraint_error)
            .def("constraint", 
                    static_cast<VectorX (DifferentiableConeMetric::*)() const>(&DifferentiableConeMetric::constraint));

    pybind11::class_<DiscreteMetric, DifferentiableConeMetric>(m, "DiscreteMetric")
        .def(pybind11::init<const Mesh<Scalar>&, const VectorX&>());

    pybind11::class_<PennerConeMetric, DifferentiableConeMetric>(m, "PennerConeMetric")
        .def(pybind11::init<
            const Mesh<Scalar>&,
            const VectorX&
        >())
        .def("flip_ccw", &PennerConeMetric::flip_ccw)
        .def("clone_cone_metric", &PennerConeMetric::clone_cone_metric)
        .def("set_metric_coordinates", &PennerConeMetric::set_metric_coordinates)
        .def("project_to_constraint", &PennerConeMetric::project_to_constraint)
        .def("make_discrete_metric", &PennerConeMetric::make_discrete_metric)
        .def("undo_flips", &PennerConeMetric::undo_flips)
        .def("change_metric", &PennerConeMetric::change_metric);

    m.def("generate_cone_metric", &generate_cone_metric, default_call_guard);
    m.def("parametrize_metric", &parametrize_metric, default_call_guard);
    m.def("view_parameterization", &view_parameterization, default_call_guard);
}

void init_field_pybind(pybind11::module& m)
{
    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    m.def("view_vector_field", &view_vector_field, default_call_guard);
    m.def("load_frame_field", &load_frame_field, default_call_guard);
    m.def("write_frame_field", &write_frame_field, default_call_guard);
    m.def("generate_frame_field", &generate_frame_field, default_call_guard);
    m.def("compute_frame_field_cones", &compute_frame_field_cones, default_call_guard);
    m.def("refine_frame_field", &refine_frame_field, default_call_guard);
}

// wrap as Python module
PYBIND11_MODULE(penner, m)
{
    m.doc() = "pybindings for penner methods";
    init_metric_pybind(m);
    init_field_pybind(m);
    init_optimization_pybind(m);
    init_holonomy_pybind(m);
    init_feature_pybind(m);
}

#endif
#endif