#include "feature/pybind.h"

#ifdef PYBIND
#ifndef MULTIPRECISION

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#ifdef WITH_MPFR
#include <unsupported/Eigen/MPRealSupport>
#endif

#include "holonomy/pybind.h"
#include "optimization/pybind.h"

#include "holonomy/core/viewer.h"

#include "feature/core/common.h"
#include "feature/core/component_mesh.h"
#include "feature/core/vf_corners.h"
#include "feature/dirichlet/constraint.h"
#include "feature/dirichlet/optimization.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "feature/dirichlet/angle_constraint_relaxer.h"
#include "feature/feature/error.h"
#include "feature/feature/features.h"
#include "feature/core/viewer.h"
#include "feature/interface.h"
#include "feature/surgery/refinement.h"
#include "feature/surgery/stitching.h"
#include "feature/core/io.h"
#include "feature/surgery/cut_metric_generator.h"
#include "feature/surgery/cut_mesh_layout.h"

namespace Penner {
namespace Feature {


void init_feature_pybind(pybind11::module& m)
{

    m.doc() = "pybind for optimization module";

    // init_optimization_pybind(m);
    // init_holonomy_pybind(m);

    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    pybind11::class_<BoundaryPath>(m, "BoundaryPath")
        .def(pybind11::init<const Mesh<Scalar>&, int>());

    pybind11::class_<FaceEdge>(m, "FaceEdge");

    pybind11::class_<IntrinsicRefinementMesh>(m, "IntrinsicRefinementMesh")
        .def(pybind11::init<const MarkedPennerConeMetric&>())
        .def("refine_face", &IntrinsicRefinementMesh::refine_face)
        .def("refine_spanning_faces", &IntrinsicRefinementMesh::refine_spanning_faces)
        .def("generate_mesh", &IntrinsicRefinementMesh::generate_mesh);

    pybind11::class_<DirichletPennerConeMetric, MarkedPennerConeMetric>(
        m,
        "DirichletPennerConeMetric")
        .def(pybind11::init<
             const MarkedPennerConeMetric&,
             const std::vector<BoundaryPath>&,
             const MatrixX&,
             const VectorX&>())
        .def(pybind11::init<const DirichletPennerConeMetric&>())
        .def("assign", &DirichletPennerConeMetric::operator=)
        .def_readwrite("ell_hat", &DirichletPennerConeMetric::ell_hat)
        .def("n_reduced_coordinates", &DirichletPennerConeMetric::n_reduced_coordinates)
        .def(
            "get_boundary_constraint_system",
            &DirichletPennerConeMetric::get_boundary_constraint_system)
        .def("set_angle_constraint_system", &DirichletPennerConeMetric::set_angle_constraint_system)
        .def("get_path_starting_vertices", &DirichletPennerConeMetric::get_path_starting_vertices);

    //pybind11::class_<BoundaryConstraintGenerator>(m, "BoundaryConstraintGenerator")
    //    .def(pybind11::init<const Mesh<Scalar>&>())
    //    .def("mark_cones_as_junctions", &BoundaryConstraintGenerator::mark_cones_as_junctions)
    //    .def(
    //        "set_uniform_feature_lengths",
    //        &BoundaryConstraintGenerator::set_uniform_feature_lengths)
    //    .def(
    //        "build_boundary_constraint_system",
    //        &BoundaryConstraintGenerator::build_boundary_constraint_system);


    pybind11::class_<FeatureFinder>(m, "FeatureFinder")
        .def(pybind11::init<const Eigen::MatrixXd&, const Eigen::MatrixXi>())
        .def("mark_dihedral_angle_features", &FeatureFinder::mark_dihedral_angle_features)
        .def("mark_features", &FeatureFinder::mark_features)
        .def("prune_cycles", &FeatureFinder::prune_cycles)
        .def("prune_greedy", &FeatureFinder::prune_greedy)
        .def("prune_refined_greedy", &FeatureFinder::prune_refined_greedy)
        .def("prune_small_features", &FeatureFinder::prune_small_features)
        .def("prune_small_components", &FeatureFinder::prune_small_components)
        .def("view_features", &FeatureFinder::view_features)
        .def("generate_feature_cut_mesh", &FeatureFinder::generate_feature_cut_mesh);

    pybind11::class_<AngleConstraintMatrixRelaxer>(m, "AngleConstraintMatrixRelaxer")
        .def(pybind11::init<>())
        .def("run", &AngleConstraintMatrixRelaxer::run);

    pybind11::class_<CutMetricGenerator>(m, "CutMetricGenerator")
        .def(pybind11::init<const Eigen::MatrixXd&, const Eigen::MatrixXi&, MarkedMetricParameters, std::vector<std::pair<int, int>>>())
        .def("get_aligned_metric", &CutMetricGenerator::get_aligned_metric)
        .def("get_fixed_aligned_metric", &CutMetricGenerator::get_fixed_aligned_metric)
        .def("get_field", &CutMetricGenerator::get_field)
        .def("set_fields", &CutMetricGenerator::set_fields)
        .def("generate_fields", &CutMetricGenerator::generate_fields);

    pybind11::class_<AlignedMetricGenerator>(m, "AlignedMetricGenerator")
        .def(pybind11::init<
            const Eigen::MatrixXd&,
            const Eigen::MatrixXi&,
            const std::vector<VertexEdge>&,
            const std::vector<VertexEdge>&,
            const Eigen::MatrixXd&,
            const Eigen::VectorXd&,
            const Eigen::MatrixXd&,
            const Eigen::MatrixXi&,
            Scalar,
            bool>())
        .def("optimize_full", &AlignedMetricGenerator::optimize_full)
        .def("optimize_relaxed", &AlignedMetricGenerator::optimize_relaxed)
        .def("compute_error", &AlignedMetricGenerator::compute_error)
        .def("get_metric", &AlignedMetricGenerator::get_metric)
        .def("parameterize", &AlignedMetricGenerator::parameterize)
        .def("get_parameterization", &AlignedMetricGenerator::get_parameterization)
        .def("get_refined_field", &AlignedMetricGenerator::get_refined_field)
        .def("get_refined_features", &AlignedMetricGenerator::get_refined_features);

    m.def("build_boundary_paths", &build_boundary_paths, default_call_guard);
    m.def("generate_refined_feature_mesh", &generate_refined_feature_mesh, default_call_guard);
    m.def("refine_corner_feature_faces", &refine_corner_feature_faces, default_call_guard);
    m.def("find_mesh_vertex_components", &find_mesh_vertex_components, default_call_guard);
    m.def("compute_component_error", &compute_component_error, default_call_guard);

    m.def("generate_overlay_cut_mask", &generate_overlay_cut_mask, default_call_guard);
    m.def("compute_mask_corners", &compute_mask_corners, default_call_guard);
    m.def("compute_feature_alignment", &compute_feature_alignment, default_call_guard);
    m.def("compute_uv_alignment", &compute_uv_alignment, default_call_guard);
    m.def("parameterize_cut_mesh", &parameterize_cut_mesh<Scalar>, default_call_guard);

#ifdef WITH_MPFR
    m.def("parameterize_cut_mesh_mpfr", &parameterize_cut_mesh<mpfr::mpreal>, default_call_guard);
#endif

    m.def("mask_difference", &mask_difference, default_call_guard);
    m.def("compute_relaxed_edges", &compute_relaxed_edges, default_call_guard);
    m.def("compute_corner_edges", &compute_corner_edges, default_call_guard);
    m.def("compute_mask_uv_alignment", &compute_mask_uv_alignment, default_call_guard);

    m.def("compute_face_edge_endpoints", &compute_face_edge_endpoints, default_call_guard);
    m.def("prune_misaligned_face_edges", &prune_misaligned_face_edges, default_call_guard);
    m.def("compute_face_edges_from_corners", &compute_face_edges_from_corners, default_call_guard);

    m.def("optimize_relaxed_angles", &optimize_relaxed_angles, default_call_guard);
    m.def("is_manifold", &is_manifold, default_call_guard);
    m.def("align_to_hard_features", &align_to_hard_features, default_call_guard);
    m.def("stitch_cut_overlay", &stitch_cut_overlay, default_call_guard);
    m.def("compute_edge_alignment", &compute_edge_alignment, default_call_guard);
    m.def("prune_misaligned_corners", &prune_misaligned_corners, default_call_guard);
    m.def("prune_misaligned_edges", &prune_misaligned_edges, default_call_guard);
    m.def("prune_redundant_edge_corners", &prune_redundant_edge_corners, default_call_guard);
    m.def("compute_seamless_error", &Holonomy::compute_seamless_error, default_call_guard);
    m.def("compute_angle_error", &Holonomy::compute_angle_error, default_call_guard);
    m.def("compute_field_direction", &compute_field_direction, default_call_guard);
    m.def("reduce_relaxed_edges", &reduce_relaxed_edges, default_call_guard);
    m.def("transfer_corner_function_to_halfedge", &transfer_corner_function_to_halfedge, default_call_guard);
    m.def("transfer_halfedge_function_to_corner", &transfer_halfedge_function_to_corner, default_call_guard);
    m.def("generate_glued_cone_vertices", &generate_glued_cone_vertices, default_call_guard);

    m.def("check_flip", &check_flip, default_call_guard);
    m.def("compute_height", &compute_height, default_call_guard);
    m.def("find_seams", &find_seams, default_call_guard);
    m.def("write_seams", &write_seams, default_call_guard);
    m.def("write_boundary", &write_boundary, default_call_guard);
    m.def("write_features", &write_features, default_call_guard);
    m.def("generate_connected_parameterization", &generate_connected_parameterization<double>, default_call_guard);
#ifdef WITH_MPFR
    m.def("generate_connected_parameterization_mpfr", &generate_connected_parameterization<mpfr::mpreal>, default_call_guard);
#endif
    m.def("load_feature_edges", &load_feature_edges, default_call_guard);
    m.def("load_mesh_edges", &load_mesh_edges, default_call_guard);
}


}
}

#endif
#endif