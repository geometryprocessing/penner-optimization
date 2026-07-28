// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "parametrization/parametrize.h"

#include "metric/projection.h"
#include "parametrization/interpolation.h"
#include "parametrization/layout.h"
#include "util/vector.h"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "parametrization/refinement.h"
#include "parametrization/error.h"

/// FIXME Do cleaning pass


namespace Penner {


template <typename OverlayScalar>
std::
    tuple<
        OverlayMesh<OverlayScalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_halfedge_metric(
        const Eigen::MatrixXd& V,
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords,
        std::vector<bool> cut_h,
        bool do_best_fit_scaling,
        bool use_uniform_bc)
{
    // Get metric target coordinates
    auto cone_metric = initial_cone_metric.set_metric_coordinates(reduced_metric_coords);
    VectorX metric_target = initial_cone_metric.get_metric_coordinates();
    VectorX metric_coords = cone_metric->get_metric_coordinates();

    // Optionally fit conformal scale factors for numerical stability
    VectorX metric_coords_scaled = metric_coords;
    VectorX scale_factors;
    scale_factors.setZero(initial_cone_metric.n_ind_vertices());
    if (do_best_fit_scaling) {
        scale_factors = best_fit_conformal(initial_cone_metric, metric_coords);
        MatrixX B = conformal_scaling_matrix(initial_cone_metric);
        metric_coords_scaled = metric_coords - B * scale_factors;
    }
    VectorX metric_diff = metric_coords_scaled - metric_target;

    // Compute interpolation overlay mesh
    Eigen::MatrixXd V_overlay;
    InterpolationMesh<OverlayScalar> interpolation_mesh, reverse_interpolation_mesh;
    spdlog::trace("Interpolating penner coordinates for mesh with {} halfedges", m.n_halfedges());
    interpolate_penner_coordinates(
        m,
        metric_coords_scaled,
        scale_factors,
        interpolation_mesh,
        reverse_interpolation_mesh);
    spdlog::trace("Interpolating vertex positions");
    interpolate_vertex_positions(
        V,
        vtx_reindex,
        interpolation_mesh,
        reverse_interpolation_mesh,
        V_overlay);
    OverlayMesh<OverlayScalar> m_o = interpolation_mesh.get_overlay_mesh();
    make_tufted_overlay(m_o);

    // Get endpoints
    std::vector<std::pair<int, int>> endpoints;
    find_origin_endpoints(m_o, endpoints);

    // Convert overlay mesh to transposed vector format
    std::vector<std::vector<OverlayScalar>> V_overlay_vec(3);
    for (int i = 0; i < 3; ++i) {
        V_overlay_vec[i].resize(V_overlay.rows());
        for (int j = 0; j < V_overlay.rows(); ++j) {
            V_overlay_vec[i][j] = OverlayScalar(V_overlay(j, i));
        }
    }

    // Get layout topology from original mesh
    std::vector<bool> is_cut = compute_layout_topology(m, cut_h);

    // Convert overlay mesh to VL format
    spdlog::trace("Getting layout");
    std::vector<int> vtx_reindex_mutable = vtx_reindex;
    std::vector<Scalar> u; // (m_o._m.Th_hat.size(), 0.0);
    convert_eigen_to_std_vector(scale_factors, u);
    // vtx_reindex_mutable, endpoints, -1); FIXME
    return layout_overlay_mesh<OverlayScalar>(
        m,
        m_o,
        vtx_reindex,
        u,
        V_overlay_vec,
        is_cut,
        {},
        use_uniform_bc);
}

std::
    tuple<
        OverlayMesh<Scalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_metric(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<Scalar>& Th_hat,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords,
        std::vector<bool> cut_h,
        bool do_best_fit_scaling)
{
    // Get mesh with vertex reindexing
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m =
        FV_to_double(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    return generate_VF_mesh_from_halfedge_metric<Scalar>(V, m, vtx_reindex, initial_cone_metric, reduced_metric_coords, cut_h, do_best_fit_scaling);
}

    std::tuple<
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<int>, // Fn_to_F_o
        std::vector<std::pair<int, int>>> // endpoints_o
    parametrize_metric(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords)
{
    // Get mesh with vertex reindexing
    std::vector<Scalar> Th_hat(V.rows(), 0); // trivial cones
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m =
        FV_to_double(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);
    m.Th_hat = initial_cone_metric.Th_hat; // copy over angle constraints

    std::vector<bool> cut_h = {};
    auto vf_res = generate_VF_mesh_from_halfedge_metric<Scalar>(V, m, vtx_reindex, initial_cone_metric, reduced_metric_coords, cut_h);

    // simplify the refined parametrization
    Eigen::MatrixXd V_o = std::get<1>(vf_res);
    Eigen::MatrixXi F_o = std::get<2>(vf_res);
    Eigen::MatrixXd uv_o = std::get<3>(vf_res);
    Eigen::MatrixXi FT_o = std::get<4>(vf_res);
    std::vector<int> fn_to_f_o = std::get<7>(vf_res);
    std::vector<std::pair<int, int>> endpoints_o = std::get<8>(vf_res);
    RefinementMesh refinement_mesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o);
    refinement_mesh.refine_mesh();
    refinement_mesh.simplify_mesh();
    return refinement_mesh.get_VF_mesh();
}

std::
    tuple<
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool> // is_cut_h
        >
    generate_VF_mesh_from_discrete_metric(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<Scalar>& Th_hat,
        const VectorX& reduced_log_edge_lengths,
        std::vector<bool> cut_h)
{
    // Get mesh with vertex reindexing
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m =
        FV_to_double(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    // Get layout topology from mesh
    std::vector<bool> is_cut = compute_layout_topology(m, cut_h);

    // Set metric for layout
    DiscreteMetric discrete_metric(m, reduced_log_edge_lengths);

    // Create trivial overlay mesh
    OverlayMesh<Scalar> m_o(discrete_metric);
    make_tufted_overlay(m_o);

    // Convert vertices to transposed vector format
    std::vector<std::vector<Scalar>> V_overlay_vec(3);
    for (int i = 0; i < 3; ++i) {
        V_overlay_vec[i].resize(V.rows());
        for (int j = 0; j < V.rows(); ++j) {
            V_overlay_vec[i][j] = V(vtx_reindex[j], i);
        }
    }

    // Compute layout
    std::vector<Scalar> u_vec(m.n_ind_vertices(), 0.0);
    //std::vector<int> vtx_reindex_mutable = vtx_reindex;
    auto layout_res = layout_overlay_mesh(
        m,
        m_o,
        vtx_reindex,
        u_vec,
        V_overlay_vec,
        is_cut,
        {});

		Eigen::MatrixXd V_l = std::get<1>(layout_res);
		Eigen::MatrixXi F_l = std::get<2>(layout_res);
		Eigen::MatrixXd uv = std::get<3>(layout_res);
		Eigen::MatrixXi FT = std::get<4>(layout_res);

		return std::make_tuple(V_l, F_l, uv, FT, cut_h);
}

template 
std::
    tuple<
        OverlayMesh<Scalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_halfedge_metric<Scalar>(
        const Eigen::MatrixXd& V,
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords,
        std::vector<bool> cut_h,
        bool do_best_fit_scaling,
        bool use_uniform_bc);

#ifdef WITH_MPFR
#ifndef MULTIPRECISION

template
std::
    tuple<
        OverlayMesh<mpfr::mpreal>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_halfedge_metric<mpfr::mpreal>(
        const Eigen::MatrixXd& V,
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords,
        std::vector<bool> cut_h,
        bool do_best_fit_scaling,
        bool use_uniform_bc);
#endif
#endif
  

} // namespace Penner 