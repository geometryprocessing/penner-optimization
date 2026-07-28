// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

/**
 * @brief Methods to build a parameterization for a halfedge mesh with the original embedding
 * geometry and a target intrinsic metric
 * 
 */

#pragma once
#include "metric/cone_metric.h"

namespace Penner {

/// Given a mesh with initial target and final optimized metric coordinates, generate a corresponding
/// overlay VF mesh with parametrization.
///
/// @param[in] V: initial mesh vertices
/// @param[in] F: initial mesh faces
/// @param[in] initial_cone_metric: initial metric
/// @param[in] reduced_metric_coords: optimized mesh metric coordinates
/// @return parametrized VF mesh with cut and topology mapping information
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
    const VectorX& reduced_metric_coords);

/// Given a mesh with initial target and final optimized metric coordinates, generate a corresponding
/// overlay VF mesh with parametrization.
///
/// @param[in] V: initial mesh vertices
/// @param[in] F: initial mesh faces
/// @param[in] Th_hat: initial target angles
/// @param[in] initial_cone_metric: initial metric
/// @param[in] reduced_metric_coords: optimized mesh metric coordinates
/// @param[in] is_cut: (optional) cut on the mesh to use for layout
/// @param[in] do_best_fit_scaling: (optional) if true, use conformal scaling for numerical stability
/// @return parametrized VF mesh with cut and topology mapping information
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
        std::vector<bool> is_cut = {},
        bool do_best_fit_scaling = false);

/// Given a mesh with final log edge length coordinates, generate a corresponding
/// overlay VF mesh with parametrization.
///
/// @param[in] V: initial mesh vertices
/// @param[in] F: initial mesh faces
/// @param[in] Th_hat: initial target angles
/// @param[in] is_cut: (optional) cut on the mesh to use for layout
/// @param[in] reduced_log_edge_lengths: optimized mesh metric coordinates
/// @return parametrized VF mesh with cut and topology mapping information
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
        std::vector<bool> cut_h = {});

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
        bool do_best_fit_scaling=false,
        bool use_uniform_bc=false);


} // namespace Penner