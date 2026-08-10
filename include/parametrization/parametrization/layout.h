// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "util/common.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

/**
 * @brief Methods to layout a uv parameterization determined by intrinsic lengths.
 * 
 */

namespace Penner {


/// Given a halfedge mesh, do a bfs on dual graph of mesh to produce a cut
///
/// Note that this only lays out the connected component containing the start halfedge.
///
/// @param m: mesh data structure
/// @param is_cut_h: (optional) pre-defined cuts to be included
/// @param start_h: (optional) the first halfedge to be laid out, can be used to control the axis-alignment for the whole patch
/// @return #h vector, mark whether the current halfedge is part of cut graph
std::vector<bool>
compute_layout_topology(const Mesh<Scalar>& m, const std::vector<bool>& is_cut_h, int start_h = -1);

/**
 * @brief Remove regular (that is, not cone) leaf vertices from a cut graph
 * 
 * @param m: mesh data structure
 * @param is_cone: per-vertex mask of cone vertices
 * @param is_cut: per-halfedge cut graph
 */
void trim_topology(const Mesh<Scalar>& m, const std::vector<bool>& is_cone, std::vector<bool>& is_cut);

/// Given a cut defined on the original or current mesh, pull it back to a cut defined on
/// an overlay mesh for the given mesh (possibly after flips)
///
/// @param[in] m_o: overlay mesh data structure
/// @param[in] is_cut_h: cuts on the original or current mesh
/// @param[in] is_original_cut: if true, use cut mask on the original mesh
/// @return cuts on the overlay mesh
template <typename Scalar>
std::vector<bool> pullback_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h,
    bool is_original_cut = true);

/**
 * @brief Core layout method for generating a parametrization from
 * a mesh with metric.
 * 
 * WARNING: this is a technical function with numerous exposed parameters.
 * In particular, it uses a metric representation augmented by conformal scale
 * factors, which is used for conformal methods and for optional numerical stability.
 * 
 * For a simpler parametrization method, use interface code in parametrize.h
 * 
 * @param _m: original mesh with initial metric and connectivity
 * @param mo: overly mesh with final metric (up to scale factors) and connectivity
 * @param vtx_reindex: map from halfedge to VF vertices
 * @param u: scale factors to apply to the final metric
 * @param V_overlay: interpolated vertices of the overlay mesh
 * @param is_cut_orig: cut on the original mesh to propagate to the final parametrization
 * @param is_cut: cut on the final mesh to use for the layout
 * @param use_uniform_bc: (optional) if true, use uniform barycentric coordinates for layout
 */
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
    layout_overlay_mesh(
        const Mesh<Scalar>& _m,
        OverlayMesh<OverlayScalar>& mo,
        const std::vector<int>& vtx_reindex,
        std::vector<Scalar>& u,
        std::vector<std::vector<OverlayScalar>>& V_overlay,
        const std::vector<bool>& is_cut_orig,
        const std::vector<bool>& is_cut,
        bool use_uniform_bc=false);

inline double signed_area(
    const Eigen::Vector2d& A,
    const Eigen::Vector2d& B,
    const Eigen::Vector2d& C)
{
    const Eigen::Vector2d AB = B - A;
    const Eigen::Vector2d AC = C - A;
    return AB.x() * AC.y() - AB.y() * AC.x();
}

/**
 * @brief Generate a mesh of the layout of the intrinsic metric.
 * 
 * @param m: mesh with intrinsic metric
 * @return mesh faces
 * @return planar layout vertices
 * @return layout faces
 */
template <typename Scalar>
std::tuple<
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
compute_layout_VF(const Mesh<Scalar>& m);

template<typename OverlayScalar>
std::tuple<Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi> build_layout_VF(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u_vec,
    const std::vector<Scalar>& v_vec);

template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> compute_layout_components(
    Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    std::vector<bool>& is_cut_h,
    int start_h = -1);
    
template <typename Scalar>
Eigen::Matrix<Scalar, 1, 2> perp_l(Eigen::Matrix<Scalar, 1, 2> a) {
    Eigen::Matrix<Scalar, 1, 2> b;
    b[0] = -a[1];
    b[1] = a[0];
    return b;
};

template <typename Scalar>
Scalar area_from_len_l(Scalar l1, Scalar l2, Scalar l3) {
    auto s = 0.5 * (l1 + l2 + l3);
    return sqrt(s * (s - l1) * (s - l2) * (s - l3));
}

template <typename Scalar>
Eigen::Matrix<Scalar, 1, 2> compute_layout_vertex(
    const Eigen::Matrix<Scalar, 1, 2>& p1,
    const Eigen::Matrix<Scalar, 1, 2>& p2,
    Scalar l0,
    Scalar l1,
    Scalar l2)
{
    return p1 + (p2 - p1) * (1 + square(l2 / l0) - square(l1 / l0)) / 2 +
                                    perp_l<Scalar>(p2 - p1) * 2 * area_from_len_l<Scalar>(1.0, l1 / l0, l2 / l0);
}

#ifdef PYBIND
#endif


} // namespace Penner
