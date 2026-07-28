// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "util/common.h"

/**
 * @brief Methods to layout a uv parameterization determined by intrinsic lengths.
 * 
 */

namespace Penner {


/// Given a VF mesh, check that the signed face areas are nonnegative
///
/// @param[in] V: mesh vertices in 2D
/// @param[in] F: mesh faces
/// @return true iff the face areas are all nonnegative
bool check_areas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

/// Given a VF mesh with uv coordinates, get the maximum error of the uv lengths
/// across cuts
///
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv coordinates in 2D
/// @param[in] F_uv: mesh uv faces
/// @return maximum uv length error across cuts
double compute_uv_length_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/// Given a VF mesh with uv coordinates, check that it satisfies fundamental uv
/// consistency constraints:
///     - uv lengths match up across cuts
///     - uv face areas are nonnegative
///
/// @param[in] V: mesh vertices in 3D
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv coordinates in 2D
/// @param[in] F_uv: mesh uv faces
/// @return true iff the mesh passes all of the tests
bool check_uv(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/**
 * @brief Compute the length of an edge between two endpoints
 * 
 * @param uv_0: first endpoint
 * @param uv_1: second endpoint
 * @return length of edge
 */
double uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1);

template <typename Mesh>
int prev_halfedge(
    const Mesh& m,
    int hij)
{
    int hli = hij;
    while (m.n[hli] != hij)
    {
        hli = m.n[hli];
    }

    return hli;
}

// compute the area of the triangle with vertices ijk
template <typename Scalar>
Scalar triangle_area(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v,
    int hij)
{
    int hjk = m.n[hij];
    int hli = prev_halfedge(m, hij);
    Eigen::Vector2d A = {u[hli], v[hli]};
    Eigen::Vector2d B = {u[hij], v[hij]};
    Eigen::Vector2d C = {u[hjk], v[hjk]};
    return signed_area(A, B, C);
}

// check that the signed area of the layout triangles are all positive
template <typename Scalar>
bool check_areas(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v)
{
    int num_halfedges = m.n_halfedges();
    Scalar min_area = triangle_area(m, u, v, 0);
    Scalar max_area = min_area;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        Scalar area_hijk = triangle_area(m, u, v, hij);
        min_area = min(area_hijk, min_area);
        max_area = max(area_hijk, max_area);
    }
    spdlog::debug("minimum overlay triangle face area: {}", min_area);
    spdlog::debug("maximum overlay triangle face area: {}", max_area);

    return (min_area >= 0);
}

// check that the difference of lengths of opposite halfedges
template <typename Scalar>
Scalar compute_uv_length_error(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v)
{
    int num_halfedges = m.n_halfedges();
    Scalar max_uv_length_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = m.opp[hij];

        // get other halfedges in the face for vertex computation
        int hki = prev_halfedge(m, hij);
        int hlj = prev_halfedge(m, hji);

        // get uv vertices on the edge
        Eigen::Vector2d uv0i = {u[hki], v[hki]};
        Eigen::Vector2d uv0j = {u[hij], v[hij]};
        Eigen::Vector2d uv1i = {u[hji], v[hji]};
        Eigen::Vector2d uv1j = {u[hlj], v[hlj]};

        // compute the length of each halfedge
        Scalar l0 = uv_length(uv0i, uv0j);
        Scalar l1 = uv_length(uv1i, uv1j);

        // determine if the max length inconsistency has increased
        if (abs(l0 - l1) > 1e-8)
        {
            spdlog::warn("uv length consistency error for edge {}, {} is {} - {}", hij, hji, l0, l1);
        }
        max_uv_length_error = max(max_uv_length_error, abs(l0 - l1));
    }

    // return the max uv length error
    return max_uv_length_error;
}



// check that two different layouts of a mesh have consistent lengths
template <typename Scalar>
bool check_uv_consistency(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u0,
    const std::vector<Scalar>& v0,
    const std::vector<Scalar>& u1,
    const std::vector<Scalar>& v1)
{
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> u_error(num_halfedges);
    std::vector<Scalar> v_error(num_halfedges);
    Scalar max_consistency_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        // get next halfedge
        int hjk = m.n[hij];

        // get uv vertices on the edge
        Eigen::Vector2d uv0j = {u0[hij], v0[hij]};
        Eigen::Vector2d uv0k = {u0[hjk], v0[hjk]};
        Eigen::Vector2d uv1j = {u1[hij], v1[hij]};
        Eigen::Vector2d uv1k = {u1[hjk], v1[hjk]};

        // compute the length of the halfedge in each metric
        Scalar l0 = uv_length(uv0j, uv0k);
        Scalar l1 = uv_length(uv1j, uv1k);

        if (abs(l0 - l1) > 1e-8)
        {
            spdlog::warn("uv length consistency error for {} with previous halfedge {} is {} - {}", hjk, hij, l0, l1);
            spdlog::warn("local face is {}, {}, {}, {}, ...", hij, hjk, m.n[hjk], m.n[m.n[hjk]]);
        }
        max_consistency_error = max(max_consistency_error, abs(l0 - l1));
    }
    spdlog::debug("max consistency error is {}", max_consistency_error);

    return (max_consistency_error < 1e-8);
}

// check the edge consistency and signed area of a mesh layout
template <typename Scalar>
bool check_uv(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v,
    const std::vector<bool>& is_cut)
{
    int num_halfedges = m.n_halfedges();
    bool is_valid = true;

    // Check faces agree in number
    if ((u.size() != num_halfedges) || (v.size() != num_halfedges)) {
        spdlog::error("uv coordinates not in correspondence with mesh halfedges");
        is_valid = false;
    }

    // Check length consistency
    Scalar uv_length_error = compute_uv_length_error(m, u, v);
    if (!float_equal(uv_length_error, 0.0, 1e-6)) {
        spdlog::warn("Inconsistent uv length error {} across edges", uv_length_error);
    }

    // Check uv face areas
    if (!check_areas(m, u, v)) {
        spdlog::error("Triangle is flipped in overlay");
        is_valid = false;
    }

    // Return true if no issues found
    return is_valid;
}


} // namespace Penner