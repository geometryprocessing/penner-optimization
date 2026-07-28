// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "parametrization/layout.h"

#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/edge_flaps.h>


namespace Penner {


bool check_areas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    Eigen::VectorXd areas;
    igl::doublearea(V, F, areas);
    double min_area = areas.minCoeff() / 2.0;
    double max_area = areas.maxCoeff() / 2.0;
    spdlog::debug("Minimum VF face area: {}", min_area);
    spdlog::debug("Maximum VF face area: {}", max_area);

    return (min_area >= 0);
}

// compute the squared length of an edge between two vertices
double uv_length_squared(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    Eigen::Vector2d difference_vector = uv_1 - uv_0;
    double length_sq = difference_vector.dot(difference_vector);
    return length_sq;
}

// compute the length of an edge between two vertices
double uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    return sqrt(uv_length_squared(uv_0, uv_1));
}

double compute_uv_length_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    // Get the edge topology for the original uncut mesh
    Eigen::MatrixXi uE, EF, EI;
    Eigen::VectorXi EMAP;
    igl::edge_flaps(F, uE, EMAP, EF, EI);

    // Iterate over edges to check the length inconsistencies
    double max_uv_length_error = 0.0;
    for (Eigen::Index e = 0; e < EF.rows(); ++e) {
        // Get face corners corresponding to the current edge
        int f0 = EF(e, 0);
        int f1 = EF(e, 1);

        // Check first face (if not boundary)
        if (f0 < 0) continue;
        int i0 = EI(e, 0); // corner vertex face index
        int v0n = F_uv(f0, (i0 + 1) % 3); // next vertex
        int v0p = F_uv(f0, (i0 + 2) % 3); // previous vertex

        // Check second face (if not boundary)
        if (f1 < 0) continue;
        int i1 = EI(e, 1); // corner vertex face index
        int v1n = F_uv(f1, (i1 + 1) % 3); // next vertex
        int v1p = F_uv(f1, (i1 + 2) % 3); // next vertex

        // Compute the length of each halfedge corresponding to the corner in the cut mesh
        double l0 = uv_length(uv.row(v0n), uv.row(v0p));
        double l1 = uv_length(uv.row(v1n), uv.row(v1p));

        // Determine if the max length inconsistency has increased
        max_uv_length_error = max(max_uv_length_error, abs(l0 - l1));
    }

    // Return the max uv length error
    return max_uv_length_error;
}

bool check_uv(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    int n_faces = F.rows();
    bool is_valid = true;

    // Check faces agree in number
    if (F_uv.rows() != n_faces) {
        spdlog::error("Mesh and uv faces are not in one to one correspondence");
        is_valid = false;
    }

    // Check length consistency
    double uv_length_error = compute_uv_length_error(F, uv, F_uv);
    if (!float_equal(uv_length_error, 0.0, 1e-6)) {
        spdlog::warn("Inconsistent uv length error {} across VF edges", uv_length_error);
    }

    // Check uv face areas
    if (!check_areas(uv, F_uv)) {
        spdlog::error("Triangle is flipped in VF");
        is_valid = false;
    }

    // Return true if no issues found
    return is_valid;
}


} // namespace Penner
