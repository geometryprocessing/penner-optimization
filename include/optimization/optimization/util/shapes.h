// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "util/common.h"

// Some good simple tests are simplex embeddings that are natural (one vertex
// at the origin and others at unit vectors) with metrics that are uniform 
// (length 1 for every edge). The uniform lengths are fully symmetric, and the
// embedded case has three symmetric edges adjacent to the origin and three
// symmetric edges not adjacent to the origin.

namespace Penner {
namespace Optimization {

void
map_to_sphere(size_t num_vertices, std::vector<Scalar>& Th_hat);

void
generate_double_triangle(Eigen::MatrixXd& V,
                         Eigen::MatrixXi& F,
                         std::vector<Scalar>& Th_hat);

void
generate_triangle(Eigen::MatrixXd& V,
                         Eigen::MatrixXi& F,
                         std::vector<Scalar>& Th_hat);

void
generate_double_triangle_mesh(Mesh<Scalar>& m, std::vector<int>& vtx_reindex);

void
generate_tetrahedron(Eigen::MatrixXd& V,
                     Eigen::MatrixXi& F,
                     std::vector<Scalar>& Th_hat);

void
generate_tetrahedron_mesh(Mesh<Scalar>& m, std::vector<int>& vtx_reindex);

std::tuple<Eigen::MatrixXd,    // V
           Eigen::MatrixXi,    // F
           std::vector<Scalar> // Th_hat
           >
generate_tetrahedron_pybind();

std::tuple<Mesh<Scalar>,    // m
           std::vector<int> // vtx_reindex
           >
generate_tetrahedron_mesh_pybind();

std::tuple<Eigen::MatrixXd,    // V
           Eigen::MatrixXi,    // F
           std::vector<Scalar> // Th_hat
           >
generate_double_triangle_pybind();

std::tuple<Mesh<Scalar>,    // m
           std::vector<int> // vtx_reindex
           >
generate_double_triangle_mesh_pybind();

} // namespace Optimization
} // namespace Penner