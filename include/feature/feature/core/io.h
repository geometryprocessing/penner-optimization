// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "feature/core/common.h"
#include "feature/core/vf_corners.h"

namespace Penner {
namespace Feature {

/**
 * @brief Methods to write/read edges to/from file, including feature, boundary, and seam edges.
 * 
 */

/**
 * @brief Write feature edges to a 0-indexed list in the following format:
 * <e00> <e01>
 * <e10> <101>
 * ...
 * 
 * @param fe_filename: filepath to edge list
 * @param feature_edges: list of edges
 */
void write_feature_edges(
    const std::string& fe_filename,
    const std::vector<VertexEdge>& feature_edges);

/**
 * @brief Load feature edges from a 0-indexed list in the following format:
 * <e00> <e01>
 * <e10> <101>
 * ...
 * 
 * @param fe_filename: filepath to edge list
 * @return list of edges
 */
std::vector<VertexEdge> load_feature_edges(const std::string& fe_filename);

/**
 * @brief Write edges to an obj file
 * 
 * @param fe_filename: filepath to obj mesh
 * @return list of edges
 */
void write_mesh_edges(
    const std::string& fe_filename,
    const std::vector<VertexEdge>& feature_edges);

/**
 * @brief Load edges from an obj file
 * 
 * @param fe_filename: filepath to obj mesh
 * @return list of edges
 */
std::vector<VertexEdge> load_mesh_edges(const std::string& fe_filename);

/**
 * @brief Write parameterization seams (that are not feature edges) to file
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_uv: mesh parameterization faces
 * @param F_is_feature: |F|x3 mask of corners opposite features
 */
void write_seams(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_feature);

/**
 * @brief Write feature edges to file
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_is_feature: |F|x3 mask of corners opposite features
 */
void write_features(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_is_feature);

/**
 * @brief Write parameterization boundary (both seams and features) to file
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_uv: mesh parameterization faces
 */
void write_boundary(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv);

/**
 * @brief Write mesh edges to file in the format:
 * v x0 y0 z0
 * v x1 y1 z1
 * ...
 * l v0 v1
 * ...
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh edges
 */
void write_edges(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E);

} // namespace Feature
} // namespace Penner