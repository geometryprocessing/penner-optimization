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
 * @brief Methods for generating cross fields as represented by four tangent vectors
 * without explicit encoding of period jumps across edges.
 * 
 */

namespace Penner {
namespace Field {

/**
 * @brief Load rawfield data format used by Directional.
 * 
 * @param filename: rawfield data file
 * @return four per-face tangent direction matrices
 */
std::array<Eigen::MatrixXd, 4> load_rawfield(const std::string& filename);

/**
 * @brief Serialize a cross field defined by an angle relative to a reference direction as
 * four directions per face (i.e., rawfield format for Directional).
 * 
 * @param output_filename: file to write cross field
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 */
void write_cross_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Load a rosy field from file.
 * 
 * The format is
 * ```
 * <num_faces>
 * 4
 * <dx> <dy> <dz>
 * ...
 * ```
 * where d is a representative direction on the face. The remaining directions can be inferred
 * by rotational symmetry.
 * 
 * @param input_filename: file location of the rosy field
 * @return per-face representative field direction
 */
Eigen::MatrixXd load_rosy_field(const std::string& input_filename);

/**
 * @brief Write a rotationally symmetric rosy field to file.
 * 
 * The format is
 * ```
 * <num_faces>
 * 4
 * <dx> <dy> <dz>
 * ...
 * ```
 * where d is a representative direction on the face. The remaining directions can be inferred
 * by rotational symmetry.
 * 
 * @param output_filename: file location to serialize the frame field
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 */
void write_rosy_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Given a representative cross field direction defined by a reference direction and an
 * offset angle, generate all four cross field direction matrices.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @return four per-face tangent direction matrices
 */
std::array<Eigen::MatrixXd, 4> generate_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Reduce the curl of a cross field
 * 
 * WARNING: Not well tested or currently in active use
 * 
 * @param V 
 * @param F 
 * @param cross_field 
 * @param fixed_faces 
 * @return std::array<Eigen::MatrixXd, 4> 
 */
std::array<Eigen::MatrixXd, 4> reduce_curl(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::array<Eigen::MatrixXd, 4>& cross_field,
    const std::vector<int>& fixed_faces);

/**
 * @brief Generate a rosy field for a mesh
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @return |F|x3 frame field of per-face field direction vectors
 * @return per-vertex cone angles corresponding to the frame field
 */
std::tuple<Eigen::MatrixXd, std::vector<Scalar>> generate_rosy_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

} // namespace Feature 
} // namespace Penner