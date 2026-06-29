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
 * @brief Methods to generate vector field directions on a surface, representated either as
 * a 3D vector or as an angle relative to a reference edge.
 * 
 * Also includes principal curvature vector field computation.
 * 
 */

namespace Penner {
namespace Field {

/**
 * @brief Generate the reference tangent direction of a face along the oriented edge opposite
 * the corner with local index {0, 1, 2}. 
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param fijk: face index
 * @param local_index: local index of the corner opposite the reference direction
 * @return tangent direction along the edge opposite the corner
 */
Eigen::Vector3d generate_reference_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int fijk,
    int local_index);

/**
 * @brief Generate a field of per-face reference directions.
 * 
 * The chosen direction for a face fijk is the directed edge eki.
 * 
 * @param V: mesh vertices 
 * @param F: mesh faces
 * @return per-face tangent direction matrix
 */
Eigen::MatrixXd generate_reference_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

/**
 * @brief Generate a field of per-face reference directions determined by opposite reference corners.
 * 
 * @param V: mesh vertices 
 * @param F: mesh faces
 * @param reference_corner: per-face local index of corners opposite the reference direciton
 * @return per-face tangent direction matrix
 */
Eigen::MatrixXd generate_reference_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& reference_corner);

/**
 * @brief Given a representative vector field direction defined by a reference direction and an
 * offset angle, generate the representative direction matrix
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @return per-face representative direction matrix
 */
Eigen::MatrixXd generate_vector_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Infer the angle offset of a direction field relative to a reference field
 * 
 * @param V: mesh vertices`
 * @param F: mesh faces
 * @param reference_corner: per-face reference corner opposite the reference direction
 * @param direction_field: per-face tangent direction matrix
 * @return offset angles of the direction field relative to the reference
 */
Eigen::VectorXd infer_theta(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& reference_corner,
    const Eigen::MatrixXd& direction_field);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
compute_facet_principal_curvature(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius=5);
    
/**
 * @brief Compute salient geometry aligned field directions for a mesh.
 * 
 * The parabolic anisotropy used for the relative threshold is ||k2| - |k1|| / max(|k1|, |k2|)
 * This measurement is near 0 for parabolic regions and near 1 for highly anisotropic regions
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param radius: (optional) vertex radius for fitting a smooth surface for field estimation
 * @param abs_threshold: (optional) minimum threshold for mean anisotropy of principal curvatures
 * @param rel_threshold: (optional) minimum threshold for parabolic anisotropy of principal curvatures
 * @return |F|x3 matrix of per face directions
 * @return per face mask indicating whether a direction is salient or not
 */
std::tuple<Eigen::MatrixXd, std::vector<bool>> compute_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius=5,
    Scalar abs_threshold=1.,
    Scalar rel_threshold=0.9,
    Scalar sample_rate=1.);



} // namespace Field
} // namespace Penner