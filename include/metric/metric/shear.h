// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/common.h"
#include "metric/cone_metric.h"

/**
 * @brief Methods to compute the shear coordinates of a metric from Penner coordinates
 * and shear coordinate bases, linearly independent from the conformal scaling space.
 * 
 * Also can compute the change in shear coordinates between two non-conformal metrics,
 * which is necessary for computing continuous mappings between arbitrary surfaces.
 * 
 */

namespace Penner {

/// Compute the per halfedge logarithmic shear values for the mesh m with
/// logarithmic lengths lambdas_he
///
/// @param[in] m: mesh
/// @param[in] he_metric_coords: metric coordinates for m
/// @return per halfedge log shear
VectorX compute_shear(const Mesh<Scalar>& m, const VectorX& he_metric_coords);

/// Compute the change in shear along each halfedge between a target metric and
/// another arbitrary metric.
///
/// @param[in] m: mesh
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[in] he_metric_target: target metric coordinates for m
/// @param[out] he_shear_change: per halfedge shear change
void compute_shear_change(
    const Mesh<Scalar>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_shear_change);

/// Compute a matrix of independent basis vectors for the shear subspace of Penner coordinate space
/// corresponding to dual shear coordinates.
///
/// @param[in] m: mesh
/// @param[in] independent_edges: set of edges corresponding to a choice of independent dual shear
/// coordinates
/// @return: matrix with basis vectors as columns
void compute_shear_dual_matrix(
    const Mesh<Scalar>& m,
    const std::vector<int>& independent_edges,
    MatrixX& shear_matrix);

/// Compute a basis for the space of shear coordinates for a given mesh of dual shear vectors.
///
/// Note that the coefficients of these basis vectors do not correspond to shear coordinates
/// but span the same linear subspace.
///
/// @param[in] m: mesh
/// @param[out] shear_basis_matrix: matrix with shear basis vectors as columns
/// @param[out] independent_edges: indices of the independent edges of the mesh used for the basis
void compute_shear_dual_basis(
    const Mesh<Scalar>& m,
    MatrixX& shear_basis_matrix,
    std::vector<int>& independent_edges);
std::tuple<MatrixX, std::vector<int>> compute_shear_dual_basis_pybind(const Mesh<Scalar>& m);

/// Compute a basis for the space of shear coordinates for a given mesh of shear vectors with log
/// shear as coordinates.
///
/// @param[in] m: mesh
/// @param[out] shear_basis_matrix: matrix with shear basis vectors as columns
/// @param[out] independent_edges: indices of the independent edges of the mesh used for the basis
void compute_shear_coordinate_basis(
    const Mesh<Scalar>& m,
    MatrixX& shear_basis_matrix,
    std::vector<int>& independent_edges);

/// Compute the coordinates for a metric in terms of a basis of the shear coordinate
/// space and conformal scale factors.
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @param[in] shear_basis_matrix: matrix with shear space basis vectors as columns
/// @param[out] shear_coords: coordinates for the shear basis
/// @param[out] scale_factors: scale factor coordinates
void compute_shear_basis_coordinates(
    const DifferentiableConeMetric& cone_metric,
    const MatrixX& shear_basis_matrix,
    VectorX& shear_coords,
    VectorX& scale_factors);

} // namespace Penner