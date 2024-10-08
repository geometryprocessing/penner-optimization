/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#pragma once

#include "optimization/core/common.h"
#include "optimization/core/cone_metric.h"

namespace Penner {
namespace Optimization {

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

} // namespace Optimization
} // namespace Penner
