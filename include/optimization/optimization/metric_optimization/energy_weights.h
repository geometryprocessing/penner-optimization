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
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "util/embedding.h"

/// \file energy_functor.h
///
/// Methods to weight energies, e.g., per element energies by element area weights

namespace Penner {
namespace Optimization {

/// @brief Given a vector of weights and a vector of values, compute the weighted 2 norm
/// as the sum of the product of the weights and squared values
///
/// @param[in] weights: per term weights
/// @param[in] values: value vector for the norm
/// @return weighted norm
Scalar compute_weighted_norm(const VectorX& weights, const VectorX& values);

/// @brief Compute per edge weights for a mesh with a given metric as 1/3 of the areas
/// of the two adjacent faces
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @param[out] edge_area_weights: weights per edge
VectorX compute_edge_area_weights(const DifferentiableConeMetric& cone_metric);

/// @brief Compute per vertex area weights for a mesh
///
/// @param[in] m: mesh
/// @return weights per vertex
VectorX compute_vertex_area_weights(const Mesh<Scalar>& m);

/// @brief Compute per independent vertex area weights for a mesh
///
/// The weights are half the sum of identified vertex weights.
///
/// @param[in] m: mesh
/// @return weights per independent vertex
VectorX compute_independent_vertex_area_weights(const Mesh<Scalar>& m);

/// @brief Compute per face area weights for a mesh
///
/// @param[in] m: mesh
/// @return weights per face
VectorX compute_face_area_weights(const Mesh<Scalar>& m);

/// Compute a vector of weights for faces adjacent to the boundary.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] reduction_maps: reduction maps
/// @param[in] bd_weight: weight to give boundary adjacent faces
/// @param[out] face_weights: weights for faces
void compute_boundary_face_weights(
    const Mesh<Scalar>& m,
    const ReductionMaps& reduction_maps,
    Scalar bd_weight,
    std::vector<Scalar>& face_weights);


} // namespace Optimization
} // namespace Penner