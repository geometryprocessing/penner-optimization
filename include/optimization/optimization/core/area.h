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

namespace CurvatureMetric {

/// Compute the squared area of the triangle with edge lengths li, lj, lk using
/// the numerically stable version of Heron's formula.
///
/// Heron's formula gives the formula s(s-a)(s-b)(s-c), where s is is the
/// semi-perimeter of the triangle. If the edge lengths do not satisfy the
/// triangle inequality this is no longer the squared area, but the formula is
/// still polynomial in the lengths.
///
/// @param[in] li: length of the first side of a triangle
/// @param[in] lj: length of the second side of a triangle
/// @param[in] lk: length of the third side of a triangle
/// @return: formal squared area of the triangle
Scalar squared_area(Scalar li, Scalar lj, Scalar lk);

/// Compute the derivative of the squared area of the triangle with given edge
/// lengths using the numerically stable version of Heron's formula with respect
/// to the first length.
///
/// Note that if the edge lengths do not satisfy the triangle inequality this is
/// no longer the squared area, but the formula is still polynomial in the
/// lengths.
///
/// @param[in] variable_length: length of the triangle to compute the derivative
/// with respect to
/// @param[in] lj: length of the second side of a triangle
/// @param[in] lk: length of the third side of a triangle
/// @return: the derivative of the squared area of the triangle
Scalar squared_area_length_derivative(Scalar variable_length, Scalar lj, Scalar lk);

/// Compute the squared area of the triangle containing each halfedge for the cone metric
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @return map from halfedges to the square of the area of the face containing it
VectorX squared_areas(const DifferentiableConeMetric& cone_metric);

/// Compute the area of the triangle containing each halfedge for the cone metric
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @param[out] he2area: map from halfedges to the area of the face containing it
VectorX areas(const DifferentiableConeMetric& cone_metric);

/// Compute the derivatives of the squared area of the triangle containing each
/// halfedge for the mesh with respect to the halfedge length coordinates.
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @return map from halfedges to the derivative of the square of the area of the face containing it
VectorX squared_area_length_derivatives(const DifferentiableConeMetric& cone_metric);

/// Compute the derivatives of the squared area of the triangle containing each
/// halfedge for the mesh with respect to the halfedge log length coordinates.
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @return map from halfedges to the derivative of the square of the area of the face containing it
VectorX squared_area_log_length_derivatives(const DifferentiableConeMetric& cone_metric);

} // namespace CurvatureMetric
