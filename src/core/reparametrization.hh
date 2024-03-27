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

#include "common.hh"

namespace CurvatureMetric {

/// Reparametrize the barycentric coordinates for the equilateral triangle by
/// translating a constant hyperbolic distance along each halfedge. If the sum
/// of translations per triangle is 0, then this corresponds to a projective map
/// over the triangle.
///
/// @param[in, out] m_o: mesh to reparametrize
/// @param[in] tau: per halfeedge hyperbolic translation distances
void bc_reparametrize_eq(OverlayMesh<Scalar>& m_o, const VectorX& tau);

/// Reparametrize points contained in equilateral reference triangles by
/// translating a constant hyperbolic distance along each halfedge. If the sum
/// of translations per triangle is 0, then this corresponds to a projective map
/// over the triangle.
///
/// @param[in, out] pts: points to reparameterize
/// @param[in] n: next halfedge array for mesh
/// @param[in] h: face to halfedge array for mesh
/// @param[in] tau: per halfeedge hyperbolic translation distances
void reparametrize_equilateral(
    std::vector<Pt<Scalar>>& pts,
    const std::vector<int>& n,
    const std::vector<int>& h,
    const VectorX& tau);

#ifdef PYBIND
#endif

} // namespace CurvatureMetric
