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
#include "optimization/core/common.h"

/// @file refinement.hh
///
/// Methods to determine if polygons are self-overlapping and triangulate them

namespace Penner {
namespace Optimization {

/// Given three vertices in the plane, compute the triangle area
///
/// @param[in] vertices: three triangle vertices
/// @return face area
Scalar compute_face_area(const std::array<Eigen::VectorXd, 3>& vertices);

/// Given three vertices in the plane, determine if the triangle they form
/// has negative orientation.
///
/// @param[in] vertices: three triangle vertices
/// @return true iff the triangle is inverted in the uv plane
bool is_inverted_triangle(const std::array<Eigen::VectorXd, 3>& vertices);

/// Given a list of vertices in the plane, determine if the polygon they
/// determine is self-overlapping.
///
/// Also generates a table indicating if the subpolygons with vertices
/// (i,...,j) are self overlapping and the corresponding splitting vertices.
///
/// @param[in] uv_vertices: list of vertices in the uv plane
/// @param[in] vertices: list of vertices (for minimum face area computation)
/// @param[out] is_self_overlapping_subpolygon: table of subpolygon self
/// 		overlapping predicate truth values
/// @param[out] splitting_vertices: table of splitting vertices for self
///     overlapping subpolygons
/// @param[out] min_face_areas: table of minimum areas of subpolygon areas
/// @return true iff the polygon is self-overlapping
bool is_self_overlapping_polygon(
    const std::vector<Eigen::VectorXd>& uv_vertices,
    const std::vector<Eigen::VectorXd>& vertices,
    std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
    std::vector<std::vector<int>>& splitting_vertices,
    std::vector<std::vector<Scalar>>& min_face_areas);

/// Given a table indicating if the subpolygons of a polygon with vertices
/// (i,...,j) are self overlapping and the corresponding splitting vertices,
/// construct a triangulation of the full polygon.
///
/// The tables can be generated by is_self_overlapping_polygon.
///
/// @param[in] is_self_overlapping_subpolygon: table of subpolygon self
/// 		overlapping predicate truth values
/// @param[in] splitting_vertices: table of splitting vertices for self
///     overlapping subpolygons
/// @param[in] min_face_areas: table of minimum areas of subpolygon areas
/// @param[out] faces: list of triangles that triangulate the polygon
void triangulate_self_overlapping_polygon(
    const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
    const std::vector<std::vector<int>>& splitting_vertices,
    const std::vector<std::vector<Scalar>>& min_face_areas,
    std::vector<std::array<int, 3>>& faces);


} // namespace Optimization
} // namespace Penner