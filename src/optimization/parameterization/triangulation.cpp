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
#include "optimization/parameterization/triangulation.h"

#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/facet_components.h>
#include <igl/flipped_triangles.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <igl/predicates/predicates.h>
#include <set>
#include <stack>
#include "optimization/core/area.h"
#include "util/io.h"
#include "optimization/parameterization/refinement.h"
#include "optimization/core/constraint.h"
#include "util/vector.h"
#include "optimization/util/viewers.h"

#if ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

/// FIXME Do cleaning pass

namespace Penner {
namespace Optimization {


template <typename Vector>
Scalar compute_face_area(const std::array<Vector, 3>& vertices)
{
    // Get edge lengths for triangle
    Scalar li = (vertices[1] - vertices[0]).norm();
    Scalar lj = (vertices[2] - vertices[1]).norm();
    Scalar lk = (vertices[0] - vertices[2]).norm();

    // Compute area from lengths
    return sqrt(max(squared_area(li, lj, lk), Scalar(0.)));
}

Scalar compute_face_area(const std::array<Eigen::VectorXd, 3>& vertices)
{
    return compute_face_area<Eigen::VectorXd>(vertices);
}

template <typename Vector>
Scalar compute_min_face_angle(const std::array<Vector, 3>& vertices)
{
    // Get edge lengths for triangle
    Scalar li = (vertices[1] - vertices[0]).norm();
    Scalar lj = (vertices[2] - vertices[1]).norm();
    Scalar lk = (vertices[0] - vertices[2]).norm();

    // Compute area from lengths
    std::array<Scalar, 3> angles = compute_triangle_angles(li, lj, lk);
    return min(angles[0], min(angles[1], angles[2]));
}

bool is_inverted_triangle(const std::array<Eigen::Vector2d, 3>& vertices, double threshold, bool use_angles)
{
    // check robust predicate
    if (igl::predicates::orient2d(vertices[0], vertices[1], vertices[2]) != igl::predicates::Orientation::POSITIVE) return true;

    return (compute_min_face_angle(vertices) < threshold);

    // Build matrix of triangle homogenous coordinates
    Eigen::Matrix<Scalar, 3, 3> tri_homogenous_coords;
    tri_homogenous_coords.col(0) << vertices[0][0], vertices[0][1], 1.0;
    tri_homogenous_coords.col(1) << vertices[1][0], vertices[1][1], 1.0;
    tri_homogenous_coords.col(2) << vertices[2][0], vertices[2][1], 1.0;

    // Triangle is flipped iff the determinant is negative
    Scalar det = tri_homogenous_coords.determinant();
    return (det <= threshold);
}


bool is_self_overlapping_polygon(
    const std::vector<Eigen::Vector2d>& uv_vertices,
    const std::vector<Eigen::Vector3d>& vertices,
    std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
    std::vector<std::vector<int>>& splitting_vertices,
    std::vector<std::vector<Scalar>>& min_face_areas,
    double threshold,
    bool use_angles)
{
    is_self_overlapping_subpolygon.clear();
    splitting_vertices.clear();
    min_face_areas.clear();
    if (vertices.size() != uv_vertices.size()) {
        spdlog::error("Inconsistent uv and 3D vertices");
        return false;
    }

    // Build a table to record if the subpolygon between vertex i and j is self
    // overlapping and the corresponding splitting vertex indices
    int face_size = uv_vertices.size();
    is_self_overlapping_subpolygon.resize(face_size);
    splitting_vertices.resize(face_size);
    min_face_areas.resize(face_size);
    for (int i = 0; i < face_size; ++i) {
        is_self_overlapping_subpolygon[i] = std::vector<bool>(face_size, false);
        splitting_vertices[i] = std::vector<int>(face_size, -1);
        min_face_areas[i] = std::vector<Scalar>(face_size, 0.0);
    }

    // Set diagonal and superdiagonal to true as trivial polygons with less than 3 faces can be
    // triangulated with positive orientation (by convention)
    for (int i = 0; i < face_size; ++i) {
        is_self_overlapping_subpolygon[i][i] = true;
        is_self_overlapping_subpolygon[i][(i + 1) % face_size] = true;
    }

    // Check for degenerate (line, point, empty) polygons
    if (face_size < 3) {
        spdlog::warn("Checking if trivial polygon is self overlapping");
        return true;
    }

    // Build table iteratively by distance between vertices in ccw order
    for (int d = 2; d < face_size; ++d) {
        for (int i = 0; i < face_size; ++i) {
            // Checking for subpolygon between i and the vertex d away ccw around
            // the polygon
            int j = (i + d) % face_size;

            // Look for a splitting vertex k between i and j
            for (int k = (i + 1) % face_size; k != j; k = (k + 1) % face_size) {
                // Check if triangle T_ikj is positively oriented
                std::array<Eigen::Vector2d, 3> uv_triangle = {
                    uv_vertices[i],
                    uv_vertices[k],
                    uv_vertices[j]};
                std::array<Eigen::Vector3d, 3> triangle = {vertices[i], vertices[k], vertices[j]};
                if (is_inverted_triangle(uv_triangle, threshold)) continue;

                // Check if the two subpolygons (i, k) and (k, j) are self overlapping
                if (!is_self_overlapping_subpolygon[i][k]) continue;
                if (!is_self_overlapping_subpolygon[k][j]) continue;

                // Compute minimum of uv and 3D triangle areas for the subpolygon ij with
                // splitting vertex k
                Scalar min_area;
                if (use_angles)
                {
                    Scalar uv_triangle_area = compute_min_face_angle(uv_triangle);
                    Scalar triangle_area = compute_min_face_angle(triangle);
                    min_area = std::min(uv_triangle_area, triangle_area);
                }
                else
                {
                    Scalar uv_triangle_area = compute_face_area(uv_triangle);
                    Scalar triangle_area = compute_face_area(triangle);
                    min_area = std::min(uv_triangle_area, triangle_area);
                }
                if (k != (i + 1) % face_size) {
                    min_area = std::min(min_area, min_face_areas[i][k]);
                }
                if (j != (k + 1) % face_size) {
                    min_area = std::min(min_area, min_face_areas[k][j]);
                }

                // check if the triangle areas are too small
                if (min_area < threshold) continue;

                // Otherwise, k is a splitting vertex and (i, j) is self overlapping
                is_self_overlapping_subpolygon[i][j] = true;

                // Set splitting vertex, overwriting existing values iff it increase the min area
                if ((splitting_vertices[i][j] < 0) || (min_face_areas[i][j] < min_area)) {
                    splitting_vertices[i][j] = k;
                    min_face_areas[i][j] = min_area;
                }
            }
        }
    }

    // Check if a subdiagonal element is true (and thus the polygon is self overlapping)
    for (int j = 0; j < face_size; ++j) {
        int i = (j + 1) % face_size; // j = i - 1
        if (is_self_overlapping_subpolygon[i][j]) {
            return true;
        }
    }

    // Polygon is not self overlapping otherwise
    return false;
}

// Helper function to triangulate subpolygons
void triangulate_self_overlapping_subpolygon(
    const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
    const std::vector<std::vector<int>>& splitting_vertices,
    int start_vertex,
    int end_vertex,
    std::vector<std::array<int, 3>>& faces)
{
    int face_size = is_self_overlapping_subpolygon.size();
    spdlog::trace(
        "Triangulating subpolygon ({}, {}) of polygon of size {}",
        start_vertex,
        end_vertex,
        face_size);

    // Get starting edge (j,i)
    int i = start_vertex;
    int j = end_vertex;

    // Do not triangulate edges or vertices
    if ((i == j) || ((i + 1) % face_size == j)) {
        spdlog::trace("Base case vertex or edge");
        return;
    }

    // Add splitting face Tikj
    int k = splitting_vertices[i][j];
    faces.push_back({i, k, j});
    spdlog::trace("Adding triangle ({}, {}, {})", i, k, j);

    // Triangulate subpolygon (i, k)
    spdlog::trace("Recursing to subpolygon ({}, {})", i, k);
    triangulate_self_overlapping_subpolygon(
        is_self_overlapping_subpolygon,
        splitting_vertices,
        i,
        k,
        faces);

    // Triangulate subpolygon (k, j)
    spdlog::trace("Recursing to subpolygon ({}, {})", k, j);
    triangulate_self_overlapping_subpolygon(
        is_self_overlapping_subpolygon,
        splitting_vertices,
        k,
        j,
        faces);
}

void triangulate_self_overlapping_polygon(
    const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
    const std::vector<std::vector<int>>& splitting_vertices,
    const std::vector<std::vector<Scalar>>& min_face_areas,
    std::vector<std::array<int, 3>>& faces)
{
    faces.clear();
    int face_size = is_self_overlapping_subpolygon.size();
    if (face_size < 3) {
        spdlog::warn("Triangulated trivial face");
        return;
    }

    // Find (j, i) so that the triangulation of the polygon containing it has maximal min area
    int optimal_j = -1;
    Scalar max_min_area = -1.0;
    for (int j = 0; j < face_size; ++j) {
        int i = (j + 1) % face_size; // j = i - 1
        if ((is_self_overlapping_subpolygon[i][j]) && (min_face_areas[i][j] > max_min_area)) {
            optimal_j = j;
            max_min_area = min_face_areas[i][j];
        }
    }

    // Call recursive subroutine on the whole polygon for the optimal edge (j, i)
    if (max_min_area <= 0) spdlog::warn("triangle area {}", max_min_area);
    int j = optimal_j;
    int i = (j + 1) % face_size; // j = i - 1
    triangulate_self_overlapping_subpolygon(
        is_self_overlapping_subpolygon,
        splitting_vertices,
        i,
        j,
        faces);

    // Check triangulation is the right size
    int num_tri = faces.size();
    if (face_size != num_tri + 2) {
        spdlog::error(
            "Invalid triangulation of size {} for polygon of size {}",
            num_tri,
            face_size);
        faces.clear();
    }
}

// Helper function to view the triangulated face
void view_triangulation(
    const std::vector<Eigen::Vector2d>& uv_vertices,
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
    const std::vector<std::vector<int>>& splitting_vertices,
    const std::vector<std::vector<Scalar>>& min_face_areas,
    const std::vector<std::array<int, 3>>& faces)
{
    spdlog::trace("uv_vertices: {}", formatted_vector(uv_vertices));
    spdlog::trace("vertices: {}", formatted_vector(vertices));
    spdlog::trace("SO table");
    for (size_t i = 0; i < is_self_overlapping_subpolygon.size(); ++i) {
        spdlog::trace("row {}: {}", i, formatted_vector(is_self_overlapping_subpolygon[i]));
    }
    spdlog::trace("splitting vertices table");
    for (size_t i = 0; i < splitting_vertices.size(); ++i) {
        spdlog::trace("row {}: {}", i, formatted_vector(splitting_vertices[i]));
    }
    spdlog::trace("min face area table");
    for (size_t i = 0; i < min_face_areas.size(); ++i) {
        spdlog::trace("row {}: {}", i, formatted_vector(min_face_areas[i]));
    }

    // Print faces
    int num_faces = faces.size();
    for (int fi = 0; fi < num_faces; ++fi) {
        spdlog::trace("Local face {} is {}, {}, {}", fi, faces[fi][0], faces[fi][1], faces[fi][2]);
    }

#if ENABLE_VISUALIZATION
    // Open viewer
    polyscope::init();

    polyscope::registerPointCloud("vertices", vertices);
    polyscope::registerPointCloud2D("uv vertices", uv_vertices);
    polyscope::registerSurfaceMesh("face mesh", vertices, faces);
    polyscope::registerSurfaceMesh2D("uv face mesh", uv_vertices, faces);
    polyscope::show();
    polyscope::removeStructure("vertices");
    polyscope::removeStructure("uv_vertices");
    polyscope::removeStructure("face mesh");
    polyscope::removeStructure("uv face mesh");
#endif // ENABLE_VISUALIZATION
}


} // namespace Optimization
} // namespace Penner