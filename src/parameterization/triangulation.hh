#include "common.hh"

/// @file refinement.hh
///
/// Methods to determine if polygons are self-overlapping and triangulate them

namespace CurvatureMetric {

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


} // namespace CurvatureMetric
