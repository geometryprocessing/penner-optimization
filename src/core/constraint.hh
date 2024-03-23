#pragma once

#include "common.hh"
#include "cone_metric.hh"

namespace CurvatureMetric {

/// Check the triangle inequality for every triangle in the mesh with respect to the
/// halfedge metric coordinates
///
/// @param[in] cone_metric: mesh with metric
/// @return: true iff each triangle in the mesh satisfies the triangle inequality
bool satisfies_triangle_inequality(const Mesh<Scalar>& cone_metric);

/// Compute the triangle angles and cotangent angles of a Delaunay (possibly
/// symmetric) mesh with metric. Angles are indexed by their opposing halfedge.
///
/// @param[in] cone_metric: mesh with metric
/// @param[out] he2angle: map from halfedges to opposing angle
/// @param[out] he2cot: map from halfedges to cotan of opposing angle
void corner_angles(const Mesh<Scalar>& cone_metric, VectorX& he2angle, VectorX& he2cot);

/// Compute the vertex cone angles of a mesh with metric and the Jacobian with respect to the reduced
/// coordinates if needed.
///
/// The mesh may be copied and modified to ensure the current halfedge coordinates are log edge lengths.
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @param[out] vertex_angles: vertex angles of m
/// @param[out] J_vertex_angles: Jacobian of the vertex_angles as a function of
/// the half edge coordinates
/// @param[in] need_jacobian: create Jacobian iff true
void vertex_angles_with_jacobian(
    const DifferentiableConeMetric& cone_metric,
    VectorX& vertex_angles,
    MatrixX& J_vertex_angles,
    bool need_jacobian = true,
    bool only_free_vertices = true);

/// Compute the difference of the vertex angles of a mesh with a metric from the target angles
/// and the Jacobian with respect to the reduced coordinates if needed.
///
/// The mesh may be copied and modified to ensure the current halfedge coordinates are log edge lengths.
///
/// @param[in] cone_metric: mesh with differentiable metric
/// @param[out] constraint: difference of the vertex angles from the target angles
/// @param[out] J_constraint: Jacobian of constraint as a function of log edge lengths
/// @param[in] need_jacobian: (optional) create Jacobian iff true
/// @return: true iff the mesh and metric coordinates are valid
bool constraint_with_jacobian(
    const DifferentiableConeMetric& cone_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian = true,
    bool only_free_vertices = true);


/// Build a map from all vertices to only vertices marked as free in the mesh
///
/// @param[in] m: mesh
/// @param[out] v_rep: map from all vertices to free vertex index or -1 for fixed vertices
/// @param[out] num_free_vertices: number of free vertices
void build_free_vertex_map(const Mesh<Scalar>& m, std::vector<int>& v_rep, int& num_free_vertices);

/// Compute the maximum cone angle constraint error.
/// 
/// @param[in] cone_metric: mesh with differentiable metric
/// @return maximum difference of the vertex angles from the target angles
Scalar compute_max_constraint(const DifferentiableConeMetric& cone_metric);

/// TODO Optionally add halfedge coordinate Jacobians

} // namespace CurvatureMetric
