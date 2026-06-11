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
 * @brief Methods to compute triangle inequality and differentiable cone angle constraints.
 * 
 * Also includes cone enumeration, and elimination of redundant or other fixed degrees of
 * freedom in the vertex cone angle constraints.
 * 
 */

namespace Penner {


/**
 * @brief Enumerate all cone vertices of a mesh, i.e., vertices whose target angle Th_hat
 * differs from the flat angle at that vertex type.
 *
 * A vertex is flat if its target angle equals the expected angle for a smooth surface point:
 * - Closed mesh: interior vertices are flat at 2*pi
 * - Open (doubled) mesh: interior vertices are flat at 4*pi; boundary vertices are flat at 2*pi
 *
 * @param[in] m: mesh with target angles Th_hat
 * @return indices of all cone vertices
 */
std::vector<int> enumerate_cone_vertices(const Mesh<Scalar>& m);

/**
 * @brief Compute vertex holonomy angles for a mesh with given angles
 *
 * @param[in] m: mesh topology
 * @param[in] alpha: per-halfedge angles for the mesh
 * @return vector of vertex holonomy angles
 */
VectorX Theta(const Mesh<Scalar>& m, const VectorX& alpha);

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

/// Build a map from independent vertices to only vertices marked as free in the mesh
///
/// @param[in] m: mesh
/// @param[out] v_mao: map from independent vertices to free vertex index or -1 for fixed vertices
/// @param[out] num_free_vertices: number of free vertices
void build_free_vertex_map(const Mesh<Scalar>& m, std::vector<int>& v_map, int& num_free_vertices);

/// Build a map from all vertices to only vertices marked as free in the mesh
///
/// @param[in] m: mesh
/// @param[out] v_rep: map from all vertices to free vertex index or -1 for fixed vertices
/// @param[out] num_free_vertices: number of free vertices
void build_free_vertex_rep(const Mesh<Scalar>& m, std::vector<int>& v_rep, int& num_free_vertices);

/// Compute the maximum cone angle constraint error.
/// 
/// @param[in] cone_metric: mesh with differentiable metric
/// @return maximum difference of the vertex angles from the target angles
Scalar compute_max_constraint(const DifferentiableConeMetric& cone_metric);

/**
 * @brief Compute the cone angles of the mesh
 * 
 * @param cone_metric: mesh with differentiable metric
 * @return cone angles
 */
VectorX compute_cone_angles(const DifferentiableConeMetric& cone_metric);

/**
 * @brief Method to compute the angles of a triangle with given lengths
 * 
 * @param l12: first edge length
 * @param l23: second edge length
 * @param l31: third edge length
 * @return corner angles at v1, v2, v3
 */
std::array<Scalar, 3> compute_triangle_angles(Scalar l12, Scalar l23, Scalar l31);

/// TODO Optionally add halfedge coordinate Jacobians

} // namespace Penner