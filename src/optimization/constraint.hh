#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {

/// Check the triangle inequality for every triangle in the mesh with
/// log lengths log_length_coords.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] log_length_coords: log edge lengths for m
/// @return: true iff each triangle in the mesh satisfies the triangle
/// inequality
bool
satisfies_triangle_inequality(const Mesh<Scalar>& m,
                              const VectorX& log_length_coords);

/// Compute the triangle angles and cotangent angles of a Delaunay (possibly
/// symmetric) mesh m with log edge lengths log_length_coords. Angles are
/// indexed by their opposing halfedge.
///
/// @param[in] m: (possibly symmetric) Delaunay mesh
/// @param[in] log_length_coords: log edge lengths for m
/// @param[out] he2angle: map from halfedges to opposing angle
/// @param[out] he2cot: map from halfedges to cotan of opposing angle
void
corner_angles(const Mesh<Scalar>& m,
              const VectorX& log_length_coords,
              VectorX& he2angle,
              VectorX& he2cot);

/// Compute the vertex angles of a Delaunay (possibly symmetric) mesh m with log
/// edge lengths log_length_coords.
///
/// @param[in] m: (possibly symmetric) Delaunay mesh
/// @param[in] log_length_coords: log edge lengths for m
/// @param[out] vertex_angles: vertex angles of m
/// @param[out] J_vertex_angles: Jacobian of the vertex_angles as a function of
/// the half edge coordinates
/// @param[in] need_jacobian: create Jacobian iff true
void
vertex_angles_with_jacobian(const Mesh<Scalar>& m,
                            const VectorX& log_length_coords,
                            VectorX& vertex_angles,
                            MatrixX& J_vertex_angles,
                            bool need_jacobian = true);

/// Compute the difference of the vertex angles of a mesh m with log edge
/// lengths metric_coords from the target angles m.Th_hat. Also compute the
/// Jacobian of the constraint if needed.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] metric_coords: metric coordinates for m
/// @param[out] constraint: difference of the vertex angles from the target
/// angles
/// @param[out] J_constraint: Jacobian of constraint as a function of log edge
/// lengths
/// @param[out] flip_seq: sequence of flips used
/// @param[in] need_jacobian: create Jacobian iff true
/// @param[in] use_edge_lengths: use edge lengths directly iff true
/// @return: true iff the mesh satisfies the triangle inequality
bool
constraint_with_jacobian(const Mesh<Scalar>& m,
                         const VectorX& metric_coords,
                         VectorX& constraint,
                         MatrixX& J_constraint,
                         std::vector<int>& flip_seq,
                         bool need_jacobian = true,
                         bool use_edge_lengths = true);

#ifdef PYBIND
// Pybind definitions
std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
vertex_angles_with_jacobian_pybind(const Mesh<Scalar>& m,
                                   const VectorX& metric_coords,
                                   bool need_jacobian = true);

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>, std::vector<int>, bool>
constraint_with_jacobian_pybind(const Mesh<Scalar>& m,
                                const VectorX& metric_coords,
                                bool need_jacobian = true,
                                bool use_edge_lengths = true);
#endif

}
