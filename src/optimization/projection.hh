#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {

/// Create matrix mapping vertex scale factors to their corresponding edges.
/// FIXME Make void with reference
///
/// @param[in] m: mesh
/// @return matrix representing the conformal scaling of edges
MatrixX
conformal_scaling_matrix(const Mesh<Scalar>& m);

/// Find conformally equivalent log edge lengths for the mesh m with initial log
/// lengths lambdas that satisfy the target angle constraints m.Th_hat.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] reduced_metric_coords: log edge lengths for the embedded original
/// mesh in m
/// @param[in, out] u: initial conformal scale factors and final scale factors
/// @param[in] proj_params: projection parameters to pass to the conformal
/// method
/// @return log edge lengths for the embedded original mesh that satisfy the
/// angle constraints to some precision defined by the proj_params
void
project_to_constraint(
  const Mesh<Scalar>& m,
  const VectorX& reduced_metric_coords,
  VectorX& reduced_metric_coords_proj,
  VectorX& u,
  std::shared_ptr<ProjectionParameters> proj_params = nullptr);

/// Find the Euclidean edge lengths for the mesh m with Penner coordinates
/// lambdas.
///
/// @param[in] m: Mesh
/// @param[in] lambdas_full: Penner coordinates for m
/// @param[out] log_edge_lengths: Euclidean log edge lengths for m
/// TODO Implement explicit checking
/// @return true if and only if the flips are all valid Euclidean flips
bool
convert_penner_coordinates_to_log_edge_lengths(const Mesh<Scalar>& m,
                                               const VectorX& lambdas_full,
                                               const VectorX& log_edge_lengths);

/// Compute the projection of the line search direction d onto the tangent space
/// of the constraint manifold with constraint function values F and Jacobian
/// J_F.
///
/// @param[in] descent_direction: direction to project onto constraint manifold
/// @param[in] constraint: constraint function values for lambdas
/// @param[in] J_constraint: Jacobian of the constraint value function for
/// lambdas
/// @return line search direction for lambdas.
VectorX
project_descent_direction(const VectorX& descent_direction,
                          const VectorX& constraint,
                          const MatrixX& J_constraint);

VectorX
project_to_constraint(
  const Mesh<Scalar>& m,
  const VectorX& lambdas,
  VectorX& u,
  std::shared_ptr<ProjectionParameters> proj_params = nullptr);

std::tuple<VectorX, VectorX>
project_to_constraint_py(
  const Mesh<Scalar>& m,
  const VectorX& lambdas,
  std::shared_ptr<ProjectionParameters> proj_params = nullptr);

/// Given the Jacobian of the constraint, compute the projection matrix to the constraint
/// tangent space
///
/// @param[in] J_constraint: Jacobian of the constraint value function for
/// lambdas
/// @return projection matrix for the descent direction
MatrixX
compute_descent_direction_projection_matrix(const MatrixX& J_constraint);

}