#pragma once

#include "common.hh"
#include "cone_metric.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric
{

  /// Create matrix mapping vertex scale factors to their corresponding edges.
  /// FIXME Make void with reference
  ///
  /// @param[in] m: mesh
  /// @return matrix representing the conformal scaling of edges
  MatrixX
  conformal_scaling_matrix(const Mesh<Scalar> &m);

  /// Find the least squares best fit conformal mapping for the metric map from
  /// the metric with given coordinates to the target metric
  ///
  /// @param[in] target_cone_metric: target mesh with differentiable metric
  /// @param[in] metric_coords: coordinates for the current metric
  /// @return best fit conformal scale factors
  VectorX
  best_fit_conformal(const DifferentiableConeMetric &target_cone_metric,
                     const VectorX &metric_coords);

  /// Find conformally equivalent log edge lengths for the mesh m with initial log
  /// lengths lambdas that satisfy the target angle constraints m.Th_hat.
  ///
  /// @param[in] m: (possibly symmetric) mesh
  /// @param[in] reduced_metric_coords: log edge lengths for the embedded original
  /// mesh in m
  /// @param[in, out] u: initial conformal scale factors and final scale factors
  /// @param[in] proj_params: projection parameters to pass to the conformal
  /// method
  /// @param[in] opt_params: optimization parameters
  /// @return flip sequence and solve statistics for the conformal projection
  std::tuple<std::vector<int>, SolveStats<Scalar>>
  compute_constraint_scale_factors(
      const DifferentiableConeMetric &cone_metric,
      VectorX &u,
      std::shared_ptr<ProjectionParameters> proj_params = nullptr,
      std::shared_ptr<OptimizationParameters> opt_params = nullptr);
  VectorX
  compute_constraint_scale_factors(
      const DifferentiableConeMetric &cone_metric,
      std::shared_ptr<ProjectionParameters> proj_params = nullptr,
      std::shared_ptr<OptimizationParameters> opt_params = nullptr);

  /// Find a conformally equivalent metric for the cone metric that satisfy the target angle constraints.
  ///
  /// @param[in] cone_metric: mesh with differentiable metric
  /// @param[in] proj_params: projection parameters to pass to the conformal method
  /// @param[in] opt_params: optimization parameters
  /// @return conformally equivalent constrained metric
  std::unique_ptr<DifferentiableConeMetric>
  project_metric_to_constraint(
      const DifferentiableConeMetric &cone_metric,
      std::shared_ptr<ProjectionParameters> proj_params = nullptr,
      std::shared_ptr<OptimizationParameters> opt_params = nullptr);

  /// Given the Jacobian of the constraint, compute the projection matrix to the constraint
  /// tangent space
  ///
  /// Note that this construction is slow and is intended for analysis, not for optimization.
  ///
  /// @param[in] J_constraint: Jacobian of the constraint value function for
  /// lambdas
  /// @return projection matrix for the descent direction
  MatrixX
  compute_descent_direction_projection_matrix(const MatrixX &J_constraint);

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
  project_descent_direction(const VectorX &descent_direction,
                            const VectorX &constraint,
                            const MatrixX &J_constraint);

  /// Project a descent direction to the tangent plane of the angle constraint
  /// manifold for the given surface with orthogonal projection.
  ///
  /// @param[in] m: surface
  /// @param[in] descent_direction: current descent direction
  /// @return descent direction after projection to the constraint
  VectorX project_descent_direction(const DifferentiableConeMetric &cone_metric, const VectorX &descent_direction);

}