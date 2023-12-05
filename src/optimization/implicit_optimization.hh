#pragma once

#include "common.hh"
#include "embedding.hh"
#include "energy_functor.hh"
#include "cone_metric.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include <filesystem>

namespace CurvatureMetric
{

  // Log for implicit optimization iteration values
  struct
      OptimizationLog
  {
    int num_iterations;                             // Number of iterations so far
    Scalar time;                                    // Step size
    Scalar beta;                                    // Step size
    Scalar energy;                                  // Energy
    Scalar error;                                   // Max angle constraint error
    int num_flips;                                  // Number of flips for MakeDelaunay
    Scalar convergence_ratio;                       // Convergence ratio
    Scalar max_change_in_metric_coords;             // Maximum change per edge in the metric coordinates
    Scalar max_total_change_in_metric_coords;       // Maximum change per edge from the target coordinates
    Scalar actual_to_unconstrained_direction_ratio; // Ratio of change in coordinates to the unconstrained direction
    Scalar max_constrained_descent_direction;       // Maximum per edge constrained descent direction
    int num_linear_solves;                          // Number of iterations so far
  };

  /// Compute the convergence ratio for implicit optimization from the unconstrained and
  /// constrained descent directions.
  ///
  /// @param[in] unconstrained_descent_direction: global descent direction
  /// @param[in] constrained_descent_direction: descent direction after projection
  /// to the constraint
  /// @return convergence ratio
  Scalar compute_convergence_ratio(
      const VectorX &unconstrained_descent_direction,
      const VectorX &constrained_descent_direction);

  /// Given a metric with reduction maps and constraints and an energy gradient
  /// functor, compute a gradient and descent direction for the energy.
  ///
  /// @param[in] cone_metric: mesh with metric
  /// @param[in] opt_energy: optimization energy
  /// @param[in] prev_gradient: previous gradient of the energy
  /// @param[in] prev_descent_direction: previous descent direction
  /// @param[out] gradient: current gradient of the energy
  /// @param[out] descent_direction: descent direction
  /// @param[in] direction_choice: (optional) type of descent direction to use
  void
  compute_descent_direction(const DifferentiableConeMetric &cone_metric,
                            const EnergyFunctor &opt_energy,
                            const VectorX &prev_gradient,
                            const VectorX &prev_descent_direction,
                            VectorX &gradient,
                            VectorX &descent_direction,
                            std::string direction_choice);

  /// Project a descent direction to the tangent plane of the angle constraint
  /// manifold for the given surface with orthogonal projection.
  ///
  /// @param[in] m: surface
  /// @param[in] descent_direction: current descent direction
  /// @return descent direction after projection to the constraint
  VectorX project_descent_direction(const DifferentiableConeMetric &cone_metric, const VectorX &descent_direction);

  /// Given a metric with reduction maps and constraints and an energy gradient
  /// functor, compute a constrained descent direction for the energy that is optimal
  /// in the tangent space to the constraint manifold.
  ///
  /// @param[in] m: surface
  /// @param[in] opt_energy: optimization energy
  /// @param[in] gradient: current gradient of the energy
  /// @return optimal descent direction in the tangent space
  VectorX
  compute_optimal_tangent_space_descent_direction(
      const DifferentiableConeMetric &m,
      const EnergyFunctor &opt_energy,
      const VectorX &gradient);

  /// Perform a line search with projection to the constraint after the step.
  ///
  /// @param[in] cone_metric: surface
  /// @param[in] opt_energy: optimization energy
  /// @param[in] descent_direction: descent direction for the line search
  /// @param[in] proj_params: projection parameters
  /// @param[in] opt_params: optimization parameters
  /// @param[in, out] beta: adaptive line step size
  /// @param[in, out] convergence_ratio: gradient convergence ratio
  /// @param[in, out] num_linear_solves: number of linear solves
  VectorX
  line_search_with_projection(const DifferentiableConeMetric &cone_metric,
                              const EnergyFunctor &opt_energy,
                              const VectorX &descent_direction,
                              std::shared_ptr<ProjectionParameters> proj_params,
                              std::shared_ptr<OptimizationParameters> opt_params,
                              Scalar &beta,
                              Scalar &convergence_ratio,
                              int &num_linear_solves);

  /// Check if the implicit optimization has converged.
  ///
  /// @param[in] opt_params: optimization parameters
  /// @param[in] beta: adaptive line step size
  /// @param[in] convergence_ratio: gradient convergence ratio
  bool check_if_converged(const OptimizationParameters &opt_params,
                          Scalar convergence_ratio,
                          Scalar beta);

  /// Optimize a metric on a mesh with respect to a target metric with a log of
  /// current iteration data.
  ///
  /// @param[in] m: surface mesh
  /// @param[in] opt_energy: energy to optimize
  /// @param[out] log: final iteration log
  /// @param[in] proj_params: projection parameters
  /// @param[in] opt_params: optimization parameters
  /// @return optimized coordinates of the metric
  VectorX
  optimize_metric_log(const DifferentiableConeMetric &m,
                      const EnergyFunctor &opt_energy,
                      OptimizationLog &log,
                      std::shared_ptr<ProjectionParameters> proj_params,
                      std::shared_ptr<OptimizationParameters> opt_params);

  /// Optimize a metric on a mesh with respect to a target metric.
  ///
  /// @param[in] m: surface mesh
  /// @param[in] opt_energy: energy to optimize
  /// @param[in] proj_params: projection parameters
  /// @param[in] opt_params: optimization parameters
  /// @return optimized coordinates of the metric
  VectorX
  optimize_metric(const DifferentiableConeMetric &m,
                  const EnergyFunctor &opt_energy,
                  std::shared_ptr<ProjectionParameters> proj_params = nullptr,
                  std::shared_ptr<OptimizationParameters> opt_params = nullptr);

}
