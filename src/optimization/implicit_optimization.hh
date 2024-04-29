#pragma once

#include "common.hh"
#include "embedding.hh"
#include "energy_functor.hh"
#include "cone_metric.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
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
    Scalar rmse;                                    // Root mean squared edge lengths
    Scalar rrmse;                                   // Relative root mean squared edge lengths
    Scalar rmsre;                                   // Root mean squared relative edge lengths
    Scalar line_step_error;                         // Max angle constraint error after the line step
    Scalar line_step_energy;                        // Energy after line step
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

  /// Optimize a metric on a mesh with respect to a target metric with a log of
  /// current iteration data.
  ///
  /// @param[in] m: surface mesh
  /// @param[in] opt_energy: energy to optimize
  /// @param[out] log: final iteration log
  /// @param[in] proj_params: projection parameters
  /// @param[in] opt_params: optimization parameters
  /// @return optimized metric
  std::unique_ptr<DifferentiableConeMetric>
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
  /// @return optimized metric
  std::unique_ptr<DifferentiableConeMetric>
  optimize_metric(const DifferentiableConeMetric &m,
                  const EnergyFunctor &opt_energy,
                  std::shared_ptr<ProjectionParameters> proj_params = nullptr,
                  std::shared_ptr<OptimizationParameters> opt_params = nullptr);

}
