#pragma once

#include "common.hh"
#include "embedding.hh"
#include "energy_functor.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include <filesystem>

namespace CurvatureMetric {

// Log for implicit optimization iteration values
struct
OptimizationLog {
  int num_iterations; // Number of iterations so far
  Scalar time; // Step size
  Scalar beta; // Step size
  Scalar energy; // Energy
  Scalar error; // Max angle constraint error
  int num_flips; // Number of flips for MakeDelaunay
  Scalar convergence_ratio; // Convergence ratio
  Scalar max_change_in_metric_coords; // Maximum change per edge in the metric coordinates
  Scalar max_total_change_in_metric_coords; // Maximum change per edge from the target coordinates
  Scalar actual_to_unconstrained_direction_ratio; // Ratio of change in coordinates to the unconstrained direction
  Scalar max_constrained_descent_direction; // Maximum per edge constrained descent direction
  int num_linear_solves; // Number of iterations so far
};

/// Compute the convergence ratio for implicit optimization from the unconstrained and
/// constrained descent directions.
///
/// @param[in] unconstrained_descent_direction: global descent direction
/// @param[in] constrained_descent_direction: descent direction after projection
/// to the constraint
/// @return convergence ratio
Scalar compute_convergence_ratio(
  const VectorX& unconstrained_descent_direction,
  const VectorX& constrained_descent_direction
);

/// Initialize data log file for implicit optimization
///
/// @param[in] data_log_path: filepath for data log output
void
initialize_data_log(
  const std::filesystem::path& data_log_path
);

/// Update the implicit data log for the given iteration.
///
/// @param[in, out] log: log to update
/// @param[in] m: surface
/// @param[in] reduction_maps: maps between metric variables
/// @param[in] opt_energy: optimization energy
/// @param[in] updated_reduced_metric_coords: coordinates of the metric at the end of the iteration
/// @param[in] reduced_metric_coords: coordinates of the metric at the start of the iteration
/// @param[in] reduced_metric_target: target coordinates for the metric
/// @param[in] unconstrained_descent_direction: descent direction before projection
/// @param[in] constrained_descent_direction: descent direction after projection
/// @param[in] convergence_ratio: gradient convergence ratio
/// @param[in] opt_params: optimization parameters
void
update_data_log(
  OptimizationLog &log,
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& updated_reduced_metric_coords,
  const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
  const VectorX& unconstrained_descent_direction,
  const VectorX& constrained_descent_direction,
  Scalar convergence_ratio,
  std::shared_ptr<OptimizationParameters> opt_params
);

/// Write iteration data log for implicit optimization to file
///
/// @param[in] log: log entry to write
/// @param[in] data_log_path: filepath for data log output
void
write_data_log_entry(
  const OptimizationLog &log,
  const std::filesystem::path& data_log_path
);

/// Given a metric with reduction maps and constraints and an energy gradient
/// functor, compute a gradient and descent direction for the energy.
///
/// @param[in] reduced_metric_coords: current coordinates of the metric
/// @param[in] reduced_metric_target: target coordinates for the metric
/// @param[in] prev_gradient: previous gradient of the energy
/// @param[in] prev_descent_direction: previous descent direction
/// @param[in] reduction_maps: maps between metric variables and per halfedge
/// values
/// @param[in] opt_energy: optimization energy
/// @param[out] gradient: current gradient of the energy
/// @param[out] descent_direction: descent direction
/// @param[in] direction_choice: type of descent direction to use
void
compute_descent_direction(const VectorX& reduced_metric_coords,
                          const VectorX& reduced_metric_target,
                          const VectorX& prev_gradient,
                          const VectorX& prev_descent_direction,
                          const ReductionMaps& reduction_maps,
                          const EnergyFunctor& opt_energy,
                          VectorX& gradient,
                          VectorX& descent_direction,
                          std::string direction_choice="gradient");

/// Project a descent direction to the tangent plane of the angle constraint
/// manifold for the given surface with orthogonal projection.
///
/// @param[in] m: surface
/// @param[in] reduced_metric_coords: current coordinates of the metric
/// @param[in] descent_direction: current descent direction
/// @param[in] reduction_maps: maps between metric variables and per halfedge
/// values
/// @param[in] opt_params: optimization parameters
/// @param[out] projected_descent_direction: descent direction after projection
/// to the constraint
void
constrain_descent_direction(const Mesh<Scalar>& m,
                            const VectorX& reduced_metric_coords,
                            const VectorX& descent_direction,
                            const ReductionMaps& reduction_maps,
                            const OptimizationParameters& opt_params,
                            VectorX& projected_descent_direction);

/// Given a metric with reduction maps and constraints and an energy gradient
/// functor, compute a constrained descent direction for the energy that is optimal
/// in the tangent space to the constraint manifold.
///
/// @param[in] m: surface
/// @param[in] reduced_metric_coords: current coordinates of the metric
/// @param[in] gradient: current gradient of the energy
/// @param[in] descent_direction: global optimal descent direction
/// @param[in] reduction_maps: maps between metric variables and per halfedge
/// values
/// @param[in] opt_params: optimization parameters
/// @param[in] opt_energy: optimization energy
/// @param[out] projected_descent_direction: optimal descent direction in the tangent space
void
compute_optimal_tangent_space_descent_direction(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
	const VectorX& gradient,
	const VectorX& descent_direction,
	const ReductionMaps& reduction_maps,
  const OptimizationParameters& opt_params,
	const EnergyFunctor& opt_energy,
	VectorX& projected_descent_direction
);

/// Perform a line search with projection to the constraint after the step.
///
/// @param[in] m: surface
/// @param[in] initial_reduced_metric_coords: initial coordinates of the metric
/// @param[in] descent_direction: descent direction for the line search
/// @param[in] reduction_maps: maps between metric variables and per halfedge
/// values
/// @param[in] proj_params: projection parameters
/// @param[in] opt_params: optimization parameters
/// @param[in] opt_energy: optimization energy
/// @param[in, out] beta: adaptive line step size
/// @param[in, out] convergence_ratio: gradient convergence ratio
/// @param[out] reduced_line_step_metric_coords: coordinates of the metric before the conformal projection
/// @param[out] u: scale factors for the conformal projection
/// @param[out] reduced_metric_coords: final coordinates of the metric
void
line_search_with_projection(const Mesh<Scalar>& m,
                            const VectorX& initial_reduced_metric_coords,
                            const VectorX& descent_direction,
                            const ReductionMaps& reduction_maps,
                            std::shared_ptr<ProjectionParameters> proj_params,
                            std::shared_ptr<OptimizationParameters> opt_params,
                            const EnergyFunctor& opt_energy,
                            Scalar& beta,
                            Scalar& convergence_ratio,
                            VectorX& reduced_line_step_metric_coords,
                            VectorX& u,
                            VectorX& reduced_metric_coords);

/// Check if the implicit optimization has converged.
///
/// @param[in] opt_params: optimization parameters
/// @param[in] beta: adaptive line step size
/// @param[in] convergence_ratio: gradient convergence ratio
bool
check_if_converged(const OptimizationParameters& opt_params,
                   Scalar convergence_ratio,
                   Scalar beta);

/// Optimize a metric on a mesh with respect to a target metric with a log of
/// current iteration data.
///
/// @param[in] m: surface mesh
/// @param[in] reduced_metric_target: target coordinates of the metric
/// @param[in] reduced_metric_init: initial coordinates of the metric
/// @param[out] optimized_reduced_metric_coords: optimized coordinates of the metric
/// @param[out] log: final iteration log
/// @param[in] proj_params: projection parameters
/// @param[in] opt_params: optimization parameters
void
optimize_metric_log(const Mesh<Scalar>& m,
                const VectorX& reduced_metric_target,
                const VectorX& reduced_metric_init,
                VectorX& optimized_reduced_metric_coords,
                OptimizationLog& log,
                std::shared_ptr<ProjectionParameters> proj_params,
                std::shared_ptr<OptimizationParameters> opt_params);

/// Optimize a metric on a mesh with respect to a target metric.
///
/// @param[in] m: surface mesh
/// @param[in] reduced_metric_target: target coordinates of the metric
/// @param[in] reduced_metric_init: initial coordinates of the metric
/// @param[out] optimized_reduced_metric_coords: optimized coordinates of the metric
/// @param[in] proj_params: projection parameters
/// @param[in] opt_params: optimization parameters
void
optimize_metric(const Mesh<Scalar>& m,
                const VectorX& reduced_metric_target,
                const VectorX& reduced_metric_init,
                VectorX& optimized_reduced_metric_coords,
                std::shared_ptr<ProjectionParameters> proj_params = nullptr,
                std::shared_ptr<OptimizationParameters> opt_params = nullptr);
              
#ifdef PYBIND
VectorX
optimize_metric_pybind(const Mesh<Scalar>& m,
                const VectorX& reduced_metric_target,
                const VectorX& reduced_metric_init,
                std::shared_ptr<ProjectionParameters> proj_params = nullptr,
                std::shared_ptr<OptimizationParameters> opt_params = nullptr);

#endif

}
