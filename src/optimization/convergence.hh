#pragma once

#include "common.hh"
#include "energy_functor.hh"
#include "embedding.hh"
#include "cone_metric.hh"

/// @file Methods to analyze the convergence of a metric to a global minimum on the
/// constraint surface.

namespace CurvatureMetric
{

  /// Given a metric and direction, compute the energy values for optimization
  /// at given (potentially negative) step sizes before and after the projection to the
  /// constraint.
  ///
  /// @param[in] m: input mesh
  /// @param[in] opt_energy: optimization energy
  /// @param[in] opt_params: parameters for the optimization method
  /// @param[in] proj_params: parameters for the projection
  /// @param[in] direction: direction for the optimization
  /// @param[in] step_sizes: step sizes to compute the energy at
  /// @param[out] unprojected_energies: energies at the step sizes before projection
  /// @param[out] projected_energies: energies at the step sizes after projection
  void
  compute_direction_energy_values(
      const DifferentiableConeMetric &m,
      const EnergyFunctor &opt_energy,
      std::shared_ptr<OptimizationParameters> opt_params,
      std::shared_ptr<ProjectionParameters> proj_params,
      const VectorX &direction,
      const VectorX &step_sizes,
      VectorX &unprojected_energies,
      VectorX &projected_energies);

  /*
  TODO
  /// Given domain metric coordinates (with corresponding basis matrix) and direction,
  /// compute
  ///
  /// @param[in] m: input mesh
  /// @param[in] reduction_maps: maps between metric variables and per halfedge values
  /// @param[in] reduced_metric_target: target metric coordinates
  /// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
  /// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
  /// @param[in] constraint_codomain_matrix: matrix with codomain basis vectors as columns
  /// @param[in] opt_params: parameters for the optimization method
  /// @param[in] proj_params: parameters for the projection
  /// @param[in] direction: direction for the optimization
  /// @param[in] step_sizes: step sizes to compute the energy at
  /// @param[out] energies: energies at the step sizes along the gradient
  void
  compute_domain_direction_energy_values(
    const DifferentiableConeMetric& m,
    const ReductionMaps& reduction_maps,
    const VectorX& reduced_metric_target,
    const VectorX& domain_coords,
    const MatrixX& constraint_domain_matrix,
    const MatrixX& constraint_codomain_matrix,
    const OptimizationParameters& opt_params,
    const std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& direction,
    const VectorX& step_sizes,
    VectorX& energies
  );

  /// Given a metric, compute various stability numerics for optimization along the projected
  /// descent direction at given (potentially negative) step sizes after the projection
  /// to the constraint.
  ///
  /// @param[in] m: input mesh
  /// @param[in] reduced_metric_coords: metric coordinates
  /// @param[in] reduced_metric_target: target metric coordinates
  /// @param[in] opt_params: parameters for the optimization method
  /// @param[in] proj_params: parameters for the projection
  /// @param[in] step_sizes: step sizes to compute the energy at
  /// @param[out] max_errors: angle errors at steps
  /// @param[out] norm_changes_in_metric_coords: norm of changes in metric coordinates at steps
  /// @param[out] convergence_ratios: convergence ratios at steps
  /// @param[out] gradient_signs: dot product of gradient and descent directions at steps
  void
  compute_projected_descent_direction_stability_values(
    const DifferentiableConeMetric& m,
    const VectorX& reduced_metric_coords,
    const VectorX& reduced_metric_target,
    const OptimizationParameters& opt_params,
    const std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& step_sizes,
    VectorX& max_errors,
    VectorX& norm_changes_in_metric_coords,
    VectorX& convergence_ratios,
    VectorX& gradient_signs
  );
  */

#ifdef PYBIND
  /*
  std::tuple<
    VectorX, // unprojected_energies
    VectorX // projected_energies
  >
  compute_direction_energy_values_pybind(
    const DifferentiableConeMetric& m,
    const VectorX& reduced_metric_coords,
    const VectorX& reduced_metric_target,
    const OptimizationParameters& opt_params,
    const std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& direction,
    const VectorX& step_sizes
  );

  std::tuple<
    VectorX, // unprojected_energies
    VectorX // projected_energies
  >
  compute_projected_descent_direction_energy_values_pybind(
    const DifferentiableConeMetric& m,
    const VectorX& reduced_metric_coords,
    const VectorX& reduced_metric_target,
    const OptimizationParameters& opt_params,
    const std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& step_sizes
  );

  VectorX
  compute_domain_direction_energy_values_pybind(
    const DifferentiableConeMetric& m,
    const ReductionMaps& reduction_maps,
    const VectorX& reduced_metric_target,
    const VectorX& domain_coords,
    const MatrixX& constraint_domain_matrix,
    const MatrixX& constraint_codomain_matrix,
    const std::shared_ptr<OptimizationParameters> opt_params,
    const std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& direction,
    const VectorX& step_sizes
  );

  std::tuple<
    VectorX, // max_errors,
    VectorX, // norm_changes_in_metric_coords,
    VectorX, // convergence_ratios,
    VectorX  // gradient_signs
  >
  compute_projected_descent_direction_stability_values_pybind(
    const DifferentiableConeMetric& m,
    const VectorX& reduced_metric_coords,
    const VectorX& reduced_metric_target,
    const OptimizationParameters& opt_params,
    const std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& step_sizes
  );
  */

#endif

}
