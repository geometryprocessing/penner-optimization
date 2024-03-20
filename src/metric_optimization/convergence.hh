#pragma once

#include "common.hh"
#include "cone_metric.hh"
#include "embedding.hh"
#include "energy_functor.hh"

/// @file Methods to analyze the convergence of a metric to a global minimum on the
/// constraint surface.

namespace CurvatureMetric {

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
void compute_direction_energy_values(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    std::shared_ptr<OptimizationParameters> opt_params,
    std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& direction,
    const VectorX& step_sizes,
    VectorX& unprojected_energies,
    VectorX& projected_energies);


} // namespace CurvatureMetric
