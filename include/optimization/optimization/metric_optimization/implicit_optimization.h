// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <filesystem>
#include "metric/common.h"
#include "metric/cone_metric.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "util/embedding.h"
#include "optimization/metric_optimization/energy_functor.h"

/**
 * @brief Method to optimize distortion with constraints using a walk-on-manifold approach in the
 * full space of metric coordinates using projection.
 * 
 * The constraints are determined by the Differentiable Cone Metric, as well as the method of projecting
 * to its constraint submanifold and projecting distortion descent directions. This is more general than
 * the explicit optimization methods, which assume the existence of a conformal basis for solving angle
 * constraints.
 * 
 */

namespace Penner {
namespace Optimization {
    
// Parameters for the optimization method
struct OptimizationParameters
{
    // Logging
    std::string output_dir = ""; // output directory for file logs
    bool use_checkpoints = false; // if true, checkpoint state to output directory

    // Convergence parameters
    Scalar min_ratio =
        0.0; // minimum ratio of projected to ambient descent direction for convergence
    int num_iter = 200; // maximum number of iterations

    // Line step choice parameters
    bool require_energy_decr = true; // if true, require energy to decrease in each iteration
    bool require_gradient_proj_negative = true; // if true, require projection of the gradient onto
    // the descent direction to remain negative
    Scalar max_angle_incr = INF; // maximum allowed angle error increase in line step
    Scalar max_energy_incr = 1e-8; // maximum allowed energy increase in iteration

    // Optimization method choices
    std::string direction_choice = "projected_gradient"; // choice of direction

    // Numerical stability parameters
    Scalar beta_0 = 1.0; // initial line step size to try
    Scalar max_beta = 1e16; // maximum allowed line step size
    Scalar max_grad_range = 10; // maximum allowed gradient range (reduce if larger)
    Scalar max_angle = INF; // maximum allowed cone angle error (reduce if larger)
};

// Log for implicit optimization iteration values
struct OptimizationLog
{
    int num_iterations; // Number of iterations so far
    Scalar time; // Step size
    Scalar beta; // Step size
    Scalar energy; // Energy
    Scalar error; // Max angle constraint error
    Scalar rmse; // Root mean squared edge lengths
    Scalar rrmse; // Relative root mean squared edge lengths
    Scalar rmsre; // Root mean squared relative edge lengths
    int num_flips; // Number of flips for MakeDelaunay
    Scalar convergence_ratio; // Convergence ratio
    Scalar max_change_in_metric_coords; // Maximum change per edge in the metric coordinates
    Scalar max_total_change_in_metric_coords; // Maximum change per edge from the target coordinates
    Scalar actual_to_unconstrained_direction_ratio; // Ratio of change in coordinates to the
                                                    // unconstrained direction
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
    const VectorX& constrained_descent_direction);

/// Optimize a metric on a mesh with respect to a target metric with a log of
/// current iteration data.
///
/// @param[in] m: surface mesh
/// @param[in] opt_energy: energy to optimize
/// @param[out] log: final iteration log
/// @param[in] proj_params: projection parameters
/// @param[in] opt_params: optimization parameters
/// @return optimized metric
std::unique_ptr<DifferentiableConeMetric> optimize_metric_log(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    OptimizationLog& log,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::shared_ptr<OptimizationParameters> opt_params);

/// Optimize a metric on a mesh with respect to a target metric.
///
/// @param[in] m: surface mesh
/// @param[in] opt_energy: energy to optimize
/// @param[in] proj_params: projection parameters
/// @param[in] opt_params: optimization parameters
/// @return optimized metric
std::unique_ptr<DifferentiableConeMetric> optimize_metric(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    std::shared_ptr<ProjectionParameters> proj_params = nullptr,
    std::shared_ptr<OptimizationParameters> opt_params = nullptr);

} // namespace Optimization
} // namespace Penner