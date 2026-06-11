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
#include "optimization/metric_optimization/energy_functor.h"

namespace Penner {
namespace Optimization {

/// Generate a mesh with initial target metric coordinates for optimization
///
/// @param[in] V: initial vertices
/// @param[in] F: initial faces
/// @param[in] uv: initial metric vertices
/// @param[in] F_uv: initial metric faces
/// @param[in] Th_hat: target angles
/// @param[out] vtx_reindex: vertices reindexing for new halfedge mesh
/// @param[in] free_cones: (optional) vertex cones to leave unconstrained
/// @param[in] fix_boundary: (optional) if true, fix boundary edge lengths
/// @param[in] use_discrete_metric: (optional) if true, use log edge lengths instead of penner coordinates
/// @return differentiable mesh with metric
std::unique_ptr<DifferentiableConeMetric> generate_initial_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    std::vector<int>& vtx_reindex,
    std::vector<int> free_cones = {},
    bool fix_boundary = false,
    bool use_discrete_metric = false);

/// Generate a distortion energy for a given target mesh.
///
/// @param[in] V: initial vertices
/// @param[in] F: initial faces
/// @param[in] Th_hat: target angles
/// @param[in] target_cone_metric: target mesh
/// @param[in] energy_choice: energy type to construct
/// @return energy functor for the chosen energy and mesh
std::unique_ptr<EnergyFunctor> generate_energy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Scalar>& Th_hat,
    const DifferentiableConeMetric& target_cone_metric,
    const EnergyChoice& energy_choice);

/// Correct cone angles that are multiples of pi/60 to machine precision.
///
/// @param[in] initial_cone_angles: target angles to correct
/// @return: corrected angles
std::vector<Scalar> correct_cone_angles(const std::vector<Scalar>& initial_cone_angles);

} // namespace Optimization
} // namespace Penner