// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/cone_metric.h"


namespace Penner {

// Parameters for constructing a cone metric.
struct ConeMetricParameters
{
    bool use_initial_zero = false; // use initial zero Penner coordinates
    bool use_log_length = false; // use initial log length coordinates instead of Penner
    bool remove_symmetry = false; // remove symmetry structure from doubled mesh
    bool free_interior = false; // remove interior cone constraints
    // TODO Implement for meshes without features
    bool use_free_cones = false; // use free cones instead of seamless constraints
};

/**
 * @brief Generate logarithmic edge lengths for a discrete metric
 * 
 * @param m: mesh with metric
 * @return per halfedge logarithmic edge lengths
 */
VectorX generate_log_edge_lengths(const Mesh<Scalar>& m);

/**
 * @brief Generate Penner coordinates for a discrete metric.
 * 
 * For a Delaunay metric, these are just logarithmic edge lengths. Otherwise, the
 * Penner coordinates for the connectivity are computed using a Delauany flip
 * algorithm.
 * 
 * @param m: mesh with metric
 * @return per halfedge Penner coordinates
 */
VectorX generate_penner_coordinates(const Mesh<Scalar>& m);

/**
 * @brief Convert a mesh into a discrete metric.
 * 
 * Note that the discrete metric supports differentiation, unlike the base mesh.
 * 
 * @param m: mesh with metric
 * @return mesh with differentiable metric
 */
DiscreteMetric generate_discrete_metric(const Mesh<Scalar>& m);

/**
 * @brief Generate a cone metric from a VF mesh with target cones.
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param Th_hat: per-vertex cone angles
 * @param free_cones: list of cones to leave free
 * @param cone_metric_params: (optional) parameters for the cone mesh construction
 * @return marked cone metric
 * @return vertex reindexing from the halfedge to VF vertices
 */
std::tuple<PennerConeMetric, std::vector<int>> generate_cone_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Scalar>& Th_hat,
    std::vector<int> free_cones,
    ConeMetricParameters cone_metric_params = ConeMetricParameters());

/**
 * @brief Generate a cone metric from a halfedge mesh.
 *
 * @param m: mesh with metric
 * @param cone_metric_params: (optional) parameters for the cone mesh construction
 * @return marked cone metric
 * @return vertex reindexing from the halfedge to VF vertices
 */
PennerConeMetric generate_cone_metric_from_mesh(
    const Mesh<Scalar>& m,
    ConeMetricParameters cone_metric_params = ConeMetricParameters());


} // namespace Penner