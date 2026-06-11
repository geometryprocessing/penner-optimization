// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "feature/experimental/energy.h"
#include "metric/constraint.h"

namespace Penner {
namespace Feature {

ConeEnergy::ConeEnergy(const PennerConeMetric& cone_metric)
    : m_cone_metric(cone_metric)
{}

Scalar ConeEnergy::energy(const VectorX& metric_coords) const
{
    // set metric coordinates
    auto cone_metric = m_cone_metric.set_metric_coordinates(metric_coords);

    // compute cone angle constraints
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = true;
    constraint_with_jacobian(
        *cone_metric,
        constraint,
        J_constraint,
        need_jacobian,
        only_free_vertices);

    return 0.5 * constraint.dot(constraint);
}

VectorX ConeEnergy::gradient(const VectorX& metric_coords) const
{
    // set metric coordinates
    auto cone_metric = m_cone_metric.set_metric_coordinates(metric_coords);

    // compute cone angle constraints with Jacobian
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = true;
    bool only_free_vertices = true;
    constraint_with_jacobian(
        *cone_metric,
        constraint,
        J_constraint,
        need_jacobian,
        only_free_vertices);

    return constraint.transpose() * J_constraint;
}

MatrixX ConeEnergy::hessian(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

MatrixX ConeEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

} // namespace Feature
} // namespace Penner