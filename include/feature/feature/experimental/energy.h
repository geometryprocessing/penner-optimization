// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "optimization/metric_optimization/energy_functor.h"

namespace Penner {
namespace Feature {


class ConeEnergy : public Optimization::EnergyFunctor
{
public:
    /// Sum of square cone error energy.
    ///
    /// @param[in] cone_metric: mesh topology
    ConeEnergy(const PennerConeMetric& cone_metric);

private:
    PennerConeMetric m_cone_metric;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};


} // namespace Feature
} // namespace Penner