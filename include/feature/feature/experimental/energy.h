
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
    ConeEnergy(const Optimization::PennerConeMetric& cone_metric);

private:
    Optimization::PennerConeMetric m_cone_metric;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};


} // namespace Feature
} // namespace Penner