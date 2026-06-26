// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "util/embedding.h"
#include "util/linear_algebra.h"
#include "metric/cone_metric.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"


/**
 * @brief Class for differentiable distortion measures supporting evaluation, gradients, and Hessians
 * 
 * Energies include:
 *   - Log length: ||l - l0||_2^2
 *   - Log Scale: ||u||_2^2, with u the best fit conformal scale factors
 *   - Symmetric Dirichlet
 *   - Quadratic Symmetric Dirichlet Approximation
 *   - Regularized Quadratic Sym. Dir. Approx.
 * 
 */

namespace Penner {
namespace Optimization {

/// Energies available for optimization
enum class EnergyChoice {
    log_length, // 2-norm of the metric coordinates
    log_scale, // scaling energy of best fit scale factors
    quadratic_sym_dirichlet, // quadratic approximation to symmetric Dirichlet
    sym_dirichlet, // symmetric Dirichlet
    p_norm // 4-norm of the metric coordinates
};

/// Functor to compute a differentiable energy over a mesh with an intrinsic
/// metric in terms of log edge or Penner coordinates.
///
/// Must support energy and gradient computation in terms of the coordinate vector,
/// and optionally supports hessian and hessian inverse methods
class EnergyFunctor
{
public:
    virtual ~EnergyFunctor() = default;

    /// Compute the current energy of the cone metric
    ///
    /// @param[in] cone_metric: mesh with metric
    /// @return current metric energy
    Scalar energy(const DifferentiableConeMetric& cone_metric) const
    {
        return energy(cone_metric.get_reduced_metric_coordinates());
    }

    /// Compute the gradient of the current energy of the cone metric with respect to the
    /// reduced metric coordinates
    ///
    /// @param[in] cone_metric: mesh with metric
    /// @return current metric energy gradient
    VectorX gradient(const DifferentiableConeMetric& cone_metric) const
    {
        return gradient(cone_metric.get_reduced_metric_coordinates());
    }

    /// Compute the hessian of the current energy of the cone metric with respect to the
    /// reduced metric coordinates
    ///
    /// Throws a runtime error if no Hessian is supported
    ///
    /// @param[in] cone_metric: mesh with metric
    /// @return current metric energy hessian
    MatrixX hessian(const DifferentiableConeMetric& cone_metric) const
    {
        return hessian(cone_metric.get_reduced_metric_coordinates());
    }

    /// Compute the inverse of the hessian of the current energy of the cone metric with respect
    /// to the reduced metric coordinates
    ///
    /// Throws a runtime error if no inverse Hessian is supported
    ///
    /// @param[in] cone_metric: mesh with metric
    /// @return current metric energy hessian inverse
    MatrixX hessian_inverse(const DifferentiableConeMetric& cone_metric) const
    {
        return hessian_inverse(cone_metric.get_reduced_metric_coordinates());
    }

private:
    virtual Scalar energy(const VectorX& metric_coords) const = 0;

    virtual VectorX gradient(const VectorX& metric_coords) const = 0;

    virtual MatrixX hessian(const VectorX& metric_coords) const
    {
        throw std::runtime_error("No hessian defined");
        return id_matrix(metric_coords.size());
    }

    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const
    {
        throw std::runtime_error("No hessian defined");
        return id_matrix(metric_coords.size());
    }
};

/// Log length energy simply given by the p-norm of the current reduced coordinates relative to
/// a target metric
///
/// To avoid unnecessary roots, the pth power of the p-norm is used
///
/// Supports hessian and hessian inverse
class LogLengthEnergy : public EnergyFunctor
{
public:
    /// Construct a log length energy functor
    ///
    /// @param[in] target_cone_metric: mesh with target metric
    /// @param[in] order: (optional) p-norm order
    LogLengthEnergy(const DifferentiableConeMetric& target_cone_metric, int order = 2);

private:
    VectorX m_metric_target;
    int m_order;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

/// Quadratic approximation to the Symmetric Dirichlet energy relative to a target metric
///
/// Supports hessian and hessian inverse
class QuadraticSymmetricDirichletEnergy : public EnergyFunctor
{
public:
    /// Construct a quadratic symmetric dirichlet energy functor
    ///
    /// Since we need to compute areas and angles for the approximation, we require both a target metric
    /// with (potentially) Penner coordinates and also a Euclidean discrete metric for the same mesh.
    ///
    /// @param[in] target_cone_metric: mesh with target metric
    /// @param[in] discrete_metric: mesh with Euclidean discrete metric for area and angle computation
    QuadraticSymmetricDirichletEnergy(
        const DifferentiableConeMetric& target_cone_metric,
        const DiscreteMetric& discrete_metric);

private:
    VectorX m_metric_target;
    MatrixX m_quadratic_energy_matrix;
    MatrixX m_quadratic_energy_matrix_inverse;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

/// Scaling energy given by the squared 2-norm of the best fit conformal scale factors (in a
/// least squares sense) mapping the current metric to a target metric
class LogScaleEnergy : public EnergyFunctor
{
public:
    /// Construct a log scale energy functor
    ///
    /// @param[in] target_cone_metric: mesh with target metric
    LogScaleEnergy(const DifferentiableConeMetric& target_cone_metric);

private:
    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    std::unique_ptr<DifferentiableConeMetric> m_target_cone_metric;
    MatrixX m_expansion_matrix;
};

/// Intrinsic Symmetric Dirichlet energy relative to a target metric
class SymmetricDirichletEnergy : public EnergyFunctor
{
public:
    /// Construct a symmetric dirichlet energy functor
    ///
    /// Since we need to compute areas and angles for the energy, we require both a target metric
    /// with (potentially) Penner coordinates and also a Euclidean discrete metric for the same mesh.
    ///
    /// @param[in] target_cone_metric: mesh with target metric
    /// @param[in] discrete_metric: mesh with Euclidean discrete metric for area and angle computation
    SymmetricDirichletEnergy(
        const DifferentiableConeMetric& target_cone_metric,
        const DiscreteMetric& discrete_metric);

private:
    std::unique_ptr<DifferentiableConeMetric> m_target_cone_metric;
    VectorX m_face_area_weights;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
};

/// Quadratic approximation to the Symmetric Dirichlet energy relative to a target metric regularized
/// by a 2-norm term weighted by the area of the target metric (and an optional additional weight)
class RegularizedQuadraticEnergy : public EnergyFunctor
{
public:
    /// Construct a regularized quadratic energy functor
    ///
    /// Since we need to compute areas and angles for the energy, we require both a target metric
    /// with (potentially) Penner coordinates and also a Euclidean discrete metric for the same mesh.
    ///
    /// @param[in] target_cone_metric: mesh with target metric
    /// @param[in] discrete_metric: mesh with Euclidean discrete metric for area and angle computation
    /// @param[in] weight: (optional) weight for the regularization term
    RegularizedQuadraticEnergy(
        const DifferentiableConeMetric& target_cone_metric,
        const DiscreteMetric& discrete_metric,
        double weight=1.);

private:
    VectorX m_metric_target;
    VectorX m_quadratic_energy_matrix;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
};

// TODO: Cone energy

} // namespace Optimization
} // namespace Penner