#pragma once

#include "holonomy/core/common.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"

#include "optimization/metric_optimization/energy_functor.h"

namespace PennerHolonomy {

/**
 * @brief Energy for the difference between halfedge lengths. For a metric, this energy
 * is always zero, but it may be nontrivial for similarity structures.
 * 
 */
class JumpEnergy : public CurvatureMetric::EnergyFunctor
{
public:
    /**
     * @brief Construct a new Jump Energy object for a given connectivity.
     * 
     * @param m: mesh connectivity
     */
    JumpEnergy(const Mesh<Scalar>& m);

private:
    std::vector<int> m_opp;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

/**
 * @brief Squared two-norm energy for a given subset of coordinates. This can be used to
 * represent a jump energy for a similarity structure by using the scaling form coordinates.
 * 
 */
class CoordinateEnergy : public CurvatureMetric::EnergyFunctor
{
public:
    /**
     * @brief Construct a new Coordinate Energy object for a given target metric.
     * 
     * @param target_cone_metric: target metric
     * @param coordinate_indices: coordinate indices to use in the energy
     */
    CoordinateEnergy(
        const DifferentiableConeMetric& target_cone_metric,
        std::vector<int> coordinate_indices);

private:
    VectorX m_metric_target;
    std::vector<int> m_coordinate_indices;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

/**
 * @brief Squared two-norm energy for the Penner coordinates for a similarity structure after scaling
 * by the integrated scaling one-form. This energy depends on the choice of cut for the integration
 * of the scaling one-form. 
 * 
 */
class IntegratedEnergy : public CurvatureMetric::EnergyFunctor
{
public:
    /**
     * @brief Construct a new Integrated Energy object.
     * 
     * @param target_similarity_metric: target similarity metric
     */
    IntegratedEnergy(const SimilarityPennerConeMetric& target_similarity_metric);

private:
    SimilarityPennerConeMetric m_target_similarity_metric;
    MatrixX m_scaling_matrix;
    MatrixX m_expansion_matrix;
    VectorX m_metric_target;

    // Precomputed matrix products
    MatrixX Axx, Axy, Ayx, Ayy;
    VectorX bx, by;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

/**
 * @brief Sum of per-face ratios of outradius to inradius. This is a rational energy.
 * 
 * TODO: Replace with inverse of current energy to ensure well defined.
 * 
 */
class TriangleQualityEnergy : public CurvatureMetric::EnergyFunctor
{
public:
    /**
     * @brief Construct a new Triangle Quality Energy object
     * 
     * @param target_marked_metric: mesh connectivity
     */
    TriangleQualityEnergy(const MarkedPennerConeMetric& target_marked_metric);

private:
    MarkedPennerConeMetric m_target_marked_metric;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

/**
 * @brief Logarithmic triangle quality measure taking the per-face sum of squared differences
 * of log edge lengths (lij + ljk - 2lki).
 * 
 */
class LogTriangleQualityEnergy : public CurvatureMetric::EnergyFunctor
{
public:
    /**
     * @brief Construct a new Log Triangle Quality Energy object
     * 
     * @param target_marked_metric: mesh connectivity
     */
    LogTriangleQualityEnergy(const MarkedPennerConeMetric& target_marked_metric);

private:
    MarkedPennerConeMetric m_target_marked_metric;

    virtual Scalar energy(const VectorX& metric_coords) const override;
    virtual VectorX gradient(const VectorX& metric_coords) const override;
    virtual MatrixX hessian(const VectorX& metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX& metric_coords) const override;
};

} // namespace PennerHolonomy
