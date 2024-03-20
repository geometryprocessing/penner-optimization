#pragma once

#include "common.hh"
#include "embedding.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "cone_metric.hh"
#include "linear_algebra.hh"

/// \file energy_functor.hh
///
/// Functor to compute a differentiable energy over a mesh with an intrinsic
/// metric in terms of log edge coordinates.

namespace CurvatureMetric
{

  // TODO Document
  class EnergyFunctor
  {
  public:
		virtual ~EnergyFunctor() = default;

    Scalar energy(const DifferentiableConeMetric& cone_metric) const
    {
      return energy(cone_metric.get_reduced_metric_coordinates());
    }

    VectorX gradient(const DifferentiableConeMetric& cone_metric) const
    {
      return gradient(cone_metric.get_reduced_metric_coordinates());
    }

    MatrixX hessian(const DifferentiableConeMetric& cone_metric) const
    {
      return hessian(cone_metric.get_reduced_metric_coordinates());
    }

    MatrixX hessian_inverse(const DifferentiableConeMetric& cone_metric) const
    {
      return hessian_inverse(cone_metric.get_reduced_metric_coordinates());
    }

  private:
    virtual Scalar energy(const VectorX &metric_coords) const = 0;

    virtual VectorX gradient(const VectorX &metric_coords) const = 0;

    virtual MatrixX hessian(const VectorX &metric_coords) const
    {
      throw std::runtime_error("No hessian defined");
      return id_matrix(metric_coords.size());
    }

    virtual MatrixX hessian_inverse(const VectorX &metric_coords) const
    {
      throw std::runtime_error("No hessian defined");
      return id_matrix(metric_coords.size());
    }

  };

  class LogLengthEnergy : public EnergyFunctor
  {
  public:
    LogLengthEnergy(const DifferentiableConeMetric &target_cone_metric, int order = 2);

  private:
    VectorX m_metric_target;
    int m_order;

    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
    virtual MatrixX hessian(const VectorX &metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX &metric_coords) const override;
  };

  class QuadraticSymmetricDirichletEnergy : public EnergyFunctor
  {
  public:
    QuadraticSymmetricDirichletEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric);

  private:
    VectorX m_metric_target;
    MatrixX m_quadratic_energy_matrix;
    MatrixX m_quadratic_energy_matrix_inverse;

    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
    virtual MatrixX hessian(const VectorX &metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX &metric_coords) const override;
  };

  class LogScaleEnergy : public EnergyFunctor
  {
  public:
    LogScaleEnergy(const DifferentiableConeMetric &target_cone_metric);
    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;

  private:
    std::unique_ptr<DifferentiableConeMetric> m_target_cone_metric;
  };

  class SymmetricDirichletEnergy : public EnergyFunctor
  {
  public:
    SymmetricDirichletEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric);

  private:
    std::unique_ptr<DifferentiableConeMetric> m_target_cone_metric;
    VectorX m_face_area_weights;

    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
  };

  class RegularizedQuadraticEnergy : public EnergyFunctor
  {
  public:
    RegularizedQuadraticEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric, double weight);

  private:
    VectorX m_metric_target;
    VectorX m_quadratic_energy_matrix;

    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
  };

  class ConeEnergy : public EnergyFunctor
  {
  public:
    ConeEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric);

  private:
    VectorX m_metric_target;
    VectorX m_quadratic_energy_matrix;

    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
  };

  // TODO WeightedQuadraticEnergy


}
