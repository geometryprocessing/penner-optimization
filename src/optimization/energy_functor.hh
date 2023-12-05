#pragma once

#include "common.hh"
#include "embedding.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "cone_metric.hh"

/// \file energy_functor.hh
///
/// Functor to compute a differentiable energy over a mesh with an intrinsic
/// metric in terms of log edge coordinates.

namespace CurvatureMetric
{

  /// Find the least squares best fit conformal mapping for the metric map from
  /// the metric with given coordinates to the target metric
  ///
  /// @param[in] target_cone_metric: target mesh with differentiable metric
  /// @param[in] metric_coords: coordinates for the current metric
  /// @return best fit conformal scale factors
  VectorX
  best_fit_conformal(const DifferentiableConeMetric &target_cone_metric,
                     const VectorX &metric_coords);

  /// Find the gradient of the best fit conformal energy.
  ///
  /// @param[in] target_cone_metric: target mesh with differentiable metric
  /// @param[in] metric_coords: coordinates for the current metric
  /// @return Jacobian of scale distortion energy per face with respect to the coordinates
  VectorX
  scale_distortion_direction(const DifferentiableConeMetric &target_cone_metric, const VectorX &metric_coords);

  /// Compute the Jacobian matrix of the change of coordinates from log edge
  /// lengths to regular edge lengths.
  ///
  /// @param[in] log_length_coords: log lengths for the  original mesh in m
  /// @param[out] J_l: Jacobian of the change of coordinates
  void
  length_jacobian(const VectorX &log_length_coords, MatrixX &J_l);

  /// Create matrix mapping edge indices to opposite face corners in a mesh
  ///
  /// @param[in] m: (possibly symmetric) mesh
  /// @return 3|F|x|E| matrix representing the reindexing.
  MatrixX
  generate_edge_to_face_he_matrix(const Mesh<Scalar> &m);

  /// Compute the per vertex function given by the maximum of the per halfedge
  /// function g on the mesh m among incoming halfedges. Assumes that g is
  /// nonnegative.
  /// FIXME Make void with reference
  ///
  /// @param[in] m: (possibly symmetric) mesh
  /// @param[in] g: per halfedge function
  /// @return:
  VectorX
  halfedge_function_to_vertices(const Mesh<Scalar> &m, const VectorX &g);

  /// Compute vertices with nonflat cone angles;
  ///
  /// @param[in] m: (possibly symmetric) mesh
  /// @param[in] reduction_maps: reduction maps
  /// @param[out] cone_vertices: list of cone vertex indices
  [[deprecated]] void compute_cone_vertices(
      const Mesh<Scalar> &m,
      const ReductionMaps &reduction_maps,
      std::vector<int> &cone_vertices);

  /// Compute a vector of weights for faces adjacent to cones.
  ///
  /// @param[in] m: (possibly symmetric) mesh
  /// @param[in] reduction_maps: reduction maps
  /// @param[in] cone_weight: weight to give cone adjacent faces
  /// @param[out] face_weights: weights for faces
  [[deprecated]] void
  compute_cone_face_weights(
      const Mesh<Scalar> &m,
      const ReductionMaps &reduction_maps,
      Scalar cone_weight,
      std::vector<Scalar> &face_weights);

  /// @brief Given a vector of weights and a vector of values, compute the weighted 2 norm
  /// as the sum of the product of the weights and squared values
  ///
  /// @param[in] weights: per term weights
  /// @param[in] values: value vector for the norm
  /// @return weighted norm
  Scalar
  compute_weighted_norm(
      const VectorX &weights,
      const VectorX &values);

  /// @brief Compute per face area weights for a mesh
  ///
  /// @param[in] m: mesh
  /// @param[in] log_edge_lengths: log edge length metric for the mesh
  /// @param[out] face_area_weights: weights per face
  void
  compute_face_area_weights(
      const Mesh<Scalar> &m,
      const VectorX &log_edge_lengths,
      VectorX &face_area_weights);

  /// @brief Compute per edge weights for a mesh with a given metric as 1/3 of the areas
  /// of the two adjacent faces
  ///
  /// @param[in] m: mesh
  /// @param[in] log_edge_lengths: log edge length metric for the mesh
  /// @param[out] edge_area_weights: weights per edge
  void
  compute_edge_area_weights(
      const Mesh<Scalar> &m,
      const VectorX &log_edge_lengths,
      VectorX &edge_area_weights);

  // TODO Document
  class EnergyFunctor
  {
  public:
		virtual ~EnergyFunctor() = default;

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

    Scalar energy(const DifferentiableConeMetric& cone_metric) const
    {
      return energy(cone_metric.get_metric_coordinates());
    }

    VectorX gradient(const DifferentiableConeMetric& cone_metric) const
    {
      return gradient(cone_metric.get_metric_coordinates());
    }

    MatrixX hessian(const DifferentiableConeMetric& cone_metric) const
    {
      return hessian(cone_metric.get_metric_coordinates());
    }

    MatrixX hessian_inverse(const DifferentiableConeMetric& cone_metric) const
    {
      return hessian_inverse(cone_metric.get_metric_coordinates());
    }
  };

  class LogLengthEnergy : public EnergyFunctor
  {
  public:
    LogLengthEnergy(const DifferentiableConeMetric &target_cone_metric, int order = 2);
    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
    virtual MatrixX hessian(const VectorX &metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX &metric_coords) const override;

  private:
    VectorX m_metric_target;
    int m_order;
  };

  class QuadraticSymmetricDirichletEnergy : public EnergyFunctor
  {
  public:
    QuadraticSymmetricDirichletEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric);
    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;
    virtual MatrixX hessian(const VectorX &metric_coords) const override;
    virtual MatrixX hessian_inverse(const VectorX &metric_coords) const override;

  private:
    VectorX m_metric_target;
    MatrixX m_quadratic_energy_matrix;
    MatrixX m_quadratic_energy_matrix_inverse;
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
    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;

  private:
    std::unique_ptr<DifferentiableConeMetric> m_target_cone_metric;
    VectorX m_face_area_weights;
  };

  class RegularizedQuadraticEnergy : public EnergyFunctor
  {
  public:
    RegularizedQuadraticEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric, double weight);
    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;

  private:
    VectorX m_metric_target;
    VectorX m_quadratic_energy_matrix;
  };

  class ConeEnergy : public EnergyFunctor
  {
  public:
    ConeEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric);
    virtual Scalar energy(const VectorX &metric_coords) const override;
    virtual VectorX gradient(const VectorX &metric_coords) const override;

  private:
    VectorX m_metric_target;
    VectorX m_quadratic_energy_matrix;
  };

  // TODO WeightedQuadraticEnergy

// FIXME Rename these variables
// FIXME Ensure all pybind functions for the entire interface are in place
#ifdef PYBIND

  MatrixX
  length_jacobian_pybind(const VectorX &lambdas_full);

#endif

}
