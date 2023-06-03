#pragma once

#include "common.hh"
#include "embedding.hh"
#include "energies.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include <filesystem>

/// @file Methods to optimize a metric satisfying angle constraints using an explicit
/// representation of the constraint manifold as a graph over a linear subspace of the
/// space of Penner coordinates. The domain of this space has |E| - |V| + d degrees of
/// freedom, where d > 0 is the number of vertices with free angles in the mesh.

namespace CurvatureMetric {

/// @brief Compute a maximal independent optimization domain for optimization from a mesh with 
/// shear subspace and conformal scaling subspace bases as well as the codomain and the initial
/// coordinates from shear and scale coordinates.
///
/// @param[in] m: mesh
/// @param[in] shear_basis_coords: shear basis coordinates of the metric
/// @param[in] scale_factors: scale factor basis coordinates of the metric
/// @param[in] shear_basis_matrix: matrix with shear basis vectors as columns
/// @param[in] scale_factor_basis_matrix: matrix with scale factor basis vectors as columns
/// @param[out] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[out] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[out] constraint_codomain_matrix: matrix with codomain basis vectors as columns
void
compute_optimization_domain(
  const Mesh<Scalar>& m,
  const VectorX& shear_basis_coords,
  const VectorX& scale_factors,
  const MatrixX& shear_basis_matrix,
  const MatrixX& scale_factor_basis_matrix,
  VectorX& domain_coords,
  MatrixX& constraint_domain_matrix,
  MatrixX& constraint_codomain_matrix
);

/// @brief Compute a metric satisfying the constraint from coordinates in the linear
/// subspace defined by the constraint domain matrix.
///
/// @param[in] m: mesh
/// @param[in] reduction_maps: maps between metric variables and per halfedge values
/// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[in] proj_params: parameters for the projection
/// @param[out] reduced_metric_coords: coordinates for a metric satisfying the constraint
void
compute_domain_coordinate_metric(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  VectorX& reduced_metric_coords);

/// @brief Compute an energy with respect to the domain coordinates.
///
/// In particular, the energy is the composition of an energy defined for Penner coordinates
/// with the map from domain coordinates to a metric on the constraint manifold.
///
/// @param[in] m: mesh
/// @param[in] reduction_maps: maps between metric variables and per halfedge values
/// @param[in] opt_energy: Penner coordinate energy to compute the gradient for
/// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[in] proj_params: parameters for the projection
/// @return energy with respect to the domain coordinates
Scalar
compute_domain_coordinate_energy(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params
);

/// @brief Compute an energy and gradient with respect to the domain coordinates.
///
/// In particular, the energy is the composition of an energy defined for Penner coordinates
/// with the map from domain coordinates to a metric on the constraint manifold.
///
/// @param[in] m: mesh
/// @param[in] reduction_maps: maps between metric variables and per halfedge values
/// @param[in] opt_energy: Penner coordinate energy to compute the gradient for
/// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[in] constraint_codomain_matrix: matrix with codomain basis vectors as columns
/// @param[in] proj_params: parameters for the projection
/// @param[in] opt_params: parameters for the optimization
/// @param[out] energy: energy with respect to the domain coordinates
/// @param[out] gradient: gradient of the energy with respect to the domain coordinates
/// @return true iff the energy computation was successful
bool
compute_domain_coordinate_energy_with_gradient(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params,
  Scalar& energy,
  VectorX& gradient
);

/// @brief Optimize a metric satisfying constraints in terms of a basis of the shear
/// space orthogonal to the space of conformal scalings.
///
/// @param[in] m: mesh 
/// @param[in] reduced_metric_target: target metric for the optimization
/// @param[in] shear_basis_coords_init: initial shear coordinate basis coefficients
/// @param[in] shear_basis_coords_init: initial scale factors
/// @param[in] shear_basis_matrix: matrix with shear coordinate basis vectors as columns
/// @param[out] reduced_metric_coords: optimized metric coordinates
/// @param[in] proj_params: parameters fro the projection to the constraint manifold
/// @param[in] opt_params: parameters for the optimization
void
optimize_shear_basis_coordinates(
  const Mesh<Scalar>& m,
  const VectorX& reduced_metric_target,
  const VectorX& shear_basis_coords_init,
  const VectorX& scale_factors_init,
  const MatrixX& shear_basis_matrix,
  VectorX& reduced_metric_coords,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params);

#ifdef PYBIND

std::tuple<
  VectorX, // domain_coords
  MatrixX, // constraint_domain_matrix
  MatrixX // constraint_codomain_matrix
>
compute_optimization_domain_pybind(
  const Mesh<Scalar>& m,
  const VectorX& shear_basis_coords,
  const VectorX& scale_factors,
  const MatrixX& shear_basis_matrix,
  const MatrixX& scale_factor_basis_matrix
);

VectorX
optimize_shear_basis_coordinates_pybind(
  const Mesh<Scalar>& m,
  const VectorX& reduced_metric_target,
  const VectorX& shear_basis_coords_init,
  const VectorX& scale_factors_init,
  const MatrixX& shear_basis_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params);

std::tuple<Scalar, VectorX>
compute_domain_coordinate_energy_with_gradient_pybind(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params
);

VectorX
compute_domain_coordinate_metric_pybind(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params);

#endif

}
