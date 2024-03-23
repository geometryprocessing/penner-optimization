#pragma once

#include <filesystem>
#include "common.hh"
#include "embedding.hh"
#include "energy_functor.hh"

/// @file Methods to optimize a metric satisfying angle constraints using an explicit
/// representation of the constraint manifold as a graph over a linear subspace of the
/// space of Penner coordinates. The domain of this space has |E| - |V| + d degrees of
/// freedom, where d > 0 is the number of vertices with free angles in the mesh.

namespace CurvatureMetric {

/// @brief Compute a maximal independent optimization domain for optimization from a mesh with
/// a specified shear subspace basis.
///
/// @param[in] cone_metric: mesh with metric
/// @param[in] shear_basis_matrix: matrix with shear basis vectors as columns
/// @param[out] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[out] constraint_codomain_matrix: matrix with codomain basis vectors as columns
/// @param[out] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[out] codomain_coords: coordinates in the basis of the constraint codomain matrix
void compute_optimization_domain(
    const DifferentiableConeMetric& cone_metric,
    const MatrixX& shear_basis_matrix,
    MatrixX& constraint_domain_matrix,
    MatrixX& constraint_codomain_matrix,
    VectorX& domain_coords,
    VectorX& codomain_coords);

/// @brief Compute a metric satisfying the constraint from coordinates in the linear
/// subspace defined by the constraint domain matrix.
///
/// Initial codomain coordinates near the actual coordinates can be supplied to improve projection.
///
/// @param[in] m: mesh
/// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[in] constraint_codomain_matrix: matrix with codomain basis vectors as columns
/// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[in] init_codomain_coords: initial coordinates in the basis of the constraint domain matrix
/// @param[in] proj_params: parameters for the projection
/// @return mesh with differentiable metric determined by the domain coordinates and constraints
std::unique_ptr<DifferentiableConeMetric> compute_domain_coordinate_metric(
    const DifferentiableConeMetric& m,
    const MatrixX& constraint_domain_matrix,
    const MatrixX& constraint_codomain_matrix,
    const VectorX& domain_coords,
    const VectorX& init_codomain_coords,
    std::shared_ptr<ProjectionParameters> proj_params);

/// @brief Compute an energy with respect to the domain coordinates.
///
/// In particular, the energy is the composition of an energy defined for Penner coordinates
/// with the map from domain coordinates to a metric on the constraint manifold.
///
/// @param[in] cone_metric: mesh with metric
/// @param[in] opt_energy: Penner coordinate energy to compute the gradient for
/// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[in] constraint_codomain_matrix: matrix with codomain basis vectors as columns
/// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[in] init_codomain_coords: initial coordinates in the basis of the constraint domain matrix
/// @param[in] proj_params: parameters for the projection
/// @return energy with respect to the domain coordinates
Scalar compute_domain_coordinate_energy(
    const DifferentiableConeMetric& cone_metric,
    const EnergyFunctor& opt_energy,
    const MatrixX& constraint_domain_matrix,
    const MatrixX& constraint_codomain_matrix,
    const VectorX& domain_coords,
    const VectorX& init_codomain_coords,
    std::shared_ptr<ProjectionParameters> proj_params);

/// @brief Compute an energy and gradient with respect to the domain coordinates.
///
/// In particular, the energy is the composition of an energy defined for Penner coordinates
/// with the map from domain coordinates to a metric on the constraint manifold.
///
/// @param[in] m: mesh
/// @param[in] opt_energy: Penner coordinate energy to compute the gradient for
/// @param[in] constraint_domain_matrix: matrix with domain basis vectors as columns
/// @param[in] constraint_codomain_matrix: matrix with codomain basis vectors as columns
/// @param[in] domain_coords: coordinates in the basis of the constraint domain matrix
/// @param[in] init_codomain_coords: initial coordinates in the basis of the constraint domain matrix
/// @param[in] proj_params: parameters for the projection
/// @param[out] energy: energy with respect to the domain coordinates
/// @param[out] gradient: gradient of the energy with respect to the domain coordinates
/// @return true iff the energy computation was successful
bool compute_domain_coordinate_energy_with_gradient(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    const MatrixX& constraint_domain_matrix,
    const MatrixX& constraint_codomain_matrix,
    const VectorX& domain_coords,
    const VectorX& init_codomain_coords,
    std::shared_ptr<ProjectionParameters> proj_params,
    Scalar& energy,
    VectorX& gradient);

VectorX optimize_domain_coordinates(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    const MatrixX& constraint_domain_matrix,
    const MatrixX& constraint_codomain_matrix,
    const VectorX& init_domain_coords,
    const VectorX& init_codomain_coords,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::shared_ptr<OptimizationParameters> opt_params);

/// @brief Optimize a metric satisfying constraints in terms of a basis of the shear
/// space orthogonal to the space of conformal scalings.
///
/// @param[in] m: mesh
/// @param[in] opt_energy: energy to optimize
/// @param[in] shear_basis_matrix: matrix with shear coordinate basis vectors as columns
/// @param[in] proj_params: parameters fro the projection to the constraint manifold
/// @param[in] opt_params: parameters for the optimization
/// @return reduced_metric_coords: optimized metric coordinates
VectorX optimize_shear_basis_coordinates(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    const MatrixX& shear_basis_matrix,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::shared_ptr<OptimizationParameters> opt_params);

} // namespace CurvatureMetric
