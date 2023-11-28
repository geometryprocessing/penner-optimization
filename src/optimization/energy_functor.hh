#pragma once

#include "common.hh"
#include "embedding.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "cone_metric.hh"

/// \file energy_functor.hh
///
/// Functor to compute a differentiable energy over a mesh with an intrinsic
/// metric in terms of log edge coordinates.

namespace CurvatureMetric {

/// Find the least squares best fit conformal mapping for the metric map from
/// lambdas_target to lambdas.
/// FIXME Make void with reference
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @param[out] conformal_scale_factors: best fit conformal scale factors
void
best_fit_conformal(const Mesh<Scalar>& m,
                   const VectorX& target_log_length_coords,
                   const VectorX& log_length_coords,
                   VectorX& conformal_scale_factors);

/// Find the gradient of the best fit conformal energy.
/// FIXME Make void with reference
/// FIXME Make explicit scale distortion energy following template of other
/// energies
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
///     mesh in m
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @param[out] direction: Jacobian of scale distortion energy per face with
///     respect to log_length_coords
void
scale_distortion_direction(const Mesh<Scalar>& m,
                           const VectorX& target_log_length_coords,
                           const VectorX& log_length_coords,
                           VectorX& direction);

/// Compute the Jacobian matrix of the change of coordinates from log edge
/// lengths to regular edge lengths.
///
/// @param[in] log_length_coords: log lengths for the  original mesh in m
/// @param[out] J_l: Jacobian of the change of coordinates
void
length_jacobian(const VectorX& log_length_coords, MatrixX& J_l);


/// Create matrix mapping edge indices to opposite face corners in a mesh
///
/// @param[in] m: (possibly symmetric) mesh
/// @return 3|F|x|E| matrix representing the reindexing.
MatrixX
generate_edge_to_face_he_matrix(const Mesh<Scalar>& m);

/// Compute the per vertex function given by the maximum of the per halfedge
/// function g on the mesh m among incoming halfedges. Assumes that g is
/// nonnegative.
/// FIXME Make void with reference
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] g: per halfedge function
/// @return:
VectorX
halfedge_function_to_vertices(const Mesh<Scalar>& m, const VectorX& g);

/// Compute vertices with nonflat cone angles;
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] reduction_maps: reduction maps
/// @param[out] cone_vertices: list of cone vertex indices
void compute_cone_vertices(
  const Mesh<Scalar> &m,
	const ReductionMaps& reduction_maps,
  std::vector<int>& cone_vertices);

/// Compute a vector of weights for faces adjacent to cones.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] reduction_maps: reduction maps
/// @param[in] cone_weight: weight to give cone adjacent faces
/// @param[out] face_weights: weights for faces
void
compute_cone_face_weights(
  const Mesh<Scalar> &m,
	const ReductionMaps& reduction_maps,
  Scalar cone_weight,
  std::vector<Scalar>& face_weights
);

/// @brief Given a vector of weights and a vector of values, compute the weighted 2 norm
/// as the sum of the product of the weights and squared values
///
/// @param[in] weights: per term weights
/// @param[in] values: value vector for the norm
/// @return weighted norm
Scalar
compute_weighted_norm(
  const VectorX &weights,
  const VectorX &values
);

/// @brief Compute the kronecker product of two vectors
///
/// @param[in] first_vector: first vector to multiply
/// @param[in] second_vector: second vector to multiply
/// @param[out] product_vector: output product vector
void
compute_kronecker_product(
  const VectorX &first_vector,
  const VectorX &second_vector,
  VectorX &product_vector
);

/// @brief Compute per face area weights for a mesh
///
/// @param[in] m: mesh
/// @param[in] log_edge_lengths: log edge length metric for the mesh
/// @param[out] face_area_weights: weights per face
void
compute_face_area_weights(
  const Mesh<Scalar> &m,
  const VectorX &log_edge_lengths,
  VectorX &face_area_weights
);

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
  VectorX &edge_area_weights
);

// TODO Document
class EnergyFunctor
{
public:
  EnergyFunctor(
    const DifferentiableConeMetric& m,
    const VectorX &metric_target,
    const OptimizationParameters& opt_params
  );

  Scalar compute_two_norm_energy(const VectorX& metric_coords) const;

  Scalar compute_p_norm_energy(const VectorX& metric_coords, int p) const;

  Scalar compute_surface_hencky_strain_energy(const VectorX& metric_coords) const;

  Scalar compute_scale_distortion_energy(const VectorX& metric_coords) const;

  Scalar compute_symmetric_dirichlet_energy(const VectorX& metric_coords) const;

  Scalar compute_cone_energy(const VectorX& metric_coords) const;

  Scalar energy(const VectorX& metric_coords) const;

  VectorX compute_cone_gradient(const VectorX& metric_coords) const;

  VectorX gradient(const VectorX& metric_coords) const;

  MatrixX hessian() const;

  MatrixX hessian_inverse() const;

private:
  // Reflection projection and embedding
  std::vector<int> m_proj;
  std::vector<int> m_embed;
  MatrixX m_projection;

  // Original mesh log edge lengths
  VectorX m_log_edge_lengths;

  // Target metric
  VectorX m_metric_target;
  
  // Weights
  Scalar m_mesh_area;
  VectorX m_face_area_weights;
  VectorX m_edge_area_weights;

  // Standard quadratic optimization energy matrix and inverse
  MatrixX m_quadratic_energy_matrix;
  MatrixX m_quadratic_energy_matrix_inverse;

  std::unique_ptr<DifferentiableConeMetric> m_mesh;
  std::string m_energy_choice;
  int m_lp_order;
  Scalar m_surface_hencky_strain_weight;
  Scalar m_two_norm_weight;
  Scalar m_cone_weight;
  Scalar m_bd_weight;
  OptimizationParameters m_opt_params;
};

// FIXME Rename these variables
// FIXME Ensure all pybind functions for the entire interface are in place
#ifdef PYBIND

MatrixX
length_jacobian_pybind(const VectorX& lambdas_full);

VectorX // conformal_scale_factors)
best_fit_conformal_pybind(const Mesh<Scalar>& m,
                   const VectorX& target_log_length_coords,
                   const VectorX& log_length_coords);
                   
#endif

}
