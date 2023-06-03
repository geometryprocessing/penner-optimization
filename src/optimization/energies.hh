#pragma once

#include "common.hh"
#include "embedding.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

/// \file energies.hh
///
/// Differentiable energy functions with gradients for the metric optimization.
/// Also provides reindexing methods to go between per edge, per vertex, per
/// face and per halfedge energies.

namespace CurvatureMetric {

/// Compute the first metric tensor invariant for the mesh m with log edge
/// lengths_full lambdas and target log edge lengths target_log_length_coords.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @param[out] f2J1: per face array for the first metric tensor invariant
/// @param[out] J_f2J1: Jacobian with respect to log_length_coords for the first
/// metric tensor invariant
/// @param[in] need_jacobian: create Jacobian iff true
void
first_invariant(const Mesh<Scalar>& m,
                const VectorX& target_log_length_coords,
                const VectorX& log_length_coords,
                VectorX& f2J1,
                MatrixX& J_f2J1,
                bool need_jacobian = false);

/// Compute the square of the second metric tensor invariant for the mesh m with
/// log edge lengths log_length_coords and target log edge lengths
/// target_log_length_coords.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @param[out] f2J2sq: per face array for the squared second metric tensor
/// invariant
/// @param[out] J_f2J2sq: Jacobian for the squared second metric tensor
/// invariant
/// @param[in] need_jacobian: create Jacobian iff true
void
second_invariant_squared(const Mesh<Scalar>& m,
                         const VectorX& target_log_length_coords,
                         const VectorX& log_length_coords,
                         VectorX& f2J2sq,
                         MatrixX& J_f2J2sq,
                         bool need_jacobian = false);

/// Compute the per face first metric tensor invariant for a VF mesh with uv embedding
/// given by the trace of the determinant.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: uv embedding vertices
/// @param[in] F_uv: uv embedding faces
/// @param[out] f2J1: map from faces to invariant values
void
first_invariant_vf(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX &f2J1
);

/// Compute the per face second metric tensor invariant for a VF mesh with uv embedding
/// given by the square root of the determinant.
///
/// This energy is also the ratio of the embedded face area to the original face area.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: uv embedding vertices
/// @param[in] F_uv: uv embedding faces
/// @param[out] f2J2: map from faces to invariant values
void
second_invariant_vf(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX &f2J2
);

/// Compute the metric distortion energy and gradient for the mesh m with log
/// edge lengths lambdas and target log edge lengths lambdas_target. This energy
/// is the per face metric distortion measure (sigma_1 - 1)^2 + (sigma_2 - 1)^2,
/// where sigma_i are the singular values of the face deformation
/// transformation.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the  original mesh in m
/// @param[out] f2energy: metric distortion energy per face
/// @param[out] J_f2energy: Jacobian of metric distortion energy per face with
/// respect to log_length_coords
/// @param[in] need_jacobian: create Jacobian iff true
void
metric_distortion_energy(const Mesh<Scalar>& m,
                         const VectorX& target_log_length_coords,
                         const VectorX& log_length_coords,
                         VectorX& f2energy,
                         MatrixX& J_f2energy,
                         bool need_jacobian = false);

/// Compute the area distortion energy and gradient for the mesh m with log edge
/// lengths lambdas and target log edge lengths lambdas_target. This energy is
/// the per face metric distortion measure (sigma_1 * sigma_2 - 1)^2, where
/// sigma_i are the singular values of the face deformation transformation.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the  original mesh in m
/// @param[out] f2energy: area distortion energy per face
/// @param[out] J_f2energy: Jacobian of area distortion energy per face with
/// respect to log_length_coords
/// @param[in] need_jacobian: create Jacobian iff true
void
area_distortion_energy(const Mesh<Scalar>& m,
                       const VectorX& target_log_length_coords,
                       const VectorX& log_length_coords,
                       VectorX& f2energy,
                       MatrixX& J_f2energy,
                       bool need_jacobian = false);

/// Compute the symmetric dirichlet energy for the mesh m with log edge lengths
/// lambdas and target log edge lengths lambdas_target. This energy is the per
/// face distortion measure sigma_1^2 + sigma_2^2 + sigma_1^(-2) + sigma_2^(-2),
/// where sigma_i
///  are the singular values of the face deformation transformation.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the  original mesh in m
/// @param[out] f2energy: symmetric dirichlet energy per face
/// @param[out] J_f2energy: Jacobian of symmetric dirichlet energy per face with
/// respect to log_length_coords
/// @param[in] need_jacobian: create Jacobian iff true
void
symmetric_dirichlet_energy(const Mesh<Scalar>& m,
                           const VectorX& target_log_length_coords,
                           const VectorX& log_length_coords,
                           VectorX& f2energy,
                           MatrixX& J_f2energy,
                           bool need_jacobian = false);

/// Compute the symmetric dirichlet energy for the uv parametrization of a mesh.
///
/// This energy is the per face distortion measure defined by
/// sigma_1^2 + sigma_2^2 + sigma_1^(-2) + sigma_2^(-2),
/// where sigma_i are the singular values of the face deformation transformation.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: uv embedding vertices
/// @param[in] F_uv: uv embedding faces
/// @param[out] f2energy: map from faces to energy
void
symmetric_dirichlet_energy_vf(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX& f2energy
);

/// Find the least squares best fit conformal mapping for the metric map from
/// lambdas_target to lambdas.
/// FIXME Make void with reference
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @return: best fit conformal scale factors
VectorX
best_fit_conformal(const Mesh<Scalar>& m,
                   const VectorX& target_log_length_coords,
                   const VectorX& log_length_coords);

/// Find the gradient of the best fit conformal energy.
/// FIXME Make void with reference
/// FIXME Make explicit scale distortion energy following template of other
/// energies
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] target_log_length_coords: target log lengths for the original
/// mesh in m
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @return Jacobian of scale distortion energy per face with respect to
/// log_length_coords
VectorX
scale_distortion_direction(const Mesh<Scalar>& m,
                           const VectorX& target_log_length_coords,
                           const VectorX& log_length_coords);

/// Compute the 3x3 energy matrix for the surface Hencky strain for a single face.
/// Length indices correspond to the opposite angle.
///
/// @param[in] lengths: triangle edge lengths
/// @param[in] cotangents: triangle angle cotangents
/// @param[in] face_area: triangle face area
/// @param[out] face_energy_matrix: 3x3 energy matrix for the face
void
triangle_surface_hencky_strain_energy(
  const std::array<Scalar, 3> &lengths,
  const std::array<Scalar, 3> &cotangents,
  Scalar face_area,
  Eigen::Matrix<Scalar, 3, 3> &face_energy_matrix
);

/// Compute the symmetric block energy matrix for the surface Hencky strain
/// energy in terms of faces. The following correspondences for matrix indices
/// are used:
///     3f + 0: m.h[f]
///     3f + 1: m.n[m.h[f]]
///     3f + 2: m.n[m.n[m.h[f]]]
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @param[out] M: 3|F|x3|F| energy matrix
void
surface_hencky_strain_energy(const Mesh<Scalar>& m,
                             const VectorX& log_length_coords,
                             MatrixX& M);

// Compute the symmetric block energy matrix for the surface Hencky strain
// energy in terms of faces. The following correspondences for matrix indices
// are used:
//     3f + 0: m.h[f]
//     3f + 1: m.n[m.h[f]]
//     3f + 2: m.n[m.n[m.h[f]]]
//
// param[in] m: (possibly symmetric) mesh
// param[in] lambdas_full: log lengths for the original mesh in m
// param[out] M: energy matrix
// FIXME Determine what this does and where it is used
VectorX
surface_hencky_strain_energy_vf(const VectorX& area,
                                const MatrixX& cot_alpha,
                                const MatrixX& l,
                                const MatrixX& delta_ll);

/// Compute the Jacobian matrix of the change of coordinates from log edge
/// lengths to regular edge lengths.
///
/// @param[in] log_length_coords: log lengths for the  original mesh in m
/// @param[out] J_l: Jacobian of the change of coordinates
void
length_jacobian(const VectorX& log_length_coords, MatrixX& J_l);


/// Create matrix mapping edge indices to opposite face corners
/// FIXME Make void with reference
///
/// @param[in] m: mesh
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
    const Mesh<Scalar> &m,
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

  Mesh<Scalar> m_mesh;
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
std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
first_invariant_pybind(const Mesh<Scalar>& C,
                       const VectorX& lambdas_target_full,
                       const VectorX& lambdas_full,
                       bool need_jacobian);

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
second_invariant_squared_pybind(const Mesh<Scalar>& C,
                                const VectorX& lambdas_target_full,
                                const VectorX& lambdas_full,
                                bool need_jacobian);

VectorX
first_invariant_vf_pybind(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv
);

VectorX
second_invariant_vf_pybind(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv
);

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
metric_distortion_energy_pybind(const Mesh<Scalar>& C,
                                const VectorX& lambdas_target_full,
                                const VectorX& lambdas_full,
                                bool need_jacobian);

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
area_distortion_energy_pybind(const Mesh<Scalar>& C,
                              const VectorX& lambdas_target_full,
                              const VectorX& lambdas_full,
                              bool need_jacobian);

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
symmetric_dirichlet_energy_pybind(const Mesh<Scalar>& C,
                                  const VectorX& lambdas_target_full,
                                  const VectorX& lambdas_full,
                                  bool need_jacobian);

MatrixX
surface_hencky_strain_energy_pybind(const Mesh<Scalar>& m,
                                    const VectorX& lambdas_full);

MatrixX
length_jacobian_pybind(const VectorX& lambdas_full);
#endif

}
