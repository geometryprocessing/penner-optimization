#pragma once

#include "common.hh"
#include "embedding.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

/// \file energies.hh
///
/// Differentiable per face energy functions with gradients for the metric
/// optimization for both halfedge and VF representations

namespace CurvatureMetric {

// ******************
// Halfedge Functions
// ******************

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

/// Compute the inverse of the symmetric block energy matrix for the surface Hencky strain
/// energy in terms of faces. The following correspondences for matrix indices
/// are used:
///     3f + 0: m.h[f]
///     3f + 1: m.n[m.h[f]]
///     3f + 2: m.n[m.n[m.h[f]]]
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] log_length_coords: log lengths for the original mesh in m
/// @param[out] inverse_M: 3|F|x3|F| energy inverse matrix
void
surface_hencky_strain_energy_inverse(const Mesh<Scalar>& m,
                             const VectorX& log_length_coords,
                             MatrixX& inverse_M);

// ************
// VF Functions
// ************

/// Compute the per face first metric tensor invariant for a VF mesh with uv embedding
/// given by the trace of the determinant.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: uv embedding vertices
/// @param[in] F_uv: uv embedding faces
/// @param[out] f2J1: map from faces to invariant values
void
first_invariant(
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
second_invariant(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX &f2J2
);

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
symmetric_dirichlet_energy(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX& f2energy
);

/// Compute the 3x3 energy matrix for the surface Hencky strain for a single face.
/// Length indices correspond to the opposite angle.
void
triangle_surface_hencky_strain_energy(
  const std::array<Scalar, 3> &lengths,
  const std::array<Scalar, 3> &cotangents,
  Scalar face_area,
  Eigen::Matrix<Scalar, 3, 3> &face_energy_matrix
);

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
// TODO Make this actually take in VF and compute area, cotalpha, and l as in python
// TODO MAKE void with reference
VectorX
surface_hencky_strain_energy(const VectorX& area,
                                const MatrixX& cot_alpha,
                                const MatrixX& l,
                                const MatrixX& delta_ll);

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
VectorX
surface_hencky_strain_energy_vf(const VectorX& area,
                                const MatrixX& cot_alpha,
                                const MatrixX& l,
                                const MatrixX& delta_ll);
#endif

}
