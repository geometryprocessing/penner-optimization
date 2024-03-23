#pragma once

#include "common.hh"
#include "cone_metric.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "embedding.hh"

/// \file energies.hh
///
/// Differentiable per face energy functions with gradients for the metric
/// optimization for both halfedge and VF representations

namespace CurvatureMetric {

// ******************
// Halfedge Functions
// ******************

/// Find the gradient of the best fit conformal energy.
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @param[in] metric_coords: coordinates for the current metric
/// @return Jacobian of scale distortion energy per face with respect to the coordinates
VectorX scale_distortion_direction(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords);

/// Compute the symmetric block energy matrix for the surface Hencky strain
/// energy in terms of faces. The following correspondences for matrix indices
/// are used:
///     3f + 0: m.h[f]
///     3f + 1: m.n[m.h[f]]
///     3f + 2: m.n[m.n[m.h[f]]]
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @return 3|F|x3|F| energy matrix
MatrixX surface_hencky_strain_energy(const DifferentiableConeMetric& target_cone_metric);

/// Compute the inverse of the symmetric block energy matrix for the surface Hencky strain
/// energy in terms of faces. The following correspondences for matrix indices
/// are used:
///     3f + 0: m.h[f]
///     3f + 1: m.n[m.h[f]]
///     3f + 2: m.n[m.n[m.h[f]]]
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @return 3|F|x3|F| energy matrix inverse
MatrixX surface_hencky_strain_energy_inverse(const DifferentiableConeMetric& target_cone_metric);

/// Compute the first metric tensor invariant for the mesh m with log edge
/// lengths_full lambdas and target log edge lengths target_log_length_coords.
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @param[in] metric_coords: metric coordinates for the mesh
/// @param[out] f2J1: per face array for the first metric tensor invariant
/// @param[out] J_f2J1: Jacobian with respect to log_length_coords for the first
/// metric tensor invariant
/// @param[in] need_jacobian: create Jacobian iff true
void first_invariant(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    VectorX& f2J1,
    MatrixX& J_f2J1,
    bool need_jacobian = false);

/// Compute the square of the second metric tensor invariant for the mesh m with
/// log edge lengths log_length_coords and target log edge lengths
/// target_log_length_coords.
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @param[in] metric_coords: metric coordinates for the mesh
/// @param[out] f2J2sq: per face array for the squared second metric tensor
/// invariant
/// @param[out] J_f2J2sq: Jacobian for the squared second metric tensor
/// invariant
/// @param[in] need_jacobian: create Jacobian iff true
void second_invariant_squared(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    VectorX& f2J2sq,
    MatrixX& J_f2J2sq,
    bool need_jacobian = false);

/// Compute the metric distortion energy and gradient for the mesh m with log
/// edge lengths lambdas and target log edge lengths lambdas_target. This energy
/// is the per face metric distortion measure (sigma_1 - 1)^2 + (sigma_2 - 1)^2,
/// where sigma_i are the singular values of the face deformation
/// transformation.
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @param[in] metric_coords: metric coordinates for the mesh
/// @param[out] f2energy: metric distortion energy per face
/// @param[out] J_f2energy: Jacobian of metric distortion energy per face with
/// respect to log_length_coords
/// @param[in] need_jacobian: create Jacobian iff true
void metric_distortion_energy(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    VectorX& f2energy,
    MatrixX& J_f2energy,
    bool need_jacobian = false);

/// Compute the area distortion energy and gradient for the mesh m with log edge
/// lengths lambdas and target log edge lengths lambdas_target. This energy is
/// the per face metric distortion measure (sigma_1 * sigma_2 - 1)^2, where
/// sigma_i are the singular values of the face deformation transformation.
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @param[in] metric_coords: metric coordinates for the mesh
/// @param[out] f2energy: area distortion energy per face
/// @param[out] J_f2energy: Jacobian of area distortion energy per face with
/// respect to log_length_coords
/// @param[in] need_jacobian: create Jacobian iff true
void area_distortion_energy(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    VectorX& f2energy,
    MatrixX& J_f2energy,
    bool need_jacobian = false);

/// Compute the symmetric dirichlet energy for the mesh m with log edge lengths
/// lambdas and target log edge lengths lambdas_target. This energy is the per
/// face distortion measure sigma_1^2 + sigma_2^2 + sigma_1^(-2) + sigma_2^(-2),
/// where sigma_i
///  are the singular values of the face deformation transformation.
///
/// @param[in] target_cone_metric: target mesh with differentiable metric
/// @param[in] metric_coords: metric coordinates for the mesh
/// @param[out] f2energy: symmetric dirichlet energy per face
/// @param[out] J_f2energy: Jacobian of symmetric dirichlet energy per face with
/// respect to log_length_coords
/// @param[in] need_jacobian: create Jacobian iff true
void symmetric_dirichlet_energy(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    VectorX& f2energy,
    MatrixX& J_f2energy,
    bool need_jacobian = false);

// ************
// VF Functions
// ************

/// Compute the 3x3 energy matrix for the surface Hencky strain for a single face.
/// Length indices correspond to the opposite angle.
void triangle_surface_hencky_strain_energy(
    const std::array<Scalar, 3>& lengths,
    const std::array<Scalar, 3>& cotangents,
    Scalar face_area,
    Eigen::Matrix<Scalar, 3, 3>& face_energy_matrix);

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
VectorX surface_hencky_strain_energy_vf(
    const VectorX& area,
    const MatrixX& cot_alpha,
    const MatrixX& l,
    const MatrixX& delta_ll);

/// Compute the per face first metric tensor invariant for a VF mesh with uv embedding
/// given by the trace of the determinant.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: uv embedding vertices
/// @param[in] F_uv: uv embedding faces
/// @param[out] f2J1: map from faces to invariant values
void first_invariant(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    VectorX& f2J1);

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
void second_invariant(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    VectorX& f2J2);

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
void symmetric_dirichlet_energy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    VectorX& f2energy);

/// Compute the root mean square error of length n vector x relative to length n x0
///
/// The root mean square error is given as sqrt( ||x - x0||_2^2 / n )
///
/// @param[in] x: vector of values
/// @param[in] x0: base vector of values
/// @return root mean square error
Scalar root_mean_square_error(const VectorX& x, const VectorX& x0);

/// Compute the relative root mean square error of length n vector x relative to length n x0
///
/// The relative root mean square error is given as sqrt( ||x - x0||_2^2 / (n ||x0||_2^2) )
///
/// @param[in] x: vector of values
/// @param[in] x0: base vector of values
/// @return relative root mean square error
Scalar relative_root_mean_square_error(const VectorX& x, const VectorX& x0);

/// Compute the root mean square relative error of length n vector x relative to length n x0
///
/// The root mean square relative error is given as sqrt( sum_i ((x - x0_i) / x0_i)^2 / n )
///
/// @param[in] x: vector of values
/// @param[in] x0: base vector of values
/// @return relative root mean square error
Scalar root_mean_square_relative_error(const VectorX& x, const VectorX& x0);

#ifdef PYBIND

std::tuple<VectorX, MatrixX> first_invariant_pybind(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    bool need_jacobian);

std::tuple<VectorX, MatrixX> second_invariant_squared_pybind(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    bool need_jacobian);

std::tuple<VectorX, MatrixX> metric_distortion_energy_pybind(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    bool need_jacobian);

std::tuple<VectorX, MatrixX> area_distortion_energy_pybind(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    bool need_jacobian);

std::tuple<VectorX, MatrixX> symmetric_dirichlet_energy_pybind(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords,
    bool need_jacobian);

VectorX first_invariant_vf_pybind(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

VectorX second_invariant_vf_pybind(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

#endif

} // namespace CurvatureMetric
