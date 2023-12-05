#pragma once

#include "common.hh"
#include "cone_metric.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

/// @file targets.hh
///
/// Methods to construct various targets for the metric and angle constraints.


namespace CurvatureMetric {

/// Compute the log edge lengths of a mesh.
///
/// Note that these will differ from the Penner coordinates if the mesh is not
/// Delaunay.
///
/// @param[in] m: mesh
/// @param[out] reduced_log_edge_lengths: log edge lengths
void
compute_log_edge_lengths(
  const Mesh<Scalar>& m,
	VectorX& reduced_log_edge_lengths
);

/// Compute the log edge lengths of a VF mesh.
///
/// Note that these will differ from the Penner coordinates if the mesh is not
/// Delaunay.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] Theta_hat: input mesh target angles
/// @param[out] reduced_log_edge_lengths: log edge lengths
void
compute_log_edge_lengths(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat,
	VectorX& reduced_log_edge_lengths
);

/// Compute the Penner coordinates of a VF mesh and the flip sequence from the
/// initial connectivity to the Delaunay connectivity with Euclidean flips.
///
/// Note that these will differ from the log edge lengths if the mesh is not
/// Delaunay.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] Theta_hat: input mesh target angles
/// @param[out] reduced_penner_coords: Penner coordinates
/// @param[out] flip_sequence: sequence of flips
void
compute_penner_coordinates(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat,
	VectorX& reduced_penner_coords,
	std::vector<int>& flip_sequence
);

/// Normalize Penner coordinates to have sum 0 (due to scale invariance)
///
/// @param[in] reduced_penner_coords: input Penner coordinates
/// @param[out] normalized_reduced_penner_coords: Penner coordinates after normalization
void
normalize_penner_coordinates(
	const VectorX& reduced_penner_coords,
	VectorX& normalized_reduced_penner_coords
);

/// Compute the shear dual coordinates of a VF mesh and the flip sequence from the
/// initial connectivity to the Delaunay connectivity with Euclidean flips.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] Theta_hat: input mesh target angles
/// @param[out] shear_dual_coords: shear dual coordinates for the mesh
/// @param[out] scale_factors: scale factors to restore the original mesh Penner coordinates
/// @param[out] shear_basis_matrix: matrix with shear basis vectors as columns
/// @param[out] independent_edges: indices of the independent edges of the mesh used for the basis
/// @param[out] flip_sequence: sequence of flips
void
compute_shear_dual_coordinates(
	const DifferentiableConeMetric& cone_metric,
	VectorX& shear_dual_coords,
	VectorX& scale_factors,
	MatrixX& shear_basis_matrix,
  std::vector<int>& independent_edges
);

#ifdef PYBIND
VectorX
compute_log_edge_lengths_pybind(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat
);

std::tuple<
  VectorX, // reduced_penner_coords
	std::vector<int> // flip_sequence
>
compute_penner_coordinates_pybind(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat
);

#endif

}
