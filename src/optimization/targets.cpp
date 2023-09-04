#include "targets.hh"
#include "shear.hh"
#include "interpolation.hh"
#include "embedding.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {
	
void
compute_log_edge_lengths(const Mesh<Scalar> &m, VectorX& reduced_log_edge_lengths)
{	
	// Build edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, he2e, e2he, proj, embed);

	// Build log edge lengths
	size_t num_reduced_edges = embed.size();
	reduced_log_edge_lengths.resize(num_reduced_edges);
	for (size_t E = 0; E < num_reduced_edges; ++E)
	{
		// Get the log edge length from the halfedge length
		size_t e = embed[E];
		size_t h = e2he[e];
		reduced_log_edge_lengths[E] = 2.0 * log(m.l[h]);
	}
}

void
compute_log_edge_lengths(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat,
	VectorX& reduced_log_edge_lengths
) {	
	// Build interpolation mesh for the given mesh
	InterpolationMesh interpolation_mesh(V, F, Theta_hat);

	// Get penner coordinates
	reduced_log_edge_lengths = interpolation_mesh.get_reduced_metric_coordinates();
}

 void
compute_penner_coordinates(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat,
	VectorX& reduced_penner_coords,
	std::vector<int>& flip_sequence
) {
	// Build interpolation mesh for the given mesh
	InterpolationMesh interpolation_mesh(V, F, Theta_hat);

	// Flip to Delaunay and then undo the Euclidean flipswith Ptolemy flips
	std::vector<int> hyperbolic_flip_sequence;
	interpolation_mesh.convert_to_hyperbolic_surface(flip_sequence, hyperbolic_flip_sequence);

	// Get penner coordinates
	reduced_penner_coords = interpolation_mesh.get_reduced_metric_coordinates();
}

void
normalize_penner_coordinates(
	const VectorX& reduced_penner_coords,
	VectorX& normalized_reduced_penner_coords
) {
	// Compute correction vector
	Scalar penner_coord_sum = reduced_penner_coords.sum();
	int num_edges = reduced_penner_coords.size();
	VectorX correction = VectorX::Constant(num_edges, penner_coord_sum / num_edges);

	// Normalize Penner coordinates
	normalized_reduced_penner_coords = reduced_penner_coords - correction;
}

void
compute_shear_dual_coordinates(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat,
	VectorX& shear_dual_coords,
	VectorX& scale_factors,
	MatrixX& shear_basis_matrix,
  std::vector<int>& independent_edges,
	std::vector<int>& flip_sequence
) {
	// Compute penner coordinates and flip sequence
	VectorX reduced_penner_coords;
	compute_penner_coordinates(V, F, Theta_hat, reduced_penner_coords, flip_sequence);

	// Compute mesh
	std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
	std::vector<int> free_cones(0);
	bool fix_boundary = false;
	Mesh<Scalar> m = FV_to_double<Scalar>(
		V,
		F,
		V,
		F,
		Theta_hat,
		vtx_reindex,
		indep_vtx,
		dep_vtx,
		v_rep,
		bnd_loops,
		free_cones,
		fix_boundary
	);

	// Compute shear dual basis and the corresponding inner product matrix
	compute_shear_dual_basis(m, shear_basis_matrix, independent_edges);

	// Compute the shear dual coordinates for this basis
	compute_shear_basis_coordinates(m, reduced_penner_coords, shear_basis_matrix, shear_dual_coords, scale_factors);
}

#ifdef PYBIND

VectorX
compute_log_edge_lengths_pybind(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat
) {
	VectorX reduced_log_edge_lengths;
	compute_log_edge_lengths(V, F, Theta_hat, reduced_log_edge_lengths);
	return reduced_log_edge_lengths;
}

std::tuple<
  VectorX, // reduced_penner_coords
	std::vector<int> // flip_sequence
>
compute_penner_coordinates_pybind(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat
) {
  VectorX reduced_penner_coords;
	std::vector<int>  flip_sequence;
	compute_penner_coordinates(V, F, Theta_hat, reduced_penner_coords, flip_sequence);
	return std::make_tuple(reduced_penner_coords, flip_sequence);
}

std::tuple<
	VectorX, // shear_dual_coords
	VectorX, // scale_factors
	MatrixX, // shear_basis_matrix
  std::vector<int>, // independent_edges
	std::vector<int> // flip_sequence
>
compute_shear_dual_coordinates_pybind(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<Scalar>& Theta_hat
) {
	VectorX shear_dual_coords;
	VectorX scale_factors;
	MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
	std::vector<int> flip_sequence;
	compute_shear_dual_coordinates(
		V,
		F,
		Theta_hat,
		shear_dual_coords,
		scale_factors,
		shear_basis_matrix,
		independent_edges,
		flip_sequence
	);
	return std::make_tuple(
		shear_dual_coords,
		scale_factors,
		shear_basis_matrix,
		independent_edges,
		flip_sequence
	);
}

#endif

}
