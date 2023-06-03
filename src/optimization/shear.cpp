#include "translation.hh"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "embedding.hh"
#include "reparametrization.hh"
#include "transitions.hh"
#include "energies.hh"
#include "projection.hh"
#include <Eigen/SparseLU>

/// FIXME Do cleaning pass

namespace CurvatureMetric {

bool
validate_spanning_tree(
  const Mesh<Scalar> &m,
  const std::vector<int> &spanning_tree_edges,
  const std::vector<int> &spanning_tree_edge_from_vertex,
  const std::vector<int> &spanning_tree_edge_to_vertex,
  const std::vector<int> &spanning_tree_vertex_in_edge
) {
  int num_vertices = m.n_ind_vertices();
  int num_edges = spanning_tree_edge_to_vertex.size();

  // Check sizes
  if ( static_cast<int>(spanning_tree_edges.size() + 1) != num_vertices ) return false;
  if ( static_cast<int>(spanning_tree_edge_from_vertex.size() + 1) != num_vertices ) return false;
  if ( static_cast<int>(spanning_tree_edge_to_vertex.size() + 1) != num_vertices ) return false;
  if ( static_cast<int>(spanning_tree_vertex_in_edge.size()) != num_vertices ) return false;

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  std::vector<bool> vertex_is_covered(num_vertices, false);
  for (int i = 0; i < num_edges; ++i)
  {
    int E = spanning_tree_edges[i];
    int h = e2he[embed[E]];
    int from_vertex = spanning_tree_edge_from_vertex[i];
    int to_vertex = spanning_tree_edge_to_vertex[i];

    // Check that the edge has correct connectivity
    if ((m.v_rep[m.to[h]] != from_vertex) && (m.v_rep[m.to[h]] != to_vertex))
    {
      spdlog::error("Spanning tree edge {} has incorrect topology", E);
      return false;
    }
    if ((m.v_rep[m.to[m.opp[h]]] != from_vertex) && (m.v_rep[m.to[m.opp[h]]] != to_vertex))
    {
      spdlog::error("Spanning tree edge {} has incorrect topology", E);
      return false;
    }

    // Check that all vertices are covered
    vertex_is_covered[from_vertex] = true;
    vertex_is_covered[to_vertex] = true;
  }

  // Check that all vertices were covered
  for (size_t i = 0; i < vertex_is_covered.size(); ++i)
  {
    if (!vertex_is_covered[i])
    {
      spdlog::error("Vertex {} is not covered", i);
      return false;
    }
  }

  return true;
}

// Build a spanning tree for a given mesh so that it includes two other edges of a face
// containing a given halfedge
void
compute_spanning_tree(
  const Mesh<Scalar> &m,
  std::vector<int> &spanning_tree_edges,
  std::vector<int> &spanning_tree_edge_from_vertex,
  std::vector<int> &spanning_tree_edge_to_vertex,
  std::vector<int> &spanning_tree_vertex_in_edge,
  int start_halfedge=0
) {
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Initialize an array to keep track of vertices (in the embedded mesh)
  int num_vertices = m.n_ind_vertices();
  std::vector<bool> is_found_vertex(num_vertices, false);

  // Initialize output data structures
  std::deque<int> vertices_to_process(0);
  spanning_tree_edges.reserve(num_vertices);
  spanning_tree_edge_from_vertex.reserve(num_vertices);
  spanning_tree_edge_to_vertex.reserve(num_vertices);
  spanning_tree_vertex_in_edge = std::vector<int>(num_vertices, -1);

  // Initialize the stack of vertices to process with two edges of the first face
  int hij = start_halfedge;
  int hjk = m.n[hij];
  int hki = m.n[hjk];
  int Ejk = proj[he2e[hjk]];
  int Eki = proj[he2e[hki]];
  int vi = m.to[hki];
  int vj = m.to[hij];
  int vk = m.to[hjk];
  vertices_to_process.push_back(vi);
  vertices_to_process.push_back(vj);
  vertices_to_process.push_back(vk);
  is_found_vertex[vi] = true;
  is_found_vertex[vj] = true;
  is_found_vertex[vk] = true;

  // Add edge jk explicitly
  spanning_tree_edges.push_back(Ejk);
  spanning_tree_edge_from_vertex.push_back(vj);
  spanning_tree_edge_to_vertex.push_back(vk);
  spanning_tree_vertex_in_edge[vj] = 0;

  // Add edge ki explicitly
  spanning_tree_edges.push_back(Eki);
  spanning_tree_edge_from_vertex.push_back(vk);
  spanning_tree_edge_to_vertex.push_back(vi);
  spanning_tree_vertex_in_edge[vk] = 1;

  // Perform breadth first search
  while (!vertices_to_process.empty())
  {
    // Get the next vertex to process
    int current_vertex = vertices_to_process.back();
    vertices_to_process.pop_back();

    // Iterate over the vertex circulator via halfedges
    int h_start = m.out[current_vertex];
    int h = h_start;
    do
    {
      // Get the vertex in the one ring at the tip of the halfedge
      int one_ring_vertex = m.v_rep[m.to[h]];
      int e = he2e[h];
      int E = proj[e];

      // Check if the edge is in the original mesh and the tip vertex hasn't been processed yet
      if ((is_embedded_edge(proj, embed, e)) && (!is_found_vertex[one_ring_vertex]))
      {
        // Add the edge to the spanning tree
        spanning_tree_vertex_in_edge[one_ring_vertex] = spanning_tree_edges.size();
        spanning_tree_edge_from_vertex.push_back(current_vertex);
        spanning_tree_edge_to_vertex.push_back(one_ring_vertex);
        spanning_tree_edges.push_back(E);

        // Mark the vertex as found and add it to the vertices to process
        vertices_to_process.push_back(one_ring_vertex);
        is_found_vertex[one_ring_vertex] = true;
      }

      // Progress to the next halfedge in the vertex circulator
      h = m.n[m.opp[h]];
    }
    while (h != h_start);
  }

  assert( validate_spanning_tree(m, spanning_tree_edges, spanning_tree_edge_from_vertex, spanning_tree_edge_to_vertex, spanning_tree_vertex_in_edge) );
}

void
compute_independent_to_full_edge_shear_matrix(
  const Mesh<Scalar> &m,
  const std::vector<int> &independent_edges,
  MatrixX &shear_matrix
) {
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Build map from independent edges to edge basis vectors
  int num_edges = e2he.size();
  int num_independent_edges = independent_edges.size();
  std::vector<T> tripletList;
  tripletList.reserve(8 * num_independent_edges);
  for (int index = 0; index < num_independent_edges; ++index)
  {
    // Get halfedge for the embedded edge
    int E = independent_edges[index];
    int e = embed[E];
    int hij = e2he[e];

    // Get triangles around the edge
    int hji = m.opp[hij];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hil = m.n[hji];
    int hlj = m.n[hil];

    // Add shear vector for the embedded edge
    tripletList.push_back(T(he2e[hjk], index, 1.0));
    tripletList.push_back(T(he2e[hki], index, -1.0));
    tripletList.push_back(T(he2e[hil], index, 1.0));
    tripletList.push_back(T(he2e[hlj], index, -1.0));

    // Add shear vector for the reflected embedded edge
    // Note that the signs are inverted due to the inversion of the next relation 
    if (m.R[hij] > 0)
    {
      tripletList.push_back(T(he2e[m.R[hjk]], index, -1.0));
      tripletList.push_back(T(he2e[m.R[hki]], index, 1.0));
      tripletList.push_back(T(he2e[m.R[hil]], index, -1.0));
      tripletList.push_back(T(he2e[m.R[hlj]], index, 1.0));
    }
  }

  // Build the matrix
  shear_matrix.resize(num_edges, num_independent_edges);
  shear_matrix.reserve(8 * num_independent_edges);
  shear_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

void
compute_shear_dual_basis(
  const Mesh<Scalar> &m,
  MatrixX &shear_basis_matrix,
  std::vector<int>& independent_edges
) {
  spdlog::debug("Computing shear dual coordinate basis");

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);
  
  // Compute the spanning tree of the mesh
  std::vector<int> spanning_tree_edges;
  std::vector<int> spanning_tree_edge_from_vertex;
  std::vector<int> spanning_tree_edge_to_vertex;
  std::vector<int> spanning_tree_vertex_in_edge;
  int start_halfedge = 0;
  compute_spanning_tree(m, spanning_tree_edges, spanning_tree_edge_from_vertex, spanning_tree_edge_to_vertex, spanning_tree_vertex_in_edge, 0);
  spdlog::debug("Spanning tree of size {} found", spanning_tree_edges.size());

  // Get the complement of the spanning tree and remove one element to get an independent set
  int num_edges = embed.size();
  std::vector<int> dependent_edges = spanning_tree_edges;
  dependent_edges.push_back(proj[he2e[start_halfedge]]);
  index_vector_complement(dependent_edges, num_edges, independent_edges);

  // Compute matrix of basis vectors
  compute_independent_to_full_edge_shear_matrix(m, independent_edges, shear_basis_matrix);
}

void
compute_shear_coordinate_basis(
  const Mesh<Scalar> &m,
  MatrixX &shear_basis_matrix,
  std::vector<int>& independent_edges
) {
	// Compute shear dual basis and independent edges
  MatrixX shear_dual_basis_matrix;
	compute_shear_dual_basis(m, shear_dual_basis_matrix, independent_edges);

  // Get the (invertible) inner product matrix for the shear basis matrix
	MatrixX inner_product_matrix = shear_dual_basis_matrix.transpose() * shear_dual_basis_matrix;
  spdlog::trace("Shear dual basis matrix is\n{}", shear_dual_basis_matrix);
  spdlog::trace("Inner product matrix is {}", inner_product_matrix);
  SPDLOG_TRACE("Condition number is {}", compute_condition_number(inner_product_matrix));

  // Solve for the shear coordiante basis
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor> rhs = shear_dual_basis_matrix.transpose();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(inner_product_matrix);
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor> solution = solver.solve(rhs);
  shear_basis_matrix = solution.transpose();
  assert( solver.info() == Eigen::Success );
}

// Method to validate shear coordinates
bool
validate_shear_basis_coordinates(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
	const VectorX& shear_dual_coords,
	const VectorX& scale_factors,
	const MatrixX& shear_basis_matrix
) {
  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

	// Get conformal scaling matrix
	MatrixX scaling_matrix = conformal_scaling_matrix(m);

	// Reconstruct the metric from the shear and scale coordinates
	VectorX shear_metric = shear_basis_matrix * shear_dual_coords;
	VectorX reconstructed_metric = shear_metric + scaling_matrix * scale_factors;
  VectorX reduced_shear_metric, reduced_reconstructed_metric;
  reduce_symmetric_function(embed, shear_metric, reduced_shear_metric);
  reduce_symmetric_function(embed, reconstructed_metric, reduced_reconstructed_metric);
  SPDLOG_TRACE("Shear metric is {}", reduced_shear_metric);
  SPDLOG_TRACE("Reconstructed metric is {}", reduced_reconstructed_metric);
  SPDLOG_TRACE("Original metric is {}", reduced_metric_coords);

	// Check if the reconstructed metric is the same as the original metirc
	if (!vector_equal(reduced_reconstructed_metric, reduced_metric_coords))
	{
		spdlog::error("Reconstructed metric and original metric differ");
		return false;
	}

	return true;
}

void
compute_shear_basis_coordinates(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
	const MatrixX& shear_basis_matrix,
	VectorX& shear_coords,
	VectorX& scale_factors
) {
  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Expand reduced metric
  VectorX metric_coords;
  expand_reduced_function(proj, reduced_metric_coords, metric_coords);
	
	// Expand reduced coordinates to halfedge coordinates
	//int num_halfedges = he2e.size();
	//VectorX he_metric_coords(num_halfedges);
	//for (int h = 0; h < num_halfedges; ++h)
	//{
	//	he_metric_coords[h] = metric_coords[he2e[h]];
	//}

	// Compute the full shear
  // FIXME
	//VectorX he_shear = compute_shear(m, he_metric_coords);

	// Compute the reduced set of independent shear coordinates
	//int num_independent_edges = independent_edges.size();
	//VectorX shear_coords(num_independent_edges);
	//for (int i = 0; i < num_independent_edges; ++i)
	//{
	//	int E = independent_edges[i];
	//	int e = embed[E];
	//	int h = e2he[e];
	//	shear_coords[i] = he_shear[h];
  //}

  // Get the (invertible) inner product matrix for the shear basis matrix
	MatrixX inner_product_matrix = shear_basis_matrix.transpose() * shear_basis_matrix;
  spdlog::trace("Shear basis matrix is\n{}", shear_basis_matrix);
  spdlog::trace("Inner product matrix is {}", inner_product_matrix);
  SPDLOG_TRACE("Condition number is {}", compute_condition_number(inner_product_matrix));

  // Solve for the shear coordiantes
  VectorX rhs = shear_basis_matrix.transpose() * metric_coords;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(inner_product_matrix);
  shear_coords = solver.solve(rhs);
  assert( solver.info() == Eigen::Success );
  assert( vector_equal( shear_coords, solver.solve(inner_product_matrix * shear_coords)) );
  SPDLOG_TRACE("Shear coordinates in range [{}, {}]", shear_coords.minCoeff(), shear_coords.maxCoeff());

	// Get the corresponding scale factors 
	VectorX shear_space_metric = shear_basis_matrix * shear_coords;
	scale_factors = best_fit_conformal(m, shear_space_metric, metric_coords);

	// Validate the result
	assert( validate_shear_basis_coordinates(m, reduced_metric_coords, shear_coords, scale_factors, shear_basis_matrix) );
}

VectorX
compute_shear(const Mesh<Scalar>& m, const VectorX& he_metric_coords)
{
  VectorX he_shear_coords(he_metric_coords.size());
  int num_halfedges = he_metric_coords.size();
  for (int h = 0; h < num_halfedges; ++h) {
    int ho = m.opp[h];
    Scalar lljk = he_metric_coords[m.n[h]];
    Scalar llki = he_metric_coords[m.n[m.n[h]]];
    Scalar llil = he_metric_coords[m.n[ho]];
    Scalar lllj = he_metric_coords[m.n[m.n[ho]]];
    he_shear_coords[h] = (lljk + llil - llki - lllj) / 2.0;
  }

  return he_shear_coords;
}

void
compute_shear_change(
  const Mesh<Scalar>& m,
  const VectorX& he_metric_coords,
  const VectorX& he_metric_target,
  VectorX& he_shear_change
) {
  VectorX he_shear_coords = compute_shear(m, he_metric_coords);
  VectorX he_shear_target = compute_shear(m, he_metric_target);
  he_shear_change = he_shear_coords - he_shear_target;
}

#ifdef PYBIND

std::tuple<
  MatrixX, // shear_basis_matrix
  std::vector<int> // independent_edges
>
compute_shear_dual_basis_pybind(
  const Mesh<Scalar> &m
) {
  MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
	compute_shear_dual_basis(m, shear_basis_matrix, independent_edges);
  return std::make_tuple(shear_basis_matrix, independent_edges);
}

std::tuple<
  MatrixX, // shear_basis_matrix
  std::vector<int> // independent_edges
>
compute_shear_coordinate_basis_pybind(
  const Mesh<Scalar> &m
) {
  MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
	compute_shear_coordinate_basis(m, shear_basis_matrix, independent_edges);
  return std::make_tuple(shear_basis_matrix, independent_edges);
}

std::tuple<
	VectorX, // shear_coords
	VectorX // scale_factors
>
compute_shear_basis_coordinates_pybind(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
	const MatrixX& shear_basis_matrix
) {
	VectorX shear_coords;
	VectorX scale_factors;
	compute_shear_basis_coordinates(m, reduced_metric_coords, shear_basis_matrix, shear_coords, scale_factors);
  return std::make_tuple(shear_coords, scale_factors);
}

VectorX
compute_shear_change_pybind(
  const Mesh<Scalar>& m,
  const VectorX& he_metric_coords,
  const VectorX& he_metric_target
) {
  VectorX he_shear_change;
  compute_shear_change(m, he_metric_coords, he_metric_target, he_shear_change);
  return he_shear_change;
}

#endif


}