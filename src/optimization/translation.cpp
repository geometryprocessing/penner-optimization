#include "translation.hh"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "embedding.hh"
#include "reparametrization.hh"
#include "shear.hh"
#include <Eigen/SparseLU>

/// FIXME Do cleaning pass

namespace CurvatureMetric {

/// Generate the Lagrangian system Lx = b for the least squares solution to the halfedge
/// translations in the hyperbolic metric needed to satisfy the per halfedge shear
/// change and face zero sum condition.
///
/// The zero sum face condition is necessary to extend the edge translations
/// to a projective transformation on the entire face.
///
/// @param[in] m: mesh
/// @param[in] halfedge_shear_change: constraint for the change in shear per halfedge
/// @param[out] lagrangian_matrix: matrix L defining the lagrangian system
/// @param[out] right_hand_side: vector b defining the right hand side of the lagrangian system
void
generate_translation_lagrangian_system(const Mesh<Scalar>& m,
                                       const VectorX& halfedge_shear_change,
                                       MatrixX& lagrangian_matrix,
                                       VectorX& right_hand_side)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get number of halfedges, edges, and faces
  int n_h = m.n.size();
  int n_e = e2he.size();
  int n_f = m.h.size();

  // Initialize matrix entry list
  std::vector<T> tripletList;
  std::vector<Scalar> rhs_vec;
  tripletList.reserve(n_h + 4 * n_e + 6 * n_f);
  rhs_vec.reserve(n_h + 4 * n_e + 6 * n_f);

  // Add identity block
  for (int h = 0; h < n_h; ++h) {
    tripletList.push_back(T(h, h, 1.0));
    rhs_vec.push_back(0.0);
  }

  // Add halfedge sum constraints
  int mu_count = 0;
  for (int e = 0; e < n_e; ++e) {
    int h = e2he[e];
    int ho = m.opp[h];

    // Add edge sum block
    tripletList.push_back(T(n_h + mu_count, h, 1.0));
    tripletList.push_back(T(n_h + mu_count, ho, 1.0));

    // Add edge sum transpose block
    tripletList.push_back(T(h, n_h + mu_count, 1.0));
    tripletList.push_back(T(ho, n_h + mu_count, 1.0));

    // Add constrained sum values to RHS
    rhs_vec.push_back(halfedge_shear_change[h]);

    // Increment number of constraints by 1
    mu_count += 1;
  }

  // Add face sum constraints (leaving one out due to redundancy)
  int nu_count = 0;
  for (int f = 1; f < n_f; ++f) {
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Add face sum block
    tripletList.push_back(T(n_h + mu_count + nu_count, hij, 1.0));
    tripletList.push_back(T(n_h + mu_count + nu_count, hjk, 1.0));
    tripletList.push_back(T(n_h + mu_count + nu_count, hki, 1.0));

    // Add face sum transpose block
    tripletList.push_back(T(hij, n_h + mu_count + nu_count, 1.0));
    tripletList.push_back(T(hjk, n_h + mu_count + nu_count, 1.0));
    tripletList.push_back(T(hki, n_h + mu_count + nu_count, 1.0));

    // Add 0 to RHS
    rhs_vec.push_back(0.0);

    // Increment constraint counter by 1
    nu_count += 1;
  }

  // Build matrix
  int n_var = n_h + mu_count + nu_count;
  lagrangian_matrix.resize(n_var, n_var);
  lagrangian_matrix.reserve(tripletList.size());
  lagrangian_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

  // Build RHS
  right_hand_side.setZero(n_var);
  for (int i = 0; i < n_var; ++i) {
    right_hand_side[i] = rhs_vec[i];
  }
}

void
compute_as_symmetric_as_possible_translations(
  const Mesh<Scalar>& m,
  const VectorX& he_metric_coords,
  const VectorX& he_metric_target,
  VectorX& he_translations
) {
  // Compute the change in shear from the target to the new metric
  spdlog::trace("Computing shear change");
  VectorX he_shear_change;
  compute_shear_change(m, he_metric_coords, he_metric_target, he_shear_change);

  // Build the lagrangian for the problem
  spdlog::trace("Computing lagrangian system");
  MatrixX lagrangian_matrix;
  VectorX right_hand_side;
  generate_translation_lagrangian_system(m, he_shear_change, lagrangian_matrix, right_hand_side);

  // Compute the solution of the lagrangian
  spdlog::trace(
    "Computing solution for {}x{} system with length {} rhs",
    lagrangian_matrix.rows(),
    lagrangian_matrix.cols(),
    right_hand_side.size()
  );
  Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(lagrangian_matrix);
  VectorX lagrangian_solution = solver.solve(-right_hand_side);

  // The desired translations are at the head of the solution vector
  spdlog::trace("Extracting halfedges");
  int num_halfedges = he_shear_change.size();
  he_translations = lagrangian_solution.head(num_halfedges);
}

}