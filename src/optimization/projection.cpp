#include "projection.hh"

#include <stack>
#include <map>
#include "constraint.hh"
#include "embedding.hh"
#include "globals.hh"
#include "delaunay.hh"
#include <igl/Timer.h>

/// FIXME Do cleaning pass

namespace CurvatureMetric {


MatrixX
conformal_scaling_matrix(const Mesh<Scalar>& m)
{
  // Create projection matrix
  int num_halfedges = m.n_halfedges();
  std::vector<T> tripletList;
  tripletList.reserve(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    int v0 = m.v_rep[m.to[h]];
    int v1 = m.v_rep[m.to[m.opp[h]]];
    tripletList.push_back(T(h, v0, 1.0));
    tripletList.push_back(T(h, v1, 1.0));
  }
  MatrixX B;
  B.resize(num_halfedges, m.n_ind_vertices());
  B.reserve(tripletList.size());
  B.setFromTriplets(tripletList.begin(), tripletList.end());

  return B;
}

std::tuple<std::vector<int>, SolveStats<Scalar>>
compute_constraint_scale_factors(
  const DifferentiableConeMetric &cone_metric,
  VectorX& u,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params)
{
  if (proj_params == nullptr)
    proj_params = std::make_shared<ProjectionParameters>();
  if (opt_params == nullptr)
    opt_params = std::make_shared<OptimizationParameters>();

  // Create parameters for conformal method using restricted set of projection
  // parameters
  AlgorithmParameters alg_params;
  LineSearchParameters ls_params;
  StatsParameters stats_params;
  alg_params.initial_ptolemy = proj_params->initial_ptolemy;
  alg_params.max_itr = proj_params->max_itr;
  alg_params.error_eps = double(proj_params->error_eps);
  alg_params.use_edge_flips = proj_params->use_edge_flips;
  ls_params.bound_norm_thres = double(proj_params->bound_norm_thres);
  ls_params.do_reduction = proj_params->do_reduction;
  std::string output_dir = opt_params->output_dir;
  if (!output_dir.empty()) {
    stats_params.error_log = true;
    stats_params.flip_count = true;
    stats_params.output_dir = output_dir;
  }

  // Run conformal method
  OverlayMesh<Scalar> mo(cone_metric);
  std::vector<int> pt_fids;
  std::vector<Eigen::Matrix<Scalar, 3, 1>> pt_bcs;
  auto conformal_out = ConformalIdealDelaunay<Scalar>::FindConformalMetric(
    mo, u, pt_fids, pt_bcs, alg_params, ls_params, stats_params);

  // Update u with conformal output
  u = std::get<0>(conformal_out);
  std::vector<int> flip_seq = std::get<1>(conformal_out);
  SolveStats<Scalar> solve_stats = std::get<2>(conformal_out);

  return std::make_tuple(flip_seq, solve_stats);
}

VectorX
compute_constraint_scale_factors(
  const DifferentiableConeMetric &cone_metric,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params)
{
  VectorX u;
  u.setZero(cone_metric.n_ind_vertices());
  compute_constraint_scale_factors(cone_metric, u, proj_params, opt_params);

  return u;
}

//bool
//convert_penner_coordinates_to_log_edge_lengths(const Mesh<Scalar>& m,
//                                               const VectorX& lambdas_full,
//                                               const VectorX& log_edge_lengths)
//{
//  // Make mesh Delaunay with (known) Ptolemy flips
//  Mesh<Scalar> m_del;
//  VectorX lambdas_full_del;
//  MatrixX J_del;
//  std::vector<int> flip_seq;
//  bool need_jacobian = false;
//  make_delaunay_with_jacobian(
//    m, lambdas_full, m_del, lambdas_full_del, J_del, flip_seq, need_jacobian);
//
//  // Copy lengths to m_del
//
//  // TODO Finish implementing
//  // Undo Ptolemy flips with Euclidean flips if possible, or return false if not
//  for (auto it = flip_seq.rbegin(); it != flip_seq.rend(); ++it) {
//  }
//}

VectorX
project_descent_direction(const VectorX& descent_direction,
                          const VectorX& constraint,
                          const MatrixX& J_constraint)
{
  // Solve for correction vector mu
  MatrixX L = J_constraint * J_constraint.transpose();
  VectorX w = -(J_constraint * descent_direction + constraint);
  igl::Timer timer;
  timer.start();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(L);
  VectorX mu = solver.solve(w);
  double time = timer.getElapsedTime();
  spdlog::info("Direction projection solve took {} s", time);
  SPDLOG_INFO("Correction mu has norm {}", mu.norm());

  // Compute lambdas line search direction
  return descent_direction + (J_constraint.transpose() * mu);
}

// Generate a sparse identity matrix of given dimension
void
generate_identity_matrix(size_t dimension,
                         MatrixX& identity)
{
  // Build identity triplets
  std::vector<Eigen::Triplet<Scalar>> entries(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    entries[i] = Eigen::Triplet<Scalar>(i, i, 1.0);
  }

  // Set hessian from the triplets
  identity.resize(dimension, dimension);
  identity.setFromTriplets(entries.begin(), entries.end());
}

MatrixX
compute_descent_direction_projection_matrix(const MatrixX& J_constraint)
{
  // Solve for correction matrix
  MatrixX L = J_constraint * J_constraint.transpose();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(L);
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor> rhs = -J_constraint;
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor> temp = solver.solve(rhs);
  MatrixX I;
  generate_identity_matrix(J_constraint.cols(), I);
  MatrixX M = J_constraint.transpose() * temp;

  // Compute lambdas line search direction projection
  return I + M;
}


// FIXME Maybe remove, and copy new code if keep
//VectorX
//conjugate_gradient_dir(const Mesh<Scalar>& m,
//                       const VectorX& lambdas_init,
//                       const VectorX& lambdas_k,
//                       const VectorX& Th_hat)
//{
  //  int n_halfedges = m.n.size();
  //  int n_edges = n_halfedges / 2;
  //  int n_vertices = Th_hat.size();
  //  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  //
  //  // Get full lambdas array from reduced input
  //  VectorX lambdas_full_init;
  //  VectorX lambdas_full_k;
  //  expand_reduced_function(m, lambdas_init, lambdas_full_init);
  //  expand_reduced_function(m, lambdas_k, lambdas_full_k);
  //
  //  VectorX F_k;
  //  MatrixX J_F_k;
  //  constraint_with_jacobian(m, lambdas_full_k, F_k, J_F_k);
  //
  //  // Find a vertex on the boundary if there is a reflection map
  //  int v_rem = 0;
  //  for (int i = 0; i < m.out.size(); ++i) {
  //    if (m.R[i] == 0)
  //      break;
  //    if (m.to[m.R[m.out[i]]] == i) {
  //      v_rem = i;
  //      break;
  //    }
  //  }
  //  std::cout << "Removing vertex " << v_rem << std::endl;
  //  v_rem = 0;
  //
  //  VectorX F_k_red(n_vertices - 1);
  //  MatrixX J_F_k_red;
  //  // Remove the an angle constraint as it is redundant due to Gauss-Bonnet
  //  for (int i = 0; i < n_vertices - 1; ++i) {
  //    F_k_red[i] = F_k[i];
  //  }
  //  J_F_k_red = J_F_k.topRows(n_vertices - 1);
  //
  //  // FIXME Might be wrong number of constraints
  //  if (m.type[0] == 0) {
  //    if (v_rem < n_vertices - 1) {
  //      F_k_red[v_rem] = F_k[n_vertices - 1];
  //      J_F_k_red.row(v_rem) = J_F_k.row(n_vertices - 1);
  //    }
  //  }
  //
  //  // Reduce Jacobian to edges from original mesh
  //  MatrixX P;
  //  build_refl_matrix(m, P);
  //  J_F_k_red = J_F_k_red * P;
  //
  //  // Get edge maps
  //  std::vector<int> he2e;
  //  std::vector<int> e2he;
  //  build_edge_maps(m, he2e, e2he);
  //
  //  // Get energy gradient
  //  VectorX g_k(n_edges);
  //  for (int e = 0; e < n_edges; ++e) {
  //    g_k[e] = lambdas_k[e] - lambdas_init[e];
  //  }
  //
  //  // Solve for correction vector mu
  //  MatrixX L_k = J_F_k_red * J_F_k_red.transpose();
  //  VectorX v_k = J_F_k_red * g_k - F_k_red;
  //  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  //  solver.compute(L_k);
  //  VectorX mu_k = -solver.solve(v_k);
  //
  //  // FIXME
  //  Scalar max_v_k = 0;
  //  for (int i = 0; i < n_vertices - 1; ++i) {
  //    if (i != v_rem) {
  //      max_v_k = std::max(abs(v_k[i] - v_k[m.to[m.R[m.out[i]]]]), max_v_k);
  //    }
  //  }
  //  std::cerr << "Max v_k err: " << max_v_k << std::endl;
  //
  //  // Compute lambdas line search direction
  //  VectorX delta_lambdas_vec_k = -g_k - (J_F_k_red.transpose() * mu_k);
  //
  //  VectorX delta_lambdas_k(lambdas_k.size());
  //  for (int e = 0; e < lambdas_k.size(); ++e) {
  //    delta_lambdas_k[e] = delta_lambdas_vec_k[e];
  //  }
  //
  //  return delta_lambdas_k;
//}

// FIXME: DEPRECATED
// Compute the projected gradient descent line search direction for the mesh m
// with embedded log edge lengths lambdas and target embedded log edge lengths
// lambdas_target. This direction is the inital direction g projected onto the
// tangent space of the constraint manifold. Assumes that the lambdas are
// already on the constraint manifold and that F and J_F are the constraint
// function and Jacobian.
//
// param[in] m: (possibly symmetric) mesh
// param[in] lambdas_target: target log lengths for the embedded original mesh
// in m param[in] lambdas: log lengths for the embedded original mesh in m
// paramiin] g: direction to project onto constraint manifold
// param[in] F: constraint function values for lambdas
// param[in] J_del: Jacobian of the constraint value function for lambdas
// return: line search direction for lambdas.
//VectorX
//line_search_direction(const Mesh<Scalar>& m,
//                      const VectorX& lambdas_target,
//                      const VectorX& lambdas,
//                      const VectorX& g,
//                      const VectorX& F_red,
//                      const MatrixX& J_F_red)
//{
  // Solve for correction vector mu
  //  MatrixX L = J_F_red * J_F_red.transpose();
  //  VectorX w = J_F_red * g - F_red;
  //  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  //  solver.compute(L);
  //  VectorX mu = -solver.solve(w);
  //
  //  // Compute lambdas line search direction
  //  VectorX delta_lambdas_vec = -g - (J_F_red.transpose() * mu);
  //
  //  // Convert direction to standard library vector
  //  VectorX delta_lambdas(lambdas.size());
  //  for (int E = 0; E < lambdas.size(); ++E) {
  //    delta_lambdas[E] = delta_lambdas_vec[E];
  //  }
  //
  //  return delta_lambdas;
//  return lambdas;
//}

}
