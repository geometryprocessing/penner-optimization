#include "energy_functor.hh"

#include "area.hh"
#include "constraint.hh"
#include "energies.hh"
#include "projection.hh"
#include "targets.hh"
#include <igl/doublearea.h>
#include <igl/cotmatrix_entries.h>
#include <igl/edge_lengths.h>

/// FIXME Do cleaning pass

namespace CurvatureMetric
{

  VectorX
  best_fit_conformal(const DifferentiableConeMetric &target_cone_metric,
                     const VectorX &metric_coords)
  {
    // Construct psuedoinverse for the conformal scaling matrix
    MatrixX B = conformal_scaling_matrix(target_cone_metric);
    MatrixX A = B.transpose() * B;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A);

    // Solve for the best fit conformal scale factor
    VectorX metric_target = target_cone_metric.get_metric_coordinates();
    VectorX w = B.transpose() * (metric_coords - metric_target);
    return solver.solve(w);
  }

  VectorX
  scale_distortion_direction(const DifferentiableConeMetric &target_cone_metric, const VectorX &metric_coords)
  {
    // Compute the psuedoinverse for the conformal scaling matrix
    MatrixX B = conformal_scaling_matrix(target_cone_metric);
    MatrixX A = B.transpose() * B;

    // Solve for the gradient of the L2 norm of the best fit conformal scale
    // factor
    VectorX metric_target = target_cone_metric.get_metric_coordinates();
    MatrixX L = A.transpose() * A;
    VectorX w = B.transpose() * (metric_coords - metric_target);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(L);
    VectorX g = solver.solve(w);
    return B * g;
  }

  void
  face_halfedge_weight_matrix(const std::vector<Scalar> &face_weights,
                              MatrixX &M)
  {
    // Iterate through faces and build diagonal energy matrix
    int num_faces = face_weights.size();
    std::vector<T> tripletList;
    tripletList.reserve(3 * num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
      // Add local entries to global matrix list
      for (Eigen::Index i = 0; i < 3; ++i)
      {
        tripletList.push_back(T(3 * f + i, 3 * f + i, face_weights[f]));
      }
    }

    // Build matrix from triplets
    M.resize(3 * num_faces, 3 * num_faces);
    M.reserve(tripletList.size());
    M.setFromTriplets(tripletList.begin(), tripletList.end());
  }

  void
  length_jacobian(const VectorX &log_length_coords, MatrixX &J_l)
  {
    VectorX J_l_vec;
    J_l_vec.setZero(log_length_coords.size());
    for (int e = 0; e < log_length_coords.size(); ++e)
    {
      J_l_vec[e] = 2.0 / exp(log_length_coords[e]);
    }

    // The Jacobian is a diagonal matrix
    J_l = J_l_vec.asDiagonal();
  }

  MatrixX
  generate_edge_to_face_he_matrix(const Mesh<Scalar> &m)
  {
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Create reindexing matrix
    int num_halfedges = m.n.size();
    int num_faces = m.h.size();
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int f = 0; f < num_faces; ++f)
    {
      // Get face halfedges
      int hij = m.h[f];
      int hjk = m.n[hij];
      int hki = m.n[hjk];

      // Get face projected edges
      int eij = he2e[hij];
      int ejk = he2e[hjk];
      int eki = he2e[hki];

      tripletList.push_back(T(3 * f + 0, eij, 1.0));
      tripletList.push_back(T(3 * f + 1, ejk, 1.0));
      tripletList.push_back(T(3 * f + 2, eki, 1.0));
    }
    MatrixX R;
    R.resize(3 * num_faces, e2he.size());
    R.reserve(tripletList.size());
    R.setFromTriplets(tripletList.begin(), tripletList.end());

    return R;
  }

  VectorX
  halfedge_function_to_vertices(const Mesh<Scalar> &m, const VectorX &g)
  {
    // Initializes vector G to zero vector of length |V|
    VectorX G;
    G.setZero(m.n_ind_vertices());

    // Iterate over halfedges to compute the maximum
    int num_halfedges = m.n.size();
    for (int h = 0; h < num_halfedges; ++h)
    {
      G[m.v_rep[m.to[h]]] = max(G[m.v_rep[m.to[h]]], g[h]);
    }

    return G;
  }

  Scalar
  compute_weighted_norm(
      const VectorX &weights,
      const VectorX &values)
  {
    int num_values = values.size();
    assert(weights.size() == num_values);

    Scalar weighted_norm = 0;
    for (int i = 0; i < num_values; ++i)
    {
      weighted_norm += weights[i] * values[i] * values[i];
    }

    return weighted_norm;
  }

  VectorX compute_face_area_weights(const DifferentiableConeMetric &cone_metric)
  {
    // Compute area per halfedges
    VectorX he2area = areas(cone_metric);

    // Reorganize areas to be per face
    int num_faces = cone_metric.h.size();
    VectorX face_area_weights(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
      face_area_weights[f] = he2area[cone_metric.h[f]];
    }
    spdlog::trace("f to areas: {}", face_area_weights.transpose());

    return face_area_weights;
  }

  VectorX compute_edge_area_weights(const DifferentiableConeMetric &cone_metric)
  {
    // Compute area per halfedges
    VectorX he2area = areas(cone_metric);

    // Compute edge weights as 1/3 of the adjacent face areas
    int num_edges = cone_metric.e2he.size();
    VectorX edge_area_weights(num_edges);
    for (int e = 0; e < num_edges; ++e)
    {
      int h = cone_metric.e2he[e];
      edge_area_weights[e] = (1.0 / 3.0) * (he2area[h] + he2area[cone_metric.opp[h]]);
    }

    return edge_area_weights;
  }

  // Compute the cone vertices of a closed mesh
  // TODO This is messy and should be cleaned. It is also particular to a given experiment and is not
  // generally used, so it should be isolated into a derived functor
  void compute_cone_vertices(
      const Mesh<Scalar> &m,
      const ReductionMaps &reduction_maps,
      std::vector<int> &cone_vertices)
  {
    cone_vertices.clear();
    int num_vertices = m.n_ind_vertices();

    // Closed meshes
    if (reduction_maps.bd_e.empty())
    {
      for (int v = 0; v < num_vertices; ++v)
      {
        if (!float_equal(m.Th_hat[v], 2 * M_PI))
        {
          cone_vertices.push_back(v);
        }
      }
    }
    // Open meshes
    else
    {
      // For open meshes, we iterate over edge endpoints with possible redundancy
      std::vector<bool> is_cone_vertex(num_vertices, false);

      // Iterate over interior edges
      int num_interior_edges = reduction_maps.int_e.size();
      for (int i = 0; i < num_interior_edges; ++i)
      {
        // Get two halfedges of the edge
        int E = reduction_maps.int_e[i];
        int e = reduction_maps.embed[E];
        int h = reduction_maps.e2he[e];
        int ho = m.opp[h];

        // Get two endpoint vertices
        int v_to = m.to[h];
        int v_fr = m.to[ho];

        // Regular interior vertices have doubled angle 4 pi
        is_cone_vertex[v_to] = (!float_equal(m.Th_hat[v_to], 4 * M_PI));
        is_cone_vertex[v_fr] = (!float_equal(m.Th_hat[v_fr], 4 * M_PI));
      }

      // Iterate over boundary edges
      int num_boundary_edges = reduction_maps.bd_e.size();
      for (int i = 0; i < num_boundary_edges; ++i)
      {
        // Get two halfedges of the edge
        int E = reduction_maps.bd_e[i];
        int e = reduction_maps.embed[E];
        int h = reduction_maps.e2he[e];
        int ho = m.opp[h];

        // Get two endpoint vertices
        int v_to = m.to[h];
        int v_fr = m.to[ho];

        // Regular boundary vertices have angle 2 pi
        is_cone_vertex[v_to] = (!float_equal(m.Th_hat[v_to], 2 * M_PI));
        is_cone_vertex[v_fr] = (!float_equal(m.Th_hat[v_fr], 2 * M_PI));
      }

      // Convert boolean list to index vector
      convert_boolean_array_to_index_vector(is_cone_vertex, cone_vertices);
    }
  }

  // Compute the cone adjacent faces of a closed mesh
  [[deprecated]]
  void compute_cone_faces(
      const Mesh<Scalar> &m,
      const ReductionMaps &reduction_maps,
      std::vector<int> &cone_faces)
  {
    // Compute the cone vertices
    std::vector<int> cone_vertices;
    compute_cone_vertices(m, reduction_maps, cone_vertices);

    // Get boolean mask for the cone vertices
    std::vector<bool> is_cone_vertex;
    int num_vertices = m.n_ind_vertices();
    convert_index_vector_to_boolean_array(cone_vertices, num_vertices, is_cone_vertex);

    // Compute the cone faces by iterating over the halfedges
    cone_faces.clear();
    int num_halfedges = m.n_halfedges();
    for (int h = 0; h < num_halfedges; ++h)
    {
      int v = m.v_rep[m.to[h]];
      if (is_cone_vertex[v])
      {
        int f = m.f[h];
        cone_faces.push_back(f);
      }
    }
  }

  void
  compute_cone_face_weights(
      const Mesh<Scalar> &m,
      const ReductionMaps &reduction_maps,
      Scalar cone_weight,
      std::vector<Scalar> &face_weights)
  {
    std::vector<int> cone_faces;
    compute_cone_faces(m, reduction_maps, cone_faces);
    spdlog::trace("Weighting {} faces with {}", cone_faces.size(), cone_weight);
    face_weights = std::vector<Scalar>(m.h.size(), 1.0);
    for (size_t i = 0; i < cone_faces.size(); ++i)
    {
      face_weights[cone_faces[i]] = cone_weight;
    }
  }

  void
  compute_boundary_face_weights(
      const Mesh<Scalar> &m,
      const ReductionMaps &reduction_maps,
      Scalar bd_weight,
      std::vector<Scalar> &face_weights)
  {
    // Initialize face weights to 1
    face_weights = std::vector<Scalar>(m.h.size(), 1.0);

    // Iterate over boundary edges
    int num_boundary_edges = reduction_maps.bd_e.size();
    for (int i = 0; i < num_boundary_edges; ++i)
    {
      // Get two halfedges of the edge
      int E = reduction_maps.bd_e[i];
      int e = reduction_maps.embed[E];
      int h = reduction_maps.e2he[e];
      int ho = m.opp[h];

      // Get two faces adjacent to the edge
      int f = m.f[h];
      int fo = m.f[ho];

      // Set face weights
      face_weights[f] = bd_weight;
      face_weights[fo] = bd_weight;
    }
  }

  LogLengthEnergy::LogLengthEnergy(const DifferentiableConeMetric &target_cone_metric, int order)
      : m_metric_target(target_cone_metric.get_metric_coordinates()), m_order(order)
  {
  }

  Scalar LogLengthEnergy::energy(const VectorX &metric_coords) const
  {
    if (m_order == 2)
    {
      VectorX difference = metric_coords - m_metric_target;
      return 0.5 * difference.squaredNorm();
    }
    else
    {
      int num_edges = metric_coords.size();
      VectorX difference = metric_coords - m_metric_target;
      Scalar energy = 0.0;
      for (int E = 0; E < num_edges; ++E)
      {
        // Iterate to compute the pth power of the edge difference absolute value
        Scalar term = 1.0;
        Scalar edge_difference = difference[E];
        for (int i = 0; i < m_order; ++i)
        {
          term *= abs(edge_difference);
        }

        // Add the term (weighted by 1/p)
        energy += term / static_cast<Scalar>(m_order);
      }
      return energy;
    }
  }

  VectorX LogLengthEnergy::gradient(const VectorX &metric_coords) const
  {
    if (m_order == 2)
    {
      VectorX difference = metric_coords - m_metric_target;
      return difference;
    }
    else
    {
      int num_edges = metric_coords.size();
      VectorX difference = metric_coords - m_metric_target;
      VectorX p_norm_gradient(num_edges);
      for (int E = 0; E < num_edges; ++E)
      {
        // Iterate to compute the gradient of the edge difference as the signed
        // p-1th power of the absolute value
        Scalar edge_difference = difference[E];
        p_norm_gradient[E] = edge_difference;
        for (int i = 2; i < m_order; ++i)
        {
          p_norm_gradient[E] *= abs(edge_difference);
        }
      }
      return p_norm_gradient;
    }
  }

  MatrixX LogLengthEnergy::hessian(const VectorX &metric_coords) const
  {
    if (m_order != 2)
    {
      throw std::runtime_error("No hessian defined");
    }

    return id_matrix(metric_coords.size());
  }

  MatrixX LogLengthEnergy::hessian_inverse(const VectorX &metric_coords) const
  {
    if (m_order != 2)
    {
      throw std::runtime_error("No hessian defined");
    }

    return id_matrix(metric_coords.size());
  }

  QuadraticSymmetricDirichletEnergy::QuadraticSymmetricDirichletEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric)
      : m_metric_target(target_cone_metric.get_metric_coordinates())
  {
    // Compute quadratic energy matrix
    MatrixX R = generate_edge_to_face_he_matrix(discrete_metric);
    MatrixX M = surface_hencky_strain_energy(discrete_metric);

    // Compute quadratic energy matrix_inverse
    MatrixX inverse_M = surface_hencky_strain_energy_inverse(discrete_metric);

    // Conjugate halfedge energy with edge to halfedge maps
    m_quadratic_energy_matrix = R.transpose() * M * R;
    m_quadratic_energy_matrix_inverse = R.transpose() * inverse_M * R;
  }

  Scalar QuadraticSymmetricDirichletEnergy::energy(const VectorX &metric_coords) const
  {
    VectorX difference = metric_coords - m_metric_target;
    return 0.5 * (difference.dot((m_quadratic_energy_matrix * difference)));
  }

  VectorX QuadraticSymmetricDirichletEnergy::gradient(const VectorX &metric_coords) const
  {
    VectorX difference = metric_coords - m_metric_target;
    return m_quadratic_energy_matrix * difference;
  }

  MatrixX QuadraticSymmetricDirichletEnergy::hessian(const VectorX &) const
  {
    return m_quadratic_energy_matrix;
  }

  MatrixX QuadraticSymmetricDirichletEnergy::hessian_inverse(const VectorX &) const
  {
    return m_quadratic_energy_matrix_inverse;
  }

  LogScaleEnergy::LogScaleEnergy(const DifferentiableConeMetric &target_cone_metric)
  : m_target_cone_metric(target_cone_metric.clone_cone_metric())
  {
  }

  Scalar LogScaleEnergy::energy(const VectorX &metric_coords) const
  {
    VectorX u = best_fit_conformal(*m_target_cone_metric, metric_coords);
    return 0.5 * (u.dot(u));
  }

  VectorX LogScaleEnergy::gradient(const VectorX &metric_coords) const
  {
    return scale_distortion_direction(*m_target_cone_metric, metric_coords);
  }

  SymmetricDirichletEnergy::SymmetricDirichletEnergy(const DifferentiableConeMetric &target_cone_metric, const DiscreteMetric &discrete_metric)
  : m_target_cone_metric(target_cone_metric.clone_cone_metric())
  , m_face_area_weights(compute_face_area_weights(discrete_metric))
  {
  }

  Scalar SymmetricDirichletEnergy::energy(const VectorX &metric_coords) const
  {
    // Get per face symmetric dirichlet energy
    VectorX f2energy;
    MatrixX J_f2energy;
    bool need_jacobian = false;
    symmetric_dirichlet_energy(
        *m_target_cone_metric,
        metric_coords,
        f2energy,
        J_f2energy,
        need_jacobian);

    // Compute face integrated symmetric dirichlet energy
    return f2energy.dot(m_face_area_weights);
  }

  VectorX SymmetricDirichletEnergy::gradient(const VectorX &metric_coords) const
  {
    // Get per face symmetric dirichlet energy
    VectorX f2energy;
    MatrixX J_f2energy;
    bool need_jacobian = true;
    symmetric_dirichlet_energy(
        *m_target_cone_metric,
        metric_coords,
        f2energy,
        J_f2energy,
        need_jacobian);

    // Compute face integrated symmetric dirichlet energy Jacobian
    return (m_face_area_weights.transpose() * J_f2energy);
  }

  RegularizedQuadraticEnergy::RegularizedQuadraticEnergy(
      const DifferentiableConeMetric &target_cone_metric,
      const DiscreteMetric &discrete_metric, double weight)
      : m_metric_target(target_cone_metric.get_metric_coordinates())
  {
    // Compute quadratic energy matrix
    MatrixX R = generate_edge_to_face_he_matrix(discrete_metric);
    MatrixX M = surface_hencky_strain_energy(discrete_metric);

    // Conjugate halfedge energy with edge to halfedge maps
    MatrixX Q = R.transpose() * M * R;
    MatrixX I = id_matrix(m_metric_target.size());

    // Compute mesh area
    VectorX face_area_weights = compute_face_area_weights(discrete_metric);
    Scalar mesh_area = face_area_weights.sum();

    // Compute regularized matrix
    m_quadratic_energy_matrix = weight * Q + mesh_area * I;
  }

  Scalar RegularizedQuadraticEnergy::energy(const VectorX &metric_coords) const
  {
    VectorX difference = metric_coords - m_metric_target;
    return 0.5 * (difference.dot((m_quadratic_energy_matrix * difference)));
  }

  VectorX RegularizedQuadraticEnergy::gradient(const VectorX &metric_coords) const
  {
    VectorX difference = metric_coords - m_metric_target;
    return m_quadratic_energy_matrix * difference;
  }

  Scalar ConeEnergy::energy(const VectorX &metric_coords) const
  {
    assert(false);
    return 0.5 * metric_coords.squaredNorm();
    // TODO
    //    VectorX constraint;
    //    MatrixX J_constraint;
    //    std::vector<int> flip_seq;
    //    bool need_jacobian = false;
    //    constraint_with_jacobian(*m_mesh,
    //                             metric_coords,
    //                             constraint,
    //                             J_constraint,
    //                             flip_seq,
    //                             need_jacobian,
    //                             m_opt_params.use_edge_lengths);
    //
    //    // Add L2 constraint error for fixed dof of the projection
    //    Scalar energy = 0;
    //    for (size_t v = 0; v < m_mesh->fixed_dof.size(); ++v)
    //    {
    //      if (m_mesh->fixed_dof[v])
    //      {
    //        energy += 0.5 * constraint[v] * constraint[v];
    //      }
    //    }
    //
    //    return energy;
  }

  VectorX ConeEnergy::gradient(const VectorX &metric_coords) const
  {
    assert(false);
    return metric_coords;
    // TODO

    //    VectorX constraint;
    //    MatrixX J_constraint;
    //    std::vector<int> flip_seq;
    //    bool need_jacobian = true;
    //    constraint_with_jacobian(*m_mesh,
    //                             metric_coords,
    //                             constraint,
    //                             J_constraint,
    //                             flip_seq,
    //                             need_jacobian,
    //                             m_opt_params.use_edge_lengths);
    //
    //    // Add L2 constraint error gradient for fixed dof of the projection
    //    VectorX cone_gradient;
    //    cone_gradient.setZero(metric_coords.size());
    //    for (size_t v = 0; v < m_mesh->fixed_dof.size(); ++v)
    //    {
    //      if (m_mesh->fixed_dof[v])
    //      {
    //        cone_gradient += constraint[v] * J_constraint.row(v);
    //      }
    //    }
    //
    //    return cone_gradient;
  }

  // TODO
  // Add cone face weights if nontrivial
//  if (!float_equal(m_cone_weight, 1.0))
//  {
//    std::vector<Scalar> face_weights;
//    compute_cone_face_weights(m, reduction_maps, m_cone_weight, face_weights);
//    MatrixX M_face_weight;
//    face_halfedge_weight_matrix(face_weights, M_face_weight);
//    M = M * M_face_weight;
//  }
//
//  // Add boundary face weights if nontrivial
//  if (!float_equal(m_bd_weight, 1.0))
//  {
//    std::vector<Scalar> face_weights;
//    compute_boundary_face_weights(m, reduction_maps, m_bd_weight, face_weights);
//    MatrixX M_face_weight;
//    face_halfedge_weight_matrix(face_weights, M_face_weight);
//    M = M * M_face_weight;
//  }

// FIXME Rename these variables
// FIXME Ensure all pybind functions for the entire interface are in place
#ifdef PYBIND

  MatrixX
  length_jacobian_pybind(const VectorX &lambdas_full)
  {
    MatrixX J_l;
    length_jacobian(lambdas_full, J_l);

    return J_l;
  }

#endif

}
