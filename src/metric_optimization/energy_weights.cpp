#include "energy_functor.hh"

#include "area.hh"
#include "constraint.hh"
#include "energies.hh"
#include "projection.hh"
#include "vector.hh"

#include <igl/doublearea.h>
#include <igl/cotmatrix_entries.h>
#include <igl/edge_lengths.h>

/// FIXME Do cleaning pass

namespace CurvatureMetric
{

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
  [[deprecated]] void compute_cone_vertices(
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
  //[[deprecated]]
  //void compute_cone_faces(
  //    const Mesh<Scalar> &m,
  //    const ReductionMaps &reduction_maps,
  //    std::vector<int> &cone_faces)
  //{
  //  // Compute the cone vertices
  //  std::vector<int> cone_vertices;
  //  compute_cone_vertices(m, reduction_maps, cone_vertices);

  //  // Get boolean mask for the cone vertices
  //  std::vector<bool> is_cone_vertex;
  //  int num_vertices = m.n_ind_vertices();
  //  convert_index_vector_to_boolean_array(cone_vertices, num_vertices, is_cone_vertex);

  //  // Compute the cone faces by iterating over the halfedges
  //  cone_faces.clear();
  //  int num_halfedges = m.n_halfedges();
  //  for (int h = 0; h < num_halfedges; ++h)
  //  {
  //    int v = m.v_rep[m.to[h]];
  //    if (is_cone_vertex[v])
  //    {
  //      int f = m.f[h];
  //      cone_faces.push_back(f);
  //    }
  //  }
  //}

  //[[deprecated]] void
  //compute_cone_face_weights(
  //    const Mesh<Scalar> &m,
  //    const ReductionMaps &reduction_maps,
  //    Scalar cone_weight,
  //    std::vector<Scalar> &face_weights)
  //{
  //  std::vector<int> cone_faces;
  //  compute_cone_faces(m, reduction_maps, cone_faces);
  //  spdlog::trace("Weighting {} faces with {}", cone_faces.size(), cone_weight);
  //  face_weights = std::vector<Scalar>(m.h.size(), 1.0);
  //  for (size_t i = 0; i < cone_faces.size(); ++i)
  //  {
  //    face_weights[cone_faces[i]] = cone_weight;
  //  }
  //}

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

  // TODO Potentially remove or refactor
  [[deprecated]] void
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


}
