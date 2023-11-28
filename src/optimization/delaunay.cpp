#include "delaunay.hh"

#include "embedding.hh"

namespace CurvatureMetric
{

  // Helper function to update a list of list representation of the make Delaunay
  // Jacobian for a given ccwedge flip at h
  // TODO Delete
  [[deprecated]] void
  update_jacobian_del(const Mesh<Scalar> &m,
                      int h,
                      const std::vector<int> &he2e,
                      std::vector<std::map<int, Scalar>> &J_del_lol,
                      Scalar zero_threshold = 1e-15)
  {
    // Get local mesh information near hd
    int hd = h;
    int hb = m.n[hd];
    int ha = m.n[hb];
    int hdo = m.opp[hd];
    int hao = m.n[hdo];
    int hbo = m.n[hao];

    // Get edges corresponding to halfedges
    int ed = he2e[hd];
    int eb = he2e[hb];
    int ea = he2e[ha];
    int eao = he2e[hao];
    int ebo = he2e[hbo];

    // Compute the shear for the edge ed
    Scalar la = m.l[ha];
    Scalar lb = m.l[hb];
    Scalar lao = m.l[hao];
    Scalar lbo = m.l[hbo];
    Scalar x = (la * lbo) / (lb * lao);

    // The matrix Pd corresponding to flipping edge ed is the identity except for
    // the row corresponding to edge ed, which has entries defined by Pd_scalars
    // in the column corresponding to the edge with the same index in Pd_edges
    std::vector<int> Pd_edges = {ed, ea, ebo, eao, eb};
    std::vector<Scalar> Pd_scalars = {
        -1.0, x / (1.0 + x), x / (1.0 + x), 1.0 / (1.0 + x), 1.0 / (1.0 + x)};

    // Compute the new row of J_del corresponding to edge ed, which is the only
    // edge that changes
    std::map<int, Scalar> J_del_d_new;
    for (int i = 0; i < 5; ++i)
    {
      int ei = Pd_edges[i];
      Scalar Di = Pd_scalars[i];
      for (auto it : J_del_lol[ei])
      {
        J_del_d_new[it.first] += Di * it.second;

        // Delete the updated entry if it is near 0
        if (abs(J_del_d_new[it.first]) < zero_threshold)
          J_del_d_new.erase(it.first);
      }
    }

    J_del_lol[ed] = J_del_d_new;
  }

  void
  make_delaunay_with_jacobian(const Mesh<Scalar> &m,
                              const VectorX &metric_coords,
                              Mesh<Scalar> &m_del,
                              VectorX &metric_coords_del,
                              MatrixX &J_del,
                              std::vector<int> &flip_seq,
                              bool need_jacobian)
  {
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Convert embedded mesh log edge lengths to a halfedge length array l for m
    int num_edges = e2he.size();
    int num_halfedges = he2e.size();
    m_del = m;
    for (int h = 0; h < num_halfedges; ++h)
    {
      m_del.l[h] = exp(metric_coords[he2e[h]] / 2.0);
    }

    // Make the copied mesh Delaunay with Ptolemy flips
    VectorX u;
    u.setZero(m_del.n_ind_vertices());
    DelaunayStats del_stats;
    SolveStats<Scalar> solve_stats;
    bool use_ptolemy = true;
    ConformalIdealDelaunay<Scalar>::MakeDelaunay(m_del, u, del_stats, solve_stats, use_ptolemy);
    flip_seq = del_stats.flip_seq;

    // If Jacobian needed, redo the flips to build it incrementally
    if (need_jacobian)
    {
      // Build a copy of the input mesh for length flip tracking
      Mesh<Scalar> m_jac = m;
      for (int h = 0; h < num_halfedges; ++h)
      {
        m_jac.l[h] = exp(metric_coords[he2e[h]] / 2.0);
      }

      // Initialize the Jacobian to the identity. We use a list of maps
      // representation for very quick row access and updates of this sparse matrix
      std::vector<std::map<int, Scalar>> J_del_lol(metric_coords.size(),
                                                   std::map<int, Scalar>());
      for (int e = 0; e < metric_coords.size(); ++e)
      {
        J_del_lol[e][e] = 1.0;
      }

      // Duplicate all flips while building the jacobian
      int num_flips = flip_seq.size();
      for (int i = 0; i < num_flips; ++i)
      {
        int hij = flip_seq[i];
        m_jac.flip_ccw(hij);
        update_jacobian_del(m_jac, hij, he2e, J_del_lol);
      }

      // Get triplet (i, j, v) for each value v = J_del_lol[i][j] in the list of
      // maps
      std::vector<T> tripletList;
      tripletList.reserve(5 * num_edges);
      for (int i = 0; i < num_edges; ++i)
      {
        for (auto it : J_del_lol[i])
        {
          tripletList.push_back(T(i, it.first, it.second));
        }
      }

      // Create the matrix from the triplets
      J_del.resize(metric_coords.size(), metric_coords.size());
      J_del.reserve(tripletList.size());
      J_del.setFromTriplets(tripletList.begin(), tripletList.end());

#if CHECK_VALIDITY
      // Check flipped meshes are consistent and return otherwise
      for (int h = 0; h < num_halfedges; ++h)
      {
        if ((!float_equal(m_del.l[h], m_jac.l[h])) || (m_del.n[h] != m_jac.n[h]))
        {
          spdlog::error("Inconsistent flipped meshes when making Delaunay");
          m_del = Mesh<Scalar>();
          metric_coords_del.setZero(0);
          J_del.setZero();
          flip_seq.clear();
          return;
        }
      }
#endif
    }

    // Convert Delaunay mesh halfedge lengths to log edge lengths
    metric_coords_del.resize(num_edges);
    for (int e = 0; e < num_edges; ++e)
    {
      metric_coords_del[e] = 2.0 * log(m_del.l[e2he[e]]);
    }
  }

#ifdef PYBIND
  [[deprecated]] std::tuple<Mesh<Scalar>,
                            VectorX,
                            Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
                            std::vector<int>>
  make_delaunay_with_jacobian_pybind(const Mesh<Scalar> &C,
                                     const VectorX &lambdas,
                                     bool need_jacobian)
  {
    Mesh<Scalar> C_del(C);
    VectorX lambdas_del;
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_del;
    std::vector<int> flip_seq;
    make_delaunay_with_jacobian(
        C, lambdas, C_del, lambdas_del, J_del, flip_seq, need_jacobian);

    return std::make_tuple(C_del, lambdas_del, J_del, flip_seq);
  }

  [[deprecated]] std::tuple<OverlayMesh<Scalar>,
                            VectorX,
                            Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
                            std::vector<int>>
  make_delaunay_with_jacobian_overlay(const OverlayMesh<Scalar> &C,
                                      const VectorX &lambdas,
                                      bool need_jacobian)
  {
    OverlayMesh<Scalar> C_del(C);
    VectorX lambdas_del;
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_del;
    std::vector<int> flip_seq;
    make_delaunay_with_jacobian(
        C, lambdas, C_del, lambdas_del, J_del, flip_seq, need_jacobian);

    return std::make_tuple(C_del, lambdas_del, J_del, flip_seq);
  }

#endif
}
