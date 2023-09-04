#include "constraint.hh"

#include "area.hh"
#include "embedding.hh"
#include "delaunay.hh"

namespace CurvatureMetric {

bool
satisfies_triangle_inequality(const Mesh<Scalar>& m,
                              const VectorX& log_length_coords)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Check triangle inequality for each halfedge
  int num_halfedges = m.n.size();
  for (int hi = 0; hi < num_halfedges; ++hi) {
    int hj = m.n[hi];
    int hk = m.n[hj];

    // Log lengths of the halfedges
    Scalar lli = log_length_coords[he2e[hi]];
    Scalar llj = log_length_coords[he2e[hj]];
    Scalar llk = log_length_coords[he2e[hk]];

    // Halfedge lengths (scaled by the average for stability)
    Scalar llijk_avg = (lli + llj + llk) / 3.0;
    Scalar li = exp((lli - llijk_avg) / 2.0);
    Scalar lj = exp((llj - llijk_avg) / 2.0);
    Scalar lk = exp((llk - llijk_avg) / 2.0);

    // Check triangle inequality
    if (li > lj + lk)
      return false;
  }

  return true;
}

void
corner_angles(const Mesh<Scalar>& m,
              const VectorX& log_length_coords,
              VectorX& he2angle,
              VectorX& he2cot)
{
  he2angle.setZero(m.n.size());
  he2cot.setZero(m.n.size());
  const Scalar cot_infty = 1e10;

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Compute maps from halfedges to opposite angles and cotangents of opposite
  // angles
  // #pragma omp parallel for
  int num_faces = m.h.size();
  for (int f = 0; f < num_faces; f++) {
    // Halfedges of face f
    int hi = m.h[f];
    int hj = m.n[hi];
    int hk = m.n[hj];

    // Log lengths of the halfedges
    Scalar lli = log_length_coords[he2e[hi]];
    Scalar llj = log_length_coords[he2e[hj]];
    Scalar llk = log_length_coords[he2e[hk]];

    // Halfedge lengths (scaled by the average for stability)
    Scalar llijk_avg = (lli + llj + llk) / 3.0;
    Scalar li = exp((lli - llijk_avg) / 2.0);
    Scalar lj = exp((llj - llijk_avg) / 2.0);
    Scalar lk = exp((llk - llijk_avg) / 2.0);

    // Compute the cotangent of the angles
    // (following "A Cotangent Laplacian for Images as Surfaces")
    Scalar Aijk4 = 4 * sqrt(std::max<Scalar>(area_squared(li, lj, lk), 0.0));
    Scalar Ijk = (-li * li + lj * lj + lk * lk);
    Scalar iJk = (li * li - lj * lj + lk * lk);
    Scalar ijK = (li * li + lj * lj - lk * lk);
    he2cot[hi] = Aijk4 == 0.0 ? copysign(cot_infty, Ijk) : (Ijk / Aijk4);
    he2cot[hj] = Aijk4 == 0.0 ? copysign(cot_infty, iJk) : (iJk / Aijk4);
    he2cot[hk] = Aijk4 == 0.0 ? copysign(cot_infty, ijK) : (ijK / Aijk4);

#define USE_ACOS
#ifdef USE_ACOS
    he2angle[hi] = acos(
      std::min<Scalar>(std::max<Scalar>(Ijk / (2.0 * lj * lk), -1.0), 1.0));
    he2angle[hj] = acos(
      std::min<Scalar>(std::max<Scalar>(iJk / (2.0 * lk * li), -1.0), 1.0));
    he2angle[hk] = acos(
      std::min<Scalar>(std::max<Scalar>(ijK / (2.0 * li * lj), -1.0), 1.0));
#else
    // atan2 is prefered for stability
    he2angle[hi] = 0.0, he2angle[hj] = 0.0, he2angle[hk] = 0.0;
    // li: l12, lj: l23, lk: l31
    Scalar l12 = li, l23 = lj, l31 = lk;
    const Scalar t31 = +l12 + l23 - l31, t23 = +l12 - l23 + l31,
                 t12 = -l12 + l23 + l31;
    // valid triangle
    if (t31 > 0 && t23 > 0 && t12 > 0) {
      const Scalar l123 = l12 + l23 + l31;
      const Scalar denom = sqrt(t12 * t23 * t31 * l123);
      he2angle[hj] = 2 * atan2(t12 * t31, denom); // a1 l23
      he2angle[hk] = 2 * atan2(t23 * t12, denom); // a2 l31
      he2angle[hi] = 2 * atan2(t31 * t23, denom); // a3 l12
    } else if (t31 <= 0)
      he2angle[hk] = pi;
    else if (t23 <= 0)
      he2angle[hj] = pi;
    else if (t12 <= 0)
      he2angle[hi] = pi;
    else
      he2angle[hj] = pi;
#endif
  }
}

void
vertex_angles_with_jacobian(const Mesh<Scalar>& m,
                            const VectorX& log_length_coords,
                            VectorX& vertex_angles,
                            MatrixX& J_vertex_angles,
                            bool need_jacobian)
{
  // Get angles and cotangent of angles of faces opposite halfedges
  VectorX he2angle;
  VectorX he2cot;
  corner_angles(m, log_length_coords, he2angle, he2cot);

  // Sum up angles around vertices
  int num_halfedges = m.n.size();
  vertex_angles.setZero(m.Th_hat.size());
  for (int h = 0; h < num_halfedges; ++h) {
    vertex_angles[m.v_rep[m.to[h]]] += he2angle[m.n[m.n[h]]];
  }

  // Build Jacobian if needed
  if (need_jacobian) {
    // Build edge to halfedge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Create list of triplets of Jacobian indices and values
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(3 * (num_halfedges / 2));
    for (int h = 0; h < num_halfedges; ++h) {
      tripletList.push_back(
        T(m.v_rep[m.to[m.n[h]]], he2e[m.n[m.n[h]]], -0.5 * he2cot[m.n[h]]));
    }
    for (int h = 0; h < num_halfedges; ++h) {
      tripletList.push_back(
        T(m.v_rep[m.to[m.n[h]]],
          he2e[h],
          0.5 * he2cot[m.n[h]] + 0.5 * he2cot[m.n[m.n[h]]]));
    }
    for (int h = 0; h < num_halfedges; ++h) {
      tripletList.push_back(
        T(m.v_rep[m.to[m.n[h]]], he2e[m.n[h]], -0.5 * he2cot[m.n[m.n[h]]]));
    }

    // Build Jacobian from triplets
    J_vertex_angles.resize(m.Th_hat.size(), num_halfedges / 2);
    J_vertex_angles.reserve(3 * (num_halfedges / 2));
    J_vertex_angles.setFromTriplets(tripletList.begin(), tripletList.end());
  }
}

bool
constraint_with_jacobian(const Mesh<Scalar>& m,
                         const VectorX& metric_coords,
                         VectorX& constraint,
                         MatrixX& J_constraint,
                         std::vector<int>& flip_seq,
                         bool need_jacobian,
                         bool use_edge_lengths)
{
  constraint.setZero(0);
  J_constraint.setZero();
  flip_seq.clear();

  // For Penner coordinates, make mesh Delaunay (and thus satisfying the
  // triangle inequality)
  Mesh<Scalar> m_tri;
  VectorX metric_coords_tri;
  VectorX vertex_angles;
  MatrixX J_vertex_angles;
  MatrixX J_del;
  if (!use_edge_lengths) {
    make_delaunay_with_jacobian(m,
                                metric_coords,
                                m_tri,
                                metric_coords_tri,
                                J_del,
                                flip_seq,
                                need_jacobian);
  }
  // Check triangle inequality explicitly otherwise
  else {
    if (!satisfies_triangle_inequality(m, metric_coords))
      return false;
    m_tri = m;
    metric_coords_tri = metric_coords;
  }
  vertex_angles_with_jacobian(
    m_tri, metric_coords_tri, vertex_angles, J_vertex_angles, need_jacobian);

  // Subtract the target angles from the vertex angles to compute constraint
  constraint.resize(vertex_angles.size());
  for (int v = 0; v < vertex_angles.size(); ++v) {
    constraint[v] = vertex_angles[v] - m.Th_hat[v];
  }

  // Compute the Jacobian of the constraint if needed
  if (need_jacobian) {
    if (use_edge_lengths)
    {
      J_constraint = J_vertex_angles;
    }
    else
    {
      J_constraint = J_vertex_angles * J_del;
    }
  }

  return true;
}

#ifdef PYBIND
// Pybind definitions
std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
vertex_angles_with_jacobian_pybind(const Mesh<Scalar>& m,
                                   const VectorX& metric_coords,
                                   bool need_jacobian)
{
  VectorX vertex_angles;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_vertex_angles;
  vertex_angles_with_jacobian(
    m, metric_coords, vertex_angles, J_vertex_angles, need_jacobian);

  return std::make_tuple(vertex_angles, J_vertex_angles);
}

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>, std::vector<int>, bool>
constraint_with_jacobian_pybind(const Mesh<Scalar>& m,
                                const VectorX& metric_coords,
                                bool need_jacobian,
                                bool use_edge_lengths)
{
  VectorX constraint;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_constraint;
  std::vector<int> flip_seq;
  bool success = constraint_with_jacobian(m,
                                          metric_coords,
                                          constraint,
                                          J_constraint,
                                          flip_seq,
                                          need_jacobian,
                                          use_edge_lengths);

  return std::make_tuple(constraint, J_constraint, flip_seq, success);
}
#endif

}
