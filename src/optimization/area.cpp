#include "area.hh"

#include "embedding.hh"

namespace CurvatureMetric {

Scalar
area_squared(Scalar li, Scalar lj, Scalar lk)
{
  // Sort the lengths for numerical stability
  Scalar a = li;
  Scalar b = lj;
  Scalar c = lk;
  if (a < b)
    swap(a, b);
  if (a < c)
    swap(a, c);
  if (b < c)
    swap(b, c);

  // Compute the area
  Scalar A = a + (b + c);
  Scalar B = c - (a - b);
  Scalar C = c + (a - b);
  Scalar D = a + (b - c);

  return A * B * C * D / 16.0;
}

Scalar
area_squared_derivative(Scalar variable_length, Scalar lj, Scalar lk)
{
  // Sort the lengths and keep track of the derivative edge.
  Scalar a = variable_length;
  Scalar b = lj;
  Scalar c = lk;
  bool deriv_a = true;
  bool deriv_b = false;
  if (a < b) {
    swap(a, b);
    deriv_a = false;
    deriv_b = true;
  }
  if (a < c) {
    swap(a, c);
    deriv_a = false;
  }
  if (b < c) {
    swap(b, c);
    // The derivative is b iff the derivate was c before the swap
    deriv_b = !(deriv_a || deriv_b);
  }

  // Compute stable factors for Heron's formula
  Scalar S = a + (b + c);
  Scalar A = c - (a - b);
  Scalar B = c + (a - b);
  Scalar C = a + (b - c);

  // Compute terms minus a term
  Scalar TmS = (A * B * C) / 16.0;
  Scalar TmA = (S * B * C) / 16.0;
  Scalar TmB = (S * A * C) / 16.0;
  Scalar TmC = (S * A * B) / 16.0;

  // Compute the derivative for li
  if (deriv_a) {
    return TmS - TmA + TmB + TmC;
  } else if (deriv_b) {
    return TmS + TmA - TmB + TmC;
  } else {
    return TmS + TmA + TmB - TmC;
  }
}

void
areas_squared_from_log_lengths(const Mesh<Scalar>& m,
                               const VectorX& log_length_coords,
                               VectorX& he2areasq)
{
  he2areasq.resize(m.n.size());

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  int num_faces = m.h.size();
  // #pragma omp parallel for
  for (int f = 0; f < num_faces; f++) {
    // Get halfedges of face f
    int hi = m.h[f];
    int hj = m.n[hi];
    int hk = m.n[hj];

    // Get log lengths of the halfedges
    Scalar lli = log_length_coords[he2e[hi]];
    Scalar llj = log_length_coords[he2e[hj]];
    Scalar llk = log_length_coords[he2e[hk]];

    // Get lengths of the halfedges
    Scalar li = exp(lli / 2.0);
    Scalar lj = exp(llj / 2.0);
    Scalar lk = exp(llk / 2.0);

    // Compute the area of the face adjacent to the halfedges
    Scalar areasq = area_squared(li, lj, lk);
    he2areasq[hi] = areasq;
    he2areasq[hj] = areasq;
    he2areasq[hk] = areasq;
  }
  spdlog::trace("he to sq areas: {}", he2areasq.transpose());
}

void
areas_from_log_lengths(const Mesh<Scalar>& m,
                       const VectorX& log_length_coords,
                       VectorX& he2area)
{
  int num_halfedges = m.n_halfedges();

  // Compute squared areas
  VectorX he2areasq;
  areas_squared_from_log_lengths(m, log_length_coords, he2areasq);
  assert( he2areasq.size() == num_halfedges );

  // Take square roots
  he2area.resize(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h)
  {
    he2area[h] = sqrt(std::max<Scalar>(he2areasq[h], 0.0));
  }
  spdlog::trace("he to areas: {}", he2area.transpose());
}

void
area_squared_derivatives_from_log_lengths(const Mesh<Scalar>& m,
                                          const VectorX& log_length_coords,
                                          VectorX& he2areasqderiv)
{
  he2areasqderiv.resize(m.n.size());

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Compute maps from halfedges to derivatives of area with respect to the edge
  // length
  int num_faces = m.h.size();
  // #pragma omp parallel for
  for (int f = 0; f < num_faces; f++) {
    // Get halfedges of face f
    int hi = m.h[f];
    int hj = m.n[hi];
    int hk = m.n[hj];

    // Get log lengths of the halfedges
    Scalar lli = log_length_coords[he2e[hi]];
    Scalar llj = log_length_coords[he2e[hj]];
    Scalar llk = log_length_coords[he2e[hk]];

    // Get halfedge lengths
    Scalar li = exp(lli / 2.0);
    Scalar lj = exp(llj / 2.0);
    Scalar lk = exp(llk / 2.0);

    // Compute the derivative of the area of f with respect to each halfedge
    he2areasqderiv[hi] = area_squared_derivative(li, lj, lk) * li / 2.0;
    he2areasqderiv[hj] = area_squared_derivative(lj, lk, li) * lj / 2.0;
    he2areasqderiv[hk] = area_squared_derivative(lk, li, lj) * lk / 2.0;
  }
}

#ifdef PYBIND
VectorX // he2areasq
areas_squared_from_log_lengths_pybind(const Mesh<Scalar>& m,
                                      const VectorX& log_length_coords)
{
  VectorX he2areasq;
  areas_squared_from_log_lengths(m, log_length_coords, he2areasq);

  return he2areasq;
}
#endif

}
