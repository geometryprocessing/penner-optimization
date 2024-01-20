#pragma once

#include "common.hh"

namespace CurvatureMetric {

/// Reparametrize the barycentric coordinates for the equilateral triangle by
/// translating a constant hyperbolic distance along each halfedge. If the sum
/// of translations per triangle is 0, then this corresponds to a projective map
/// over the triangle.
///
/// @param[in, out] m_o: mesh to reparametrize
/// @param[in] tau: per halfeedge hyperbolic translation distances
void
bc_reparametrize_eq(OverlayMesh<Scalar>& m_o, const VectorX& tau);

/// Reparametrize points contained in equilateral reference triangles by
/// translating a constant hyperbolic distance along each halfedge. If the sum
/// of translations per triangle is 0, then this corresponds to a projective map
/// over the triangle.
///
/// @param[in, out] pts: points to reparameterize
/// @param[in] n: next halfedge array for mesh
/// @param[in] h: face to halfedge array for mesh
/// @param[in] tau: per halfeedge hyperbolic translation distances
void
reparametrize_equilateral(std::vector<Pt<Scalar>>& pts,
                          const std::vector<int>& n,
                          const std::vector<int>& h,
                          const VectorX& tau);

#ifdef PYBIND
#endif

}
