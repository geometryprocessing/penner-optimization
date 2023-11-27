#include "reparametrization.hh"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "embedding.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

// Reparametrize the barycentric coordinates for the equilateral triangle by
// translating a constant hyperbolic distance along each halfedge. If the sum
// of translations per triangle is 0, then this corresponds to a projective map
// over the triangle.
//
// param[in, out] m_o: mesh to reparametrize
// param[in] tau: per halfedge hyperbolic translation distances
void
bc_reparametrize_eq(OverlayMesh<Scalar>& m_o, const VectorX& tau)
{
	int num_halfedges = m_o.seg_bcs.size();
	spdlog::trace("Reparametrizing {} halfedges", num_halfedges);

  for (int h = 0; h < num_halfedges; h++) {
    if (m_o.edge_type[m_o.e(h)] == ORIGINAL_EDGE)
      continue;

		if (m_o.edge_type[m_o.e(h)] == -1)
		{
			spdlog::warn("Invalid edge index");
		}

    // Get origin halfedge and translation for the halfedge
    int _hij = m_o.origin[h];
    Scalar tij = tau[_hij];

    // Compute the coordinate for the image of the midpoint
    Scalar Dij = exp(tij);
    Scalar dij = exp(-tij);

    // Compute new barycentric coordinates
    if (Dij > 1e10) {
      m_o.seg_bcs[h][0] = 1.0;
      m_o.seg_bcs[h][1] = 0.0;
      continue;
    }
    if (dij > 1e10) {
      m_o.seg_bcs[h][0] = 0.0;
      m_o.seg_bcs[h][1] = 1.0;
      continue;
    }
    m_o.seg_bcs[h][0] *= Dij;
    m_o.seg_bcs[h][1] *= dij;
    Scalar sum = m_o.seg_bcs[h][0] + m_o.seg_bcs[h][1];
		if (sum < 1e-10)
		{
			spdlog::warn("Barycentric coordinate sum {} is numerically unstable", sum);
		}
    m_o.seg_bcs[h][0] /= sum;
    m_o.seg_bcs[h][1] /= sum;
  }
}

void
reparametrize_equilateral(std::vector<Pt<Scalar>>& pts,
                          const std::vector<int>& n,
                          const std::vector<int>& h,
                          const VectorX& tau)
{
	spdlog::trace("Reparametrizing {} points", pts.size());

  for (size_t i = 0; i < pts.size(); i++) {
    int fid = pts[i].f_id;
    int hij = h[fid];
    int hjk = n[hij];
    int hki = n[hjk];
    Scalar tij = tau[hij];
    Scalar tjk = tau[hjk];
    Scalar tki = tau[hki];
    Scalar Si = exp((-tij - tki) / 2.0);
    Scalar Sj = exp((-tij + tjk) / 2.0);
    Scalar Sk = exp((tjk + tki) / 2.0);
    pts[i].bc(0) *= Si;
    pts[i].bc(1) *= Sj;
    pts[i].bc(2) *= Sk;
    pts[i].bc /= pts[i].bc.sum();
  }
}

#ifdef PYBIND
#endif
}
