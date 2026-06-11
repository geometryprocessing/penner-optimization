// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "metric/reparametrization.h"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "util/embedding.h"

/// FIXME Do cleaning pass

namespace Penner {

template <typename OverlayScalar>
void bc_reparametrize_eq(OverlayMesh<OverlayScalar>& m_o, const VectorX& tau)
{
    int num_halfedges = m_o.seg_bcs.size();
    spdlog::trace("Reparametrizing {} halfedges", num_halfedges);

    for (int h = 0; h < num_halfedges; h++) {
        if (m_o.edge_type[m_o.e(h)] == ORIGINAL_EDGE) continue;

        if (m_o.edge_type[m_o.e(h)] == -1) {
            spdlog::warn("Invalid edge index");
        }

        // Get origin halfedge and translation for the halfedge
        int _hij = m_o.origin[h];
        OverlayScalar tij(tau[_hij]);

        // Compute the coordinate for the image of the midpoint
        OverlayScalar Dij = exp(tij);
        OverlayScalar dij = exp(-tij);

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
        OverlayScalar sum = m_o.seg_bcs[h][0] + m_o.seg_bcs[h][1];
        if (sum < 1e-10) {
            spdlog::warn("Barycentric coordinate sum {} is numerically unstable", sum);
        }
        m_o.seg_bcs[h][0] /= sum;
        m_o.seg_bcs[h][1] /= sum;
    }
}

template <typename OverlayScalar>
void reparametrize_equilateral(
    std::vector<Pt<OverlayScalar>>& pts,
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
        OverlayScalar tij(tau[hij]);
        OverlayScalar tjk(tau[hjk]);
        OverlayScalar tki(tau[hki]);
        OverlayScalar Si = exp((-tij - tki) / 2.0);
        OverlayScalar Sj = exp((-tij + tjk) / 2.0);
        OverlayScalar Sk = exp((tjk + tki) / 2.0);
        pts[i].bc(0) *= Si;
        pts[i].bc(1) *= Sj;
        pts[i].bc(2) *= Sk;
        pts[i].bc /= pts[i].bc.sum();
    }
}

#ifdef PYBIND
#endif

template void bc_reparametrize_eq<Scalar>(OverlayMesh<Scalar>& m_o, const VectorX& tau);
template void reparametrize_equilateral(
    std::vector<Pt<Scalar>>& pts,
    const std::vector<int>& n,
    const std::vector<int>& h,
    const VectorX& tau);

#ifdef WITH_MPFR
#ifndef MULTIPRECISION

template void bc_reparametrize_eq<mpfr::mpreal>(OverlayMesh<mpfr::mpreal>& m_o, const VectorX& tau);
template void reparametrize_equilateral(
    std::vector<Pt<mpfr::mpreal>>& pts,
    const std::vector<int>& n,
    const std::vector<int>& h,
    const VectorX& tau);
#endif
#endif

} // namespace Penner