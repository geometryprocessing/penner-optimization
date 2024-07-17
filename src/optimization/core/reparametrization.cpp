/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#include "optimization/core/reparametrization.h"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "util/embedding.h"

/// FIXME Do cleaning pass

namespace Penner {
namespace Optimization {

void bc_reparametrize_eq(OverlayMesh<Scalar>& m_o, const VectorX& tau)
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
        if (sum < 1e-10) {
            spdlog::warn("Barycentric coordinate sum {} is numerically unstable", sum);
        }
        m_o.seg_bcs[h][0] /= sum;
        m_o.seg_bcs[h][1] /= sum;
    }
}

void reparametrize_equilateral(
    std::vector<Pt<Scalar>>& pts,
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

} // namespace Optimization
} // namespace Penner