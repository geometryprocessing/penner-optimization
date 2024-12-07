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
#include "optimization/parameterization/layout.h"

#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/edge_flaps.h>
#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Layout.hh"
#include "util/embedding.h"
#include "optimization/parameterization/interpolation.h"
#include "optimization/core/viewer.h"
#include "optimization/core/projection.h"
#include "optimization/core/constraint.h"
#include "optimization/parameterization/refinement.h"
#include "optimization/parameterization/translation.h"
#include "util/vector.h"
#include "util/vf_mesh.h"
#include "util/io.h"
#include "optimization/util/viewers.h"

// TODO: cleaning pass

namespace Penner {
namespace Optimization {

template <typename OverlayScalar>
OverlayMesh<Scalar> add_overlay(const Mesh<Scalar>& m, const VectorX& reduced_metric_coords)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Build refl projection and embedding
    std::vector<int> proj;
    std::vector<int> embed;
    build_refl_proj(m, he2e, e2he, proj, embed);

    // Build overlay mesh from mesh m
    Mesh<Scalar> m_l = m;

    // Convert mesh Penner coordinates to a halfedge length array l for m
    int num_halfedges = he2e.size();
    for (int h = 0; h < num_halfedges; ++h) {
        m_l.l[h] = exp(reduced_metric_coords[proj[he2e[h]]] / 2.0);
    }

    OverlayMesh<Scalar> mo(m_l);

    return mo;
}

template <typename OverlayScalar>
void make_tufted_overlay(OverlayMesh<OverlayScalar>& mo)
{
    auto& m = mo._m;
    if (m.type[0] == 0) return; // nothing to do for closed mesh

    int n_ind_v = m.n_ind_vertices();
    int n_he = m.n_halfedges();

    // Modify the to and out arrays to identify dependent vertices with their reflection
    m.out = std::vector<int>(n_ind_v);
    for (int i = 0; i < n_he; ++i)
    {
        m.out[m.v_rep[m.to[i]]] = i;
        m.to[i] = m.v_rep[m.to[i]];
    }
    m.v_rep = range(0, n_ind_v);
}

bool check_areas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    Eigen::VectorXd areas;
    igl::doublearea(V, F, areas);
    double min_area = areas.minCoeff() / 2.0;
    double max_area = areas.maxCoeff() / 2.0;
    spdlog::debug("Minimum VF face area: {}", min_area);
    spdlog::debug("Maximum VF face area: {}", max_area);

    return (min_area >= 0);
}

// compute the squared length of an edge between two vertices
double uv_length_squared(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    Eigen::Vector2d difference_vector = uv_1 - uv_0;
    double length_sq = difference_vector.dot(difference_vector);
    return length_sq;
}

// compute the length of an edge between two vertices
double uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    return sqrt(uv_length_squared(uv_0, uv_1));
}

double compute_uv_length_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    // Get the edge topology for the original uncut mesh
    Eigen::MatrixXi uE, EF, EI;
    Eigen::VectorXi EMAP;
    igl::edge_flaps(F, uE, EMAP, EF, EI);

    // Iterate over edges to check the length inconsistencies
    double max_uv_length_error = 0.0;
    for (Eigen::Index e = 0; e < EF.rows(); ++e) {
        // Get face corners corresponding to the current edge
        int f0 = EF(e, 0);
        int f1 = EF(e, 1);

        // Check first face (if not boundary)
        if (f0 < 0) continue;
        int i0 = EI(e, 0); // corner vertex face index
        int v0n = F_uv(f0, (i0 + 1) % 3); // next vertex
        int v0p = F_uv(f0, (i0 + 2) % 3); // previous vertex

        // Check second face (if not boundary)
        if (f1 < 0) continue;
        int i1 = EI(e, 1); // corner vertex face index
        int v1n = F_uv(f1, (i1 + 1) % 3); // next vertex
        int v1p = F_uv(f1, (i1 + 2) % 3); // next vertex

        // Compute the length of each halfedge corresponding to the corner in the cut mesh
        double l0 = uv_length(uv.row(v0n), uv.row(v0p));
        double l1 = uv_length(uv.row(v1n), uv.row(v1p));

        // Determine if the max length inconsistency has increased
        max_uv_length_error = max(max_uv_length_error, abs(l0 - l1));
    }

    // Return the max uv length error
    return max_uv_length_error;
}

bool check_uv(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    int n_faces = F.rows();
    bool is_valid = true;

    // Check faces agree in number
    if (F_uv.rows() != n_faces) {
        spdlog::error("Mesh and uv faces are not in one to one correspondence");
        is_valid = false;
    }

    // Check length consistency
    double uv_length_error = compute_uv_length_error(F, uv, F_uv);
    if (!float_equal(uv_length_error, 0.0, 1e-6)) {
        spdlog::warn("Inconsistent uv length error {} across VF edges", uv_length_error);
    }

    // Check uv face areas
    if (!check_areas(uv, F_uv)) {
        spdlog::error("Triangle is flipped in VF");
        is_valid = false;
    }

    // Return true if no issues found
    return is_valid;
}

// signed area of a triangle ABC
double signed_area(
    const Eigen::Vector2d& A,
    const Eigen::Vector2d& B,
    const Eigen::Vector2d& C)
{
    Eigen::Matrix<double, 3, 3> tet;
    tet.row(0) << A(0), A(1), 1.;
    tet.row(1) << B(0), B(1), 1.;
    tet.row(2) << C(0), C(1), 1.;

    return tet.determinant();
}

// find the previous halfedge of a halfedge in a mesh
template <typename Scalar>
int prev_halfedge(
    const OverlayMesh<Scalar>& m,
    int hij)
{
    int hli = hij;
    while (m.n[hli] != hij)
    {
        hli = m.n[hli];
    }

    return hli;
}

// compute the area of the triangle with vertices ijk
template <typename Scalar>
Scalar triangle_area(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v,
    int hij)
{
    int hjk = m.n[hij];
    int hli = prev_halfedge(m, hij);
    Eigen::Vector2d A = {u[hli], v[hli]};
    Eigen::Vector2d B = {u[hij], v[hij]};
    Eigen::Vector2d C = {u[hjk], v[hjk]};
    return signed_area(A, B, C);
}

// check that the signed area of the layout triangles are all positive
template <typename Scalar>
bool check_areas(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v)
{
    int num_halfedges = m.n_halfedges();
    Scalar min_area = triangle_area(m, u, v, 0);
    Scalar max_area = min_area;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        Scalar area_hijk = triangle_area(m, u, v, hij);
        min_area = min(area_hijk, min_area);
        max_area = max(area_hijk, max_area);
    }
    spdlog::debug("minimum overlay triangle face area: {}", min_area);
    spdlog::debug("maximum overlay triangle face area: {}", max_area);

    return (min_area >= 0);
}

// check that the difference of lengths of opposite halfedges
template <typename Scalar>
Scalar compute_uv_length_error(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v)
{
    int num_halfedges = m.n_halfedges();
    Scalar max_uv_length_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = m.opp[hij];

        // get other halfedges in the face for vertex computation
        int hki = prev_halfedge(m, hij);
        int hlj = prev_halfedge(m, hji);

        // get uv vertices on the edge
        Eigen::Vector2d uv0i = {u[hki], v[hki]};
        Eigen::Vector2d uv0j = {u[hij], v[hij]};
        Eigen::Vector2d uv1i = {u[hji], v[hji]};
        Eigen::Vector2d uv1j = {u[hlj], v[hlj]};

        // compute the length of each halfedge
        Scalar l0 = uv_length(uv0i, uv0j);
        Scalar l1 = uv_length(uv1i, uv1j);

        // determine if the max length inconsistency has increased
        if (abs(l0 - l1) > 1e-8)
        {
            spdlog::warn("uv length consistency error for edge {}, {} is {} - {}", hij, hji, l0, l1);
        }
        max_uv_length_error = max(max_uv_length_error, abs(l0 - l1));
    }

    // return the max uv length error
    return max_uv_length_error;
}

// check the edge consistency and signed area of a mesh layout
template <typename Scalar>
bool check_uv(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    const std::vector<Scalar>& v,
    const std::vector<bool>& is_cut)
{
    int num_halfedges = m.n_halfedges();
    bool is_valid = true;

    // Check faces agree in number
    if ((u.size() != num_halfedges) || (v.size() != num_halfedges)) {
        spdlog::error("uv coordinates not in correspondence with mesh halfedges");
        is_valid = false;
    }

    // Check length consistency
    Scalar uv_length_error = compute_uv_length_error(m, u, v);
    if (!float_equal(uv_length_error, 0.0, 1e-6)) {
        spdlog::warn("Inconsistent uv length error {} across edges", uv_length_error);
    }

    // Check uv face areas
    if (!check_areas(m, u, v)) {
        spdlog::error("Triangle is flipped in overlay");
        is_valid = false;
    }

    // Return true if no issues found
    return is_valid;
}

// check that two different layouts of a mesh have consistent lengths
template <typename Scalar>
bool check_uv_consistency(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u0,
    const std::vector<Scalar>& v0,
    const std::vector<Scalar>& u1,
    const std::vector<Scalar>& v1)
{
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> u_error(num_halfedges);
    std::vector<Scalar> v_error(num_halfedges);
    Scalar max_consistency_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        // get next halfedge
        int hjk = m.n[hij];

        // get uv vertices on the edge
        Eigen::Vector2d uv0j = {u0[hij], v0[hij]};
        Eigen::Vector2d uv0k = {u0[hjk], v0[hjk]};
        Eigen::Vector2d uv1j = {u1[hij], v1[hij]};
        Eigen::Vector2d uv1k = {u1[hjk], v1[hjk]};

        // compute the length of the halfedge in each metric
        Scalar l0 = uv_length(uv0j, uv0k);
        Scalar l1 = uv_length(uv1j, uv1k);

        if (abs(l0 - l1) > 1e-8)
        {
            spdlog::warn("uv length consistency error for {} with previous halfedge {} is {} - {}", hjk, hij, l0, l1);
            spdlog::warn("local face is {}, {}, {}, {}, ...", hij, hjk, m.n[hjk], m.n[m.n[hjk]]);
        }
        max_consistency_error = max(max_consistency_error, abs(l0 - l1));
    }
    spdlog::debug("max consistency error is {}", max_consistency_error);

    return (max_consistency_error < 1e-8);
}


std::vector<bool>
compute_layout_topology(const Mesh<Scalar>& m, const std::vector<bool>& is_cut_h, int start_h)
{
    // Determine if an initial cut is given
    bool cut_given = !is_cut_h.empty();
    auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);

    // set starting point - use a boundary edge if none given
    int h = 0;
    if (start_h == -1) {
        for (int i = 0; i < m.n_halfedges(); i++) {
            if (m.type[i] == 1 && m.type[m.opp[i]] == 2) {
                h = i;
            }
        }
    }

    // Keep a record of faces that have been laid out
    auto done = std::vector<bool>(m.n_faces(), false);

    // discard part 2 faces by marking as done
    for (size_t i = 0; i < done.size(); i++) {
        int hh = m.h[i];
        if (m.type[hh] == 2 && m.type[m.n[hh]] == 2 && m.type[m.n[m.n[hh]]] == 2) {
            done[i] = true;

            // mark edges of face as cut
            for (int _h : { hh, m.n[hh], m.n[m.n[hh]]} )
            {
                is_cut_h_gen[_h] = true;
                is_cut_h_gen[m.opp[_h]] = true;
            }
        }
    }

    // Initialize queue and record of faces to process
    std::queue<int> Q;
    Q.push(h);
    done[m.f[h]] = true;

    while (!Q.empty()) {
        // Get next halfedge to process
        h = Q.front();
        Q.pop();

        // Get other triangle edges
        int hn = m.n[h];
        int hp = m.n[hn];
        int hno = m.opp[hn];
        int hpo = m.opp[hp];
        int ho = m.opp[h];

        // Check if next edge triangle should be laid out
        if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn])) {
            done[m.f[hno]] = true;
            Q.push(hno);
        } else {
            is_cut_h_gen[hn] = true;
            is_cut_h_gen[m.opp[hn]] = true;
        }

        // Check if previous edge triangle should be laid out
        if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp])) {
            done[m.f[hpo]] = true;
            Q.push(hpo);
        } else {
            is_cut_h_gen[hp] = true;
            is_cut_h_gen[m.opp[hp]] = true;
        }

        // Check if current edge triangle should be laid out
        // NOTE: Should only be used once for original edge
        if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho])) {
            done[m.f[ho]] = true;
            Q.push(ho);
        }
    }

    // Check how many faces seen and edges cut
    // TODO Move to separate validity check function
    int num_done = std::count(done.begin(), done.end(), true);
    int num_cut = std::count(is_cut_h_gen.begin(), is_cut_h_gen.end(), true);
    spdlog::debug("{}/{} faces seen", num_done, m.n_faces());
    spdlog::debug("{}/{} halfedges cut", num_cut, is_cut_h_gen.size());
    auto is_found_vertex = std::vector<bool>(m.n_vertices(), false);
    for (int hi = 0; hi < m.n_halfedges(); ++hi) {
        int vi = m.to[hi];
        if (is_cut_h_gen[hi]) {
            is_found_vertex[vi] = true;
        }
    }
    int num_found_vertices = std::count(is_found_vertex.begin(), is_found_vertex.end(), true);
    spdlog::debug("{}/{} vertices seen", num_found_vertices, m.n_vertices());

    return is_cut_h_gen;
}

// FIXME Remove once fix halfedge origin
template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> compute_layout_components(
    Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    std::vector<bool>& is_cut_h,
    int start_h = -1)
{
    int num_halfedges = is_cut_h.size();

    auto _u = std::vector<Scalar>(m.n_halfedges(), 0.0);
    auto _v = std::vector<Scalar>(m.n_halfedges(), 0.0);

    bool cut_given = !is_cut_h.empty();
    auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);

    auto phi = std::vector<Scalar>(m.n_halfedges(), 0.0);
    auto xi = std::vector<Scalar>(m.n_halfedges(), 0.0);
    for (int i = 0; i < m.n_halfedges(); i++) {
        xi[i] = u[m.to[i]] - u[m.to[m.opp[i]]];
    }

    // set starting point - use a boundary edge
    int h = 0;
    if (start_h == -1) {
        for (int i = 0; i < m.n_halfedges(); i++) {
            if (m.type[i] == 0) break;
            if (m.type[i] == 1 && m.type[m.opp[i]] == 2) {
                h = m.n[m.n[i]];
                spdlog::debug("Using edge {} as layout start", h);
                break;
            }
        }
    } else {
        assert(m.f[start_h] != -1);
        h = m.n[m.n[start_h]];
    }

    _u[h] = 0.0;
    _v[h] = 0.0;
    phi[h] = 0.0;
    h = m.n[h];
    assert(m.f[h] != -1);

    phi[h] = xi[h];
    _u[h] = m.l[h] * exp((u[m.to[h]] + u[m.to[m.opp[h]]]) / 2);
    _v[h] = 0.0;
    auto done = std::vector<bool>(m.n_faces(), false);

    // discard part 2
    for (int i = 0; i < m.n_faces(); i++) {
        int hh = m.h[i];
        if (m.type[hh] == 2 || m.type[m.n[hh]] == 2 || m.type[m.n[m.n[hh]]] == 2) {
            // TODO
            //done[i] = true;
        }
    }
    // set edge type 2 as cut
    for (int i = 0; i < num_halfedges; i++) {
        if (m.type[i] == 2) {
            // TODO
            //is_cut_h[i] = true;
        }
    }

    std::queue<int> Q;
    Q.push(h);
    done[m.f[h]] = true;

    auto perp = [](Eigen::Matrix<Scalar, 1, 2> a) {
        Eigen::Matrix<Scalar, 1, 2> b;
        b[0] = -a[1];
        b[1] = a[0];
        return b;
    };

    auto area_from_len = [](Scalar l1, Scalar l2, Scalar l3) {
        auto s = 0.5 * (l1 + l2 + l3);
        return sqrt(s * (s - l1) * (s - l2) * (s - l3));
    };

    auto square = [](Scalar x) { return x * x; };

    while (!Q.empty()) {
        h = Q.front();
        Q.pop();
        int hn = m.n[h];
        int hp = m.n[hn];
        phi[hn] = phi[h] + xi[hn];
        Eigen::Matrix<Scalar, 1, 2> p1;
        p1[0] = _u[hp];
        p1[1] = _v[hp];
        Eigen::Matrix<Scalar, 1, 2> p2;
        p2[0] = _u[h];
        p2[1] = _v[h];
        assert(m.l[h] != 0.0);
        Scalar l0 = Scalar(1.0);
        Scalar l1 = exp((phi[hn] - phi[hp]) / 2) * (m.l[hn] / m.l[h]);
        Scalar l2 = exp((phi[hn] - phi[h]) / 2) * (m.l[hp] / m.l[h]);
        Eigen::Matrix<Scalar, 1, 2> pn = p1 +
                                         (p2 - p1) * (1 + square(l2 / l0) - square(l1 / l0)) / 2 +
                                         perp(p2 - p1) * 2 * area_from_len(1.0, l1 / l0, l2 / l0);
#ifdef CHECK_VALIDITY
        if (!float_equal((p1 - p2).norm(), m.l[h])) spdlog::error("inconsistent lengths {}, {}", (p1 - p2).norm(), m.l[h]);
        if (!float_equal((pn - p2).norm(), m.l[hn])) spdlog::error("inconsistent lengths {}, {}", (pn - p2).norm(), m.l[hn]);
        if (!float_equal((pn - p1).norm(), m.l[hp])) spdlog::error("inconsistent lengths {}, {}", (pn - p1).norm(), m.l[hp]);
#endif
        _u[hn] = pn[0];
        _v[hn] = pn[1];
        int hno = m.opp[hn];
        int hpo = m.opp[hp];
        int ho = m.opp[h];

        if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn])) {
            done[m.f[hno]] = true;
            phi[hno] = phi[h];
            phi[m.n[m.n[hno]]] = phi[hn];
            _u[hno] = _u[h];
            _v[hno] = _v[h];
            _u[m.n[m.n[hno]]] = _u[hn];
            _v[m.n[m.n[hno]]] = _v[hn];
            Q.push(hno);
        } else {
            is_cut_h_gen[hn] = true;
            is_cut_h_gen[m.opp[hn]] = true;
        }

        if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp])) {
            done[m.f[hpo]] = true;
            phi[hpo] = phi[hn];
            phi[m.n[m.n[hpo]]] = phi[hp];
            _u[hpo] = _u[hn];
            _v[hpo] = _v[hn];
            _u[m.n[m.n[hpo]]] = _u[hp];
            _v[m.n[m.n[hpo]]] = _v[hp];
            Q.push(hpo);
        } else {
            is_cut_h_gen[hp] = true;
            is_cut_h_gen[m.opp[hp]] = true;
        }

        if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho])) {
            done[m.f[ho]] = true;
            phi[ho] = phi[hp];
            phi[m.n[m.n[ho]]] = phi[h];
            _u[ho] = _u[hp];
            _v[ho] = _v[hp];
            _u[m.n[m.n[ho]]] = _u[h];
            _v[m.n[m.n[ho]]] = _v[h];
            Q.push(ho);
        }

        // If Q is empty and there are faces left, continue layout with a new halfedge
        if (Q.empty()) {
            for (int fi = 0; fi < m.n_faces(); ++fi) {
                if (!done[fi]) {
                    int h = m.h[fi];
                    _u[h] = 0.0;
                    _v[h] = 0.0;
                    phi[h] = 0.0;
                    h = m.n[h];
                    assert(m.f[h] != -1);

                    phi[h] = xi[h];
                    _u[h] = m.l[h] * exp((u[m.to[h]] + u[m.to[m.opp[h]]]) / 2);
                    _v[h] = 0.0;

                    Q.push(h);
                    done[fi] = true;
                    spdlog::debug("restarting layout from {}", h);
                    break;
                }
            }
        }
    }

    return std::make_tuple(_u, _v, is_cut_h_gen);
};

// Pull back cut on the current mesh to the overlay
template <typename Scalar>
std::vector<bool> pullback_current_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h)
{
    int num_halfedges = m_o.n_halfedges();
    std::vector<bool> is_cut_o = std::vector<bool>(num_halfedges, false);
    for (int hi = 0; hi < num_halfedges; ++hi) {
        if (m_o.edge_type[hi] == CURRENT_EDGE) {
            is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
        } else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE) {
            is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
        }
        // Don't cut edges not in the current mesh
        else if (m_o.edge_type[hi] == ORIGINAL_EDGE) {
            continue;
        }
    }

    return is_cut_o;
}

template <typename Scalar>
std::vector<bool> pullback_original_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h)
{
    int num_halfedges = m_o.n_halfedges();
    const auto& mc = m_o.cmesh();
    int num_current_halfedges = mc.n_halfedges();
    std::vector<bool> is_cut_o = std::vector<bool>(num_halfedges, false);
    std::vector<std::vector<int>> origin_halfedge_list(num_current_halfedges);
    for (int hi = 0; hi < num_halfedges; ++hi) {
        // Don't cut edges not in the original mesh
        if (m_o.edge_type[hi] == CURRENT_EDGE) {
            continue;
        } else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE) {
            int _h = m_o.origin_of_origin[hi];
            is_cut_o[hi] = is_cut_h[_h];
            origin_halfedge_list[_h].push_back(hi);
        } else if (m_o.edge_type[hi] == ORIGINAL_EDGE) {
            int _h = m_o.origin[hi];
            is_cut_o[hi] = is_cut_h[_h];
            origin_halfedge_list[_h].push_back(hi);
        }
    }

    for (int hij = 0; hij < num_current_halfedges; ++hij)
    {
        if (mc.opp[hij] < hij) continue;
        int list_size = origin_halfedge_list[hij].size();
        for (int i = 0; i < list_size - 1; ++i)
        {
            int _h = origin_halfedge_list[hij][i];
            is_cut_o[_h] = true;
            is_cut_o[m_o.opp[_h]] = true;
        }
    }

    return is_cut_o;
}

template <typename Scalar>
std::vector<bool> pullback_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h,
    bool is_original_cut)
{
    if (is_original_cut) {
        return pullback_original_cut_to_overlay(m_o, is_cut_h);
    } else {
        return pullback_current_cut_to_overlay(m_o, is_cut_h);
    }
}

// Helper function to determine if any faces in a triangle mesh are flipped
template <typename Scalar>
void check_if_flipped(Mesh<Scalar>& m, const std::vector<Scalar>& u, const std::vector<Scalar>& v)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> uv(u.size(), 2);
    Eigen::MatrixXi F_uv(m.n_faces(), 3);

    for (size_t i = 0; i < u.size(); ++i) {
        uv(i, 0) = static_cast<double>(u[i]);
        uv(i, 1) = static_cast<double>(v[i]);
    }

    for (int fi = 0; fi < m.n_faces(); ++fi) {
        int h = m.h[fi];
        F_uv(fi, 0) = h;
        F_uv(fi, 1) = m.n[h];
        F_uv(fi, 2) = m.n[m.n[h]];
    }

    Eigen::VectorXi flipped_f;
    igl::flipped_triangles(uv, F_uv, flipped_f);
    spdlog::debug("{} flipped elements in mesh", flipped_f.size());
    for (int i = 0; i < flipped_f.size(); ++i) {
        int fi = flipped_f[i];
        spdlog::debug("Face {} is flipped", F_uv.row(fi));
        spdlog::debug(
            "Vertices {}, {}, {}",
            uv.row(F_uv(fi, 0)),
            uv.row(F_uv(fi, 1)),
            uv.row(F_uv(fi, 2)));
    }
}

template <typename Scalar>
void view_halfedge_mesh_type(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u_vec,
    const std::vector<Scalar>& v_vec,
    const std::vector<char>& type)
{
    Eigen::VectorXd u, v;
    convert_std_to_eigen_vector(u_vec, u);
    convert_std_to_eigen_vector(v_vec, v);
    Eigen::MatrixXd uv(u.size(), 3);
    uv.col(0) = u;
    uv.col(1) = v;
    Eigen::MatrixXi F(m.n_faces(), 3);
    for (int f = 0; f < m.n_faces(); ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];
        F(f, 0) = hij;
        F(f, 1) = hjk;
        F(f, 2) = hki;
    }
#if ENABLE_VISUALIZATION
    spdlog::info("Viewing layout");
    polyscope::init();
    polyscope::registerSurfaceMesh2D("layout", uv, F)
        ->addVertexScalarQuantity("type", type);
    polyscope::show();
#endif
}

void check_angles(const Mesh<Scalar>& m)
{
    VectorX he2angle, he2cot;
    corner_angles(m, he2angle, he2cot);
    VectorX vertex_angles(m.n_vertices());
    for (int h = 0; h < m.n_halfedges(); ++h) {
        int v = m.to[h];
        vertex_angles[v] += he2angle[m.n[m.n[h]]] / (M_PI / 2.);
    }
    spdlog::info("Vertex angles: {}", vertex_angles.transpose());
}

/**
 * @brief Given overlay mesh with associated flat metric compute the layout
 *
 * @tparam Scalar double/mpfr::mpreal
 * @param m_o, overlay mesh
 * @param u_vec, per-vertex scale factor
 * @param singularities, list of singularity vertex ids
 * @param use_uniform_bc, (optional) use uniform edge barycentric coordinates where possible
 * @return u_o, v_o, is_cut_h (per-corner u/v assignment of overlay mesh and marked cut edges)
 */
template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>, std::vector<bool>>
get_consistent_layout(
    OverlayMesh<Scalar>& m_o,
    const std::vector<char>& type,
    const std::vector<Scalar>& u_vec,
    std::vector<int> singularities,
    const std::vector<bool>& is_cut_orig,
    const std::vector<bool>& is_cut,
    bool use_uniform_bc)
{
    // Get original overlay face labels
    auto f_labels = get_overlay_face_labels(m_o);

    // Compute layout of the underlying flipped mesh
    auto mc = m_o.cmesh();
    m_o.garbage_collection();
    if (!m_o.check(&mc)) {
        spdlog::error("Invalid overlay mesh for layout");
        return std::make_tuple(
            std::vector<Scalar>(),
            std::vector<Scalar>(),
            std::vector<bool>(),
            std::vector<bool>());
    }
    if (!overlay_has_all_original_halfedges(m_o)) {
        spdlog::error("Overlay mesh is missing an original edge");
        return std::make_tuple(
            std::vector<Scalar>(),
            std::vector<Scalar>(),
            std::vector<bool>(),
            std::vector<bool>());
    }

    auto mc_type = mc.type;
    mc.type = std::vector<char>(mc.n_halfedges(), 0);
    //std::vector<bool> _is_cut_place_holder; // TODO Remove
    // auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder, 0);
    std::vector<bool> _is_cut_place_holder = is_cut;
    auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder);
    auto _u_c = std::get<0>(layout_res);
    auto _v_c = std::get<1>(layout_res);
    auto is_cut_c = std::get<2>(layout_res);
    mc.type = mc_type;

#ifdef CHECK_VALIDITY
    if (!check_uv(mc, _u_c, _v_c, is_cut_c)) {
        spdlog::error("Inconsistent uvs in cut mesh");
    }
#endif

    std::vector<Scalar> u_o, v_o, u_unif, v_unif;

    // Interpolate layout to the overlay mesh
    Eigen::Matrix<Scalar, -1, 1> u_eig;
    u_eig.resize(u_vec.size());
    for (size_t i = 0; i < u_vec.size(); i++) {
        u_eig(i) = u_vec[i];
    }
    m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);

    // determine if coordinates are degenerate
    bool is_bc_degenerate;
    for (int i = 0; i < mc.n_halfedges(); i++)
    {
        if ((mc.type[mc.e(i)] == 3) || (mc.type[mc.e(i)] == 4))
        {
            continue; // skip edges crossing boundary
        }

        int h_prev = m_o.first_segment[i];
        int h_last = m_o.last_segment(i);
        if (h_prev == h_last) continue;

        int h = m_o.next_segment(h_prev);
        do {
            if (m_o.seg_bcs[h_prev][0] < 1e-12)
            {
                spdlog::info("Degenerate barycentric coordinates found");
                is_bc_degenerate = true;
                break;
            }
            if (float_equal(m_o.seg_bcs[h_prev][0], m_o.seg_bcs[h][0]))
            {
                spdlog::info("Degenerate barycentric coordinates found");
                is_bc_degenerate = true;
                break;
            }
            h = m_o.next_segment(h);
        }
        while (h != h_last);

        if (is_bc_degenerate) break;
    }

    if ((use_uniform_bc) || (is_bc_degenerate))
    {
        for (int i = 0; i < mc.n_halfedges(); i++)
        {
            if ((mc.type[mc.e(i)] == 3) || (mc.type[mc.e(i)] == 4))
            {
                continue; // skip edges crossing boundary
            }

            int h = m_o.first_segment[i];
            int h_last = m_o.last_segment(i);
            int num_crossings = 0;
            while (h != h_last)
            {
                ++num_crossings;
                h = m_o.next_segment(h);
            }

            h = m_o.first_segment[i];
            for (int j = 0; j < num_crossings; ++j)
            {
                if (m_o.vertex_type[m_o.to[h]] == ORIGINAL_VERTEX)
                {
                    spdlog::error("Changing bc of original vertex");
                }
                spdlog::trace("Initial coordiantes are {}, {}", m_o.seg_bcs[h][0], m_o.seg_bcs[h][1]);
                m_o.seg_bcs[h][0] = (num_crossings - j) / (num_crossings + 1.);
                m_o.seg_bcs[h][1] = (j + 1.) / (num_crossings + 1.);
                spdlog::trace("Setting coordiantes to {}, {}", m_o.seg_bcs[h][0], m_o.seg_bcs[h][1]);
                h = m_o.next_segment(h);
            }
        }
    }

    // check alignment
    for (int hij = 0; hij < mc.n_halfedges(); ++hij)
    {
      if (m_o.first_segment[hij] == m_o.last_segment(hij))
        continue;
      int hji = mc.opp[hij];

      // get the vertices on the _h side
      std::vector<Scalar> lambdas;
      int current_seg = m_o.first_segment[hij];
      while (m_o.vertex_type[m_o.to[current_seg]] != ORIGINAL_VERTEX)
      {
        std::vector<Scalar> tmp = m_o.seg_bcs[current_seg];
        lambdas.push_back(tmp[1]);
        current_seg = m_o.next_segment(current_seg);
      }
      // check the vertices on the _hopp side

      current_seg = m_o.first_segment[hji];
      int cnt = lambdas.size() - 1;
      while (m_o.vertex_type[m_o.to[current_seg]] != ORIGINAL_VERTEX)
      {
        std::vector<Scalar> tmp = m_o.seg_bcs[current_seg];
        if (abs(lambdas[cnt] - tmp[0]) > 1e-10)
        {
          spdlog::error("alignment problem: {}, {}", lambdas[cnt], tmp[0]);
        }
        cnt--;
        current_seg = m_o.next_segment(current_seg);
      }
    }

    u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _u_c);
    v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _v_c);
    //u_o = m_o.interpolate_along_c(_u_c);
    //v_o = m_o.interpolate_along_c(_v_c);
    spdlog::trace("Interpolate on overlay mesh done.");

#ifdef CHECK_VALIDITY
    if (!check_uv(m_o, u_o, v_o, is_cut_c)) {
        spdlog::error("Inconsistent uvs in overlay");
    }
#endif

    // Build a new mesh directly from the triangulated overlay mesh
    Mesh<Scalar> m;
    m.n = m_o.n;
    m.opp = m_o.opp;
    m.f = m_o.f;
    m.h = m_o.h;
    m.out = m_o.out;
    m.to = m_o.to;
    m.l = std::vector<Scalar>(m.n.size(), 0.0);
    for (int i = 0; i < m.n_halfedges(); i++) {
        int h0 = i;
        int h1 = h0;
        do {
            if (m.n[h1] == h0) break;
            h1 = m.n[h1];
        } while (h0 != h1);
        if (m.to[m.opp[h0]] != m.to[h1]) {
            spdlog::error("h0 h1 picked wrong.");
            exit(0);
        }
        m.l[h0] = sqrt(
            (u_o[h0] - u_o[h1]) * (u_o[h0] - u_o[h1]) + (v_o[h0] - v_o[h1]) * (v_o[h0] - v_o[h1]));
    }
    m.type = std::vector<char>(m.n.size(), 0);
    int num_m_halfedges = m.n.size();
    for (int hi = 0; hi < num_m_halfedges; ++hi) {
        if (!float_equal(m.l[hi], m.l[m.opp[hi]])) {
            spdlog::error("inconsistent edge lengths");
        }
        if (m_o.edge_type[hi] == CURRENT_EDGE) {
            m.type[hi] = 4;
        } else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE) {
            m.type[hi] = type[m_o.origin_of_origin[hi]];
        } else if (m_o.edge_type[hi] == ORIGINAL_EDGE) {
            m.type[hi] = type[m_o.origin[hi]];
        }
    }
    m.type_input = m.type;

    triangulate_polygon_mesh(m, u_o, v_o, f_labels);
    m.R = std::vector<int>(m.n.size(), 0);
    m.v_rep = range(0, m.out.size());
    m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);
    m.type.resize(m.n.size(), 4);
    OverlayMesh<Scalar> m_o_tri(m);
    for (int i = m_o.n_halfedges(); i < m_o_tri.n_halfedges(); i++) {
        m_o_tri.edge_type[i] = CURRENT_EDGE; // make sure do not use the new diagonal
    }
    bool view_layouts = false;
    if (view_layouts) {
        //view_halfedge_mesh_layout(m, u_o, v_o);
    }

    // Pullback cut on the original mesh to the overlay
    bool is_original_cut = !(is_cut_orig.empty());
    std::vector<bool> is_cut_poly;
    is_cut_poly = pullback_cut_to_overlay(m_o, is_cut_orig, is_original_cut);

    // Check validity (not needed)
    //Eigen::MatrixXi F_poly, F_uv_poly;
    //compute_layout_faces(mc.n_vertices(), m_o, is_cut_poly, F_poly, F_uv_poly);
    //int num_poly_components = count_components(F_uv_poly);
    //if (num_poly_components != 1) {
    //    spdlog::error("Overlay connectivity has {} components", num_poly_components);
    //}

    // Extend the overlay cut to the triangulated mesh
    // WARNING: Assumes triangulation halfedges added to the end
    std::vector<bool> is_cut_o = std::vector<bool>(m.n_halfedges(), false);
    for (int h = 0; h < m_o.n_halfedges(); ++h) {
        is_cut_o[h] = is_cut_poly[h];
    }

    // Check validity (not needed)
    //Eigen::MatrixXi F_tri, F_uv_tri;
    //compute_layout_faces(mc.n_vertices(), m, is_cut_o, F_tri, F_uv_tri);
    //int num_tri_components = count_components(F_uv_tri);
    //if (num_tri_components != 1) {
    //    spdlog::error("Triangulated overlay connectivity has {} components", num_tri_components);
    //}

    // Now directly do layout on triangulated overlay mesh
    std::vector<Scalar> phi(m.n_vertices(), 0.0);
    auto overlay_layout_res = compute_layout_components(m, phi, is_cut_o);
    std::vector<Scalar> _u_o = std::get<0>(overlay_layout_res);
    std::vector<Scalar> _v_o = std::get<1>(overlay_layout_res);
    is_cut_o = std::get<2>(overlay_layout_res);
    if (view_layouts) {
        //view_halfedge_mesh_type(m, _u_o, _v_o, m.type);
        //view_halfedge_mesh_layout(m, _u_o, _v_o);
    }

#ifdef CHECK_VALIDITY
    if (!check_uv(m, _u_o, _v_o, is_cut_o)) {
        spdlog::error("inconsistent uvs in triangulated overlay with {} haledges", m.n_halfedges());
    }
#endif

    // Restrict back to original overlay
    // WARNING: Assumes triangulation halfedges added to the end
    // WARNING: triangulation halfedges may be cut, but they are adjacent to flat vertices
    // and should be consistent if the input cut is accurate
    _u_o.resize(m_o.n.size());
    _v_o.resize(m_o.n.size());
    is_cut_o.resize(m_o.n.size());

    // Trim unnecessary branches of the cut graph
    bool do_trim = true;
    if (do_trim) {
        trim_open_branch(m_o, f_labels, singularities, is_cut_o);
    }

#ifdef CHECK_VALIDITY
    // Check for validity
    if (!check_uv(m_o, _u_o, _v_o, is_cut_o)) {
        spdlog::error("inconsistent uvs in overlay with {} haledges", m_o.n_halfedges());
    }
    if (!check_uv_consistency(m_o, _u_o, _v_o, u_o, v_o)) {
        spdlog::error("second overlay layout inconsistent for {} halfedges", m_o.n_halfedges());
    }
#endif

    return std::make_tuple(_u_o, _v_o, is_cut_c, is_cut_o);
}

template <typename OverlayScalar>
std::
    tuple<
        OverlayMesh<OverlayScalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    consistent_overlay_mesh_to_VL(
        const Mesh<Scalar>& _m,
        OverlayMesh<OverlayScalar>& mo,
        const std::vector<int>& vtx_reindex,
        const std::vector<bool>& is_bd,
        std::vector<Scalar>& u,
        std::vector<std::vector<OverlayScalar>>& V_overlay,
        std::vector<std::pair<int, int>>& endpoints,
        const std::vector<bool>& is_cut_orig,
        const std::vector<bool>& is_cut,
        bool use_uniform_bc)
{
    const auto& m = mo.cmesh();

    // get cones and bd
    std::vector<int> cones, bd;
    for (size_t i = 0; i < is_bd.size(); i++) {
        if (is_bd[i]) {
            bd.push_back(i);
        }
    }
    Scalar flat_angle = (m.type[0] != 0) ? (4. * M_PI) : (2. * M_PI);
    for (size_t i = 0; i < m.Th_hat.size(); i++) {
        if ((!is_bd[i]) && abs(m.Th_hat[i] - flat_angle) > 1e-12) {
            cones.push_back(i);
        }
    }

    std::vector<int> f_labels = get_overlay_face_labels(mo);

    SPDLOG_TRACE("Cone angles: {}", formatted_vector(m.Th_hat, "\n", 16));
    spdlog::trace("#bd_vt: {}", bd.size());
    spdlog::trace("#cones: {}", cones.size());
    spdlog::trace("mc.out size: {}", mo.cmesh().out.size());

    // get layout
    auto layout_res = get_consistent_layout<OverlayScalar>(
        mo,
        _m.type,
        convert_vector_type<Scalar, OverlayScalar>(u),
        cones,
        is_cut_orig,
        is_cut,
        use_uniform_bc);
    auto u_o = std::get<0>(layout_res);
    auto v_o = std::get<1>(layout_res);
    auto is_cut_h = std::get<2>(layout_res);
    auto is_cut_o = std::get<3>(layout_res);

    // get output VF and metric
    auto FVFT_res = get_FV_FTVT(mo, endpoints, is_cut_o, V_overlay, u_o, v_o);
    auto v3d = std::get<0>(FVFT_res);
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);
    auto Fn_to_F = std::get<5>(FVFT_res);
    auto remapped_endpoints = std::get<6>(FVFT_res);

    // v3d_out = v3d^T
    std::vector<std::vector<Scalar>> v3d_out(v3d[0].size());
    for (size_t i = 0; i < v3d[0].size(); i++) {
        v3d_out[i].resize(3);
        for (int j = 0; j < 3; j++) {
            v3d_out[i][j] = Scalar(v3d[j][i]);
        }
    }

    // reindex back
    auto u_o_out_copy = u_o_out;
    auto v_o_out_copy = v_o_out;
    auto v3d_out_copy = v3d_out;
    auto endpoints_out = remapped_endpoints;
    int num_vertices = vtx_reindex.size();
    for (size_t i = 0; i < F_out.size(); i++) {
        for (int j = 0; j < 3; j++) {
            if (F_out[i][j] < num_vertices) {
                F_out[i][j] = vtx_reindex[F_out[i][j]];
            }
            if (FT_out[i][j] < num_vertices) {
                FT_out[i][j] = vtx_reindex[FT_out[i][j]];
            }
        }
    }
    for (size_t i = 0; i < vtx_reindex.size(); i++) {
        u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
        v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
        v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
    }
    for (size_t i = vtx_reindex.size(); i < endpoints_out.size(); i++) {
        int a = vtx_reindex[endpoints_out[i].first];
        int b = vtx_reindex[endpoints_out[i].second];
        endpoints_out[i] = std::make_pair(a, b);
    }

    // Convert vector formats to matrices
    Eigen::MatrixXd V_o, uv_o;
    Eigen::VectorXd u_o_col, v_o_col;
    Eigen::MatrixXi F_o, FT_o;
    convert_std_to_eigen_matrix(v3d_out, V_o);
    convert_std_to_eigen_matrix(F_out, F_o);
    convert_std_to_eigen_matrix(FT_out, FT_o);
    convert_std_to_eigen_vector(u_o_out, u_o_col);
    convert_std_to_eigen_vector(v_o_out, v_o_col);
    uv_o.resize(u_o_col.size(), 2);
    uv_o.col(0) = u_o_col;
    uv_o.col(1) = v_o_col;

#ifdef CHECK_VALIDITY
    // Check for validity
    if (!check_uv(V_o, F_o, uv_o, FT_o)) {
        spdlog::error("Inconsistent uvs in VF");
    }
#endif

    return std::make_tuple(mo, V_o, F_o, uv_o, FT_o, is_cut_h, is_cut_o, Fn_to_F, endpoints_out);
}

[[deprecated]]
void compute_layout_faces(
    int origin_size,
    const OverlayMesh<Scalar>& m,
    const std::vector<bool>& is_cut_o,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& F_uv)
{
    int num_halfedges = m.n.size();
    std::vector<int> h_group(m.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    std::vector<int> to_group(origin_size, -1);
    std::vector<int> which_to_group(m.out.size(), -1);
    for (int i = 0; i < origin_size; i++) {
        to_group[i] = i;
        which_to_group[i] = i;
    }
    for (int i = 0; i < num_halfedges; i++) {
        if (h_group[i] != -1) continue;
        if (which_to_group[m.to[i]] == -1) {
            which_to_group[m.to[i]] = to_group.size();
            to_group.push_back(m.to[i]);
        }
        if (m.to[i] < origin_size && to_map[m.to[i]] == -1) {
            h_group[i] = m.to[i];
            to_map[m.to[i]] = i;
        } else {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = m.n[i];
        while (is_cut_o[cur] == false && m.opp[cur] != i) {
            cur = m.opp[cur];
            h_group[cur] = h_group[i];
            cur = m.n[cur];
        }
        cur = m.opp[i];
        while (is_cut_o[cur] == false && m.prev[cur] != i) {
            cur = m.prev[cur];
            h_group[cur] = h_group[i];
            cur = m.opp[cur];
        }
    }

    std::vector<std::vector<int>> F_out;
    std::vector<std::vector<int>> FT_out;
    int num_faces = m.h.size();
    for (int i = 0; i < num_faces; i++) {
        int h0 = m.h[i];
        int hc = m.n[h0];
        std::vector<int> hf;
        hf.push_back(h0);
        while (hc != h0) {
            hf.push_back(hc);
            hc = m.n[hc];
        }
        for (size_t j = 0; j < hf.size() - 2; j++) {
            FT_out.push_back(std::vector<int>{h_group[h0], h_group[hf[j + 1]], h_group[hf[j + 2]]});
            F_out.push_back(std::vector<int>{
                which_to_group[m.to[h0]],
                which_to_group[m.to[hf[j + 1]]],
                which_to_group[m.to[hf[j + 2]]]});
        }
    }

    convert_std_to_eigen_matrix(F_out, F);
    convert_std_to_eigen_matrix(FT_out, F_uv);
}


#ifdef PYBIND

#endif

template void make_tufted_overlay<Scalar>(OverlayMesh<Scalar>& mo);
template
std::
    tuple<
        OverlayMesh<Scalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    consistent_overlay_mesh_to_VL<Scalar>(
        const Mesh<Scalar>& _m,
        OverlayMesh<Scalar>& mo,
        const std::vector<int>& vtx_reindex,
        const std::vector<bool>& is_bd,
        std::vector<Scalar>& u,
        std::vector<std::vector<Scalar>>& V_overlay,
        std::vector<std::pair<int, int>>& endpoints,
        const std::vector<bool>& is_cut_orig,
        const std::vector<bool>& is_cut,
        bool use_uniform_bc);

#ifdef WITH_MPFR
#ifndef MULTIPRECISION

template void make_tufted_overlay<mpfr::mpreal>(OverlayMesh<mpfr::mpreal>& mo);

template
std::
    tuple<
        OverlayMesh<mpfr::mpreal>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    consistent_overlay_mesh_to_VL<mpfr::mpreal>(
        const Mesh<Scalar>& _m,
        OverlayMesh<mpfr::mpreal>& mo,
        const std::vector<int>& vtx_reindex,
        const std::vector<bool>& is_bd,
        std::vector<Scalar>& u,
        std::vector<std::vector<mpfr::mpreal>>& V_overlay,
        std::vector<std::pair<int, int>>& endpoints,
        const std::vector<bool>& is_cut_orig,
        const std::vector<bool>& is_cut,
        bool use_uniform_bc);
#endif
#endif


} // namespace Optimization
} // namespace Penner