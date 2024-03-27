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
#include "cone_metric.hh"

#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "constraint.hh"
#include "projection.hh"

// TODO: Clean code

namespace CurvatureMetric {

DifferentiableConeMetric::DifferentiableConeMetric(const Mesh<Scalar>& m)
    : Mesh<Scalar>(m)
    , m_is_discrete_metric(false)
    , m_flip_seq(0)
{
    // Build edge maps
    build_edge_maps(m, he2e, e2he);
    m_identification = build_edge_matrix(he2e, e2he);
}

VectorX DifferentiableConeMetric::get_metric_coordinates() const
{
    int num_halfedges = n_halfedges();
    VectorX metric_coords(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        metric_coords[h] = 2.0 * log(l[h]);
    }

    return metric_coords;
}

void DifferentiableConeMetric::get_corner_angles(VectorX& he2angle, VectorX& he2cot) const
{
    if (m_is_discrete_metric) {
        corner_angles(*this, he2angle, he2cot);
    }
}

std::unique_ptr<DifferentiableConeMetric> DifferentiableConeMetric::project_to_constraint(
    std::shared_ptr<ProjectionParameters> proj_params) const
{
    SolveStats<Scalar> solve_stats;
    return project_to_constraint(solve_stats, proj_params);
}

bool DifferentiableConeMetric::constraint(
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices) const
{
    return constraint_with_jacobian(
        *this,
        constraint,
        J_constraint,
        need_jacobian,
        only_free_vertices);
}

void DifferentiableConeMetric::undo_flips()
{
    std::vector<int> flip_seq = m_flip_seq;
    for (auto h_iter = flip_seq.rbegin(); h_iter != flip_seq.rend(); ++h_iter) {
        int h = *h_iter;
        flip_ccw(h);
        flip_ccw(h);
        flip_ccw(h);
    }
    m_flip_seq = {};
}

int DifferentiableConeMetric::n_reduced_coordinates() const
{
    return get_reduced_metric_coordinates().size();
}

MatrixX DifferentiableConeMetric::change_metric_to_reduced_coordinates(
    const std::vector<int>& I,
    const std::vector<int>& J,
    const std::vector<Scalar>& V,
    int num_rows) const
{
    assert(I.size() == V.size());
    assert(J.size() == V.size());
    int num_entries = V.size();

    // Build triplet list
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_entries);
    for (int k = 0; k < num_entries; ++k) {
        tripletList.push_back(T(I[k], J[k], V[k]));
    }

    // Use triplet list method
    return change_metric_to_reduced_coordinates(tripletList, num_rows);
}

MatrixX DifferentiableConeMetric::change_metric_to_reduced_coordinates(
    const std::vector<Eigen::Triplet<Scalar>>& tripletList,
    int num_rows) const
{
    // Build Jacobian from triplets
    int num_entries = tripletList.size();
    int num_halfedges = n_halfedges();
    MatrixX halfedge_jacobian(num_rows, num_halfedges);
    halfedge_jacobian.reserve(num_entries);
    halfedge_jacobian.setFromTriplets(tripletList.begin(), tripletList.end());

    // Use matrix method
    return change_metric_to_reduced_coordinates(halfedge_jacobian);
}


MatrixX DifferentiableConeMetric::change_metric_to_reduced_coordinates(
    const MatrixX& halfedge_jacobian) const
{
    MatrixX J_transition = get_transition_jacobian();
    return halfedge_jacobian * J_transition;
}

PennerConeMetric::PennerConeMetric(const Mesh<Scalar>& m, const VectorX& metric_coords)
    : DifferentiableConeMetric(m)
    , m_need_jacobian(true)
    , m_transition_jacobian_lol(m.n_edges())
{
    build_refl_proj(m, he2e, e2he, m_proj, m_embed);
    m_projection = build_refl_matrix(m_proj, m_embed);
    expand_metric_coordinates(metric_coords);
}

VectorX PennerConeMetric::get_reduced_metric_coordinates() const
{
    return reduce_metric_coordinates(get_metric_coordinates());
}

std::unique_ptr<DifferentiableConeMetric> PennerConeMetric::clone_cone_metric() const
{
    return std::make_unique<PennerConeMetric>(PennerConeMetric(*this));
}

std::unique_ptr<DifferentiableConeMetric> PennerConeMetric::set_metric_coordinates(
    const VectorX& metric_coords) const
{
    return std::make_unique<PennerConeMetric>(PennerConeMetric(*this, metric_coords));
}

bool PennerConeMetric::flip_ccw(int halfedge_index, bool Ptolemy)
{
    // Perform the (Ptolemy) flip in the base class
    int _h = halfedge_index;
    m_flip_seq.push_back(_h);
    bool success = Mesh<Scalar>::flip_ccw(_h, Ptolemy);

    if (m_need_jacobian) {
        // Get local mesh information near hd
        int hd = _h;
        int hb = n[hd];
        int ha = n[hb];
        int hdo = opp[hd];
        int hao = n[hdo];
        int hbo = n[hao];

        // Get edges corresponding to halfedges
        int ed = he2e[hd];
        int eb = he2e[hb];
        int ea = he2e[ha];
        int eao = he2e[hao];
        int ebo = he2e[hbo];

        // Compute the shear for the edge ed
        // TODO Check if edge or halfedge coordinate
        Scalar la = l[ha];
        Scalar lb = l[hb];
        Scalar lao = l[hao];
        Scalar lbo = l[hbo];
        Scalar x = (la * lbo) / (lb * lao);

        // The matrix Pd corresponding to flipping edge ed is the identity except for
        // the row corresponding to edge ed, which has entries defined by Pd_scalars
        // in the column corresponding to the edge with the same index in Pd_edges
        std::vector<int> Pd_edges = {ed, ea, ebo, eao, eb};
        std::vector<Scalar> Pd_scalars =
            {-1.0, x / (1.0 + x), x / (1.0 + x), 1.0 / (1.0 + x), 1.0 / (1.0 + x)};
        m_transition_jacobian_lol.multiply_by_matrix(Pd_edges, Pd_scalars, ed);
    }

    return success;
}

std::unique_ptr<DifferentiableConeMetric> PennerConeMetric::scale_conformally(
    const VectorX& u) const
{
    int num_reduced_coordinates = m_embed.size();
    VectorX reduced_metric_coords(num_reduced_coordinates);
    for (int E = 0; E < num_reduced_coordinates; ++E) {
        int h = e2he[m_embed[E]];
        reduced_metric_coords[E] = 2.0 * log(l[h]) + (u[v_rep[to[h]]] + u[v_rep[to[opp[h]]]]);
    }

    return set_metric_coordinates(reduced_metric_coords);
}

std::unique_ptr<DifferentiableConeMetric> PennerConeMetric::project_to_constraint(
    SolveStats<Scalar>& solve_stats,
    std::shared_ptr<ProjectionParameters> proj_params) const
{
    VectorX u;
    u.setZero(n_ind_vertices());
    auto projection_out = compute_constraint_scale_factors(*this, u, proj_params);
    solve_stats = std::get<1>(projection_out);
    return scale_conformally(u);
}

void PennerConeMetric::make_discrete_metric()
{
    // Make the copied mesh Delaunay with Ptolemy flips
    VectorX u;
    u.setZero(n_ind_vertices());
    DelaunayStats del_stats;
    SolveStats<Scalar> solve_stats;
    bool use_ptolemy = true;
    ConformalIdealDelaunay<Scalar>::MakeDelaunay(*this, u, del_stats, solve_stats, use_ptolemy);
    m_is_discrete_metric = true;
    return;
}

MatrixX PennerConeMetric::get_transition_jacobian() const
{
    MatrixX transition_jacobian = get_flip_jacobian();
    return m_identification * (transition_jacobian * m_projection);
}

void PennerConeMetric::reset()
{
    // Initialize jacobian to the identity
    m_transition_jacobian_lol.reset();
    m_flip_seq.clear();
}

MatrixX PennerConeMetric::get_expansion_matrix() const
{
    return m_identification * m_projection;
}

VectorX PennerConeMetric::reduce_metric_coordinates(const VectorX& metric_coords) const
{
    int num_reduced_coordinates = m_embed.size();
    VectorX reduced_metric_coords(num_reduced_coordinates);
    for (int E = 0; E < num_reduced_coordinates; ++E) {
        int h = e2he[m_embed[E]];
        reduced_metric_coords[E] = metric_coords[h];
    }

    return reduced_metric_coords;
}

void PennerConeMetric::expand_metric_coordinates(const VectorX& metric_coords)
{
    int num_embedded_edges = m_embed.size();
    int num_edges = e2he.size();
    int num_halfedges = he2e.size();

    // Embedded edge lengths provided
    if (metric_coords.size() == num_embedded_edges) {
        for (int h = 0; h < num_halfedges; ++h) {
            l[h] = exp(metric_coords[m_proj[he2e[h]]] / 2.0);
        }
    } else if (metric_coords.size() == num_edges) {
        for (int h = 0; h < num_halfedges; ++h) {
            l[h] = exp(metric_coords[he2e[h]] / 2.0);
        }
    } else if (metric_coords.size() == num_halfedges) {
        for (int h = 0; h < num_halfedges; ++h) {
            l[h] = exp(metric_coords[h] / 2.0);
        }
    }
    // TODO error
}

DiscreteMetric::DiscreteMetric(const Mesh<Scalar>& m, const VectorX& log_length_coords)
    : DifferentiableConeMetric(m)
{
    m_is_discrete_metric = true;
    build_refl_proj(m, he2e, e2he, m_proj, m_embed);
    m_projection = build_refl_matrix(m_proj, m_embed);
    expand_metric_coordinates(log_length_coords);
}

VectorX DiscreteMetric::get_reduced_metric_coordinates() const
{
    return reduce_metric_coordinates(get_metric_coordinates());
}

bool DiscreteMetric::flip_ccw(int halfedge_index, bool Ptolemy)
{
    // Perform the Euclidean flip in the base class
    // TODO Add warning and validity check
    int _h = halfedge_index;
    m_flip_seq.push_back(_h);
    bool success = Mesh<Scalar>::flip_ccw(_h, Ptolemy);
    return success;
}

std::unique_ptr<DifferentiableConeMetric> DiscreteMetric::clone_cone_metric() const
{
    return std::make_unique<DiscreteMetric>(DiscreteMetric(*this));
}

std::unique_ptr<DifferentiableConeMetric> DiscreteMetric::set_metric_coordinates(
    const VectorX& metric_coords) const
{
    return std::make_unique<DiscreteMetric>(DiscreteMetric(*this, metric_coords));
}

std::unique_ptr<DifferentiableConeMetric> DiscreteMetric::scale_conformally(const VectorX& u) const
{
    int num_reduced_coordinates = m_embed.size();
    VectorX reduced_metric_coords(num_reduced_coordinates);
    for (int E = 0; E < num_reduced_coordinates; ++E) {
        int h = e2he[m_embed[E]];
        reduced_metric_coords[E] = 2.0 * log(l[h]) + (u[v_rep[to[h]]] + u[v_rep[to[opp[h]]]]);
    }

    return set_metric_coordinates(reduced_metric_coords);
}

std::unique_ptr<DifferentiableConeMetric> DiscreteMetric::project_to_constraint(
    SolveStats<Scalar>& solve_stats,
    std::shared_ptr<ProjectionParameters> proj_params) const
{
    VectorX u;
    u.setZero(n_ind_vertices());
    auto projection_out = compute_constraint_scale_factors(*this, u, proj_params);
    solve_stats = std::get<1>(projection_out);
    return scale_conformally(u);
}

void DiscreteMetric::make_discrete_metric()
{
    // Always true
    return;
}

MatrixX DiscreteMetric::get_transition_jacobian() const
{
    return m_identification * m_projection;
}

MatrixX DiscreteMetric::get_expansion_matrix() const
{
    return m_identification * m_projection;
}

VectorX DiscreteMetric::reduce_metric_coordinates(const VectorX& metric_coords) const
{
    int num_reduced_coordinates = m_embed.size();
    VectorX reduced_metric_coords(num_reduced_coordinates);
    for (int E = 0; E < num_reduced_coordinates; ++E) {
        int h = e2he[m_embed[E]];
        reduced_metric_coords[E] = metric_coords[h];
    }

    return reduced_metric_coords;
}

void DiscreteMetric::expand_metric_coordinates(const VectorX& metric_coords)
{
    int num_embedded_edges = m_embed.size();
    int num_edges = e2he.size();
    int num_halfedges = he2e.size();

    // Embedded edge lengths provided
    if (metric_coords.size() == num_embedded_edges) {
        for (int h = 0; h < num_halfedges; ++h) {
            l[h] = exp(metric_coords[m_proj[he2e[h]]] / 2.0);
        }
    } else if (metric_coords.size() == num_edges) {
        for (int h = 0; h < num_halfedges; ++h) {
            l[h] = exp(metric_coords[he2e[h]] / 2.0);
        }
    } else if (metric_coords.size() == num_halfedges) {
        for (int h = 0; h < num_halfedges; ++h) {
            l[h] = exp(metric_coords[h] / 2.0);
        }
    }
    // TODO error
}

} // namespace CurvatureMetric
