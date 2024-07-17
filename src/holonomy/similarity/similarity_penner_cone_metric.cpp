#include "holonomy/similarity/similarity_penner_cone_metric.h"

#include "holonomy/similarity/conformal.h"
#include "holonomy/similarity/constraint.h"
#include "holonomy/holonomy/holonomy.h"

#include <set>

#include "optimization/core/area.h"
#include "optimization/core/constraint.h"

namespace Penner {
namespace Holonomy {

void similarity_corner_angles(
    const SimilarityPennerConeMetric& similarity_metric,
    VectorX& he2angle,
    VectorX& he2cot)
{
    int num_halfedges = similarity_metric.n_halfedges();
    he2angle.setZero(num_halfedges);
    he2cot.setZero(num_halfedges);
    const Scalar cot_infty = 1e10;

    // Get integrated metric coordinates
    auto [metric_coords, u, is_cut_h] = similarity_metric.get_integrated_metric_coordinates();

    // Compute maps from halfedges to opposite angles and cotangents of opposite
    // angles
    int num_faces = similarity_metric.h.size();
    for (int f = 0; f < num_faces; f++) {
        // Halfedges of face f
        int hi = similarity_metric.h[f];
        int hj = similarity_metric.n[hi];
        int hk = similarity_metric.n[hj];

        // Lengths of the halfedges
        Scalar li = exp(metric_coords[hi] / 2.0);
        Scalar lj = exp(metric_coords[hj] / 2.0);
        Scalar lk = exp(metric_coords[hk] / 2.0);

        // Compute the cotangent of the angles
        Scalar Aijk4 = 4 * sqrt(std::max<Scalar>(Optimization::squared_area(li, lj, lk), 0.0));
        Scalar Ijk = (-li * li + lj * lj + lk * lk);
        Scalar iJk = (li * li - lj * lj + lk * lk);
        Scalar ijK = (li * li + lj * lj - lk * lk);
        he2cot[hi] = Aijk4 == 0.0 ? copysign(cot_infty, Ijk) : (Ijk / Aijk4);
        he2cot[hj] = Aijk4 == 0.0 ? copysign(cot_infty, iJk) : (iJk / Aijk4);
        he2cot[hk] = Aijk4 == 0.0 ? copysign(cot_infty, ijK) : (ijK / Aijk4);

#define USE_ACOS_HOLONOMY
#ifdef USE_ACOS_HOLONOMY
        he2angle[hi] = acos(std::min<Scalar>(std::max<Scalar>(Ijk / (2.0 * lj * lk), -1.0), 1.0));
        he2angle[hj] = acos(std::min<Scalar>(std::max<Scalar>(iJk / (2.0 * lk * li), -1.0), 1.0));
        he2angle[hk] = acos(std::min<Scalar>(std::max<Scalar>(ijK / (2.0 * li * lj), -1.0), 1.0));
#else
        // atan2 is prefered for stability
        he2angle[hi] = 0.0, he2angle[hj] = 0.0, he2angle[hk] = 0.0;
        // li: l12, lj: l23, lk: l31
        Scalar l12 = li, l23 = lj, l31 = lk;
        const Scalar t31 = +l12 + l23 - l31,
                    t23 = +l12 - l23 + l31,
                    t12 = -l12 + l23 + l31;
        // valid triangle
        if (t31 > 0 && t23 > 0 && t12 > 0)
        {
            const Scalar l123 = l12 + l23 + l31;
            const Scalar denom = sqrt(t12 * t23 * t31 * l123);
            he2angle[hj] = 2 * atan2(t12 * t31, denom); // a1 l23
            he2angle[hk] = 2 * atan2(t23 * t12, denom); // a2 l31
            he2angle[hi] = 2 * atan2(t31 * t23, denom); // a3 l12
        }
        else if (t31 <= 0)
            he2angle[hk] = M_PI;
        else if (t23 <= 0)
            he2angle[hj] = M_PI;
        else if (t12 <= 0)
            he2angle[hi] = M_PI;
        else
            he2angle[hj] = M_PI;
#endif
    }
}

Scalar ell(Scalar l, Scalar u0, Scalar u1, Scalar offset = 0)
{
    return l * exp((u0 + u1) / 2 - offset);
}

bool NonDelaunay(SimilarityPennerConeMetric& m, int e, SolveStats<Scalar>& solve_stats)
{
    if (m.type[m.h0(e)] == 4) return false; // virtual diagonal of symmetric trapezoid
    solve_stats.n_checks++;
    int hij = m.h0(e);
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hji = m.h1(e);
    int him = m.n[hji];
    int hmj = m.n[him];

    // triangles: hij, hjk, hki
    //            hji, him, hmj
    const VectorX& xi = m.get_one_form();
    if (abs(xi[hij] + xi[hjk] + xi[hki]) > 1e-10 || abs(xi[hji] + xi[him] + xi[hmj]) > 1e-10) {
        std::cerr << "error! xi not closed." << std::endl;
        std::cerr << "f" << m.f[hij] << ": " << hij << "," << hjk << "," << hki << std::endl;
        std::cerr << "f" << m.f[hji] << ": " << hji << "," << him << "," << hmj << std::endl;
        std::cerr << xi[hij] << ", " << xi[hjk] << ", " << xi[hki] << ", "
                  << xi[hij] + xi[hjk] + xi[hki] << std::endl;
        std::cerr << xi[hji] << ", " << xi[him] << ", " << xi[hmj] << ", "
                  << xi[hji] + xi[him] + xi[hmj] << std::endl;
        exit(0);
    }

    Scalar ui = 0;
    Scalar uj = xi[hij];
    Scalar uk = -xi[hki];
    Scalar um = xi[him];

    Scalar uijk_avg = (ui + uj + uk) / 3.0;
    Scalar ujim_avg = (uj + ui + um) / 3.0;
    Scalar ljk = ell(m.l[m.e(hjk)], uj, uk, uijk_avg);
    Scalar lki = ell(m.l[m.e(hki)], uk, ui, uijk_avg);
    Scalar lij = ell(m.l[m.e(hij)], ui, uj, uijk_avg);
    Scalar lji = ell(m.l[m.e(hji)], uj, ui, ujim_avg);
    Scalar lmj = ell(m.l[m.e(hmj)], um, uj, ujim_avg);
    Scalar lim = ell(m.l[m.e(him)], ui, um, ujim_avg);

    bool pre_flip_check = (ljk / lki + lki / ljk - (lij / ljk) * (lij / lki)) +
                              (lmj / lim + lim / lmj - (lji / lmj) * (lji / lim)) <
                          0;

    // additionally check whether delaunay is violated after flip
    // we consider the configuration to 'violate delaunay condition' only if
    // it does not satisfy delaunay check AND post-flip configuration satisfies delaunay condition.
    Scalar umki_avg = (um + uk + ui) / 3.0;
    Scalar ukmj_avg = (uk + um + uj) / 3.0;
    Scalar _lkm_non_scaled =
        (m.l[m.e(hjk)] * m.l[m.e(him)] + m.l[m.e(hki)] * m.l[m.e(hmj)]) / m.l[m.e(hij)];
    Scalar _lkm = ell(_lkm_non_scaled, uk, um, ukmj_avg);
    Scalar _lmj = ell(m.l[m.e(hmj)], um, uj, ukmj_avg);
    Scalar _ljk = ell(m.l[m.e(hjk)], uj, uk, ukmj_avg);
    Scalar _lmk = ell(_lkm_non_scaled, um, uk, umki_avg);
    Scalar _lki = ell(m.l[m.e(hki)], uk, ui, umki_avg);
    Scalar _lim = ell(m.l[m.e(him)], ui, um, umki_avg);
    bool post_flip_check = (_lki / _lim + _lim / _lki - (_lmk / _lki) * (_lmk / _lim)) +
                               (_ljk / _lmj + _lmj / _ljk - (_lkm / _ljk) * (_lkm / _lmj)) <
                           0;
    return pre_flip_check && !post_flip_check;
}

bool EdgeFlip(
    std::set<int>& q,
    Mesh<Scalar>& m,
    int e,
    int tag,
    DelaunayStats& delaunay_stats,
    bool Ptolemy = true)
{
    FlipStats flip_stats;
    bool success = ::Penner::EdgeFlip<Scalar>(
        m,
        e,
        tag,
        delaunay_stats.flip_seq,
        q,
        flip_stats,
        Ptolemy);
    delaunay_stats.n_flips += flip_stats.n_flips;
    delaunay_stats.n_flips_12 += flip_stats.n_flips_12;
    delaunay_stats.n_flips_q += flip_stats.n_flips_q;
    delaunay_stats.n_flips_s += flip_stats.n_flips_s;
    delaunay_stats.n_flips_t += flip_stats.n_flips_t;
    return success;
}

void MakeSimilarityDelaunay(
    SimilarityPennerConeMetric& m,
    DelaunayStats& delaunay_stats,
    SolveStats<Scalar>& solve_stats,
    bool Ptolemy)
{
    std::set<int> q;
    for (int i = 0; i < m.n_halfedges(); i++) {
        if (m.opp[i] < i) // Only consider halfedges with lower index to prevent duplication
            continue;
        int type0 = m.type[m.h0(i)];
        int type1 = m.type[m.h1(i)];
        if (type0 == 0 || type0 == 1 || type1 == 1 ||
            type0 == 3) // type 22 edges are flipped below; type 44 edges (virtual diagonals) are
                        // never flipped.
            q.insert(i);
    }
    while (!q.empty()) {
        int e = *(q.begin());
        q.erase(q.begin());
        int type0 = m.type[m.h0(e)];
        int type1 = m.type[m.h1(e)];
        if (!(type0 == 2 && type1 == 2) && !(type0 == 4) && NonDelaunay(m, e, solve_stats)) {
            int Re = -1;
            if (type0 == 1 && type1 == 1) Re = m.e(m.R[m.h0(e)]);
            if (!EdgeFlip(q, m, e, 0, delaunay_stats, Ptolemy)) continue;
            if (type0 == 1 && type1 == 1) // flip mirror edge on sheet 2
            {
                int e = Re;
                if (Re == -1) spdlog::info("Negative index");
                if (!EdgeFlip(q, m, e, 1, delaunay_stats, Ptolemy)) continue;
            }
            // checkR();
        }
    }
}

SimilarityPennerConeMetric::SimilarityPennerConeMetric(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa,
    const VectorX& harmonic_form_coords)
    : MarkedPennerConeMetric(m, metric_coords, homology_basis_loops, kappa)
    , m_harmonic_form_coords(harmonic_form_coords)
{
    MatrixX harmonic_form_matrix =
        build_dual_loop_basis_one_form_matrix(*this, m_homology_basis_loops);
    m_one_form = harmonic_form_matrix * m_harmonic_form_coords;
    m_one_form_direction = VectorX::Zero(m_one_form.size());
}

SimilarityPennerConeMetric::SimilarityPennerConeMetric(
    const Mesh<Scalar>& m,
    const VectorX& reduced_metric_coords,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa)
    : SimilarityPennerConeMetric(
          m,
          reduced_metric_coords.head(m.n_edges()),
          homology_basis_loops,
          kappa,
          reduced_metric_coords.tail(reduced_metric_coords.size() - m.n_edges()))
{}

VectorX SimilarityPennerConeMetric::get_reduced_metric_coordinates() const
{
    int num_length_coordinates = m_embed.size();
    int num_form_coordinates = m_harmonic_form_coords.size();
    VectorX reduced_metric_coords(num_length_coordinates + num_form_coordinates);
    for (int E = 0; E < num_length_coordinates; ++E) {
        int h = e2he[m_embed[E]];
        reduced_metric_coords[E] = 2.0 * log(l[h]);
    }
    for (int i = 0; i < num_form_coordinates; ++i) {
        reduced_metric_coords[num_length_coordinates + i] = m_harmonic_form_coords[i];
    }

    return reduced_metric_coords;
}

void SimilarityPennerConeMetric::get_corner_angles(VectorX& he2angle, VectorX& he2cot) const
{
    similarity_corner_angles(*this, he2angle, he2cot);
}

std::unique_ptr<DifferentiableConeMetric> SimilarityPennerConeMetric::set_metric_coordinates(
    const VectorX& reduced_metric_coords) const
{
    VectorX metric_coords;
    VectorX harmonic_form_coords;
    separate_coordinates(reduced_metric_coords, metric_coords, harmonic_form_coords);
    return std::make_unique<SimilarityPennerConeMetric>(SimilarityPennerConeMetric(
        *this,
        metric_coords,
        m_homology_basis_loops,
        kappa_hat,
        harmonic_form_coords));
}

// TODO Make projection to constraint, including projection of harmonic form coordinates
std::unique_ptr<DifferentiableConeMetric> SimilarityPennerConeMetric::scale_conformally(
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

bool SimilarityPennerConeMetric::constraint(
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices) const
{
    if (!only_free_vertices) {
        spdlog::warn("Similarity metric only supports free vertices");
    }
    compute_similarity_constraint_with_jacobian(*this, constraint, J_constraint, need_jacobian);
    return true;
}

std::unique_ptr<DifferentiableConeMetric> SimilarityPennerConeMetric::project_to_constraint(
    SolveStats<Scalar>& solve_stats,
    std::shared_ptr<Optimization::ProjectionParameters> proj_params) const
{
    solve_stats.n_solves++; // TODO Make accurate
    AlgorithmParameters alg_params;
    LineSearchParameters ls_params;
    alg_params.initial_ptolemy = proj_params->initial_ptolemy;
    alg_params.max_itr = proj_params->max_itr;
    alg_params.error_eps = double(proj_params->error_eps);
    alg_params.use_edge_flips = proj_params->use_edge_flips;
    ls_params.bound_norm_thres = double(proj_params->bound_norm_thres);
    ls_params.do_reduction = proj_params->do_reduction;

    // TODO Expose to interface
    ls_params.lambda0 = 1;
    ls_params.reset_lambda = false;

    // Get projected metric
    SimilarityPennerConeMetric projected_similarity_metric = *this;
    compute_conformal_similarity_metric(projected_similarity_metric, alg_params, ls_params);
    projected_similarity_metric.undo_flips();

    return std::make_unique<SimilarityPennerConeMetric>(
        projected_similarity_metric.scale_by_one_form());
}

void SimilarityPennerConeMetric::make_delaunay(std::vector<int>& flip_seq)
{
    // Make the copied mesh Delaunay with Ptolemy flips
    DelaunayStats del_stats;
    SolveStats<Scalar> solve_stats;
    bool use_ptolemy = true;
    MakeSimilarityDelaunay(*this, del_stats, solve_stats, use_ptolemy);
    flip_seq = del_stats.flip_seq;
    return;
}

void SimilarityPennerConeMetric::make_discrete_metric()
{
    // Make the copied mesh Delaunay with Ptolemy flips
    DelaunayStats del_stats;
    SolveStats<Scalar> solve_stats;
    bool use_ptolemy = true;
    MakeSimilarityDelaunay(*this, del_stats, solve_stats, use_ptolemy);
    m_is_discrete_metric = true;
    return;
}

std::tuple<VectorX, VectorX, std::vector<bool>>
SimilarityPennerConeMetric::get_integrated_metric_coordinates(std::vector<bool> cut_h) const
{
    std::vector<bool> is_cut_h;
    VectorX u = integrate_one_form(*this, m_one_form, cut_h, is_cut_h);
    VectorX metric_coords = get_metric_coordinates();
    metric_coords = scale_halfedges_by_integrated_one_form(*this, metric_coords, u);
    return std::make_tuple(metric_coords, u, is_cut_h);
}

VectorX SimilarityPennerConeMetric::reduce_one_form(const VectorX& one_form) const
{
    // Build psuedoinverse system
    MatrixX closed_one_form_matrix =
        build_closed_one_form_matrix(*this, m_homology_basis_loops, true);
    MatrixX A = closed_one_form_matrix.transpose() * closed_one_form_matrix;
    VectorX b = closed_one_form_matrix.transpose() * one_form;

    // Solve for the reduced basis coefficients
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A);
    VectorX coefficients = solver.solve(b);
    SPDLOG_TRACE(
        "Error is {}",
        sup_norm(one_form - closed_one_form_matrix * coefficients));
    return coefficients;
}

SimilarityPennerConeMetric SimilarityPennerConeMetric::scale_by_one_form() const
{
    // Get reduced 1-form coefficients
    VectorX coefficients = reduce_one_form(m_one_form);

    // Extract conformal scale factors and harmonic coefficients
    int num_vertices = n_ind_vertices();
    int num_loops = m_homology_basis_loops.size();
    VectorX u;
    u.setZero(num_vertices);
    std::vector<int> v_map;
    int num_angles;
    Optimization::build_free_vertex_map(*this, v_map, num_angles);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (v_map[vi] >= 0) {
            u[vi] = coefficients[v_map[vi]];
        }
    }
    VectorX harmonic_form_coords = coefficients.tail(num_loops);
    SPDLOG_TRACE("Minimum scale factor is {}", u.minCoeff());
    SPDLOG_TRACE("Maximum scale factor is {}", u.maxCoeff());

    // Solve for scaled metric coords
    int num_reduced_coordinates = m_embed.size();
    VectorX reduced_metric_coords(num_reduced_coordinates);
    for (int E = 0; E < num_reduced_coordinates; ++E) {
        int h = e2he[m_embed[E]];
        reduced_metric_coords[E] = 2.0 * log(l[h]) + (u[v_rep[to[h]]] + u[v_rep[to[opp[h]]]]);
    }

    return SimilarityPennerConeMetric(
        *this,
        reduced_metric_coords,
        m_homology_basis_loops,
        kappa_hat,
        harmonic_form_coords);
}

bool SimilarityPennerConeMetric::flip_ccw(int _h, bool Ptolemy)
{
    // Perform the flip in the base class
    bool success = MarkedPennerConeMetric::flip_ccw(_h, Ptolemy);

    // Update one form
    m_one_form[_h] = -(m_one_form[n[_h]] + m_one_form[n[n[_h]]]);
    m_one_form[opp[_h]] = -m_one_form[_h];

    // Update one form direction
    m_one_form_direction[_h] = -(m_one_form_direction[n[_h]] + m_one_form_direction[n[n[_h]]]);
    m_one_form_direction[opp[_h]] = -m_one_form_direction[_h];

    return success;
}

void SimilarityPennerConeMetric::separate_coordinates(
    const VectorX& reduced_metric_coords,
    VectorX& metric_coords,
    VectorX& harmonic_form_coords) const
{
    int num_coords = reduced_metric_coords.size();
    int num_basis_loops = m_homology_basis_loops.size();
    metric_coords = reduced_metric_coords.head(num_coords - num_basis_loops);
    harmonic_form_coords = reduced_metric_coords.tail(num_basis_loops);
}

} // namespace Holonomy
} // namespace Penner