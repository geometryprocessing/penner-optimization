#include "holonomy/holonomy/marked_penner_cone_metric.h"

#include "util/vector.h"
#include "holonomy/core/viewer.h"
#include "holonomy/holonomy/constraint.h"
#include "holonomy/holonomy/holonomy.h"
#include "holonomy/holonomy/newton.h"

#include "optimization/core/constraint.h"
#include "optimization/core/projection.h"

#include "conformal_ideal_delaunay/ConformalInterface.hh"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

bool is_reflection_structure_valid(
    const std::vector<int>& next,
    const std::vector<int>& prev,
    const std::vector<int>& opp,
    const std::vector<int>& R,
    const std::vector<char>& type)
{
    int num_halfedges = next.size();
    if (prev.size() != next.size()) return false;

    // check reflection structure
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // check R has order 2
        if (R[R[hij]] != hij) {
            spdlog::warn("{} is not invariant under R-R", hij);
            return false;
        }

        // check R preserves opp
        if (R[opp[hij]] != opp[R[hij]]) {
            spdlog::warn("{} under opp is not preserved by R", hij);
            return false;
        }

        // check R inverts next
        // TODO: Fails for quads
        //if ((type[hij] != 4) && (R[prev[hij]] != next[R[hij]])) {
        //    spdlog::warn("{} under next is not inverted by R", hij);
        //    spdlog::warn("{}, {}, {}, {}", prev[hij], R[prev[hij]], R[hij], next[R[hij]]);
        //    return false;
        //}

        // check edge typing
        if ((type[hij] == 1) && (type[R[hij]] != 2))
        {
            spdlog::warn("{} type (1) is not changed by R", hij);
            return false;
        }
        if ((type[hij] == 2) && (type[R[hij]] != 1))
        {
            spdlog::warn("{} type (2) is not changed by R", hij);
            return false;
        }
        if ((type[hij] == 3) && (type[R[hij]] != 3))
        {
            spdlog::warn("{} type (3) is not fixed by R", hij);
            return false;
        }
        if ((type[hij] == 4) && (type[R[hij]] != 4))
        {
            spdlog::warn("{} type (4) is not fixed by R", hij);
            return false;
        }
    }

    return true;
}

bool is_valid_mesh(const Mesh<Scalar>& m)
{
    if (m.n_halfedges() == 0) return true;

    // build previous map
    std::vector<int> prev = invert_map(m.n);

    // check edge, face, and vertex conditions
    if (!are_polygon_mesh_edges_valid(m.n, prev))
    {
        spdlog::warn("Edges are invalid");
        return false;
    }
    if (!are_polygon_mesh_faces_valid(m.n, m.f, m.h))
    {
        spdlog::warn("Faces are invalid");
        return false;
    }
    if (!are_polygon_mesh_vertices_valid(m.opp, prev, m.to, m.out))
    {
        spdlog::warn("Vertices are invalid");
        return false;
    }

    // check reflection if doubled mesh
    if ((m.type[0] != 0) && (!is_reflection_structure_valid(m.n, prev, m.opp, m.R, m.type)))
    {
        spdlog::warn("Symmetry structure is invalid");
        return false;
    }

    return true;
}

MarkedPennerConeMetric::MarkedPennerConeMetric()
    : m_dual_loop_manager(0)
{}

MarkedPennerConeMetric::MarkedPennerConeMetric(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa)
    : Optimization::PennerConeMetric(m, metric_coords)
    , kappa_hat(kappa)
    , m_dual_loop_manager(m.n_edges())
{
    assert(is_valid_mesh(m));

    int num_basis_loops = homology_basis_loops.size();
    m_homology_basis_loops.reserve(num_basis_loops);
    for (int i = 0; i < num_basis_loops; ++i) {
        m_homology_basis_loops.push_back(homology_basis_loops[i]->clone());
        m_dual_loop_manager.register_loop_edges(i, m, *homology_basis_loops[i]);
    }

    // TODO
    int num_halfedges = m.n_halfedges();
    original_coords.resize(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        original_coords[h] = 2. * log(l[h]);
    }
}


MarkedPennerConeMetric::MarkedPennerConeMetric(const MarkedPennerConeMetric& marked_metric)
    : MarkedPennerConeMetric(
          marked_metric,
          marked_metric.get_metric_coordinates(),
          marked_metric.get_homology_basis_loops(),
          marked_metric.kappa_hat)
{
    assert(is_valid_mesh(marked_metric));
}

void MarkedPennerConeMetric::copy_connectivity(const MarkedPennerConeMetric& m)
{
    n = m.n;
    to = m.to;
    f = m.f;
    h = m.h;
    out = m.out;
    opp = m.opp;
    l = m.l;
    type = m.type;
    type_input = m.type_input;
    R = m.R;
    v_rep = m.v_rep;
    Th_hat = m.Th_hat;
    fixed_dof = m.fixed_dof;
    pts = m.pts;
    pt_in_f = m.pt_in_f;

    he2e = m.he2e;
    e2he = m.e2he;
}

void MarkedPennerConeMetric::copy_metric(const MarkedPennerConeMetric& m)
{
    m_is_discrete_metric = m.m_is_discrete_metric;
    m_flip_seq = m.m_flip_seq;
    m_identification = m.m_identification;
    m_embed = m.m_embed;
    m_proj = m.m_proj;
    m_projection = m.m_projection;
    m_need_jacobian = m.m_need_jacobian;
    m_transition_jacobian_lol = m.m_transition_jacobian_lol;
}

void MarkedPennerConeMetric::copy_holonomy(const MarkedPennerConeMetric& m)
{
    kappa_hat = m.kappa_hat;
    m_dual_loop_manager = m.m_dual_loop_manager;
    int num_basis_loops = m.get_homology_basis_loops().size();
    m_homology_basis_loops.clear();
    m_homology_basis_loops.reserve(num_basis_loops);
    for (int i = 0; i < num_basis_loops; ++i) {
        m_homology_basis_loops.push_back(m.get_homology_basis_loops()[i]->clone());
    }
}

void MarkedPennerConeMetric::operator=(const MarkedPennerConeMetric& m)
{
    copy_connectivity(m);
    copy_metric(m);
    copy_holonomy(m);
}


void MarkedPennerConeMetric::reset_connectivity(const MarkedPennerConeMetric& m)
{
    // Halfedge arrays
    int num_halfedges = n_halfedges();
    for (int h = 0; h < num_halfedges; ++h) {
        n[h] = m.n[h];
        to[h] = m.to[h];
        f[h] = m.f[h];
        l[h] = m.l[h];
        type[h] = m.type[h];
        R[h] = m.R[h];

        // opp, he2e, e2he do not change
    }

    // Vertex arrays
    int num_vertices = n_vertices();
    for (int v = 0; v < num_vertices; ++v) {
        out[v] = m.out[v];

        // v_rep, Th_hat, fixed_dof do not change
    }

    // Face arrays
    int num_faces = n_faces();
    for (int f = 0; f < num_faces; ++f) {
        h[f] = m.h[f];
    }
}

void MarkedPennerConeMetric::reset_markings(const MarkedPennerConeMetric& m)
{
    // Loop data
    int num_basis_loops = m.n_homology_basis_loops();
    m_homology_basis_loops.resize(num_basis_loops);
    for (int i = 0; i < num_basis_loops; ++i) {
        m_homology_basis_loops[i] = m.get_homology_basis_loops()[i]->clone();

    }
    kappa_hat = m.kappa_hat;
}

void MarkedPennerConeMetric::reset_marked_metric(const MarkedPennerConeMetric& m)
{
    // Reset connectivity and markings
    reset_connectivity(m);
    reset_markings(m);

    // Clear flip data
    m_is_discrete_metric = false;
    m_flip_seq.clear();
    PennerConeMetric::reset();
}

void MarkedPennerConeMetric::change_metric(
    const MarkedPennerConeMetric& m,
    const VectorX& metric_coords,
    bool need_jacobian,
    bool do_repeat_flips)
{
    // Restore connectivity to that of m
    reset_connectivity(m);

    // Change metric coordinates
    PennerConeMetric::expand_metric_coordinates(metric_coords);
    std::vector<int> flip_seq = m_flip_seq;
    spdlog::debug("Repeating {} flips", m_flip_seq.size());
    m_flip_seq.clear();
    m_is_discrete_metric = false;
    m_need_jacobian = need_jacobian;
    PennerConeMetric::reset();

    // TODO
    int num_halfedges = m.n_halfedges();
    original_coords.resize(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        original_coords[h] = 2. * log(l[h]);
    }

    // Flip back to current connectivity if flag set
    if (do_repeat_flips) {
        for (int h : flip_seq) {
            PennerConeMetric::flip_ccw(h);
        }
        spdlog::debug("{} flips performed", m_flip_seq.size());
    }
    // Reset markings if using original connectivity
    else {
        reset_markings(m);
    }
}

std::unique_ptr<DifferentiableConeMetric> MarkedPennerConeMetric::set_metric_coordinates(
    const VectorX& metric_coords) const
{
    return std::make_unique<MarkedPennerConeMetric>(
        MarkedPennerConeMetric(*this, metric_coords, m_homology_basis_loops, kappa_hat));
}

bool MarkedPennerConeMetric::constraint(
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices) const
{
    compute_metric_constraint_with_jacobian(
        *this,
        constraint,
        J_constraint,
        need_jacobian,
        only_free_vertices);
    return true;
}

Scalar MarkedPennerConeMetric::max_constraint_error() const
{
    VectorX cons;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = true;
    constraint(cons, J_constraint, need_jacobian, only_free_vertices);
    return cons.cwiseAbs().maxCoeff();
}

VectorX MarkedPennerConeMetric::constraint(const VectorX& angles)
{
    return compute_metric_constraint(*this, angles);
}

MatrixX MarkedPennerConeMetric::constraint_jacobian(const VectorX& cotangents)
{
    return compute_metric_constraint_jacobian(*this, cotangents);
}

std::unique_ptr<DifferentiableConeMetric> MarkedPennerConeMetric::project_to_constraint(
    SolveStats<Scalar>& solve_stats,
    std::shared_ptr<Optimization::ProjectionParameters> proj_params) const
{
    // Copy parameters
    NewtonParameters alg_params;
    alg_params.max_itr = proj_params->max_itr;
    alg_params.bound_norm_thres = proj_params->bound_norm_thres;
    alg_params.error_eps = proj_params->error_eps;
    alg_params.do_reduction = proj_params->do_reduction;
    alg_params.output_dir = proj_params->output_dir;
    alg_params.solver = "ldlt";
    spdlog::info("Doing projection");
    spdlog::info("Output to {}", alg_params.output_dir);

    // Optimize metric angles
    MatrixX identity = id_matrix(n_reduced_coordinates());
    NewtonLog log;
    MarkedPennerConeMetric optimized_metric =
        optimize_subspace_metric_angles_log(*this, identity, alg_params, log);

    // Return output
    solve_stats.n_solves = log.num_iter;
    return std::make_unique<MarkedPennerConeMetric>(optimized_metric);
}

bool MarkedPennerConeMetric::flip_ccw(int _h, bool Ptolemy)
{
    // Flip the homology basis loops
    bool do_bypass_manager = false;
    if (do_bypass_manager) {
        for (auto& homology_basis_loop : m_homology_basis_loops) {
            homology_basis_loop->update_under_ccw_flip(*this, _h);
        }
    } else {
        for (int loop_index : m_dual_loop_manager.get_edge_loops(he2e[_h])) {
            // Update loop
            m_homology_basis_loops[loop_index]->update_under_ccw_flip(*this, _h);

            // Add all adjacent edges (conservative guess at actual adjacent edges)
            for (int adj_h : {n[_h], n[n[_h]], n[opp[_h]], n[n[opp[_h]]]}) {
                m_dual_loop_manager.add_loop(he2e[adj_h], loop_index);
            }
        }
    }

    // Perform the flip in the base class
    bool success = PennerConeMetric::flip_ccw(_h, Ptolemy);

    return success;
}

void MarkedPennerConeMetric::write_status_log(std::ostream& stream, bool write_header)
{
    if (write_header) {
        stream << "num_flips,";
        stream << std::endl;
    }

    stream << num_flips() << ",";
    stream << std::endl;
}


} // namespace Holonomy
} // namespace Penner
