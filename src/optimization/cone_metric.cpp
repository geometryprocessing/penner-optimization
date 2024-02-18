#include "cone_metric.hh"
#include "constraint.hh"
#include "projection.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"

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

std::unique_ptr<DifferentiableConeMetric> DifferentiableConeMetric::project_to_constraint(
    std::shared_ptr<ProjectionParameters> proj_params) const
{
    SolveStats<Scalar> solve_stats;
    return project_to_constraint(solve_stats, proj_params);
}

bool DifferentiableConeMetric::flip_ccw(int _h, bool Ptolemy)
{
    m_flip_seq.push_back(_h);
    return Mesh<Scalar>::flip_ccw(_h, Ptolemy);
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


bool DifferentiableConeMetric::constraint(
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices) const
{
    return constraint_with_jacobian(*this, constraint, J_constraint, need_jacobian, only_free_vertices);
}

int DifferentiableConeMetric::n_reduced_coordinates() const
{
    return get_reduced_metric_coordinates().size();
}

PennerConeMetric::PennerConeMetric(const Mesh<Scalar>& m, const VectorX& metric_coords)
    : DifferentiableConeMetric(m)
{
    build_refl_proj(m, he2e, e2he, m_proj, m_embed);
    build_refl_matrix(m_proj, m_embed, m_projection);
    expand_metric_coordinates(metric_coords);

    // Initialize jacobian to the identity
    int num_edges = e2he.size();
    m_transition_jacobian_lol =
        std::vector<std::vector<std::pair<int, Scalar>>>(num_edges, std::vector<std::pair<int, Scalar>>());
    for (int e = 0; e < num_edges; ++e) {
        m_transition_jacobian_lol[e].push_back(std::make_pair(e, 1.0));
    }
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

std::unique_ptr<DifferentiableConeMetric> PennerConeMetric::clone_cone_metric() const
{
    return std::make_unique<PennerConeMetric>(PennerConeMetric(*this));
}

std::unique_ptr<DifferentiableConeMetric> PennerConeMetric::set_metric_coordinates(
    const VectorX& metric_coords) const
{
    return std::make_unique<PennerConeMetric>(PennerConeMetric(*this, metric_coords));
}

VectorX PennerConeMetric::get_reduced_metric_coordinates() const
{
    return reduce_metric_coordinates(get_metric_coordinates());
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

MatrixX PennerConeMetric::get_transition_jacobian() const
{
    std::vector<T> tripletList;
    int num_edges = e2he.size();
    tripletList.reserve(5 * num_edges);
    for (int i = 0; i < num_edges; ++i) {
        for (auto it : m_transition_jacobian_lol[i]) {
            tripletList.push_back(T(i, it.first, it.second));
        }
    }

    // Create the matrix from the triplets
    MatrixX transition_jacobian;
    transition_jacobian.resize(num_edges, num_edges);
    transition_jacobian.reserve(tripletList.size());
    transition_jacobian.setFromTriplets(tripletList.begin(), tripletList.end());

    return m_identification * (transition_jacobian * m_projection);
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

bool PennerConeMetric::flip_ccw(int _h, bool Ptolemy)
{
    // Perform the flip in the base class
    bool success = DifferentiableConeMetric::flip_ccw(_h, Ptolemy);

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
    int num_entries = 0;
    for (int i = 0; i < 5; ++i) {
        int ei = Pd_edges[i];
        num_entries += m_transition_jacobian_lol[ei].size();
    }

    // Compute the new row of J_del corresponding to edge ed, which is the only
    // edge that changes
    std::vector<std::pair<int, Scalar>> J_del_d_new;
    J_del_d_new.reserve(num_entries);
    for (int i = 0; i < 5; ++i) {
        int ei = Pd_edges[i];
        Scalar Di = Pd_scalars[i];
        for (auto it : m_transition_jacobian_lol[ei]) {
            J_del_d_new.push_back(std::make_pair(it.first, Di * it.second));
        }
    }

    m_transition_jacobian_lol[ed] = J_del_d_new;

    return success;
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
    build_refl_matrix(m_proj, m_embed, m_projection);
    expand_metric_coordinates(log_length_coords);
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

std::unique_ptr<DifferentiableConeMetric> DiscreteMetric::clone_cone_metric() const
{
    return std::make_unique<DiscreteMetric>(DiscreteMetric(*this));
}

std::unique_ptr<DifferentiableConeMetric> DiscreteMetric::set_metric_coordinates(
    const VectorX& metric_coords) const
{
    return std::make_unique<DiscreteMetric>(DiscreteMetric(*this, metric_coords));
}

VectorX DiscreteMetric::get_reduced_metric_coordinates() const
{
    return reduce_metric_coordinates(get_metric_coordinates());
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


MatrixX DiscreteMetric::get_transition_jacobian() const
{
    return m_identification * m_projection;
}

void DiscreteMetric::make_discrete_metric()
{
    // Always true
    return;
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

bool DiscreteMetric::flip_ccw(int _h, bool Ptolemy)
{
    // Perform the flip in the base class
    // TODO Add warning and validity check
    return DifferentiableConeMetric::flip_ccw(_h, Ptolemy);
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
