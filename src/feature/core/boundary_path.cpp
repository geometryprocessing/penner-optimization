#include "feature/core/boundary_path.h"
#include "util/boundary.h"

namespace Penner {
namespace Feature {

BoundaryPath::BoundaryPath(const Mesh<Scalar>& m, int vertex_index)
    : m_start_vertex(vertex_index)
{
    assert(vertex_index < m.n_vertices());

    // Get a starting halfedge near the symmetry line emanating from the vertex
    int vi = vertex_index;
    int hij = m.out[vi];
    if (m.opp[m.n[m.n[hij]]] != hij) {
        // Iterate ccw until we have an edge in the primal mesh
        int h_start = hij;
        while (m.type[hij] != 1) {
            hij = m.opp[m.n[m.n[hij]]];

            // Safety check for full circulation
            if (hij == h_start) {
                throw std::runtime_error("Invalid vertex index: no type 1 edge");
                return;
            }
        }

        // Iterate clockwise until we are opposite a copy halfedge
        h_start = hij;
        while (m.type[m.opp[hij]] != 2) {
            hij = m.n[m.opp[hij]];

            // Safety check for full circulation
            if (hij == h_start) {
                throw std::runtime_error("Invalid vertex index: no type 2 edge");
                return;
            }
        }
    }

    // From starting halfedge, follow path to next boundary halfedge.
    // There are two options for this path:
    //     1. A single primal halfedge
    //     2a. Two dual halfedges adjacent to a common type 3 edge
    //     2b. The same as (2a) with a sequence of quad halfedges in between
    // We identify which case we are in and proceed accordingly
    m_halfedge_path.clear();
    m_transverse_edges.clear();

    // Case 1
    if (m.type[hij] == 1) {
        // Add only edge
        m_halfedge_path.push_back(hij);
    }
    // Case 2
    else {
        // Add starting boundary edge
        int h_bd = hij;
        m_halfedge_path.push_back(h_bd);

        // Move to next edge, which is necessarily type 3
        int h_perp = m.n[h_bd];
        m_transverse_edges.push_back(h_perp);
        assert(m.type[h_perp] == 3);

        while (true) {
            // Circulate clockwise
            int hjk = m.n[m.opp[h_perp]];

            // case: current edge is a diagonal
            if (m.type[hjk] == 4) {
                // Continue circulating to quad boundary edge
                h_bd = m.n[m.opp[hjk]];
                m_halfedge_path.push_back(h_bd);

                // Get next type 3 edge
                h_perp = m.n[h_bd];
                m_transverse_edges.push_back(h_perp);
                assert(m.type[h_perp] == 3);
            }
            // case: next edge is a diagonal
            else if (m.type[m.n[hjk]] == 4) {
                h_bd = hjk;
                m_halfedge_path.push_back(h_bd);

                // Get next type 3 edge, skipping diagonal
                h_perp = m.n[m.opp[m.n[h_bd]]];
                m_transverse_edges.push_back(h_perp);
                assert(m.type[h_perp] == 3);
            }
            // case: neither edge is a diagonal, so in terminating triangle
            else {
                // add final edge and break loop
                h_bd = hjk;
                m_halfedge_path.push_back(h_bd);
                break;
            }
        }
    }

    // Record end vertex
    m_end_vertex = m.to[m_halfedge_path.back()];

    assert(is_valid_boundary_path(m));
}

Scalar BoundaryPath::compute_length(const Mesh<Scalar>& m) const
{
    int num_segments = m_halfedge_path.size();

    // special case: empty path
    if (num_segments == 0) return 0.;

    // special case: single segment; use length directly
    if (num_segments == 1) {
        int hij = m_halfedge_path[0];
        return m.l[hij];
    }

    // add contribution from first triangle
    int h_bd = m_halfedge_path.front();
    int h_perp = m_transverse_edges.front();
    Scalar length = 0.0;
    length += compute_tri_length(m.l[h_bd], m.l[h_perp]);

    // add contributions for any quads
    for (int i = 1; i < num_segments - 1; ++i) {
        int h_bd = m_halfedge_path[i];
        int h_perp_0 = m.opp[m_transverse_edges[i - 1]];
        int h_perp_1 = m_transverse_edges[i];
        length += compute_quad_length(m.l[h_bd], m.l[h_perp_0], m.l[h_perp_1]);
    }

    // add contribution from last triangle
    h_bd = m_halfedge_path.back();
    h_perp = m.opp[m_transverse_edges.back()];
    length += compute_tri_length(m.l[h_bd], m.l[h_perp]);

    return length;
}

Scalar BoundaryPath::compute_log_length(const Mesh<Scalar>& m) const
{
    return 2. * log(compute_length(m));
}


std::vector<std::pair<int, Scalar>> BoundaryPath::compute_length_jacobian(
    const Mesh<Scalar>& m) const
{
    int num_segments = m_halfedge_path.size();
    std::vector<std::pair<int, Scalar>> length_jacobian = {};
    length_jacobian.reserve(4 * num_segments);

    // special case: empty path
    if (num_segments == 0) return length_jacobian;

    // special case: single segment
    if (num_segments == 1) {
        int hij = m_halfedge_path[0];
        length_jacobian.push_back(std::make_pair(hij, m.l[hij] / 2.));
        return length_jacobian;
    }

    // add contribution from first triangle
    int h_bd = m_halfedge_path.front();
    int h_perp = m_transverse_edges.front();
    Scalar bd_deriv = compute_tri_side_derivative(m.l[h_bd], m.l[h_perp]);
    Scalar perp_deriv = compute_tri_base_derivative(m.l[h_bd], m.l[h_perp]);
    length_jacobian.push_back(std::make_pair(h_bd, bd_deriv));
    length_jacobian.push_back(std::make_pair(h_perp, perp_deriv));

    // add contributions for any quads
    for (int i = 1; i < num_segments - 1; ++i) {
        int h_bd = m_halfedge_path[i];
        int h_perp_0 = m_transverse_edges[i - 1];
        int h_perp_1 = m_transverse_edges[i];

        // derivative with respect to side of quadrilateral
        Scalar bd_deriv = compute_quad_side_derivative(m.l[h_bd], m.l[h_perp_0], m.l[h_perp_1]);
        length_jacobian.push_back(std::make_pair(h_bd, bd_deriv));

        // derivatives with respect to transverse edges of quadrilateral
        Scalar perp_deriv_0 =
            compute_quad_base_derivative(m.l[h_bd], m.l[h_perp_0], m.l[h_perp_1]);
        Scalar perp_deriv_1 =
            compute_quad_base_derivative(m.l[h_bd], m.l[h_perp_1], m.l[h_perp_0]);
        length_jacobian.push_back(std::make_pair(h_perp_0, perp_deriv_0));
        length_jacobian.push_back(std::make_pair(h_perp_1, perp_deriv_1));
    }

    // add contribution from last triangle
    h_bd = m_halfedge_path.back();
    h_perp = m.opp[m_transverse_edges.back()];
    bd_deriv = compute_tri_side_derivative(m.l[h_bd], m.l[h_perp]);
    perp_deriv = compute_tri_base_derivative(m.l[h_bd], m.l[h_perp]);
    length_jacobian.push_back(std::make_pair(h_bd, bd_deriv));
    length_jacobian.push_back(std::make_pair(h_perp, perp_deriv));

    return length_jacobian;
}

std::vector<std::pair<int, Scalar>> BoundaryPath::compute_log_length_jacobian(
    const Mesh<Scalar>& m) const
{
    int num_segments = m_halfedge_path.size();

    // special case: empty path
    if (num_segments == 0) return {};

    // special case: single segment
    if (num_segments == 1) {
        int hij = m_halfedge_path[0];
        return {std::make_pair(hij, 1.)};
    }

    // compute length Jacobian and simply scale by 2 / length
    Scalar length = compute_length(m);
    Scalar scaling_factor = 2. / length;
    auto length_jacobian = compute_length_jacobian(m);
    int n = length_jacobian.size();
    for (int i = 0; i < n; ++i) {
        length_jacobian[i].second *= scaling_factor;
    }

    return length_jacobian;
}

bool BoundaryPath::is_valid_boundary_path(const Mesh<Scalar>& m) const
{
    int num_edges = m_halfedge_path.size();

    // Check path of halfedges is connected
    for (int i = 1; i < num_edges; ++i) {
        int v_tip = m.to[m_halfedge_path[i - 1]];
        int v_tail = m.to[m.opp[m_halfedge_path[i]]];
        if (v_tip != v_tail) {
            spdlog::error("Halfedges in boundary path are not connected");
            return false;
        }

        // check transverse edge emenates from tip
        if (m.to[m.opp[m_transverse_edges[i - 1]]] != v_tip) {
            spdlog::error("transverse edges in boundary path not connected");
            return false;
        }
    }

    // Check start vertex is correct
    if (m.to[m.opp[m_halfedge_path.front()]] != m_start_vertex) {
        spdlog::error("Boundary path does not emanate from start vertex");
        return false;
    }

    // Check end vertex is correct
    if (m.to[m_halfedge_path.back()] != m_end_vertex) {
        spdlog::error("Boundary path does not terminate at end vertex");
        return false;
    }

    // check all path edges are type 2
    if (num_edges > 1) {
        for (int h : m_halfedge_path)
        {
            if (m.type[h] != 2)
            {
                spdlog::error("Nontrivial boundary path not in double mesh");
                return false;
            }
        }
    } else {
        if (m.type[m_halfedge_path[0]] != 1)
        {
            spdlog::error("Single boundary edge not in primal mesh");
            return false;
        }
    }

    // check all transverse edges are type 3
    for (int h : m_transverse_edges)
    {
        if (m.type[h] != 3)
        {
            spdlog::error("transverse edges do not cross symmetry line");
            return false;
        }
    }

    return true;
}

// compute the height of an isosceles triangle with given base and side lengths
Scalar BoundaryPath::compute_tri_length(Scalar side_length, Scalar base_length) const
{
    return sqrt(max(square(side_length) - (square(base_length) / 4.), 0.));
}

// compute the height of an isoscelese trapezoid with given base and side lengths
Scalar BoundaryPath::compute_quad_length(
    Scalar side_length,
    Scalar first_base_length,
    Scalar second_base_length) const
{
    Scalar base_diff = first_base_length - second_base_length;
    return sqrt(max(square(side_length) - (square(base_diff) / 4.), 0.));
}

// compute triangle derivative with respect to side length
Scalar BoundaryPath::compute_tri_side_derivative(Scalar side_length, Scalar base_length) const
{
    Scalar length = compute_tri_length(side_length, base_length);
    return square(side_length) / (2. * length);
}

// compute triangle derivative with respect to base length
Scalar BoundaryPath::compute_tri_base_derivative(Scalar side_length, Scalar base_length)
    const
{
    Scalar length = compute_tri_length(side_length, base_length);
    return -square(base_length) / (8. * length);
}

// compute quad derivative with respect to side length
Scalar BoundaryPath::compute_quad_side_derivative(
    Scalar side_length,
    Scalar first_base_length,
    Scalar second_base_length) const
{
    Scalar length = compute_quad_length(side_length, first_base_length, second_base_length);
    return square(side_length) / (2. * length);
}

// compute quad derivative with respect to first base length; second is fixed
Scalar BoundaryPath::compute_quad_base_derivative(
    Scalar side_length,
    Scalar first_base_length,
    Scalar second_base_length) const
{
    Scalar length = compute_quad_length(side_length, first_base_length, second_base_length);
    return -first_base_length * (first_base_length - second_base_length) / (8. * length);
}


std::tuple<std::vector<BoundaryPath>, MatrixX> build_boundary_paths(const Mesh<Scalar>& m)
{
    // find and index boundary halfedges
    int num_halfedges = m.n_halfedges();
    std::vector<int> boundary_he = find_primal_boundary_halfedges(m);
    spdlog::info("{} boundary he found", boundary_he.size());

    // build boundary paths for each boundary halfedge
    int num_bd = boundary_he.size();
    std::vector<BoundaryPath> boundary_paths = {};
    boundary_paths.reserve(num_bd);
    for (int i = 0; i < num_bd; ++i) {
        int vi = m.to[m.opp[boundary_he[i]]];
        boundary_paths.push_back(BoundaryPath(m, vi));
    }

    // make map from boundary paths to halfedges
    typedef Eigen::Triplet<int> ScalarTrip;
    std::vector<ScalarTrip> trips;
    for (int i = 0; i < num_bd; ++i) {
        trips.push_back(ScalarTrip(boundary_he[i], i, 1.));
    }
    MatrixX boundary_map(num_halfedges, num_bd);
    boundary_map.setFromTriplets(trips.begin(), trips.end());

    return std::make_tuple(boundary_paths, boundary_map);
}

} // namespace Feature
} // namespace Penner