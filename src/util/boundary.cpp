#include "util/boundary.h"

#include "util/vector.h"

namespace Penner {

int circulate_ccw_to_boundary(const Mesh<Scalar>& m, int halfedge_index)
{
    // circulate until doubled mesh component changes (e.g., primal to copy)
    int h = halfedge_index;
    while (m.type[h] == m.type[halfedge_index]) {
        h = m.opp[m.n[h]];

        // check for full circulation
        if (h == halfedge_index)
        {
            spdlog::error("Circulating to boundary around interior vertex");
            return -1;
        }
    }

    return h;
}

std::vector<int> find_primal_boundary_halfedges(const Mesh<Scalar>& m)
{
    // Closed mesh case
    if (m.R[0] == 0) return {};

    // General case
    int num_halfedges = m.n_halfedges();
    std::vector<int> boundary_halfedges = {};
    boundary_halfedges.reserve(m.n_ind_vertices());
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = m.opp[hij];
        if ((m.type[hij] == 1) && (m.type[hji] == 2)) {
            boundary_halfedges.push_back(hij);
        }
    }

    return boundary_halfedges;
}

std::vector<int> find_boundary_vertices(
    const Mesh<Scalar>& m
) {
    // get tip of primal boundary halfedges
    std::vector<int> boundary_halfedges = find_primal_boundary_halfedges(m);
    std::vector<int> boundary_vertices = {};
    boundary_vertices.reserve(boundary_halfedges.size());
    for (int hij : boundary_halfedges)
    {
        boundary_vertices.push_back(m.to[hij]);
    }

    return boundary_vertices;
}

std::vector<int> find_boundary_vertices(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex)
{
    std::vector<int> boundary_vertices = find_boundary_vertices(m);
    std::vector<int> ind_boundary_vertices = vector_compose(m.v_rep, boundary_vertices);
    std::vector<int> reindexed_boundary_vertices = vector_compose(vtx_reindex, ind_boundary_vertices);

    return reindexed_boundary_vertices;
}

std::vector<bool> compute_boundary_vertices(const Mesh<Scalar>& m)
{
    // Get the boundary vertices
    auto bd_vertices = find_boundary_vertices(m);

    // Make list of boundary vertices into boolean mask
    int num_vertices = m.n_vertices();
    std::vector<bool> is_boundary_vertex(num_vertices, false);
    int num_bd_vertices = bd_vertices.size();
    for (int i = 0; i < num_bd_vertices; ++i)
    {
        int vi = bd_vertices[i];
        is_boundary_vertex[vi] = true;
    }

    return is_boundary_vertex;
}

std::vector<int> build_boundary_component(const Mesh<Scalar>& m, int halfedge_index)
{
    std::vector<int> component = {};
    int h = halfedge_index;
    do {
        component.push_back(h);

        // Circulate to next boundary edge 
        h = circulate_ccw_to_boundary(m, h);
        h = m.opp[h];
    } while (h != halfedge_index);

    return component;
}

std::vector<int> find_boundary_components(const Mesh<Scalar>& m)
{
    // Closed mesh case
    if (m.R[0] == 0) return {};

    // Get boundary edges
    int num_halfedges = m.n_halfedges();
    std::vector<int> boundary_components = {};
    std::vector<bool> is_seen(num_halfedges, false);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (is_seen[hij]) continue;
        is_seen[hij] = true;

        // TODO Handle flipped case
        if (m.type[hij] == 3) {
            spdlog::error("Cannot find boundary vertices for flipped mesh");
            return {};
        }

        // Only process primal edges
        if (m.type[hij] != 1) continue;

        // Check for edges on symmetry line
        // TODO Fix all this
        if (m.opp[m.R[hij]] == hij) {
            // mark boundary component
            boundary_components.push_back(hij);

            // add all edges in component to seen list
            std::vector<int> component = build_boundary_component(m, hij);
            for (int h : component) is_seen[h] = true;
        }
    }

    return boundary_components;
}

} // namespace Penner
