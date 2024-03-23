#include "embedding.hh"

namespace CurvatureMetric {

ReductionMaps::ReductionMaps(const Mesh<Scalar>& m, bool fix_bd_lengths)
{
    // Build embedding and halfedge information
    build_edge_maps(m, he2e, e2he);
    build_refl_proj(m, he2e, e2he, proj, embed);
    projection = build_refl_matrix(proj, embed);

    // Get element counts from constructed maps
    num_edges = e2he.size();
    num_halfedges = he2e.size();
    num_reduced_edges = embed.size();

    // Get fixed and free vertices from the mesh
    fixed_v.clear();
    free_v.clear();
    fixed_v.reserve(m.fixed_dof.size());
    free_v.reserve(m.fixed_dof.size());
    for (size_t v = 0; v < m.fixed_dof.size(); ++v) {
        if (m.fixed_dof[v]) {
            fixed_v.push_back(v);
        } else {
            free_v.push_back(v);
        }
    }

    // Identify boundary edges as the fixed points of the reflection map
    bd_e.clear();
    int_e.clear();
    bd_e.reserve(num_reduced_edges);
    int_e.reserve(num_reduced_edges);
    for (size_t E = 0; E < num_reduced_edges; ++E) {
        size_t e = embed[E];
        size_t h = e2he[e];
        size_t Rh = m.R[h];
        size_t Re = he2e[Rh];
        // Note: We check if edge type is 0 as in this case there is no reflection map
        if ((m.type[h] != 0) && (e == Re)) {
            bd_e.push_back(E);
        } else {
            int_e.push_back(E);
        }
    }

    fixed_e.clear();
    free_e.clear();
    // Optionally set boundary as fixed
    if (fix_bd_lengths) {
        fixed_e = bd_e;
        free_e = int_e;
    }
    // Otherwise, all edges are free
    else {
        arange(num_reduced_edges, free_e);
    }
}

void build_edge_maps(const Mesh<Scalar>& m, std::vector<int>& he2e, std::vector<int>& e2he)
{
    int num_halfedges = m.opp.size();
    int num_edges = num_halfedges / 2;
    he2e.resize(num_halfedges);
    e2he.clear();
    e2he.reserve(num_edges);

    // First build map from edges to the lower index halfedges, which is a
    // bijection
    for (int h = 0; h < num_halfedges; ++h) {
        if (h < m.opp[h]) {
            e2he.push_back(h);
        }
    }

    // Construct 2-to-1 map from halfedges to edges
    for (int e = 0; e < num_edges; ++e) {
        int h = e2he[e];
        he2e[h] = e;
        he2e[m.opp[h]] = e;
    }
}

void build_refl_proj(
    const Mesh<Scalar>& m,
    const std::vector<int>& he2e,
    const std::vector<int>& e2he,
    std::vector<int>& proj,
    std::vector<int>& embed)
{
    // Resize arrays
    proj.resize(e2he.size());
    embed.clear();
    embed.reserve(e2he.size());

    // Build injective map from edges that are not type 2 to double mesh
    int num_edges = e2he.size();
    for (int e = 0; e < num_edges; ++e) {
        int h0 = m.h0(e2he[e]);
        int h1 = m.h1(e2he[e]);
        if ((m.type[h0] != 2) || (m.type[h1] != 2)) {
            embed.push_back(e);
        }
    }

    // Construct map from double mesh to the embedded mesh
    // Map reflection of edges in the image of the embedding to the original edge
    int num_embedded_edges = embed.size();
    for (int E = 0; E < num_embedded_edges; ++E) {
        int e = embed[E];
        int Re = he2e[m.R[e2he[e]]];
        proj[Re] = E;
    }

    // Map embedded edge to itself. Note that if E is identified with e = embed[E]
    // then this implies proj[E] = E and the map is a projection.
    for (int E = 0; E < num_embedded_edges; ++E) {
        int e = embed[E];
        proj[e] = E;
    }
}

void build_refl_he_proj(
    const Mesh<Scalar>& m,
    const std::vector<int>& he2e,
    const std::vector<int>& e2he,
    std::vector<int>& he_proj,
    std::vector<int>& he_embed)
{
    // Resize arrays
    int num_halfedges = he2e.size();
    int num_edges = e2he.size();
    he_proj.resize(num_halfedges);
    he_embed.clear();
    he_embed.reserve(num_edges);

    // Build injective map from halfedges that are not type 2 to double mesh
    for (int h = 0; h < num_halfedges; ++h) {
        if (m.type[h] != 2) {
            he_embed.push_back(h);
        }
    }

    // Construct map from double mesh to the embedded mesh
    // Map reflection of halfedges in the image of the embedding to the original
    // halfedge
    int num_embedded_halfedges = he_embed.size();
    for (int H = 0; H < num_embedded_halfedges; ++H) {
        int h = he_embed[H];
        int Rh = m.R[h];
        he_proj[Rh] = H;
    }
    // Map embedded halfedge to itself. Note that if H is identified with h =
    // embed[H] then this implies proj[H] = H and the map is a projection.
    for (int H = 0; H < num_embedded_halfedges; ++H) {
        int h = he_embed[H];
        he_proj[h] = H;
    }
}

MatrixX build_edge_matrix(const std::vector<int>& he2e, const std::vector<int>& e2he)
{
    int num_halfedges = he2e.size();
    int num_edges = e2he.size();
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        tripletList.push_back(T(h, he2e[h], 1.0));
    }
    MatrixX identification(num_halfedges, num_edges);
    identification.reserve(tripletList.size());
    identification.setFromTriplets(tripletList.begin(), tripletList.end());
    return identification;
}

MatrixX build_refl_matrix(const std::vector<int>& proj, const std::vector<int>& embed)
{
    // Convert linear projection function e -> P(e) to a matrix
    int num_edges = proj.size();
    int num_embedded_edges = embed.size();
    std::vector<T> tripletList;
    tripletList.reserve(num_edges);
    for (int e = 0; e < num_edges; ++e) {
        tripletList.push_back(T(e, proj[e], 1.0));
    }
    MatrixX projection;
    projection.resize(num_edges, num_embedded_edges);
    projection.reserve(tripletList.size());
    projection.setFromTriplets(tripletList.begin(), tripletList.end());
    return projection;
}

bool is_embedded_edge(const std::vector<int>& proj, const std::vector<int>& embed, int e)
{
    return (embed[proj[e]] == e);
}

void reduce_symmetric_function(
    const std::vector<int>& embed,
    const VectorX& symmetric_function,
    VectorX& reduced_function)
{
    // Define f[E] by the value of f_full at e = embed[E]
    int num_embedded_edges = embed.size();
    reduced_function.resize(num_embedded_edges);
    for (int E = 0; E < num_embedded_edges; ++E) {
        int e = embed[E];
        reduced_function[E] = symmetric_function[e];
    }
}

void expand_reduced_function(
    const std::vector<int>& proj,
    const VectorX& reduced_function,
    VectorX& symmetric_function)
{
    // Define f_full[e] by the value of f at E = proj[e]. This implies that
    // f_full[embed[e]] = f[proj[embed[e]] = f[e]
    int num_edges = proj.size();
    symmetric_function.resize(num_edges);
    for (int e = 0; e < num_edges; ++e) {
        int E = proj[e];
        symmetric_function[e] = reduced_function[E];
    }
}

void restrict_he_func(const std::vector<int>& e2he, const VectorX& f_he, VectorX& f_e)
{
    // Restrict f_he to edges
    int n_edges = e2he.size();
    f_e.resize(n_edges);
    for (int e = 0; e < n_edges; ++e) {
        int h = e2he[e];
        f_e[e] = f_he[h];
    }
}

void expand_edge_func(const std::vector<int>& he2e, const VectorX& f_e, VectorX& f_he)
{
    // Expand f_e to halfedges
    int n_halfedges = he2e.size();
    f_he.resize(n_halfedges);
    for (int h = 0; h < n_halfedges; ++h) {
        int e = he2e[h];
        f_he[h] = f_e[e];
    }
}

bool is_valid_halfedge(const Mesh<Scalar>& m)
{
    // Build prev array (inverse of n if n is a permutation)
    int num_halfedges = m.n.size();
    std::vector<int> prev(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        prev[m.n[h]] = h;
    }

    // Check halfedge conditions
    for (int h = 0; h < num_halfedges; ++h) {
        // opp^2 is the identity
        if (m.opp[m.opp[h]] != h) return false;
        // opp has no fixed points
        if (m.opp[h] == h) return false;
        // Prev is left inverse
        if (prev[m.n[h]] != h) return false;
        // Prev is right inverse
        if (m.n[prev[h]] != h) return false;
    }

    // Check vertex conditions
    int num_vertices = m.out.size();
    for (int v = 0; v < num_vertices; ++v) {
        // opp reverses the tail and tip of out
        if (m.to[m.opp[m.out[v]]] != v) return false;
    }

    return true;
}

bool is_valid_symmetry(const Mesh<Scalar>& m)
{
    bool is_valid = true;

    // Check mesh is valid
    if (!is_valid_halfedge(m)) {
        std::cerr << "Invalid halfedge" << std::endl;
        is_valid = false;
    }

    // Check halfedge conditions
    int num_halfedges = m.n.size();
    for (int h = 0; h < num_halfedges; ++h) {
        // Reversing n
        if (m.n[m.n[m.R[h]]] != m.R[m.n[h]]) {
            std::cerr << "R does not reverse n at " << h << std::endl;
            is_valid = false;
        }

        // Commuting with opp
        if (m.opp[m.R[h]] != m.R[m.opp[h]]) {
            std::cerr << "R does not commute with opp at " << h << std::endl;
            is_valid = false;
        }

        // Halfedges partitioned
        if ((m.type[h] == 1) && (m.type[m.R[h]] != 2)) {
            std::cerr << "Type 1 halfedge does not map to a type 1 halfedge at " << h << std::endl;
            is_valid = false;
        }
        if ((m.type[h] == 2) && (m.type[m.R[h]] != 1)) {
            std::cerr << "Type 1 halfedge does not map to a type 1 halfedge at " << h << std::endl;
            is_valid = false;
        }
        if ((m.type[h] == 3) && (m.R[h] != h)) {
            std::cerr << "Type 3 halfedge does not map to itself at " << h << std::endl;
            is_valid = false;
        }
    }

    // Check face conditions
    int num_faces = m.h.size();
    for (int f = 0; f < num_faces; ++f) {
        // Face maps to itself
        if (m.f[m.R[m.h[f]]] == f) {
            continue;
        }
        int h = m.h[f];
        if (h == 1) {
            if ((m.n[h] != 1) || (m.n[m.n[h]] != 1)) {
                std::cerr << "Interior face not labelled consistently at " << f << std::endl;
                is_valid = false;
            }
            if ((m.R[h] != 2) || (m.n[m.R[h]] != 2) || (m.n[m.n[m.R[h]]] != 2)) {
                std::cerr << "Interior face not labelled consistently at " << f << std::endl;
                is_valid = false;
            }
        }
        if (h == 2) {
            if ((m.n[h] != 2) || (m.n[m.n[h]] != 2)) {
                std::cerr << "Interior face not labelled consistently at " << f << std::endl;
                is_valid = false;
            }
            if ((m.R[h] != 1) || (m.n[m.R[h]] != 1) || (m.n[m.n[m.R[h]]] != 1)) {
                std::cerr << "Interior face not labelled consistently at " << f << std::endl;
                is_valid = false;
            }
        }
    }

    return is_valid;
}

} // namespace CurvatureMetric
