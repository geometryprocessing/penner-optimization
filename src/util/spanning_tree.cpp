
#include "util/spanning_tree.h"

#include "util/vector.h"

#include <queue>

namespace Penner {

void cut_boundary_edges(const Mesh<Scalar>& m, std::vector<bool>& is_cut)
{
    // Do nothing if not doubled mesh
    if (m.type[0] == 0) return;

    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (m.opp[m.R[hij]] == hij) {
            is_cut[hij] = true;
        }
    }
}

void cut_copy_edges(const Mesh<Scalar>& m, std::vector<bool>& is_cut)
{
    // Do nothing if not doubled mesh
    if (m.type[0] == 0) return;

    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if ((m.type[hij] == 2) && (m.type[m.opp[hij]] == 2)) {
            is_cut[hij] = true;
        }
    }
}


bool Forest::is_valid_forest(const Mesh<Scalar>& m) const
{
    int num_edges = m_halfedges.size();
    int num_vertices = m_out.size();
    if (m_to.size() != m_halfedges.size()) {
        spdlog::error(
            "to and edges have inconsistent sizes {} and {}",
            m_to.size(),
            m_halfedges.size());
        return false;
    }
    if (m_from.size() != m_halfedges.size()) {
        spdlog::error(
            "from and edges have inconsistent sizes {} and {}",
            m_from.size(),
            m_halfedges.size());
        return false;
    }

    // Check number of vertices and edges compatible for a spanning tree
    // TODO Add check for double mesh
    if ((num_vertices != num_edges + 1) && (m.type[0] == 0)) {
        spdlog::error("Spanning tree has {} edges and {} vertices", num_edges, num_vertices);
        return false;
    }

    // Check edge conditions
    for (int eij = 0; eij < num_edges; ++eij) {
        // Check out and from are inverse
        if (m_out[m_from[eij]] != eij) {
            spdlog::error(
                "Edge {} is from {} with out edge {}",
                eij,
                m_from[eij],
                m_out[m_from[eij]]);
            return false;
        }

        // Check edges are all in tree
        if (!m_edge_is_in_forest[he2e[m_halfedges[eij]]]) {
            spdlog::error("Edge {} not marked in tree", eij);
            return false;
        }
    }

    return true;
}

PrimalTree::PrimalTree(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    int root,
    bool use_shortest_path)
{
    // Generate minimal spanning tree data
    int num_halfedges = m.n_halfedges();
    std::vector<bool> is_cut(num_halfedges, false);
    cut_copy_edges(m, is_cut);

    std::vector<int> halfedge_from_vertex =
        build_primal_forest(m, weights, is_cut, root, use_shortest_path);

    // Initialize Primal Tree data structures
    initialize_primal_tree(m, halfedge_from_vertex);

    assert(is_valid_primal_tree(m));
}

void PrimalTree::initialize_primal_tree(
    const Mesh<Scalar>& m,
    const std::vector<int>& halfedge_from_vertex)
{
    // Get edge maps
    build_edge_maps(m, he2e, e2he);

    // Initialize data structures
    int num_vertices = m.n_vertices();
    m_halfedges.reserve(num_vertices);
    m_from.reserve(num_vertices);
    m_to.reserve(num_vertices);
    m_out = std::vector<int>(num_vertices, -1);
    m_edge_is_in_forest = std::vector<bool>(e2he.size(), false);

    for (int vj = 0; vj < num_vertices; ++vj) {
        // Get edge to vertex (skipping the root vertex with no incoming edge)
        int hij = halfedge_from_vertex[vj];
        if (hij < 0) continue;
        int eij = he2e[hij];
        int vi = m.to[m.opp[hij]];

        // Add the edge to the spanning tree
        m_out[vj] = m_halfedges.size();
        m_from.push_back(vj);
        m_to.push_back(vi);
        m_halfedges.push_back(hij);
        m_edge_is_in_forest[eij] = true;
    }
}

bool PrimalTree::is_valid_primal_tree(const Mesh<Scalar>& m) const
{
    if (!is_valid_forest(m)) return false;

    // Check edge conditions
    int num_edges = m_halfedges.size();
    for (int i = 0; i < num_edges; ++i) {
        int h0 = m_halfedges[i];
        int h1 = m.opp[h0];

        // Check vertices adjacent to each edge are actually adjacent to the edge
        int v0 = m.to[h0];
        int v1 = m.to[h1];
        if (!((m_to[i] == v0) && (m_from[i] == v1)) && !((m_to[i] == v1) && (m_from[i] == v0))) {
            return false;
        }
    }

    return true;
}

DualTree::DualTree(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    int root,
    bool use_shortest_path)
{
    // Generate minimal spanning tree data
    int num_halfedges = m.n_halfedges();
    std::vector<bool> is_cut(num_halfedges, false);
    cut_boundary_edges(m, is_cut);

    std::vector<int> halfedge_from_face =
        build_dual_forest(m, weights, is_cut, root, use_shortest_path);

    // Initialize Dual Tree data structures
    initialize_dual_tree(m, halfedge_from_face);

    assert(is_valid_dual_tree(m));
}

void DualTree::initialize_dual_tree(
    const Mesh<Scalar>& m,
    const std::vector<int>& halfedge_from_face)
{
    // Get edge maps
    build_edge_maps(m, he2e, e2he);

    // Initialize dual tree data structures
    int num_faces = m.n_faces();
    m_halfedges.reserve(num_faces);
    m_from.reserve(num_faces);
    m_to.reserve(num_faces);
    m_out = std::vector<int>(num_faces, -1);
    m_edge_is_in_forest = std::vector<bool>(e2he.size(), false);

    for (int fj = 0; fj < num_faces; ++fj) {
        // Get edge to vertex (skipping the root vertex with no incoming edge)
        int hij = halfedge_from_face[fj];
        if (hij < 0) continue;
        int eij = he2e[hij];
        int fi = m.f[hij];

        // Add the edge to the spanning tree
        m_out[fj] = m_halfedges.size();
        m_from.push_back(fj);
        m_to.push_back(fi);
        m_halfedges.push_back(hij);
        m_edge_is_in_forest[eij] = true;
    }
}

bool DualTree::is_valid_dual_tree(const Mesh<Scalar>& m) const
{
    if (!is_valid_forest(m)) return false;

    // Check edge conditions
    int num_edges = m_halfedges.size();
    for (int i = 0; i < num_edges; ++i) {
        int h0 = m_halfedges[i];
        int h1 = m.opp[h0];

        // Check dual vertices adjacent to each dual edge are actually adjacent to the edge
        int f0 = m.f[h0];
        int f1 = m.f[h1];
        if (!((m_to[i] == f0) && (m_from[i] == f1)) && !((m_to[i] == f1) && (m_from[i] == f0))) {
            spdlog::error("Faces {} and {} are not adjacent", f0, f1);
            return false;
        }
    }

    return true;
}

PrimalCotree::PrimalCotree(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const DualTree& dual_tree,
    int root,
    bool use_shortest_path)
{
    // Get edge maps
    build_edge_maps(m, he2e, e2he);

    // Generate maximal spanning tree data that does not intersect the primal tree
    int num_halfedges = he2e.size();
    std::vector<bool> is_cut(num_halfedges, false);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (dual_tree.is_edge_in_tree(he2e[hij])) {
            is_cut[hij] = true;
        }
    }
    cut_copy_edges(m, is_cut);

    std::vector<int> halfedge_from_vertex =
        build_primal_forest(m, weights, is_cut, root, use_shortest_path);

    // Initialize Primal Tree data structures
    initialize_primal_tree(m, halfedge_from_vertex);

    assert(is_valid_primal_cotree(m, dual_tree));
}

bool PrimalCotree::is_valid_primal_cotree(const Mesh<Scalar>& m, const DualTree& dual_tree) const
{
    // Check if valid primal tree structure
    if (!is_valid_primal_tree(m)) return false;

    // Check that it does not intersect the primal tree
    int num_edges = n_edges();
    for (int i = 0; i < num_edges; ++i) {
        int ei = edge(i);

        // Check edge is in the primal tree and not in the dual tree
        if (dual_tree.is_edge_in_tree(ei)) {
            spdlog::error("Primal Cotree edge {} also in dual tree", ei);
            return false;
        }
    }

    return true;
}

DualCotree::DualCotree(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const PrimalTree& primal_tree,
    int root,
    bool use_shortest_path)
{
    // Get edge maps
    build_edge_maps(m, he2e, e2he);

    // Generate maximal spanning tree data that does not intersect the primal tree
    int num_halfedges = he2e.size();
    std::vector<bool> is_cut(num_halfedges, false);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (primal_tree.is_edge_in_tree(he2e[hij])) {
            is_cut[hij] = true;
        }
    }
    cut_boundary_edges(m, is_cut);

    std::vector<int> halfedge_from_face =
        build_dual_forest(m, weights, is_cut, root, use_shortest_path);

    // Initialize Dual Tree data structures
    initialize_dual_tree(m, halfedge_from_face);

    assert(is_valid_dual_cotree(m, primal_tree));
}


bool DualCotree::is_valid_dual_cotree(const Mesh<Scalar>& m, const PrimalTree& primal_tree) const
{
    // Check if valid dual tree structure
    if (!is_valid_dual_tree(m)) return false;

    // Check that it does not intersect the primal tree
    int num_edges = n_edges();
    for (int i = 0; i < num_edges; ++i) {
        int ei = edge(i);

        // Check edge is in the dual tree and not in the primal tree
        if (primal_tree.is_edge_in_tree(ei)) {
            spdlog::error("Dual Cotree edge {} also in primal tree", ei);
            return false;
        }
    }

    return true;
}

// Generic method to build a forest given a circulator for halfedges around a vertex and maps
// between vertices and halfedges
std::vector<int> build_forest(
    const std::vector<int>& circ,
    const std::vector<int>& v2h,
    const std::vector<int>& h2v,
    const std::vector<Scalar>& weights,
    const std::vector<bool>& is_cut,
    int v_start,
    bool use_shortest_path)
{
    // Initialize an array to keep track of vertices
    int num_vertices = v2h.size();
    Scalar max_cost = vector_max(weights);
    std::vector<bool> is_processed_vertex(num_vertices, false);
    std::vector<int> halfedge_from_vertex(num_vertices, -1);

    // Initialize vertex cost with value above maximum possible weight
    Scalar inf_cost = std::max((num_vertices * max_cost) + 1.0, max_cost + 1.0);
    std::vector<Scalar> vertex_cost(num_vertices, inf_cost);

    // Mark root with 0 (or cost lower than other edges for all negative weights)
    Scalar root_cost = min(max_cost, 0.0);
    vertex_cost[v_start] = root_cost;

    // Define a custom comparison function for the priority queue
    typedef std::pair<int, Scalar> WeightedVertex;
    auto vertex_compare = [](const WeightedVertex& left, const WeightedVertex& right) {
        return left.second > right.second;
    };

    // Initialize the stack of vertices to process with all vertices
    std::priority_queue<WeightedVertex, std::vector<WeightedVertex>, decltype(vertex_compare)>
        vertices_to_process(vertex_compare);
    for (int vi = 0; vi < num_vertices; ++vi) {
        vertices_to_process.push(std::make_pair(vi, vertex_cost[vi]));
    }

    // Perform Prim or Dijkstra algorithm
    while (!vertices_to_process.empty()) {
        // Get the next vertex to process
        auto [vi, vi_cost] = vertices_to_process.top();
        vertices_to_process.pop();

        // Skip already processed vertices
        if (is_processed_vertex[vi]) continue;
        is_processed_vertex[vi] = true;

        // Check if vertex has uninitialized cost and give it the root cost if so
        if (vertex_cost[vi] == inf_cost) {
            vertex_cost[vi] = root_cost;
        }

        // Iterate over the vertex circulator via halfedges
        int h_start = v2h[vi];
        int hij = h_start;
        do {
            // Get the vertex in the one ring at the tip of the halfedge
            int vj = h2v[hij];

            // Get candidate edge cost (either path length or edge weight)
            Scalar candidate_cost;
            if (use_shortest_path) {
                candidate_cost = weights[hij] + vertex_cost[vi];
            } else {
                candidate_cost = weights[hij];
            }

            // Check if the edge to the tip vertex is the best seen so far
            if ((!is_cut[hij]) && (!is_processed_vertex[vj]) &&
                (vertex_cost[vj] >= candidate_cost)) {
                halfedge_from_vertex[vj] = hij;
                vertex_cost[vj] = candidate_cost;
                vertices_to_process.push(std::make_pair(vj, vertex_cost[vj]));
            }

            // Progress to the next halfedge in the vertex circulator
            hij = circ[hij];
        } while (hij != h_start);
    }

    return halfedge_from_vertex;
}

std::vector<int> build_primal_forest(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const std::vector<bool>& is_cut,
    int v_start,
    bool use_shortest_path)
{
    // Construct vertex circulator
    std::vector<int> circ = vector_compose(m.n, m.opp);

    // Construct maps from vertices to and from halfedges
    const std::vector<int>& v2h = m.out;
    std::vector<int> h2v = m.to;

    // Use generic forest constructor
    return build_forest(circ, v2h, h2v, weights, is_cut, v_start, use_shortest_path);
}

std::vector<int> build_dual_forest(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const std::vector<bool>& is_cut,
    int f_start,
    bool use_shortest_path)
{
    // Face circulator is next halfedge
    const std::vector<int>& circ = m.n;

    // Construct maps from faces to and from halfedges
    const std::vector<int>& f2h = m.h;
    std::vector<int> h2f = vector_compose(m.f, m.opp);

    // Use generic forest constructor
    return build_forest(circ, f2h, h2f, weights, is_cut, f_start, use_shortest_path);
}

std::vector<int> find_shortest_path(const Mesh<Scalar>& m, int v_start, int v_end)
{
    // build primal tree with shortest path from all vertices to the target end
    bool use_shortest_path = true;
    PrimalTree primal_tree(m, m.l, v_end, use_shortest_path);

    // build the path from the start vertex to the end vertex
    int v_curr = v_start;
    std::vector<int> path = {};
    while (!primal_tree.is_root(v_curr)) {
        int e = primal_tree.out(v_curr);
        path.push_back(primal_tree.halfedge(e));
        v_curr = primal_tree.to(e);
    }

    return path;
}

} // namespace Penner