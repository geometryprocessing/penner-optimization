#include "holonomy/core/homology_basis.h"

#include "holonomy/core/vector.h"

#include <random>
#include "optimization/core/embedding.h"

namespace PennerHolonomy {

// Construct a clockwise sequence of dual segments around a vertex in the mesh
std::vector<DualSegment> build_clockwise_vertex_dual_segment_sequence(
    const Mesh<Scalar>& m,
    int vertex_index)
{
    std::vector<DualSegment> dual_loop(0);

    // Circulate clockwise around the vertex to build the loop
    int h_start = m.opp[m.out[vertex_index]];
    int h_iter = h_start;
    do {
        // Build clockwise dual segment in current face
        int h_next = m.n[h_iter];
        dual_loop.push_back({h_iter, h_next});

        // Iterate to next face
        h_iter = m.opp[h_next];
    } while (h_iter != h_start);

    return dual_loop;
}

// Construct a counterclockwise sequence of dual segments around a vertex in the mesh
std::vector<DualSegment> build_counterclockwise_vertex_dual_segment_sequence(
    const Mesh<Scalar>& m,
    int vertex_index)
{
    // Use reverse of the clockwise dual segment
    return reverse_dual_path(build_clockwise_vertex_dual_segment_sequence(m, vertex_index));
}

HomotopyBasisGenerator::HomotopyBasisGenerator(const Mesh<Scalar>& m, int root, Weighting weighting)
    : m_mesh(m)
{
    // Build halfedge to edge maps
    CurvatureMetric::build_edge_maps(m, m_he2e, m_e2he);

    // Build spanning tree and cotree with given weighting
    if (weighting == Weighting::minimal_homotopy) {
        // Compute dual tree with shortest path tree 
        std::vector<Scalar> dual_edge_lengths = compute_dual_edge_lengths(m);
        m_dual_tree = DualTree(m, dual_edge_lengths, root, true);

        // Compute primal cotree with maximal dual loop lengths
        std::vector<Scalar> dual_loop_lengths =
                compute_dual_loop_length_weights(m, dual_edge_lengths, m_dual_tree);
        m_primal_tree = PrimalCotree(m, vector_negate(dual_loop_lengths), m_dual_tree);
    } 
    // Build 
    else if (weighting == Weighting::maximal_homotopy) {
        // Compute dual tree with shortest path tree 
        std::vector<Scalar> dual_edge_lengths = compute_dual_edge_lengths(m);
        m_dual_tree = DualTree(m, vector_negate(dual_edge_lengths), root, true);

        // Compute primal cotree with maximal dual loop lengths
        std::vector<Scalar> dual_loop_lengths =
                compute_dual_loop_length_weights(m, dual_edge_lengths, m_dual_tree);
        m_primal_tree = PrimalCotree(m, dual_loop_lengths, m_dual_tree);
    } 
    // Use min-max trees with dual edge weights
    else if (weighting == Weighting::dual_min_primal_max) {
        std::vector<Scalar> dual_edge_lengths = compute_dual_edge_lengths(m);
        m_dual_tree = DualTree(m, dual_edge_lengths, root);
        m_primal_tree = PrimalCotree(m, vector_negate(dual_edge_lengths), m_dual_tree);
    }
    // Use min-max trees with primal edge weights
    else if (weighting == Weighting::primal_min_dual_max) {
        m_primal_tree = PrimalTree(m, m.l, root);
        m_dual_tree = DualCotree(m, vector_negate(m.l), m_primal_tree);
    }

    // Find homotopy basis handle edges that are not in the tree or cotree
    int num_edges = m_e2he.size();
    for (int ei = 0; ei < num_edges; ++ei) {
        int h = m_e2he[ei];

        // Skip edges in double or boundary
        if ((m_mesh.type[h] == 2) || (m_mesh.opp[m_mesh.R[h]] == h)) continue;

        if ((!m_primal_tree.is_edge_in_tree(ei)) && (!m_dual_tree.is_edge_in_tree(ei))) {
            m_homotopy_basis_edge_handles.push_back(ei);
        }
    }
};

// Trace a dual vertex back to the root of the dual tree
std::tuple<std::vector<int>, std::vector<int>> HomotopyBasisGenerator::trace_dual_vertex_to_root(int face_index) const
{
    std::vector<int> dual_path = {face_index};
    std::vector<int> dual_edges = {};

    // Trace up the dual tree until a root is reached
    int curr_face_index = face_index;
    while (!m_dual_tree.is_root(curr_face_index)) {
        // Get parent face of current face
        int edge_index = m_dual_tree.out(curr_face_index);
        curr_face_index = m_dual_tree.to(edge_index);
        assert(m_dual_tree.from(edge_index) == dual_path.back());

        // Add face and edge to the path
        dual_path.push_back(curr_face_index);
        dual_edges.push_back(m_dual_tree.edge(edge_index));
    }

    return std::make_tuple(dual_path, dual_edges);
}

std::tuple<std::vector<int>, std::vector<int>> HomotopyBasisGenerator::construct_homotopy_basis_edge_loop(int index) const
{
    assert(index >= 0);
    assert(index < n_homology_basis_loops());

    // Construct path from both sides of the edge to the root face of the dual tree
    int handle_edge = m_homotopy_basis_edge_handles[index];
    int left_halfedge = m_e2he[handle_edge];
    int right_halfedge = m_mesh.opp[left_halfedge];
    int left_face = m_mesh.f[left_halfedge];
    int right_face = m_mesh.f[right_halfedge];
    auto [left_dual_path, left_dual_edges] = trace_dual_vertex_to_root(left_face);
    auto [right_dual_path, right_dual_edges] = trace_dual_vertex_to_root(right_face);

    // Find common root path of the left and right paths
    int left_path_size = left_dual_path.size();
    int right_path_size = right_dual_path.size();

    // Combine dual paths to generate a simple loop
    std::vector<int> dual_loop(0);
    std::vector<int> dual_edges(0);
    dual_loop.reserve(left_path_size + right_path_size);
    dual_edges.reserve(left_path_size + right_path_size);
    dual_edges.push_back(handle_edge);
    for (int i = 0; i < left_path_size-1; ++i) {
        dual_loop.push_back(left_dual_path[i]); // Add left path to root (exclusive)
        dual_edges.push_back(left_dual_edges[i]); // Add left path to root (exclusive)
    }
    dual_loop.push_back(left_dual_path[left_path_size - 1]); // Add common root
    for (int i = right_path_size - 2; i >= 0; --i) {
        dual_loop.push_back(right_dual_path[i]); // Add right path from root (exclusive)
        dual_edges.push_back(right_dual_edges[i]); // Add left path to root (exclusive)
    }

    return std::make_tuple(dual_loop, dual_edges);
}

std::tuple<std::vector<int>, std::vector<int>> HomotopyBasisGenerator::construct_homology_basis_edge_loop(int index) const
{
    assert(index >= 0);
    assert(index < n_homology_basis_loops());

    // Construct path from both sides of the edge to the root face of the dual tree
    int handle_edge = m_homotopy_basis_edge_handles[index];
    int left_halfedge = m_e2he[handle_edge];
    int right_halfedge = m_mesh.opp[left_halfedge];
    int left_face = m_mesh.f[left_halfedge];
    int right_face = m_mesh.f[right_halfedge];
    auto [left_dual_path, left_dual_edges] = trace_dual_vertex_to_root(left_face);
    auto [right_dual_path, right_dual_edges] = trace_dual_vertex_to_root(right_face);

    // Find common root path of the left and right paths
    int trim_root_offset = 0;
    int left_path_size = left_dual_path.size();
    int right_path_size = right_dual_path.size();
    while ((trim_root_offset < left_path_size) && (trim_root_offset < right_path_size) &&
           (left_dual_path[left_path_size - 1 - trim_root_offset] ==
            right_dual_path[right_path_size - 1 - trim_root_offset])) {
        trim_root_offset++;
    }
    assert(
        left_dual_path[left_path_size - trim_root_offset] ==
        right_dual_path[right_path_size - trim_root_offset]);

    // Combine dual paths and trim common path to root to generate a simple loop
    std::vector<int> dual_loop(0);
    std::vector<int> dual_edges(0);
    dual_loop.reserve(left_path_size + right_path_size);
    dual_edges.reserve(left_path_size + right_path_size);
    dual_edges.push_back(handle_edge);
    for (int i = 0; i < left_path_size - trim_root_offset; ++i) {
        dual_loop.push_back(left_dual_path[i]); // Add left path to trim root
        dual_edges.push_back(left_dual_edges[i]);
    }
    dual_loop.push_back(left_dual_path[left_path_size - trim_root_offset]); // Add trim root
    for (int i = right_path_size - 1 - trim_root_offset; i >= 0; --i) {
        dual_loop.push_back(right_dual_path[i]); // Add right path from trim root
        dual_edges.push_back(right_dual_edges[i]);
    }

    return std::make_tuple(dual_loop, dual_edges);
}

std::vector<int> HomotopyBasisGenerator::construct_homology_basis_loop(int index) const
{
    return std::get<0>(construct_homology_basis_edge_loop(index));
}

} // namespace PennerHolonomy