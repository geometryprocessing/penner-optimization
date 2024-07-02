#pragma once

#include "holonomy/core/common.h"

namespace PennerHolonomy {

/**
 * @brief Base representation for a forest (primal or dual) on a mesh
 *
 * Supports queries to get parents of edges and vertices, determine if a vertex is
 * a root, and determine if an edge in the mesh is in the forest
 *
 */
class Forest
{
public:
    Forest() {}

    /**
     * @brief Get the number of edges in the forest
     *
     * @return number of edges
     */
    int n_edges() const { return m_edges.size(); }

    /**
     * @brief Get the number of vertices in the forest
     *
     * @return number of vertices
     */
    int n_vertices() const { return m_out.size(); }

    /**
     * @brief Get the mesh edge corresponding to an edge in the forest
     *
     * @param index: index of an edge in the forest
     * @return index of the edge in the mesh
     */
    int edge(int index) const
    {
        assert(is_valid_index(index));
        return m_edges[index];
    }

    /**
     * @brief Get the parent vertex of an edge in the rooted forest
     *
     * @param index: index of an edge in the forest
     * @return index of the parent vertex
     */
    int to(int index) const
    {
        assert(is_valid_index(index));
        return m_to[index];
    }

    /**
     * @brief Get the child vertex of an edge in the rooted forest
     *
     * @param index: index of an edge in the forest
     * @return index of the child vertex
     */
    int from(int index) const
    {
        assert(is_valid_index(index));
        return m_from[index];
    }

    /**
     * @brief Get the parent edge of a vertex in the rooted forest
     *
     * @param vertex_index: index of a vertex in the forest
     * @return index of the parent edge
     */
    int out(int vertex_index) const
    {
        assert(is_valid_vertex_index(vertex_index));
        return m_out[vertex_index];
    }

    /**
     * @brief Determine if a vertex is a root of the forest
     *
     * @param vertex_index: index of a vertex in the forest
     * @return true if the vertex is a root
     * @return false otherwise
     */
    bool is_root(int vertex_index) const
    {
        assert(is_valid_vertex_index(vertex_index));
        return (m_out[vertex_index] < 0);
    }

    /**
     * @brief Determine if a mesh edge is in the forest
     *
     * @param edge_index: index of an edge in the mesh
     * @return true if the edge is in the forest
     * @return false otherwise
     */
    bool is_edge_in_forest(int edge_index) const
    {
        assert(is_valid_edge_index(edge_index));
        return m_edge_is_in_forest[edge_index];
    }

protected:
    std::vector<int> m_edges;
    std::vector<bool> m_edge_is_in_forest;

    // Edge to vertex maps
    std::vector<int> m_to;
    std::vector<int> m_from;

    // Vertex to edge maps
    std::vector<int> m_out;

    // Index validation
    int num_indices() const { return m_edges.size(); }
    int num_edges() const { return m_edge_is_in_forest.size(); }
    int num_vertices() const { return m_out.size(); }
    bool is_valid_index(int index) const {
        return ((index >= 0) && (index < num_indices()));
    }
    bool is_valid_vertex_index(int vertex_index) const
    {
        return ((vertex_index >= 0) && (vertex_index < num_vertices()));
    }
    bool is_valid_edge_index(int edge_index) const
    {
        return ((edge_index >= 0) && (edge_index < num_edges()));
    }

    // Forest validation
    bool is_valid_forest(const Mesh<Scalar>& m) const;
};

/**
 * @brief Representation for a primal minimal spanning tree (on vertices and edges) on a mesh
 *
 */
class PrimalTree : public Forest
{
public:
    /**
     * @brief Construct an empty Primal Tree object
     *
     */
    PrimalTree() {}

    /**
     * @brief Construct a new Primal Tree object on a mesh
     *
     * @param m: underlying mesh
     * @param weights: per-halfedge weights (assumes halfedge pairs have same weight)
     * @param root: (optional) starting vertex root for the tree construction
     * @param use_shortest_path: (optional) use shortest path tree instead of minimal weight
     * tree
     */
    PrimalTree(
        const Mesh<Scalar>& m,
        const std::vector<Scalar>& weights,
        int root = 0,
        bool use_shortest_path = false);
        
    /**
     * @brief Determine if a mesh edge is in the tree
     *
     * @param edge_index: index of an edge in the mesh
     * @return true if the edge is in the tree
     * @return false otherwise
     */
    bool is_edge_in_tree(int edge_index) const
    {
        return is_edge_in_forest(edge_index);
    }

protected:
    void initialize_primal_tree(
        const Mesh<Scalar>& m,
        const std::vector<int>& vertex_from_halfedge);
    bool is_valid_primal_tree(const Mesh<Scalar>& m) const;
};

/**
 * @brief Representation for a dual minimal spanning tree (on faces and edges) on a mesh
 *
 */
class DualTree : public Forest
{
public:
    /**
     * @brief Construct an empty Dual Tree object
     *
     */
    DualTree() {}

    /**
     * @brief Construct a new Dual Tree object on a mesh
     *
     * @param m: underlying mesh
     * @param weights: per-halfedge weights (assumes halfedge pairs have same weight)
     * @param root: (optional) starting face root for the tree construction
     * @param use_shortest_path: (optional) use shortest path tree instead of minimal weight
     * tree
     */
    DualTree(
        const Mesh<Scalar>& m,
        const std::vector<Scalar>& weights,
        int root = 0,
        bool use_shortest_path = false);

    /**
     * @brief Determine if a mesh edge is in the tree
     *
     * @param edge_index: index of an edge in the mesh
     * @return true if the edge is in the tree
     * @return false otherwise
     */
    bool is_edge_in_tree(int edge_index) const
    {
        return is_edge_in_forest(edge_index);
    }

protected:
    void initialize_dual_tree(const Mesh<Scalar>& m, const std::vector<int>& face_from_halfedge);
    bool is_valid_dual_tree(const Mesh<Scalar>& m) const;
};

/**
 * @brief Representation for a primal minimal spanning cotree on a mesh that is disjoint from a
 * given dual tree
 *
 */
class PrimalCotree : public PrimalTree
{
public:
    /**
     * @brief Construct an empty Primal Cotree object
     *
     */
    PrimalCotree() {}

    /**
     * @brief Construct a new Primal Cotree object on a mesh disjoint from the given dual tree
     *
     * @param m: underlying mesh
     * @param weights: per-halfedge weights (assumes halfedge pairs have same weight)
     * @param dual_tree: dual tree on the mesh to keep disjoint
     * @param root: (optional) starting vertex root for the tree construction
     * @param use_shortest_path: (optional) use shortest path tree instead of minimal weight
     * tree
     */
    PrimalCotree(
        const Mesh<Scalar>& m,
        const std::vector<Scalar>& weights,
        const DualTree& dual_tree,
        int root = 0,
        bool use_shortest_path = false);

private:
    bool is_valid_primal_cotree(const Mesh<Scalar>& m, const DualTree& dual_tree) const;
};

/**
 * @brief Representation for a dual minimal spanning cotree on a mesh that is disjoint from a
 * given primal tree
 *
 */
class DualCotree : public DualTree
{
public:
    /**
     * @brief Construct an empty Dual Cotree object
     * 
     */
    DualCotree() {}

    /**
     * @brief Construct a new Dual Cotree object on a mesh disjoint from the given primal tree
     *
     * @param m: underlying mesh
     * @param weights: per-halfedge weights (assumes halfedge pairs have same weight)
     * @param primal_tree: primal tree on the mesh to keep disjoint
     * @param root: (optional) starting face root for the tree construction
     * @param use_shortest_path: (optional) use shortest path tree instead of minimal weight
     * tree
     */
    DualCotree(
        const Mesh<Scalar>& m,
        const std::vector<Scalar>& weights,
        const PrimalTree& primal_tree,
        int root = 0,
        bool use_shortest_path = false);

private:
    bool is_valid_dual_cotree(const Mesh<Scalar>& m, const PrimalTree& primal_tree) const;
};

/**
 * @brief Build a minimal spanning forest on a mesh with respect to given edge weights
 * and cut along certain edges.
 *
 * @param m: underlying mesh
 * @param weights: per-halfedge weights (assumes halfedge pairs have same weight)
 * @param is_cut: mask for cut mesh edges
 * @param v_start: starting root for the forest construction
 * @param use_shortest_path: use shortest path forest instead of minimal weight forest
 * @return map from vertices vj to the halfedge hij connecting them to its parent vi in the forest
 */
std::vector<int> build_primal_forest(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const std::vector<bool>& is_cut,
    int v_start = 0,
    bool use_shortest_path = false);

/**
 * @brief Build a minimal dual spanning forest on a mesh with respect to given edge weights
 * and cut along certain dual edges.
 *
 * @param m: underlying mesh
 * @param weights: per-halfedge weights (assumes halfedge pairs have same weight)
 * @param is_cut: mask for cut mesh dual edges
 * @param f_start: starting root for the forest construction
 * @param use_shortest_path: use shortest path forest instead of minimal weight forest
 * @return map from faces fj to the halfedge (adjacent to fi) connecting it to its parent fi in the
 * forest
 */
std::vector<int> build_dual_forest(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const std::vector<bool>& is_cut,
    int f_start = 0,
    bool use_shortest_path = false);

/**
 * @brief Compute dual edge lengths using the DEC formulation with dual vertices at circumcenters.
 *
 * @param m: mesh with primal edge lengths
 * @return per-halfedge dual edge lengths
 */
std::vector<Scalar> compute_dual_edge_lengths(const Mesh<Scalar>& m);

/**
 * @brief Compute the edge weights for a primal tree given by the length of the root cycle
 * generated by adding an edge to a dual spanning tree (or 0 if the edge is in the spanning tree)
 *
 * @param m: underlying mesh
 * @param weights: weights on the dual mesh for loop lengths
 * @param dual_tree: spanning dual tree
 * @return per-halfedge dual loop length weights
 */
std::vector<Scalar> compute_dual_loop_length_weights(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const DualTree& dual_tree);

} // namespace PennerHolonomy