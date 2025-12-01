
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/dual_loop.h"
#include "util/spanning_tree.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Construct a clockwise dual path around a vertex in the mesh.
 *
 * @param m: mesh
 * @param vertex_index: index of the vertex to build a dual loop around
 * @return dual loop composed of dual segments
 */
std::vector<DualSegment> build_clockwise_vertex_dual_segment_sequence(
    const Mesh<Scalar>& m,
    int vertex_index);

/**
 * @brief Construct a counterclockwise dual path around a vertex in the mesh.
 *
 * @param m: mesh
 * @param vertex_index: index of the vertex to build a dual loop around
 * @return dual loop composed of dual segments
 */
std::vector<DualSegment> build_counterclockwise_vertex_dual_segment_sequence(
    const Mesh<Scalar>& m,
    int vertex_index);

/**
 * @brief Class to generate a homotopy basis via a tree-cotree decomposition.
 */
class HomotopyBasisGenerator
{
public:
    // Weighting scheme for the tree-cotree construction
    enum Weighting {
        minimal_homotopy, // Use a shortest path dual tree with maximal dual loop length
                          // primal cotree (default choice with good theoretical properties)
        maximal_homotopy, // Use a longest path dual tree with minimal dual loop length
                          // primal cotree
        dual_min_primal_max, // Use dual edge lengths with a minimal dual tree and maximal primal cotree
        primal_min_dual_max // Use primal edge lengths with a minimal primal tree and maximal dual cotree
    };

    /**
     * @brief Construct a new Homotopy Basis Generator object on a mesh
     * 
     * @param m: mesh
     * @param root (optional) root vertex (or dual-vertex) for the tree construction
     * @param weighting (optional) weighting for the tree-cotree construction
     */
    HomotopyBasisGenerator(
        const Mesh<Scalar>& m,
        int root = 0,
        Weighting weighting = Weighting::minimal_homotopy);

    /**
     * @brief Get number of homology basis loops (twice the genus)
     * 
     * @return number of homology basis loops
     */
    int n_homology_basis_loops() const { return m_homotopy_basis_edge_handles.size(); }

    /**
     * @brief Construct a dual loop corresponding to the homotopy basis loop with given index
     * 
     * NOTE: The dual loop is rooted and thus generally is not simple
     * 
     * @param index: index of the homotopy basis loop
     * @return sequence of faces defining the dual loop
     * @return sequence of edges between the dual loop faces
     */
    std::tuple<std::vector<int>, std::vector<int>> construct_homotopy_basis_edge_loop(
        int index) const;

    /**
     * @brief Construct a dual loop corresponding to the homotopy basis loop with given index with
     * the path to the root contracted to make the loop simple
     * 
     * @return sequence of faces defining the dual loop
     * @return sequence of edges between the dual loop faces
     */
    std::tuple<std::vector<int>, std::vector<int>> construct_homology_basis_edge_loop(
        int index) const;

    /**
     * @brief Construct a dual loop corresponding to the homotopy basis loop with given index with
     * the path to the root contracted to make the loop simple
     * 
     * @return sequence of faces defining the dual loop
     */
    std::vector<int> construct_homology_basis_loop(int index) const;

    /**
     * @brief Get the edge handle for the homotopy basis loop with the given index that forms a
     * homotopy basis cycle when added to the dual tree
     * 
     * @param index: index of the homotopy basis loop
     * @return edge index in the mesh of the handle edge
     */
    int homotopy_basis_handle(int index) const { return m_homotopy_basis_edge_handles[index]; }

	/**
	 * @brief Get the primal tree object used for the tree-cotree construction
	 * 
	 * @return primal tree reference
	 */
    const PrimalTree& get_primal_tree() const { return m_primal_tree; }

	/**
	 * @brief Get the dual tree object used for the tree-cotree construction
	 * 
	 * @return dual tree reference
	 */
    const DualTree& get_dual_tree() const { return m_dual_tree; }

    std::vector<DualSegment> construct_homology_basis_dual_path(int index) const;

private:
    Mesh<Scalar> m_mesh;
    std::vector<int> m_he2e;
    std::vector<int> m_e2he;

    PrimalTree m_primal_tree;
    DualTree m_dual_tree;
    std::vector<int> m_homotopy_basis_edge_handles;

    std::tuple<std::vector<int>, std::vector<int>> trace_dual_vertex_to_root(int face_index) const;
};

typedef HomotopyBasisGenerator HomologyBasisGenerator;


} // namespace Holonomy
} // namespace Penner