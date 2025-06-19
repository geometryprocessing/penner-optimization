
#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"

namespace Penner {
namespace Feature {

/**
 * @brief Generate angle relaxation matrix for the given mesh and relaxed edges.
 * 
 * Assumes that each vertex of the mesh is adjacent to some unrelaxed edge.
 * 
 * @param m: underlying mesh
 * @param relaxed_edges: halfedge pairs (hij, hji) of relaxed edges
 * @return matrix mapping vertex cone angles to a relaxed system
 */
MatrixX compute_relaxed_angle_constraint_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::pair<int, int>>& relaxed_edges);

/**
 * @brief Class to construct joined corner systems for angle relaxation and create the
 * relaxed holonomy angle prescription matrices.
 */
class AngleConstraintMatrixRelaxer
{
public:
    /**
     * @brief Construct a trivial Angle Constraint Matrix Relaxer object
     *
     */
    AngleConstraintMatrixRelaxer();

    /**
     * @brief Generate angle relaxation matrix for the given mesh and relaxed edges.
     * 
     * Assumes that each vertex of the mesh is adjacent to some unrelaxed edge.
     * 
     * @param m: underlying mesh
     * @param relaxed_edges: halfedge pairs (hij, hji) of relaxed edges
     * @return matrix mapping vertex cone angles to a relaxed system
     */
    MatrixX run(const Mesh<Scalar>& m, const std::vector<std::pair<int, int>>& relaxed_edges);

    /**
     * @brief View the relaxed edges and joined corners on the mesh.
     * 
     * @param m: underlying mesh
     * @param vtx_reindex: map from halfedge vertex indices to VF vertex indices
     * @param V: mesh vertex positions
     */
    void view(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex, const Eigen::MatrixXd& V)
        const;

private:
    /**
     * @brief Representation for collections of contiguous corner angles at a vertex in a cut graph.
     *
     */
    class JoinedCorners
    {
    public:
        typedef Eigen::Triplet<Scalar> T;

        /**
         * @brief Construct a trivial Joined Corners object.
         *
         */
        JoinedCorners()
            : m_corners({})
            , m_edges({})
        {}

        /**
         * @brief Join the two corners across a given edge at vertex vj.
         *
         * The edge eij should contain the two primal halfedges in the uncut mesh. Note that this
         * means these edges are not opposite each other in the cut mesh m.
         *
         * @param m: underlying cut mesh
         * @param eij: edge (hij, hji) separating the two corners to join
         */
        JoinedCorners(const Mesh<Scalar>& m, const std::pair<int, int>& eij)
            : JoinedCorners()
        {
            join_adjacent_corner(m, eij);
        }

        /**
         * @brief Get number of edges in the interior of the joined corners
         *
         * @return number of interior edges
         */
        int n_edges() const { return m_edges.size(); }

        /**
         * @brief Get the number of joined corners in the collection
         *
         * @return number of edges
         */
        int n_corners() const { return m_corners.size(); }

        /**
         * @brief Get the corners in the joined corner collection
         *
         * @return const reference to corners
         */
        const std::deque<int>& get_corners() const { return m_corners; }

        /**
         * @brief Get the interior edges in the joined corner collection
         *
         * @return const reference to edges of format (hij, hji)
         */
        const std::deque<std::pair<int, int>>& get_edges() const { return m_edges; }

        /**
         * @brief Get the corners in the joined collection prior to the given edge.
         *
         * @param eij: edge (hij, hji) in the interior of the joined corner
         * @return corners in the prefix
         */
        std::vector<int> get_prefix(const std::pair<int, int>& eij);

        /**
         * @brief Get the corners in the joined collection after the given edge.
         *
         * @param eij: edge (hij, hji) in the interior of the joined corner
         * @return corners in the suffix
         */
        std::vector<int> get_suffix(const std::pair<int, int>& eij);

        /**
         * @brief Add corner adjacent to the given joined corner collection.
         *
         * @param eij: edge between the two corners (halfedges may be in separate components)
         */
        void join_adjacent_corner(const Mesh<Scalar>& m, const std::pair<int, int>& eij);

        /**
         * @brief Merge another joined corner collection across an edge
         *
         * The joined corners to merge can be either clockwise or counterclockwise.
         *
         * @param m: underlying cut mesh
         * @param eij: edge separating two joined corners
         * @param adjacent_corners: corners to merge into current collection
         */
        void merge(
            const Mesh<Scalar>& m,
            const std::pair<int, int>& eij,
            const JoinedCorners& adjacent_corners);

    private:
        std::deque<int> m_corners;
        std::deque<std::pair<int, int>> m_edges;

        bool is_valid_joined_corners(const Mesh<Scalar>& m) const;
    };

    std::vector<JoinedCorners> joined_corners;
    std::vector<int> joined_corner_map;

    void join_tip(const Mesh<Scalar>& m, const std::pair<int, int>& eij);
    void join_tail(const Mesh<Scalar>& m, const std::pair<int, int>& eij);
    void
    add_joined_vertex_constraints(const Mesh<Scalar>& m, std::vector<T>& tripletList, int& count);
    void add_edge_holonomy_constraints(
        const Mesh<Scalar>& m,
        const std::vector<std::pair<int, int>>& relaxed_edges,
        std::vector<T>& tripletList,
        int& count);
    std::pair<int, int> swap_edge_vertex(const std::pair<int, int>& eij) const;
    bool is_joined_corner(int ci) const;
};

} // namespace Feature
} // namespace Penner