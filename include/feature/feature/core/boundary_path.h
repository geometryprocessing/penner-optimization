#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Representation of a boundary edge path. If the boundary edge is unflipped and lies along
 * the symmetry line, then this is just a single halfedge in the primal mesh. If the edge is
 * flipped, then this is a path of halfedges in the mesh copy (type 2) homotopic to the symmetry
 * line segment between the two vertices on the boundary.
 *
 */
class BoundaryPath
{
public:
    /**
     * @brief Construct a new Boundary Path object from the given boundary vertex
     * to the next ccw vertex on the boundary.
     *
     * @param m: underlying mesh
     * @param vertex_index: starting vertex index (must be on the boundary)
     */
    BoundaryPath(const Mesh<Scalar>& m, int vertex_index);

    /**
     * @brief Compute the number of edges in the boundary path
     * 
     * @return number of edges
     */
    int size() const { return m_halfedge_path.size(); }

    /**
     * @brief Compute the length of the boundary path in the given mesh
     *
     * @param m: underlying mesh
     * @return length of the current path in the mesh
     */
    Scalar compute_length(const Mesh<Scalar>& m) const;

    /**
     * @brief Compute the (doubled) log length of the boundary path in the given mesh
     *
     * @param m: underlying mesh
     * @return log length of the current path in the mesh
     */
    Scalar compute_log_length(const Mesh<Scalar>& m) const;

    /**
     * @brief Compute the Jacobian of the length of the boundary path in the given mesh with
     * respect to the halfedge lengths
     *
     * The Jacobian is represented as a sparse vector with index, value pairs.
     *
     * @param m: underlying mesh
     * @return jacobian of length
     */
    std::vector<std::pair<int, Scalar>> compute_length_jacobian(const Mesh<Scalar>& m) const;

    /**
     * @brief Compute the Jacobian of the log length of the boundary path in the given mesh with
     * respect to the halfedge lengths
     *
     * The Jacobian is represented as a sparse vector with index, value pairs.
     *
     * @param m: underlying mesh
     * @return jacobian of log length
     */
    std::vector<std::pair<int, Scalar>> compute_log_length_jacobian(const Mesh<Scalar>& m) const;

    /**
     * @brief Check if the boundary path is valid
     *
     * @param m: underlying mesh
     * @return true if the path is valid
     * @return false otherwise
     */
    bool is_valid_boundary_path(const Mesh<Scalar>& m) const;

    /**
     * @brief Get the start vertex of the path
     * 
     * @return start vertex index
     */
    int get_start_vertex() const { return m_start_vertex; }

    /**
     * @brief Get the ending vertex of the path, which is the next original boundary vertex
     * counterclockwise on the symmetry line.
     * 
     * @return end vertex index
     */
    int get_end_vertex() const { return m_end_vertex; }

    /**
     * @brief Get the representative path of halfedges from the start to end vertex;
     * 
     * This is either a single primal halfedge or a path of doubled halfedges bounding 
     * the facets crossing the symmetry line between the vertices.
     * 
     * @return path from the start to end vertex
     */
    const std::vector<int>& get_halfedge_path() const { return m_halfedge_path; }

    /**
     * @brief Get transverse halfedges crossing the line of symmetry.
     * 
     * One halfedge per transverse edge is returned.
     * 
     * @return transverse halfedges.
     */
    const std::vector<int>& get_transverse_edges() const { return m_transverse_edges; }

private:
    int m_start_vertex;
    int m_end_vertex;

    Scalar compute_tri_length(Scalar side_length, Scalar base_length) const;
    Scalar compute_quad_length(
        Scalar side_length,
        Scalar first_base_length,
        Scalar second_base_length) const;
    Scalar compute_tri_side_derivative(Scalar side_length, Scalar base_length) const;
    Scalar compute_tri_base_derivative(Scalar side_length, Scalar base_length) const;
    Scalar compute_quad_side_derivative(
        Scalar side_length,
        Scalar first_base_length,
        Scalar second_base_length) const;
    Scalar compute_quad_base_derivative(
        Scalar side_length,
        Scalar first_base_length,
        Scalar second_base_length) const;

    // Paths between boundary vertices contained in the primal mesh
    std::vector<int> m_halfedge_path;

    // Transverse edges crossing the symmetry line
    std::vector<int> m_transverse_edges;
};

/**
 * @brief Build all boundary paths for a mesh and a map to the corresponding halfedges.
 * 
 * @param m: underlying mesh
 * @return list of boundary paths
 * @return map from boundary path indices to halfedge indices
 */
std::tuple<std::vector<BoundaryPath>, MatrixX> build_boundary_paths(const Mesh<Scalar>& m);

} // namespace Feature
} // namespace Penner
