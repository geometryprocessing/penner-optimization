#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Representation of an edge as the two opposite face corners
 * 
 */
class FaceEdge {
public:
    /**
     * @brief Construct a new Face Edge object from corner indices
     * 
     * @param fijk: first adjacent face
     * @param k: first local corner index opposite the edge eij
     * @param fjil: second adjacent face
     * @param l: second local corner index opposite the edge eij
     */
    FaceEdge(int fijk, int k, int fjil, int l)
    {
        faces = {fijk, fjil};
        local_indices = {k, l};
    }

    /**
     * @brief Get the face and local index for the face to the right of the edge
     * 
     * @return face index
     * @return local corner index
     */
    std::pair<int, int> right_corner() const
    {
        return {faces[0], local_indices[0]};
    }

    /**
     * @brief Get the face and local index for the face to the left of the edge
     * 
     * @return face index 
     * @return local corner index
     */
    std::pair<int, int> left_corner() const
    {
        return {faces[1], local_indices[1]};
    }

private:
    std::array<int, 2> faces;
    std::array<int, 2> local_indices;
};

/**
 * @brief Find the mask of corners in only the first mask.
 * 
 * @param F_is_in: corners to include
 * @param F_is_out: corners to exclude
 * @return corner mask of corners in the first mask and not in the second
 */
Eigen::MatrixXi mask_difference(
    const Eigen::MatrixXi& F_is_in,
    const Eigen::MatrixXi& F_is_out);

/**
 * @brief Compute a list of corners with nonzero value in a corner mask.
 * 
 * @param F_mask: |F|x3 mask of mesh corners 
 * @return list of nonzero corner index pairs (fijk, i)
 */
std::vector<std::pair<int, int>> compute_mask_corners(const Eigen::MatrixXi& F_mask);

/**
 * @brief Compute a mask marking the face corners.
 * 
 * @param num_faces: number of faces in the mesh
 * @param mask_corners: list of nonzero corner index pairs (fijk, i)
 * @return |F|x3 corner mask
 */
Eigen::MatrixXi compute_mask_from_corners(
    int num_faces,
    const std::vector<std::pair<int, int>>& mask_corners);

/**
 * @brief Given a list of mesh corners, produce a list of the opposite halfedges (vi, vj).
 * 
 * @param corners: list of face and local corner indices (fijk, i)
 * @param F: mesh faces
 * @return list of edge endpoints opposite the given corners
 */
std::vector<VertexEdge> compute_corner_edges(
    const std::vector<std::pair<int, int>>& corners,
    const Eigen::MatrixXi& F);

/**
 * @brief Given a list of mesh halfedges (vi, vj), produce a list of the opposite corners.
 * 
 * @param edges: list of edge endpoints
 * @param F: mesh faces 
 * @return list of corners opposite the given oriented edges
 */
std::vector<std::pair<int, int>> compute_edge_corners(
    const std::vector<VertexEdge>& edges,
    const Eigen::MatrixXi& F);

/**
 * @brief Given a a list of corners, build a list of face edges opposite the corners
 * 
 * WARNING: A face edge is only added if both opposite corners are given.
 * 
 * @param F: mesh faces
 * @param corners: list of face and local corner index pairs
 * @return list of edges opposite the corners
 */
std::vector<FaceEdge> compute_face_edges_from_corners(
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& corners
);

/**
 * @brief Given a list of face edges, generate a list of the edge endpoints.
 * 
 * WARNING: Only one (arbitrary) endpoint pair is given, not both orientations
 * 
 * @param face_edges: list of face edge representations
 * @param F: mesh faces
 * @return list of edge endpoints (vi, vj)
 */
std::vector<VertexEdge> compute_face_edge_endpoints(
    const std::vector<FaceEdge>& face_edges,
    const Eigen::MatrixXi& F);

/**
 * @brief Compute a matrix mapping oriented vertex pairs (vi, vj) to halfedges.
 * 
 * Halfedge indices are converted to 1-indexing to differentiate from 0 entries, and
 * primal halfedges are used for doubled meshes.
 * 
 * @param m: halfedge mesh
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @return map from vertex pairs to halfedge indices (+1 due to one-indexing)
 */
Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex);

/**
 * @brief Compute a matrix mapping oriented vertex pairs (vi, vj) in a glued mesh
 * to halfedges in the cut halfedge mesh.
 * 
 * Halfedge indices are converted to 1-indexing to differentiate from 0 entries.
 * 
 * @param m: cut halfedge mesh
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @return map from uncut vertex pairs to cut primal halfedge indices (+1 due to one-indexing)
 */
Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

/**
 * @brief Given a list of relaxed corners, find the halfedges in the cut mesh corresponding to the
 * opposite edge in the glued mesh
 * 
 * @param relaxed_corners:list of face and local relaxed corner indices (fijk, i)
 * @param m: cut halfedge mesh
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @param F_cut: VF faces of the cut mesh
 * @return primal halfedge pairs (hij, hji) corresponding to the relaxed corners 
 */
std::vector<std::pair<int, int>> compute_relaxed_edges(
    const std::vector<std::pair<int, int>>& relaxed_corners,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXi& F_cut);

/**
 * @brief Generate the cut mask for an overlay mesh from the original cut information.
 *
 * @param F_overlay: overlay mesh faces
 * @param endpoints: endpoints of the inserted edge vertices in the overlay
 * @param F_cut: original cut mesh faces
 * @param F_is_cut: corner mask for opposite cut edges in the original mesh
 * @return corner mask for opposite cut edges in the overlay mesh
 */
Eigen::MatrixXi generate_overlay_cut_mask(
    const Eigen::MatrixXi& F_overlay,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::MatrixXi& F_cut,
    const Eigen::MatrixXi& F_is_cut);

/**
 * @brief Given a list of corners opposite edges, remove redundant corners referring to
 * the same edge.
 * 
 * @param F: mesh faces
 * @param corners: list of face and local relaxed corner indices (fijk, i) opposite edges
 * @return independent list of corners opposite edges
 */
std::vector<std::pair<int, int>> prune_redundant_edge_corners(
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& corners);

/**
 * @brief Given a function defined on the corners of a VF mesh, propagate it to the opposite
 * halfedges of a cut halfedge mesh
 * 
 * @param m: cut halfedge mesh
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @param F: glued mesh faces
 * @param corner_func: |F|x3 matrix of corner function values
 * @return vector of halfedge values
 */
VectorX transfer_corner_function_to_halfedge(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& corner_func);

/**
 * @brief Given a function defined on the halfedges of a cut mesh, propagate it to the opposite
 * corners of a glued VF mesh
 * 
 * @param m: cut halfedge mesh
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @param F: glued mesh faces
 * @param corner_func: matrix of halfedge function values
 * @return |F|x3 matrix of corner values
 */
Eigen::MatrixXd transfer_halfedge_function_to_corner(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXi& F,
    const VectorX& halfedge_func);

/**
 * @brief Find the seams of the parameterization of a closed mesh.
 * 
 * TODO: Extend to open meshes
 * 
 * @param F: mesh faces
 * @param FT: mesh layout faces
 * @return |F|x3 mask of corners opposite seam edges
 */
Eigen::MatrixXi find_seams(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& FT);

/**
 * @brief Given a mask of corners opposite edges, generate the corresponding edge geometry.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_is_edge: |F|x3 matrix of corners opposite edges
 * @return edge vertices
 * @return edges
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_edges(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_is_edge);

} // namespace Feature
} // namespace Penner