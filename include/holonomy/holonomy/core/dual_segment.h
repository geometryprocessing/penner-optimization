#pragma once

#include "holonomy/core/common.h"

namespace PennerHolonomy {

typedef std::array<int, 2> DualSegment;

/**
 * @brief Check if a dual segment is valid.
 *
 * The only condition is that both halfedges of the segment belong to the same face.
 *
 * @param m: mesh
 * @param dual_segment: pair of halfedge indices specifying a dual segment in a face
 * @return true if the dual segment is valid
 * @return false otherwise
 */
bool is_valid_dual_segment(const Mesh<Scalar>& m, const DualSegment& dual_segment);

/**
 * @brief Determine if a vector of dual segments specifies a valid dual path.
 *
 * The conditions are:
 *   - each dual segment is valid
 *   - sequential dual segments are adjacent in the mesh
 *
 * @param m: mesh
 * @param dual_path: vector of dual segments specifying a dual path
 * @return true if the dual path is valid
 * @return false otherwise
 */
bool is_valid_dual_path(const Mesh<Scalar>& m, const std::vector<DualSegment>& dual_path);

/**
 * @brief Determine if a vector of dual segments specifies a valid dual loop.
 *
 * The conditions are that the loop is a nontrivial valid dual path and the last segment
 * is adjacent to the first.
 *
 * @param m: mesh
 * @param dual_loop: vector of dual segments specifying a dual loop
 * @return true if the dual loop is valid
 * @return false otherwise
 */
bool is_valid_dual_loop(const Mesh<Scalar>& m, const std::vector<DualSegment>& dual_loop);

/**
 * @brief Get the face index containing a dual segment.
 * 
 * @param m: mesh
 * @param dual_segment: pair of halfedge indices specifying a dual segment in a face
 * @return face containing the segment
 */
int compute_dual_segment_face(const Mesh<Scalar>& m, const DualSegment& dual_segment);

/**
 * @brief Reverse the orientation of a dual segment.
 *
 * @param dual_segment: pair of halfedge indices specifying a dual segment in a face
 * @return reversed dual segment
 */
DualSegment reverse_dual_segment(const DualSegment& dual_segment);

/**
 * @brief Reverse the orientation of a dual path.
 *
 * @param dual_path: vector of dual segments specifying a dual path
 * @return reversed dual path
 */
std::vector<DualSegment> reverse_dual_path(const std::vector<DualSegment>& dual_path);

/**
 * @brief Construct a sequence of faces on a mesh from a dual loop path.
 *
 * @param m: mesh
 * @param dual_path: dual path composed of dual segments
 * @return sequence of faces in the mesh (must be adjacent)
 */
std::vector<int> build_face_sequence_from_dual_path(
    const Mesh<Scalar>& m,
    const std::vector<DualSegment>& dual_path);

/**
 * @brief Construct a sequence of dual segments from a dual loop face sequence
 * 
 * NOTE: The face sequence must constitute a closed dual loop and not just a dual path.
 * 
 * @param m: underlying mesh
 * @param dual_loop_faces: closed dual loop on the mesh
 * @return vector of dual segments specifying a dual path
 */
std::vector<DualSegment> build_dual_path_from_face_sequence(
    const Mesh<Scalar>& m,
    const std::vector<int>& dual_loop_faces);

/**
 * @brief Update the dual loop for a mesh to be flipped at a given halfedge.
 *
 * NOTE: Dual path sequence must be a closed loop
 * 
 * @param m: mesh before flip
 * @param halfedge_index: halfedge to be flipped
 * @param dual_loop: dual loop to be modified
 */
void update_dual_loop_under_ccw_flip(
    const Mesh<Scalar>& m,
    int halfedge_index,
    std::vector<DualSegment>& dual_loop);

/**
 * @brief Update the dual loop for a mesh to be flipped according to a sequence.
 *
 * NOTE: Dual path sequence must be a closed loop
 * 
 * @param m: mesh before flips
 * @param flip_seq: sequence of flips to perform
 * @param dual_loop: dual loop to be modified
 */
void update_dual_loop_under_ccw_flip_sequence(
    const Mesh<Scalar>& m,
    const std::vector<int>& flip_seq,
    std::vector<DualSegment>& dual_loop);

/**
 * @brief View a dual loop on a mesh.
 *
 * @param V: vertices of the mesh
 * @param F: faces of the mesh
 * @param m: mesh
 * @param dual_loop
 */
void view_dual_path(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Mesh<Scalar>& m,
    const std::vector<DualSegment>& dual_path);

} // namespace PennerHolonomy