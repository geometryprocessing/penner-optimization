#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/halfedge.h"

namespace Penner {
namespace Resolution {


/**
 * @brief Compute seamless error measure for the rotation that identifies a halfedge
 * with its paired halfedge in the uv plane.
 * 
 * The error measure is determined from the fact that the 2x2 rotation matrix
 * in the plane R has order that divides 4. The error is the max deviation of the
 * coefficients of R^4 from the identity coefficients.
 * 
 * @param halfedge: halfedge data structure
 * @param uv: per-halfedge tip uv coordinates
 * @param hij: halfedge to compute the rotation across
 * @return seamless error measure
 */
double compute_seamless_error(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij);

/**
 * @brief Determine if a 2x2 matrix corresponds to a seamless rotation, up to some
 * numerical tolerance
 * 
 * @param R: 2x2 rotation matrix
 * @return true if the matrix is numerically seamless
 * @return false otherwise
 */
bool is_seamless_rotation(const Eigen::Matrix2d& R);

/**
 * @brief Determine the rotation to identify a halfedge with its paired edge, up to 
 * translation
 * 
 * @param halfedge: halfedge data structure
 * @param uv: per-halfedge tip uv coordinates
 * @param hij: halfedge to compute the rotation across
 * @return 2x2 rotation matrix
 */
Eigen::Matrix2d infer_float_rotation(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij
);

/**
 * @brief Determine the rotation to identify a halfedge with its paired edge, up to 
 * translation, rounding the coefficients to the nearest integer.
 * 
 * WARNING: this snapping is only safe for seamless rotations, which are theoretically
 * integer valued
 * 
 * @param halfedge: halfedge data structure
 * @param uv: per-halfedge tip uv coordinates
 * @param hij: halfedge to compute the rotation across
 * @return 2x2 rotation matrix
 */
Eigen::Matrix2d infer_rotation(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij
);

/**
 * @brief Rigidly transform the face opposite the face containing the given halfedge
 * to be adjacent to the halfedge.
 * 
 * WARNING: assumes the parametrization is seamless so that the rotation for the 
 * transformation is a numerically stable integer valued matrix
 * 
 * @param halfedge: halfedge data structure
 * @param uv: per-halfedge tip uv coordinates
 * @param hij: halfedge to join with its mate (this halfedge is fixed; the mate is moved)
 * @return rotation matrix used in the transformation
 */
Eigen::Matrix2d transform_opposite_face(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int hij
);

/**
 * @brief Rotate a vector in the plane of a face of a halfedge matrix
 * 
 * @param halfedge: halfedge data structure
 * @param V: per-halfedge tip 3D coordinates
 * @param R: rotation to apply
 * @param d; 3D vector, assumed tangent to given face
 * @param face_index: face index
 * @return rotated vector
 */
Eigen::Vector3d rotate_in_face_plane(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector3d>& V,
    const Eigen::Matrix2d& R,
    const Eigen::Vector3d& d,
    int face_index);

/**
 * @brief Rigidly transform the face opposite the face containing the given halfedge
 * to be adjacent to the halfedge, and also transform two per-face vector fields 
 * representing target uv gradients accordingly so the norm of the gradients from the
 * targets is invariant under the transformation
 * 
 * WARNING: assumes the parametrization is seamless so that the rotation for the 
 * transformation is a numerically stable integer valued matrix
 * 
 * @param halfedge: halfedge data structure
 * @param uv: per-halfedge tip uv coordinates
 * @param V: per-halfedge tip 3D coordinates
 * @param PD1: target u gradient per face
 * @param PD2: target v gradient per face
 * @param hij: halfedge to join with its mate (this halfedge is fixed; the mate is moved)
 * @return rotation matrix used in the transformation
 */
Eigen::Matrix2d transform_opposite_face(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V,
    std::vector<Eigen::Vector3d>& PD1,
    std::vector<Eigen::Vector3d>& PD2,
    int hij);

}
}
