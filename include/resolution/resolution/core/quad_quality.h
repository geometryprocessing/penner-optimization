
#pragma once
#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

Scalar compute_triangle_angle(
    const Vector3& x0,
    const Vector3& x1,
    const Vector3& x2);


/**
 * @brief Compute the four per-face corner angles of a quad mesh
 * 
 * @param V: quad mesh vertices
 * @param F: quad mesh faces
 * @return per-face arrays of four corner angles
 */
std::vector<std::array<Scalar, 4>> compute_quad_angles(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
);

/**
 * @brief Compute the four per-face edge lengths of a quad mesh
 * 
 * @param V: quad mesh vertices
 * @param F: quad mesh faces
 * @return per-face arrays of four edge lengths
 */
std::vector<std::array<Scalar, 4>> compute_quad_lengths(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
);

/**
 * @brief Compute the per-face scaled jacobian quality measure of a quad mesh.
 * 
 * The values lie between [-1, 1], with [0.3, 1] considered acceptable
 * 
 * @param V: quad mesh vertices
 * @param F: quad mesh faces
 * @return per-face scaled Jacobian values
 */
std::vector<Scalar> compute_scaled_jacobian(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
);

}
}