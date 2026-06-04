
#pragma once

#include "holonomy/core/common.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Load rawfield data format used by Directional.
 * 
 * @param filename: rawfield data file
 * @return four per-face tangent direction matrices
 */
std::array<Eigen::MatrixXd, 4> load_rawfield(const std::string& filename);

/**
 * @brief Serialize a cross field defined by an angle relative to a reference direction as
 * four directions per face (i.e., rawfield format for Directional).
 * 
 * @param output_filename: file to write cross field
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 */
void write_cross_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Given a representative cross field direction defined by a reference direction and an
 * offset angle, generate all four cross field direction matrices.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @return four per-face tangent direction matrices
 */
std::array<Eigen::MatrixXd, 4> generate_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Reduce the curl of a cross field
 * 
 * WARNING: Not well tested or currently in active use
 * 
 * @param V 
 * @param F 
 * @param cross_field 
 * @param fixed_faces 
 * @return std::array<Eigen::MatrixXd, 4> 
 */
std::array<Eigen::MatrixXd, 4> reduce_curl(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::array<Eigen::MatrixXd, 4>& cross_field,
    const std::vector<int>& fixed_faces);

} // namespace Feature 
} // namespace Penner
