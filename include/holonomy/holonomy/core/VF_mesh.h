#pragma once

#include "holonomy/core/common.h"

namespace PennerHolonomy {

/**
 * @brief Generate a list of boundary vertex indices in a VF mesh
 *
 * @param F: mesh faces
 * @return list of boundary vertex indices
 */
std::vector<int> find_boundary_vertices(const Eigen::MatrixXi& F, int num_vertices);

/**
 * @brief Compute the boundary vertices of a VF mesh.
 *
 * @param F: mesh faces
 * @return boolean mask of boundary vertices
 */
std::vector<bool> compute_boundary_vertices(const Eigen::MatrixXi& F, int num_vertices);

/**
 * @brief Inflate a mesh by displacing vertices along the mesh surface normal.
 *
 * @param V: input mesh vertices
 * @param F: mesh faces
 * @param inflation_distance: (optional) distance to displace vertices
 * @return inflated mesh vertices
 */
Eigen::MatrixXd
inflate_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double inflation_distance = 1e-8);

} // namespace PennerHolonomy
