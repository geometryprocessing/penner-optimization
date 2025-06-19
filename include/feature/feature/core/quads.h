
#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Compute the vertex valences of a given (tri or quad) mesh
 * 
 * @param F: mesh faces
 * @return valences of the mesh vertices
 */
std::vector<int> compute_valences(const Eigen::MatrixXi& F);

} // namespace Feature
} // namespace Penner