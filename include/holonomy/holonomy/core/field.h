
#pragma once

#include "holonomy/core/common.h"

namespace PennerHolonomy {

/**
 * @brief Generate a cross field for a mesh
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @return |F|x3 frame field of per-face field direction vectors
 * @return per-vertex cone angles corresponding to the frame field
 */
std::tuple<Eigen::MatrixXd, std::vector<Scalar>> generate_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);


} // namespace PennerHolonomy
