#pragma once

#include <array>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "optimization/interface.h"
#include "optimization/core/common.h"
#include "optimization/core/cone_metric.h"
#include "util/vector.h"

namespace Penner {
namespace Holonomy {

// Typedefs
using Optimization::DifferentiableConeMetric;
typedef Eigen::Matrix<int, Eigen::Dynamic, 2> RowVectors2i;


/**
 * @brief Compute the square of a numeric value.
 *
 * @tparam type of object (must support multiplication)
 * @param a: value to square
 * @return squared value
 */
template <typename Type>
Type square(const Type& a)
{
    return a * a;
}

/**
 * @brief Compute the real modulus of x mod y
 * 
 * @param x: positive number to mod
 * @param y: positive modulus
 * @return x (mod y)
 */
inline
Scalar pos_fmod(Scalar x, Scalar y) { return (0 == y) ? x : x - y * floor(x / y); }

/**
 * @brief Compute the Euler characteristic of the mesh
 * 
 * @param m: mesh
 * @return Euler characteristic
 */
int compute_euler_characteristic(const Mesh<Scalar>& m);

/**
 * @brief Compute the genus of the mesh
 * 
 * @param m: mesh
 * @return genus
 */
int compute_genus(const Mesh<Scalar>& m);

/**
 * @brief Compute the map from vertex-vertex edges to primal mesh halfedges
 * 
 * WARNING: 1-indexing is used for halfedges instead of the usual 0 indexing.
 * 
 * @param m mesh
 * @return vertex-vertex to halfedge map
 */
Eigen::SparseMatrix<int> compute_vv_to_halfedge_matrix(const Mesh<Scalar>& m);

} // namespace Holonomy
} // namespace Penner