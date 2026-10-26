// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "holonomy/interface.h"
#include "holonomy/core/common.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"

#include "field/intrinsic_field.h"

namespace Penner {
namespace Resolution {

using Holonomy::DualLoop;
using Holonomy::NewtonParameters;
using Holonomy::MarkedPennerConeMetric;
using Holonomy::MarkedMetricParameters;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> GeometryData;
typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> ImageData;

// TODO: migrate these functions to lower level modules


/**
 * @brief Generate the map from VF vertex edge pairs to halfedges in m.
 * 
 * If m is a double cover mesh, this method only considers primal edges.
 * 
 * WARNING: Uses 1-indexing for halfedges instead of 0-indexing.
 * 
 * @param m: halfedge mesh
 * @param vtx_reindex: map from vertices in the halfedge to VF mesh
 * @return sparse matrix mapping vertex pairs to halfedges
 */
Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex);

/**
 * @brief Generates a halfedge function on a cover mesh from a function on the opposite corners
 * of a VF mesh.
 * 
 * @tparam function type
 * @param m: halfedge mesh
 * @param vtx_reindex: map from vertices in the halfedge to VF mesh
 * @param F: VF mesh faces
 * @param corner_func: function on the VF mesh corners
 * @return function on the halfedges
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> transfer_corner_function_to_halfedge(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& corner_func);


/**
 * @brief Compute measures of seamless metric failure in halfedge uv coordinates.
 * 
 * @param m: halfedge mesh
 * @param uv: coordinates of uv coordinates at the tip of each halfedge
 * @return per-halfedge uv edge length deviation for paired halfeges
 * @return per-halfedge uv edge rotation deviation from a multiple of pi/2 for paired halfeges
 */
std::tuple<VectorX, VectorX> compute_seamless_error(
    const Mesh<Scalar> &m,
    const Eigen::MatrixXd &uv);

/**
 * @brief Print an error message and return false
 * 
 * @param message: message to print to log
 * @return false
 */
inline bool errmess(const std::string& message)
{
    spdlog::error(message);
    assert(false);
    return false;
}

/**
 * @brief Get the next segment in an overlay mesh, ignoring current only segments
 * 
 * @param m_o: overlay mesh
 * @param _h: segment in overlay mesh
 * @return next original segment in the overlay mesh
 */
int next_orig_segment(const OverlayMesh<Scalar> &m_o, int _h);

/**
 * @brief Remove unreferenced vertices from a VF mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 */
void remove_unreferenced_inplace(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

Eigen::Vector3d project_orthogonal_to(
    const Eigen::Vector3d& v,
    const Eigen::Vector3d& u);

/**
 * @brief Compute the area of a triangle with the given vertices
 * 
 * @param v1: first vertex
 * @param v2: second vertex
 * @param v3: third vertex
 * @return area of triangle spanned by the vertices
 */
double compute_triangle_area(
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3);

/**
 * @brief Generate rotation for a pi/2 ccw rotation.
 * 
 * @return rotation matrix
 */
Eigen::Matrix2d generate_ccw_rotation();

} // namespace Resolution
} // namespace Penner
