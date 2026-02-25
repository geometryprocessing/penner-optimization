#pragma once

#include "holonomy/interface.h"
#include "holonomy/core/common.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/newton.h"

namespace Penner {
namespace Feature {

using Holonomy::DualLoop;
using Holonomy::MarkedPennerConeMetric;
using Holonomy::NewtonParameters;
using Holonomy::MarkedMetricParameters;
typedef std::array<int, 2> VertexEdge;

/**
 * @brief Compute the square of a scalar.
 *
 * @param x: value to square
 * @return squared value
 */
inline Scalar square(Scalar x)
{
    return x * x;
}

/**
 * @brief Compute the one ring of halfedges emenating from a mesh vertex.
 *
 * @param m: mesh
 * @param vertex_index: index of a vertex in the mesh
 * @return halfedges emenating from a given vertex
 */
std::vector<int> generate_vertex_one_ring(const Mesh<Scalar>& m, int vertex_index);

/**
 * @brief Determine if a VF mesh is manifold
 * @param F: mesh faces
 * @return true iff the mesh is edge and vertex manifold
 */
bool is_manifold(const Eigen::MatrixXi& F);

/**
 * @brief Reindex VF mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param vtx_reindex: map from old to new vertex indices
 * @return reindexed mesh vertices
 * @return reindexed mesh faces
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> reindex_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& vtx_reindex);

/**
 * @brief Reindex list of edge endpoints under vertex reindexing.
 * 
 * @param endpoints: list of vertex endpoints before reindexing
 * @param vtx_reindex: map from old to new vertex indices
 * @return list of reindexed vertex endpoints
 */
std::vector<std::pair<int, int>> reindex_endpoints(
    const std::vector<std::pair<int, int>>& endpoints,
    const std::vector<int>& vtx_reindex);


} // namespace Feature
} // namespace Penner
