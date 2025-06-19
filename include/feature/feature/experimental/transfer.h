
#pragma once

#include "feature/core/common.h"
 
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/interface.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "feature/util/union_find.h"

namespace Penner {
namespace Feature {

/**
 * @brief Transfer edge data stored per halfedge from a domain mesh to a target mesh,
 * both obtained from a common base mesh by cutting edges.
 * 
 * @param marked_metric_0: target mesh with metric
 * @param vtx_reindex_0: VV to halfedge indexing for target mesh
 * @param V_map_0: vertex identification for target mesh
 * @param marked_metric: domain mesh with metric
 * @param vtx_reindex: VV to halfedge indexing for domain mesh
 * @param V_map: vertex identification for domain mesh
 * @param halfedge_data: data stored on domain mesh
 * @param is_signed: if true, treat halfedge data as having opposite signs for opposite halfedges
 * @return halfedge data transfered to target mesh
 */
VectorX transfer_edge_data(
    const Mesh<Scalar>& marked_metric_0,
    const std::vector<int>& vtx_reindex_0,
    const Eigen::VectorXi& V_map_0,
    const Mesh<Scalar>& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const VectorX& halfedge_data,
    bool is_signed=false);

/**
 * @brief Transfer metric coordinates from a domain mesh to a target mesh.
 * 
 * @param marked_metric_0: target mesh with metric
 * @param vtx_reindex_0: VV to halfedge indexing for target mesh
 * @param V_map_0: vertex identification for target mesh
 * @param marked_metric: domain mesh with metric
 * @param vtx_reindex: VV to halfedge indexing for domain mesh
 * @param V_map: vertex identification for domain mesh
 */
void transfer_metric(
    MarkedPennerConeMetric& marked_metric_0,
    const std::vector<int>& vtx_reindex_0,
    const Eigen::VectorXi& V_map_0,
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

/**
 * @brief Transfer rotation form from a domain mesh to a target mesh.
 * 
 * @param marked_metric_0: target mesh with metric
 * @param vtx_reindex_0: VV to halfedge indexing for target mesh
 * @param V_map_0: vertex identification for target mesh
 * @param marked_metric: domain mesh with metric
 * @param vtx_reindex: VV to halfedge indexing for domain mesh
 * @param V_map: vertex identification for domain mesh
 * @param rotation_form: rotation form on domain mesh
 * @return rotation form transfered to target mesh
 */
VectorX transfer_rotation_form(
    const Mesh<Scalar>& m_0,
    const std::vector<int>& vtx_reindex_0,
    const Eigen::VectorXi& V_map_0,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const VectorX& rotation_form);


} // namespace Feature
} // namespace Penner