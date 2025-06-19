
#pragma once

#include "feature/core/common.h"
 
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "feature/util/union_find.h"

namespace Penner {
namespace Feature {


/**
 * @brief Count the total number of vertices in set of mesh components.
 * 
 * @param mesh_vertex_components: list of vertex position matrices for components
 * @return total number of vertices
 */
int count_total_vertices(std::vector<Eigen::MatrixXd>& mesh_vertex_components);

/**
 * @brief Count the total number of faces in set of mesh components.
 * 
 * @param mesh_face_components: list of face matrices for components
 * @return total number of faces
 */
int count_total_faces(std::vector<Eigen::MatrixXi>& mesh_face_components);

/**
 * @brief Given a vector of mesh components, combine them into a single mesh
 * 
 * @param meshes: mesh components to union
 * @return union of meshes
 */
Mesh<Scalar> union_meshes(const std::vector<Mesh<Scalar>>& meshes);

/**
 * @brief Given a vector of marked metric components, combine them into a single mesh
 * 
 * @param marked_metrics: marked metric components to union
 * @return union of metrics 
 */
MarkedPennerConeMetric union_marked_metrics(const std::vector<MarkedPennerConeMetric>& marked_metrics);

} // namespace Feature
} // namespace Penner