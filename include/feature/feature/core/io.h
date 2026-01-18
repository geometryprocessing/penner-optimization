#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Load feature edges from a 0-indexed list in the following format:
 * <edge count>
 * ...
 * <e00> <e01>
 * <e10> <101>
 * ...
 * 
 * @param fe_filename: filepath to edge list
 * @return list of edges
 */
std::vector<VertexEdge> load_feature_edges(const std::string& fe_filename);

/**
 * @brief Load edges from an obj file
 * 
 * @param fe_filename: filepath to obj mesh
 * @return list of edges
 */
std::vector<VertexEdge> load_mesh_edges(const std::string& fe_filename);

/**
 * @brief Write parameterization seams (that are not feature edges) to file
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_uv: mesh parameterization faces
 * @param F_is_feature: |F|x3 mask of corners opposite features
 */
void write_seams(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_feature);

/**
 * @brief Write feature edges to file
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_is_feature: |F|x3 mask of corners opposite features
 */
void write_features(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_is_feature);

/**
 * @brief Write parameterization boundary (both seams and features) to file
 * 
 * @param filename: output edge geometry file
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param F_uv: mesh parameterization faces
 */
void write_boundary(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv);

void write_edges(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E);

} // namespace Feature
} // namespace Penner
