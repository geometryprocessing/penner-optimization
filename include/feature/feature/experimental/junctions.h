
#pragma once

#include "feature/core/common.h"
#include "feature/feature/features.h"

namespace Penner {
namespace Feature {

class JunctionFeatureFinder : public FeatureFinder
{
public:
    JunctionFeatureFinder(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

    /**
     * @brief Remove feature edges adjacent to junctions of 3 or more features.
     *
     */
    void prune_junctions();

    /**
     * @brief Remove feature edges adjacent to a given vertex
     *
     */
    void prune_junction(int vertex_index);

    /**
     * @brief Remove feature edges adjacent to the junction closest to the given vertex
     *
     * @param vertex_index: vertex in the mesh (could be junction or not)
     */
    void prune_closest_junction(int vertex_index);

    /**
     * @brief Remove an edge from a closed loop feature
     *
     */
    void prune_closed_loops();

    /**
     * @brief View the junctions of the feature graph.
     *
     * @param show: (optional) if true, open polyscope viewer
     */
    void view_junctions(bool show = true);

    /**
     * @brief Get junctions of the feature graph
     * 
     * @return list of junction vertices
     */
    std::vector<int> compute_junctions() const;
private:
    std::tuple<std::vector<int>, std::vector<bool>> compute_feature_forest(
        const std::vector<int>& flood_seeds) const;

};

/**
 * @brief Remove junctions in a feature finder according to a thresholded component
 * function heuristc, which relaxes components with a function value above some threshold.
 *
 * Note that we generally assume the cut mesh edges are the feature edges, but the method
 * is robust to different cut structures as long as the original uncut mesh is the same.
 *
 * @param feature_finder: features to modify
 * @param m: cut mesh
 * @param vtx_reindex: cut mesh to cut VF reindexing
 * @param V_map: map from cut VF vertices to original VF vertices
 * @param vertex_component: map from cut mesh vertices to component id
 * @param component_function: function on cut mesh components
 * @param relative_threshold: (optional) threshold for relaxing a component
 *     (relative to maximum of the component function)
 * @param do_remove_all_junctions: (optional) if true, remove all junctions in a component
 */
void remove_threshold_junctions(
    JunctionFeatureFinder& feature_finder,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::VectorXi& vertex_component,
    const VectorX& component_function,
    Scalar relative_threshold = 0.1,
    bool do_remove_all_junctions = true);

} // namespace Feature
} // namespace Penner