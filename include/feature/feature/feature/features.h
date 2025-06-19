
#pragma once

#include "feature/core/common.h"

#include "feature/util/union_find.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace Penner {
namespace Feature {

class FeatureFinder
{
public:
    /**
     * @brief Construct a new Feature Finder object
     *
     * @param V: input mesh vertices
     * @param F: input mesh faces
     */
    FeatureFinder(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

    /**
     * @brief Get a list of the feature edges, represented as a list of edge endpoints
     * 
     * @return list of feature edges
     */
    std::vector<VertexEdge> get_features() const;

    /**
     * @brief Remove all feature markings
     */
    void reset_features();

    /**
     * @brief Mark edges in the VF mesh represented by edge endpoints as features.
     * 
     * @param feature_edges: list of edges to mark as features
     */
    void mark_features(const std::vector<VertexEdge>& feature_edges);

    /**
     * @brief Mark all edges with a dihedral angle larger than some threshold
     *
     * @param feature_angle: (optional) consider an edge a feature if its angle (in degrees) is
     * larger than this threshold
     */
    void mark_dihedral_angle_features(Scalar feature_angle = 30.);

    /**
     * @brief Remove features smaller than the given size
     *
     * @param feature_size: minimum feature size
     */
    void prune_small_features(int feature_size);

    /**
     * @brief Remove boundaries of feature components smaller than a given size
     *
     * @param min_component_size: minimum vertex count in a component
     */
    void prune_small_components(int min_component_size);

    /**
     * @brief Remove cycle edges to form a spanning forest of the feature graph.
     *
     */
    void prune_cycles();

    /**
     * @brief Greedily attempt to prune halfedges without producing isolated vertices.
     * 
     */
    void prune_greedy();

    /**
     * @brief Greedily attempt to prune halfedges without producing isolated vertices or
     * isolated feature boundary components.
     * 
     * @param cut_feature_components: sets of cut mesh component features
     */
    void prune_refined_greedy(UnionFind& cut_feature_components);

    /**
     * @brief Generate unions of feature halfedges in the same connected component.
     * 
     * @return sets of halfedges in connected features
     */
    UnionFind compute_feature_components() const;

    /**
     * @brief Generate vertices of the feature cut graph, which are represented as unions
     * of halfedges opposite triangle corners
     * 
     * @return sets of halfedges representing cut mesh vertices
     */
    UnionFind compute_vertices() const;

    /**
     * @brief Generate unions of faces in the same connected component of the feature cut mesh.
     * 
     * @return sets of face components
     */
    UnionFind compute_face_components() const;

    /**
     * @brief Generate unions of feature halfedges in the same connected component of the 
     * feature cut mesh.
     * 
     * @return sets of cut mesh component features
     */
    UnionFind compute_cut_feature_components() const;

    /**
     * @brief Compute degree of feature graph vertices.
     * 
     * @return feature graph degree of mesh vertex (0 for non-feature vertices)
     */
    std::vector<int> compute_feature_degrees() const;

    /**
     * @brief Cut the mesh along feature lines and produce a corresponding VF mesh.
     * 
     * TODO: Add option for soft constraints and return soft mask
     *
     * @return cut mesh vertices
     * @return cut mesh faces
     * @return map from cut mesh vertices to original vertex indices
     * @return face corner mask indicating if opposite halfedge is cut
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::VectorXi, Eigen::MatrixXi>
    generate_feature_cut_mesh() const;

    /**
     * @brief View the feature edge graph and the forest that pruning cycles would produce.
     *
     * @param show: (optional) if true, open polyscope viewer
     */
    void view_features(bool show = true);

    /**
     * @brief View the features in the parametric domain with given vertex positions.
     *
     * @param uv: vertex parametric positions
     * @param vtx_reindex: map from halfedge vertices to uv vertices
     */
    void view_parametric_features(const Eigen::MatrixXd& uv, const std::vector<int>& vtx_reindex)
        const;

    /**
     * @brief Determine if a halfedge is a feature.
     * 
     * @param halfedge_index: mesh halfedge index
     * @return true iff the halfedge is a feature
     */
    bool is_feature_halfedge(int halfedge_index) const
    {
        return m_is_feature_halfedge[halfedge_index];
    }

    /**
     * @brief Mark edge containing the given halfedge as a feature
     * 
     * @param halfedge_index: mesh halfedge index to mark
     * @param is_feature: (optional) if false, set to not a feature
     */
    void set_feature_edge(int halfedge_index, bool is_feature=true);

    /**
     * @brief Compute a spanning forest of the feature halfedge graph.
     * 
     * @return mask for forest halfedges
     */
    std::vector<bool> compute_feature_forest_halfedges() const;

    // getters
    const Mesh<Scalar>& get_mesh() const { return m_mesh; }
    const Eigen::MatrixXd& get_vertex_positions() const { return m_vertex_positions; }
    const std::vector<int>& get_vertex_reindex() const { return m_vtx_reindex; }
    const std::vector<int>& get_vertex_reindex_inverse() const { return m_vtx_reindex_inverse; }
    const std::vector<bool>& get_feature_halfedges() const { return m_is_feature_halfedge; }
    std::vector<bool>& get_feature_halfedges() { return m_is_feature_halfedge; }
    const Eigen::MatrixXi& get_faces() const { return m_faces; }

private:
    // VF mesh
    Eigen::MatrixXd m_vertex_positions;
    Eigen::MatrixXi m_faces;

    // Halfedge mesh
    Mesh<Scalar> m_mesh;
    std::vector<int> m_vtx_reindex;
    std::vector<int> m_vtx_reindex_inverse;

    // Additional geometry
    Eigen::MatrixXd m_face_normals;

    // Feature markings
    std::vector<bool> m_is_feature_halfedge;

    // feature angle utility 
    Scalar compute_signed_dihedral_angle(int halfedge_index) const;
    Scalar compute_dihedral_angle(int halfedge_index) const;

    // utility for pruning cycles
    std::vector<int> compute_feature_forest() const;

    // forest weighting schemes
    std::vector<Scalar> compute_forest_weights() const;
    std::vector<Scalar> compute_dihedral_weights() const;
    std::vector<Scalar> compute_interior_biased_weights() const;
    std::vector<Scalar> compute_random_weights() const;
};


} // namespace Feature
} // namespace Penner