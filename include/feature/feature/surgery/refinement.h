#pragma once

#include "feature/core/boundary_path.h"
#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "feature/feature/features.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace Penner {
namespace Feature {

class IntrinsicRefinementMesh
{
public:
    /**
     * @brief Construct a new Intrinsic Refinement Mesh object
     *
     * @param m: initial mesh for refinement
     *
     */
    IntrinsicRefinementMesh(const Mesh<Scalar>& m);

    virtual ~IntrinsicRefinementMesh() = default;

    /**
     * @brief Refine a given face by adding a new midpoint connected to all vertices.
     *
     * @param face_index: index of the face to refine
     * @return index of the new face midpoint
     */
    int refine_face(int face_index);

    /**
     * @brief Refined a given halfedge (and its pair) by adding a midpoint.
     *
     * @param halfedge_index: halfedge index of the edge to refine
     * @return index of the new edge midpoint
     */
    int refine_halfedge(int halfedge_index);

    /**
     * @brief Refine all faces with all vertices on the boundary.
     *
     */
    void refine_spanning_faces();

    /**
     * @brief Generate an embedding of the refined mesh from the original mesh embedding.
     *
     * Use average midpoint for refined faces and edges.
     *
     * WARNING: Currently assumes that refinement is simple, meaning only edges or faces
     * are refined (not both), and no edge or face is refined multiple times.
     *
     * @param V: original mesh vertices
     * @param vtx_reindex: reindexing of the halfedge mesh vertices
     * @return refined mesh vertices
     * @return refined mesh faces
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_mesh(
        const Eigen::MatrixXd& V,
        const std::vector<int>& vtx_reindex) const;

    /**
     * @brief View the refined mesh
     *
     * @param V: original mesh vertices
     * @param vtx_reindex: reindexing of the halfedge mesh vertices
     */
    void view_refined_mesh(const Eigen::MatrixXd& V, const std::vector<int>& vtx_reindex) const;

    /**
     * @brief Get the mesh object
     *
     * @return reference to the mesh
     */
    Mesh<Scalar>& get_mesh() { return m_mesh; }
    const Mesh<Scalar>& get_mesh() const { return m_mesh; }

protected:
    int get_new_face();
    int get_new_vertex();
    std::pair<int, int> get_new_edge();
    int get_new_independent_vertex(int vi);
    int get_new_independent_vertex(int vi, int Rvi);

    Mesh<Scalar> m_mesh;
    std::map<int, std::pair<int, int>> m_endpoints;

private:
    virtual int refine_single_face(int face_index);
    virtual int refine_single_halfedge(int halfedge_index);
};

/**
 * @brief Refine all faces in a mesh with features that have more than one tagged feature edge.
 *
 * @param feature_finder: mesh with tagged features
 * @return refined vertex positions
 * @return refined face indices
 * @return refined feature edges
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<VertexEdge>> refine_corner_feature_faces(
    const FeatureFinder& feature_finder);

/**
 * @brief Refine halfedge to ensure all cut mesh feature components have one feature in the
 * feature forest (either spanning or minimal forest).
 * 
 * @param feature_finder: mesh with tagged features
 * @param use_minimal_forest: (optional) if true, use minimal feature forest instead of spanning tree
 * @return refined vertex positions
 * @return refined face indices
 * @return refined feature edges
 * @return refined feature forest edges (spanning or minimal)
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<VertexEdge>, std::vector<VertexEdge>>
refine_feature_components(const FeatureFinder& feature_finder, bool use_minimal_forest = false);

} // namespace Feature
} // namespace Penner