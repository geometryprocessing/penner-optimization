#pragma once

#include "resolution/core/common.h"
#include "field/forms.h"

namespace Penner {
namespace Resolution {

/**
 * @brief Set the rotation across a halfedge, maintaining edge consistency.
 * 
 * @param m: underlying mesh
 * @param rotation_form: rotation of cross field across halfdges
 * @param halfedge: halfedge to update
 * @param kappa: new rotation across halfedge
 */
void set_rotation_form(const Mesh<Scalar>& m, VectorX& rotation_form, int halfedge_index, Scalar kappa);

/**
 * @brief Align the field in the faces adjacent to the halfedge, making the rotation
 * across the edge trivial.
 * 
 * The average angle is used for the aligned faces.
 * 
 * @param m: underlying mesh
 * @param rotation_form: rotation of cross field across halfdges
 * @param halfedge_index: halfedge of the rotation to modify.
 */
void align_rotation_form_faces(const Mesh<Scalar>& m, VectorX& rotation_form, int halfedge_index);

class IntrinsicSimplificationMesh
{
public:
    bool use_random_seed = false;
    bool maintain_simplicial = true;

    /**
     * @brief Construct a new Intrinsic Simplification Mesh object
     *
     * @param m: initial mesh for simplification
     */
    IntrinsicSimplificationMesh(const Mesh<Scalar>& m);

    void set_rotation_form(const VectorX& rotation_form);

    void set_dual_loop(const std::vector<std::unique_ptr<DualLoop>>& dual_loops);

    /**
     * @brief Simplify the mesh by removing as many flat vertices as possible.
     *
     */
    void simplify_mesh();

    /**
     * @brief Simplify the mesh by collapsing as many edges as possible.
     *
     */
    void collapse_edges();

    bool collapse_edge(int edge_index);

    /**
     * @brief Compute the degree of a given vertex
     *
     * @param vertex_index: index of a vertex
     * @return degree of the vertex
     */
    int compute_degree(int vertex_index) const;

    /**
     * @brief Remove a given vertex if possible
     *
     * @param vertex_index: index of the vertex to remove
     * @return true iff the vertex is successfully removed
     */
    bool remove_vertex(int vertex_index);

    /**
     * @brief Get the mesh object
     *
     * @return reference to the mesh
     */
    Mesh<Scalar>& get_mesh() { return m_mesh; }
    const Mesh<Scalar>& get_mesh() const { return m_mesh; }

    const std::vector<int>& get_e2h() const { return m_e2h; }
    const std::vector<int>& get_h2e() const { return m_h2e; }

    /**
     * @brief Get the holonomy coordinates vector.
     * 
     * @return vector of per corner holonomy angle coordinates
     */
    VectorX& get_holonomy_coordinates() { return m_alpha; }
    const VectorX& get_holonomy_coordinates() const { return m_alpha; }
    VectorX get_opposite_angles() const;

    /**
     * @brief Generate the simplified mesh object.
     * 
     * @return mesh reindexed to remove elements 
     */
    Mesh<Scalar> get_simplified_mesh() const;

    /**
     * @brief Get the simplified marked metric object with consistent holonomy.
     * 
     * Currently uses initial 0 coordinates for the metric.
     * 
     * @return metric for mesh with removed elements
     */
    MarkedPennerConeMetric get_simplified_marked_metric(
        const MarkedMetricParameters& marked_metric_params
    ) const;

    /**
     * @brief Get the rotation of the field across a given halfedge
     * 
     * @param halfedge_index: index of the halfedge to query
     */
    Scalar get_rotation(int halfedge_index) const
    {
        return m_rotation_form[halfedge_index];
    }
    VectorX& get_rotation_form() { return m_rotation_form; }
    const VectorX& get_rotation_form() const { return m_rotation_form; }
    int vertex_parent(int vertex_index) const { return m_vertex_parent[vertex_index]; }

    /**
     * @brief Set the rotation across a given halfedge, maintaining the invariant
     * that the rotation of opposite halfedges sum to 0.
     * 
     * @param halfedge_index: index of the halfedge to modify
     * @param kappa: new rotation value 
     */
    void set_rotation_form(int halfedge_index, Scalar kappa)
    {
        m_rotation_form[halfedge_index] = kappa;
        m_rotation_form[m_mesh.opp[halfedge_index]] = -kappa;
    }

    /**
     * @brief Align the field in the faces adjacent to the halfedge, making the rotation
     * across the edge trivial.
     * 
     * The average angle is used for the aligned faces.
     * 
     * @param halfedge_index: halfedge of the rotation to modify.
     */
    void align_faces(int halfedge_index);

    /**
     * @brief Generate an embedding of the simplified mesh from the original mesh embedding.
     *
     * @param V: original mesh vertices
     * @param vtx_reindex: reindexing of the halfedge mesh vertices
     * @return simplified mesh vertices
     * @return simplified mesh faces
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_mesh(
        const Eigen::MatrixXd& V,
        const std::vector<int>& vtx_reindex) const;

    /**
     * @brief View the simplified mesh
     *
     * @param V: original mesh vertices
     * @param vtx_reindex: reindexing of the halfedge mesh vertices
     */
    void view_simplified_mesh(const Eigen::MatrixXd& V, const std::vector<int>& vtx_reindex) const;

    std::vector<std::vector<Holonomy::DualSegment>>& get_dual_loops() { return m_dual_loops; }
    const std::vector<std::vector<Holonomy::DualSegment>>& get_dual_loops() const { return m_dual_loops; }

    /**
     * @brief Determine whether the holonomy signature of the homology basis and cone angles are consistent
     * with the simplified rotation form and holonomy coordinates.
     * 
     * @return true iff the rotation form is consistent;
     */
    bool is_rotation_consistent() const;

    void get_dual_loops(
        std::vector<std::unique_ptr<DualLoop>>& dual_loops) const;

    VectorX get_simplified_rotation_form() const;

    VectorX get_simplified_holonomy_coordinates() const;

    Scalar simplification_factor;
    void garbage_collection();

private:
    Mesh<Scalar> m_mesh;
    std::vector<bool> m_is_halfedge_deleted;
    std::vector<bool> m_is_vertex_deleted;
    std::vector<bool> m_is_face_deleted;
    std::vector<bool> m_is_edge_deleted;
    std::vector<int> m_vertex_parent;
    std::vector<int> m_e2h;
    std::vector<int> m_h2e;
    VectorX m_rotation_form;
    VectorX m_alpha;
    VectorX m_kappa;

    std::vector<int> prev;

    std::vector<std::vector<Holonomy::DualSegment>> m_dual_loops;
    int removed_vert_count;
    int initial_vertices;


    // topology modification with success checks
    bool increase_degree(int vertex_index);
    bool decrease_degree(int vertex_index);
    bool flip_to_degree(int vertex_index, int target_degree);
    bool remove_degree_3_vertex(int vertex_index);
    bool splice_halfedges(int h, int g);
    bool join_halfedge_endpoints(int halfedge_index);
    bool join_edge_faces(int edge_index);
    void flip_ccw(int hij);
    void rotate_edge(int halfedge_index, Scalar theta);

    void delete_halfedge(int halfedge_index);
    void delete_edge(int edge_index);
    void delete_vertex(int vertex_index);
    void delete_face(int face_index);

    int create_face();
    int create_vertex();

    // predicates
    bool will_increase_degree(int hij) const;
    bool will_decrease_degree(int hij) const;
    bool is_collapse_possible(int edge_index) const;
    bool is_flat_vertex(int vertex_index) const;
    bool is_degree_1_vertex(int vertex_index) const;
    bool is_degree_3_vertex(int vertex_index) const;
    bool is_flippable_edge(int halfedge_index) const;
    bool is_deleted_vertex(int vertex_index) const { return m_is_vertex_deleted[vertex_index]; }
    bool is_deleted_face(int face_index) const { return m_is_face_deleted[face_index]; }
    bool is_deleted_halfedge(int halfedge_index) const { return m_is_halfedge_deleted[halfedge_index]; }
};

} // namespace Resolution
} // namespace Penner
