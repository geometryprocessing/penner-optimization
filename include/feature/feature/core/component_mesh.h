
#pragma once

#include "feature/core/common.h"

#include "feature/core/union_meshes.h"
#include "feature/util/union_find.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace Penner {
namespace Feature {

/**
 * @brief Representation for a mesh with multiple components. This structure
 * decomposes the mesh into individual component meshes while saving the maps from
 * the component mesh faces and vertices to the global mesh. Given overlay parameterizations
 * for the original components, a global overlay parameterization can be constructed.
 *
 */
class ComponentMesh
{
public:
    /**
     * @brief Construct a collection of components for the global (possibly disconnected) mesh
     *
     * @param m: (possibly disconnected) manifold mesh
     */
    ComponentMesh(const Mesh<Scalar>& m);

    ComponentMesh(const Mesh<Scalar>& m, const Eigen::VectorXi& components);

    /**
     * @brief Get the individual connected components
     *
     * @return list of component meshes
     */
    const std::vector<Mesh<Scalar>>& get_mesh_components() const {return m_mesh_components;}

    /**
     * @brief Get the maps from component halfedges to global halfedges
     *
     * @return list of halfedge maps
     */
    const std::vector<std::vector<int>>& get_halfedge_maps() const {return m_he_maps;}

    const std::vector<std::vector<int>>& get_vertex_maps() const {return m_v_maps;}

    const std::vector<std::vector<int>>& get_halfedge_projections() const {return m_he_proj;}

    /**
     * @brief Combine a list of overlay meshes into a single mesh with the individual meshes as
     * components.
     *
     * No attempt at combining components geometrically is made; only concatenation and reindexing
     * is done
     *
     * @param face_reindex:
     * @param mesh_vertex_components: list of mesh vertex positions
     * @param mesh_face_components: list of mesh face indices
     * @param fn_to_f_components: list of mesh refined to original face maps
     * @param endpoint_components: list of vertex endpoint maps
     * @return combined mesh vertices
     * @return combined mesh faces
     * @return combined mesh refined to original face maps
     * @return combined vertex endpoints
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<int>, std::vector<std::pair<int, int>>>
    combine_refined_components(
        const std::vector<int>& face_reindex,
        std::vector<Eigen::MatrixXd>& mesh_vertex_components,
        std::vector<Eigen::MatrixXi>& mesh_face_components,
        std::vector<std::vector<int>>& fn_to_f_components,
        std::vector<std::vector<std::pair<int, int>>>& endpoint_components) const;

private:
    std::vector<Mesh<Scalar>> m_mesh_components;
    std::vector<std::vector<int>> m_he_maps;
    std::vector<std::vector<int>> m_v_maps;
    std::vector<std::vector<int>> m_f_maps;
    std::vector<std::vector<int>> m_he_proj;
    std::vector<std::vector<int>> m_v_proj;
    std::vector<std::vector<int>> m_f_proj;
    int m_num_vertices;

    void build_mesh_component(
        const Mesh<Scalar>& m,
        const Eigen::VectorXi& components,
        int component_index);
    void build_halfedge_data(
        Mesh<Scalar>& component_mesh,
        const Mesh<Scalar>& m,
        const std::vector<bool>& he_in_component,
        const std::vector<int>& component_he,
        const std::vector<int>& he_map,
        const std::vector<int>& v_map,
        const std::vector<int>& f_map);
    void build_face_data(
        Mesh<Scalar>& component_mesh,
        const Mesh<Scalar>& m,
        const std::vector<int>& component_f,
        const std::vector<int>& he_map);
    void build_vertex_data(
        Mesh<Scalar>& component_mesh,
        const Mesh<Scalar>& m,
        const std::vector<int>& component_v,
        const std::vector<int>& he_map,
        const std::vector<int>& ind_v_map);
    void build_independent_vertex_data(
        Mesh<Scalar>& component_mesh,
        const Mesh<Scalar>& m,
        const std::vector<int>& component_ind_v);
    bool is_valid_component_mesh() const;
};

/**
 * @brief Compute component indices for all mesh faces
 *
 * @param m: underlying mesh
 * @return face component labels
 */
Eigen::VectorXi find_mesh_face_components(const Mesh<Scalar>& m);

/**
 * @brief Compute component indices for all mesh vertices
 *
 * @param m: underlying mesh
 * @return vertex component labels
 */
Eigen::VectorXi find_mesh_vertex_components(const Mesh<Scalar>& m);

/**
 * @brief Compute the maximum values per component of a given vertex function
 *
 * @param m: underlying mesh
 * @param vertex_component: vector of vertex component ids
 * @param v: per vertex function
 * @return maximum vertex function value per component
 */
VectorX compute_vertex_component_max(
    const Mesh<Scalar>& m,
    const Eigen::VectorXi& vertex_component,
    const VectorX& v);

/**
 * @brief Given a mesh with component face labels, build a given component
 *
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param components: list of mesh face indices
 * @param component_index: index of the component to extract
 * @return component mesh vertices
 * @return component mesh faces
 * @return map from component mesh faces to original faces
 * @return map from component mesh vertices to original vertices
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<int>, Eigen::VectorXi> build_component(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& components,
    int component_index);

/**
 * @brief Combine a list of meshes into a single mesh with the individual meshes as components. 
 * 
 * No attempt at combining components geometrically is made.
 * 
 * @param mesh_vertex_components: list of mesh vertex positions
 * @param mesh_face_components: list of mesh face indices
 * @return combined mesh vertices
 * @return combined mesh faces
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> combine_mesh_components(
    std::vector<Eigen::MatrixXd>& mesh_vertex_components,
    std::vector<Eigen::MatrixXi>& mesh_face_components
);

/**
 * @brief Add a cone pair to a given component of a cut mesh.
 * 
 * WARNING: Experimental method; not currently in use
 * 
 * @param m: mesh
 * @param vertex_component: vertex component labels
 * @param component: target component
 * @param only_interior: (optional) only modify interior vertices if true
 */
void add_component_cone_pair(
    Mesh<Scalar>& m,
    Eigen::VectorXi vertex_component,
    int component,
    bool only_interior = true);


} // namespace Feature
} // namespace Penner