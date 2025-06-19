#pragma once

#include "feature/core/common.h"

// TODO: Cleaning pass

namespace Penner {
namespace Feature {

/**
 * @brief Compute a halfedge function for a viewer from per-corner data
 * 
 * The indexing scheme is (f, i) -> h = 3 * f + ((i + 1)%3)
 * 
 * @param corner_values: values indexee by opposite face corners
 * @return: values indexed by halfedge indices
 */
template <typename Type>
Eigen::Matrix<Type, Eigen::Dynamic, 1>
    compute_polyscope_halfedge_from_corner_function(const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& corner_values
) {
    int num_faces = corner_values.rows();
    Eigen::Matrix<Type, Eigen::Dynamic, 1> halfedge_values(3 * num_faces);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int h = 3 * fijk + j;
            halfedge_values[h] = corner_values(fijk, i);
        }
    }

    return halfedge_values;
}

/**
 * @brief Generate glued sum of sector cones for a cut mesh with geometry and angle defect.
 * 
 * @param V: glued mesh vertices
 * @param m: cut mesh with sector cones
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @return cone positions
 * @return cone angle defects
 */
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_glued_cone_vertices(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

/**
 * @brief View the glued sum of sector cones on a cut mesh.
 * 
 * @param m: cut mesh with sector cones
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @param V: glued mesh vertices
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_glued_mesh_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXd& V,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View the alignment of the given edges of a mesh with the uv isolines
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: uv vertices
 * @param F_uv: uv faces
 * @param F_is_aligned: face corner mask indicating if the opposite edge should be aligned
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_uv_alignment(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_aligned,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View a feature edge curve network.
 * 
 * @param V: feature edge vertices
 * @param E: feature edges
 * @param mesh_handle: (optional) name for edge mesh
 * @param show: (optional) if true, open viewer
 */
void view_feature_edges(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View a cross field on a mesh.
 * 
 * TODO: Move to holonomy
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @param kappa: per-corner rotation angle of the reference direction field across the opposite edge
 * @param period_jump: per-corner period jump of the cross field across the opposite edge
 * @param mesh_handle: (optional) name for edge mesh
 */
void view_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    std::string mesh_handle = "intrinsic_field_mesh");

/**
 * @brief View a quad mesh and its irregular vertices.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_quad_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View the cut on a VF mesh.
 * 
 * @param V_cut: cut mesh vertices
 * @param F_cut: cut mesh faces
 * @param F_is_cut: corner mask for cut edges opposite corners
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_cut(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    const Eigen::MatrixXi& F_is_cut,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View the endpoints of an overlay mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param endpoints: map from vertices to endpoints (or (-1, -1) for orginal vertices)
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_endpoints(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& endpoints,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View a direction field on a mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param direction_field: per-face reference tangent direction matrix
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_direction_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& direction_field,
    std::string mesh_handle="direction_field_mesh",
    bool show=true);

/**
 * @brief View principal curvature values and directions on a mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param radius: vertex radius used for smooth surface approximation for curvature
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_principal_curvature(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius=5,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View a direction field on a mesh with salient direction mask.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param direction: per-face reference tangent direction matrix
 * @param is_fixed_direction: per-face mask for salient directions
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_fixed_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& direction,
    const std::vector<bool>& is_fixed_direction,
    std::string mesh_handle="",
    bool show=true);

/**
 * @brief View the best fit per-vertex conformal scaling to deform the embedding metric
 * to the parameterization metric.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: uv mesh vertices
 * @param F_uv: uv mesh faces
 * @param mesh_handle: (optional) name for surface mesh
 * @param show: (optional) if true, open viewer
 */
void view_conformal_scaling(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    std::string mesh_handle="",
    bool show=true);

} // namespace Feature
} // namespace Penner