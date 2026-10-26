#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/halfedge.h"

#include <nlohmann/json.hpp>

namespace Penner {
namespace Resolution {


Eigen::VectorXd symmetric_dirichlet(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT);

class MeshImprover
{
public:
    using json = nlohmann::json;

    json results;

    // skip steps of the mesh improvement
    bool skip_smooth = false;
    bool skip_flips = false;
    bool skip_split = false;
    bool skip_collapse = false;
    //double sym_dir_thres = 10.;
    bool show_smooth = false;

    // weights for uv local optimization
    double symdir_weight = 1e-3;
    double alignment_weight = 1;
    int norm_p = 1;

    // minimum area for regularizing symmetric Dirichlet energy
    double min_rel_area = 1e-5;

    // saved mesh area for later use
    double avg_area = 0;

    /**
     * @brief Initialize the Mesh Improver
     * 
     * @param V: mesh vertices
     * @param F: mesh faces
     * @param uv: mesh uv vertices
     * @param FT: mesh uv faces
     */
    MeshImprover(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& FT
    );

    /**
     * @brief Set the combed field to maintain during improvement
     * 
     * @param PD1: first combed field direction (u gradient target)
     * @param PD2: second combed field direction (v gradient target)
     */
    void set_combed_field(
        const Eigen::MatrixXd& PD1,
        const Eigen::MatrixXd& PD2
    );
    
    /**
     * @brief Get the combed field maintain during improvement
     * 
     * @return first combed field direction (u gradient target)
     * @return second combed field direction (v gradient target)
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
    get_combed_field() const;

    /**
     * @brief Run iterations of improvement.
     * 
     * The improvement alternates between steps of:
     *   - local per-vertex smoothing to improve energy
     *   - splitting long edges
     *   - flipping nearly flat edges
     *   - collapsing short edges
     * The latter two operations are only performed if they improve the triangle quality
     * in both the 3D and uv mesh
     * 
     * @param num_iter: number of iterations of improvement to perform
     */
    void improve(int num_iter);

    /**
     * @brief Create a VF mesh with parametrization for the improved mesh, transforming
     * faces so that the parametrization is a single connected component
     * 
     * @return improved mesh vertices
     * @return improved mesh faces 
     * @return improved mesh uv vertices
     * @return improved mesh uv faces
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi>
    get_improved_mesh();

    /**
     * @brief Get the uv coordinates of the mesh (assuming no connectivtity changes)
     * 
     * WARNING: only works if no mesh connectivity operations are performed
     *
     * @param uv: optimized uv vertices
     * @param FT: optimized uv faces
     */
    void get_uv(
        Eigen::MatrixXd& uv,
        Eigen::MatrixXi& FT
    ) const;

    /**
     * @brief Set the uv coordinates of the mesh (assuming no connectivtity changes)
     * 
     * WARNING: only works if no mesh connectivity operations are performed
     *
     * @param uv: uv vertices
     * @param FT: uv faces
     */
    void set_uv(
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& FT
    );

    /**
     * @brief Transform the face opposite the given halfedge to make the two paired
     * halfedges adjacent in the plane
     * 
     * Also updates field data if present.
     * 
     * @param hij: halfedge to join with pair
     */
    void transform_opposite_face(int hij);

    /**
     * @brief Transform the faces adjacent to the given vertex to make them adjacent in the plane
     * 
     * Also updates field data if present.
     * 
     * @param vertex_index: vertex to locally relayout
     */
    void transform_vertex_adjacent_faces(int vertex_index);

    /**
     * @brief Compute 3x3 gradient operator with respect to spatial coordinates
     * for functions defined on the vertices of the face.
     * 
     * WARNING: The indexing of the vertices is determined by the halfedge f2h and n operators
     * 
     * @param face_index: face index
     * @return gradient operator
     */
    Eigen::Matrix<double, 3, 3> compute_face_gradient(int face_index) const;

    /**
     * @brief Compute 2x3 gradient operator with respect to local tangent plane coordinates
     * for functions defined on the vertices of the face.
     * 
     * For small faces, the face is normalized to an equilateral triangle with target minimal area.
     * 
     * WARNING: The indexing of the vertices is determined by the halfedge f2h and n operators
     * 
     * @param face_index: face index
     * @return tangent gradient operator
     */
    Eigen::Matrix<double, 2, 3> compute_face_tangent_gradient(int face_index) const;

    /**
     * @brief Compute the area of the face with given index in the improved mesh
     * 
     * @param face_index: face index
     * @return face area 
     */
    double compute_face_area(int face_index) const;

    /**
     * @brief Compute the total surface area of the improved mesh
     * 
     * @return mesh area 
     */
    double compute_total_area() const;

    /**
     * @brief Compute the optimization energy associated with the given face.
     * 
     * This energy is a weighted comment of a symmetric Dirichlet energy and a field
     * alignment term
     * 
     * @param face_index: index of a face in the improved mesh
     * @return face energy
     */
    double compute_face_energy(int face_index) const;

    /**
     * @brief Compute the optimization energy associated with the faces adjacent to a vertex
     * 
     * This energy is a weighted comment of a symmetric Dirichlet energy and a field
     * alignment term
     * 
     * @param vertex_index: index of a vertex in the improved mesh
     * @return vertex energy
     */
    double compute_vertex_energy(int vertex_index) const;

    /**
     * @brief Compute the optimization energy for the current improved mesh
     * 
     * This energy is a weighted comment of a symmetric Dirichlet energy and a field
     * alignment term
     * 
     * @return total energy
     */
    double compute_energy() const;

    /**
     * @brief Compute the current energy at a vertex and a descent direction for the vertex
     * to improve the energy
     * 
     * This descent direction is a local Newton direction.
     * 
     * @param vertex_index: vertex to improve
     * @return energy at vertex
     * @return descent direction for the vertex
     */
    std::tuple<double, Eigen::Vector2d> vertex_energy_direction(int vertex_index) const;

    /**
     * @brief Optimize the energy at a vertex with a local backtracking line search.
     * 
     * @param vertex_index: vertex to improve
     */
    void local_energy_opt(int vertex_index);

    /**
     * @brief Flip a halfedge counterclockwise while preserving the parametrization and averaging
     * the field directions of the flipped faces, projecting to the tangent plane
     * 
     * @param halfedge_index: edge to flip
     */
    void flip_edge_ccw(int halfedge_index);

    /**
     * @brief Flip a halfedge clockwise while preserving the parametrization and averaging
     * the field directions of the flipped faces, projecting to the tangent plane
     * 
     * @param halfedge_index: edge to flip
     */
    void flip_edge_clockwise(int halfedge_index);

    /**
     * @brief Collapse the given halfedge, projecting field directions to tangents.
     * 
     * The operation is rejected if the the collapse reduces triangle quality or flips elements
     * 
     * @param halfedge_index: halfedge to collapse
     * @return true iff the collapse is performed succesfully
     */
    bool collapse_edge(int halfedge_index);

    /**
     * @brief Split the given halfedge at the midpoint, duplicating field data for split faces.
     * 
     * @param halfedge_index: halfedge to split
     */
    void split_edge(int halfedge_index);

    /**
     * @brief Get the uv vertices of the face
     * 
     * WARNING: The indexing of the vertices is determined by the halfedge f2h and n operators,
     * and is consistent with the gradient operators
     * 
     * @param face_index 
     * @return Eigen::Matrix<double, 3, 2> 
     */
    Eigen::Matrix<double, 3, 2> compute_face_uv(int face_index) const;

private:
    HalfedgeMesh halfedge;
    std::vector<int> vtx_reindex;
    std::vector<std::pair<int, int>> he_to_corner;
    std::vector<bool> is_cone;

    std::vector<Eigen::Vector2d> halfedge_uv;
    std::vector<Eigen::Vector3d> V_r;

    // cross field
    std::vector<Eigen::Vector3d> PD1_r;
    std::vector<Eigen::Vector3d> PD2_r;

    igl::Timer timer, method_timer;
    double avg_edge_length, abs_collapse_edge_thres;

    // experimental
    void smooth_vertex(int vertex_index);
};
    
}
}
