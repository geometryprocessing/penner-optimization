
#pragma once

#include "holonomy/core/common.h"
#include "field/intrinsic_field.h"

/**
 * @brief TODO: Redistribute this code
 * 
 */

namespace Penner {
namespace Holonomy {

/**
 * @brief Generate the reference tangent direction of a face along the oriented edge opposite
 * the corner with local index {0, 1, 2}. 
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param fijk: face index
 * @param local_index: local index of the corner opposite the reference direction
 * @return tangent direction along the edge opposite the corner
 */
Eigen::Vector3d generate_reference_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int fijk,
    int local_index);

/**
 * @brief Generate a field of per-face reference directions.
 * 
 * The chosen direction for a face fijk is the directed edge eki.
 * 
 * @param V: mesh vertices 
 * @param F: mesh faces
 * @return per-face tangent direction matrix
 */
Eigen::MatrixXd generate_reference_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

/**
 * @brief Generate a field of per-face reference directions determined by opposite reference corners.
 * 
 * @param V: mesh vertices 
 * @param F: mesh faces
 * @param reference_corner: per-face local index of corners opposite the reference direciton
 * @return per-face tangent direction matrix
 */
Eigen::MatrixXd generate_reference_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& reference_corner);

/**
 * @brief Given a representative cross field direction defined by a reference direction and an
 * offset angle, generate the representative direction matrix
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @return per-face representative direction matrix
 */
Eigen::MatrixXd generate_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Compute salient geometry aligned field directions for a mesh.
 * 
 * The parabolic anisotropy used for the relative threshold is ||k2| - |k1|| / max(|k1|, |k2|)
 * This measurement is near 0 for parabolic regions and near 1 for highly anisotropic regions
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param radius: (optional) vertex radius for fitting a smooth surface for field estimation
 * @param abs_threshold: (optional) minimum threshold for mean anisotropy of principal curvatures
 * @param rel_threshold: (optional) minimum threshold for parabolic anisotropy of principal curvatures
 * @return |F|x3 matrix of per face directions
 * @return per face mask indicating whether a direction is salient or not
 */
std::tuple<Eigen::MatrixXd, std::vector<bool>> compute_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius=5,
    Scalar abs_threshold=1.,
    Scalar rel_threshold=0.9,
    Scalar sample_rate=1.);
    
/**
 * @brief Optimize a cross field on a mesh.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param field_params: parameters for the field optimization
 * @return field reference directions in R3
 * @return field angles relative to the reference directions
 * @return angles across edges between reference directions
 * @return jump in period across edges
 */
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi>
generate_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const FieldParameters& field_params);

/**
 * @brief Write a frame field to file.
 * 
 * The format is
 * ```
 * <num_faces>
 * <dx> <dy> <dz> <theta> <k0> <k1> <k2> <p0> <p1> <p2>
 * ...
 * ```
 * where d is the reference field direction, theta is the offset angle, k is the rotation of the
 * reference directions across the edge opposite corner i, and p is corresponding the period jump.
 * 
 * @param output_filename: file location to serialize the frame field
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 * @param kappa: per-corner rotation angle of the reference direction field across the opposite edge
 * @param period_jump: per-corner period jump of the cross field across the opposite edge
 */
void write_frame_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump);

/**
 * @brief Write a rosy field to file.
 * 
 * The format is
 * ```
 * <num_faces>
 * 4
 * <dx> <dy> <dz>
 * ...
 * ```
 * where d is a representative direction on the face.
 * 
 * @param output_filename: file location to serialize the frame field
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param reference_field: per-face reference tangent direction matrix
 * @param theta: offset angles of a representative cross field direction relative to the reference
 */
void write_rosy_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta);

/**
 * @brief Write a combed frame field to file.
 * 
 * <d1x> <d1y> <d1z> <d2x> <d2y> <d2z>
 * ...
 * ```
 * where d1 and d2 are the two combed field direction
 * 
 * @param output_filename: file location to serialize the frame field
 * @param PD1: per-face u direction matrix
 * @param PD2: per-face b direction matrix
 */
void write_combed_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2);
    
std::tuple<int, Eigen::MatrixXd, Eigen::MatrixXd>
maximize_combed_frame_alignment(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
comb_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& thetas,
    const Eigen::MatrixXi& period_jumps);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
load_combed_field(const std::string& ffield_file);

/**
 * @brief Load a rosy field from file.
 * 
 * The format is
 * ```
 * <num_faces>
 * 4
 * <dx> <dy> <dz>
 * ...
 * ```
 * where d is a representative direction on the face.
 * 
 * @param input_filename: file location of the rosy field
 * @return per-face representative field direction
 */
Eigen::MatrixXd load_rosy_field(const std::string& input_filename);

/**
 * @brief Load a frame field from file.
 * 
 * The format is
 * ```
 * <num_faces>
 * <dx> <dy> <dz> <theta> <k0> <k1> <k2> <p0> <p1> <p2>
 * ...
 * ```
 * where d is the reference field direction, theta is the offset angle, k is the rotation of the
 * reference directions across the edge opposite corner i, and p is corresponding the period jump.
 * 
 * @param output_filename: file location of the frame field to load
 * @return per-face reference tangent direction matrix
 * @return offset angles of a representative cross field direction relative to the reference
 * @return per-corner rotation angle of the reference direction field across the opposite edge
 * @return per-corner period jump of the cross field across the opposite edge
 */
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi>
load_frame_field(const std::string& output_filename);

/**
 * @brief Infer the angle offset of a direction field relative to a reference field
 * 
 * @param V: mesh vertices`
 * @param F: mesh faces
 * @param reference_corner: per-face reference corner opposite the reference direction
 * @param direction_field: per-face tangent direction matrix
 * @return offset angles of the direction field relative to the reference
 */
Eigen::VectorXd infer_theta(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& reference_corner,
    const Eigen::MatrixXd& direction_field);

/**
 * @brief Extend a frame field defined on a base mesh to a refined mesh.
 * 
 * @param F: refined mesh faces
 * @param F_uv: refined mesh parameterization
 * @param Fn_to_F: map from refined to base mesh faces
 * @param endpoints: endpoint base mesh vertices of the refined edge vertices 
 * @param F_base: base mesh faces
 * @param reference_vector: base per-face reference tangent direction matrix
 * @param theta: base offset angles of a representative cross field direction relative to the reference
 * @param kappa: base per-corner rotation angle of the reference direction field across the opposite edge
 * @param period_jump: base per-corner period jump of the cross field across the opposite edge
 * @return refined per-face reference tangent direction matrix
 * @return refined offset angles of a representative cross field direction relative to the reference
 * @return refined per-corner rotation angle of the reference direction field across the opposite edge
 * @return refined per-corner period jump of the cross field across the opposite edge
 */
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi> refine_frame_field(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::MatrixXi& F_base,
    const Eigen::MatrixXd& reference_vector,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump);

Eigen::VectorXd rotate_vector(
                    const Eigen::VectorXd& vec,
                    double angle,
                    const Eigen::VectorXd& B1,
                    const Eigen::VectorXd& B2);

std::vector<Scalar> compute_cone_angle( 
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump);

Eigen::VectorXi transfer_period_jumps_to_halfedge(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F, 
    const Eigen::MatrixXi& corner_period_jump);

} // namespace Feature 
} // namespace Penner
