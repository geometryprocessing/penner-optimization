#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {

/// A mesh structure to perform interpolation of points in a surface.
///
/// While the overlay mesh supports changes in triangulation, this mesh
/// also supports changes in the underlying metric. Also, it creates a
/// a more distinct representation of Hyperbolic and Euclidean surfaces to
/// ensure invalid operations are prevented from occurring. This is important
/// for interpolation as the barycentric coordinates introduce state and
/// require initialization to go between hyperbolic and Euclidean representations.
class InterpolationMesh
{
public:
    /// Construct a trivial invalid interpolation mesh.
    InterpolationMesh();

    /// Construct a Euclidean interpolation mesh from a VF mesh with possibly distinct
    /// metric mesh determining edge lengths.
    ///
    /// @param[in] V: input mesh vertices
    /// @param[in] F: input mesh faces
    /// @param[in] uv: input metric mesh vertices
    /// @param[in] F_uv: input metric mesh faces
    /// @param[in] Theta_hat: input mesh target angles
    InterpolationMesh(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& F_uv,
        const std::vector<Scalar>& Theta_hat);

    /// Construct an interpolation mesh from a base mesh.
    ///
    /// @param[in] cone_metric: input mesh
    /// @param[in] scale_factors: input scale factors
    /// @param[in] is_hyperbolic: initialize the mesh as a hyperbolic surface if true
    InterpolationMesh(
        const Mesh<Scalar>& m,
        const VectorX& scale_factors,
        bool is_hyperbolic = true);

    /// Add points to interpolate to the surface.
    ///
    /// Overwrites any previous points.
    ///
    /// @param[in] pt_fids: input point face ids
    /// @param[in] pt_bcs: input point barycentric coordinates
    void add_points(
        const std::vector<int>& pt_fids,
        const std::vector<Eigen::Matrix<Scalar, 3, 1>>& pt_bcs);

    /// Get interpolated points on the surface.
    ///
    /// @param[in] pt_fids: output point face ids
    /// @param[in] pt_bcs: output point barycentric coordinates
    void get_points(std::vector<int>& pt_fids, std::vector<Eigen::Matrix<Scalar, 3, 1>>& pt_bcs);

    /// Convert the mesh to a hyperbolic surface with ideal Delaunay triangulation.
    ///
    /// This method changes the triangulation and coordinates but not the underlying metric.
    /// It is (up to numerical error) inverse to converting to a euclidean surface.
    ///
    /// @param[out] flip_sequence: sequence of flips used to make the mesh Delaunay
    void convert_to_delaunay_hyperbolic_surface(std::vector<int>& flip_sequence);

    /// Convert the mesh to a Euclidean surface with ideal Delaunay triangulation.
    ///
    /// This method changes the triangulation and coordinates but not the underlying metric
    /// It is (up to numerical error) inverse to converting to a hyperbolic surface.
    ///
    /// @param[out] flip_sequence: sequence of flips used to make the mesh Delaunay
    void convert_to_delaunay_euclidean_surface(std::vector<int>& flip_sequence);

    /// Convert the mesh to a Hyperbolic surface with the same metric and trianguation.
    ///
    /// This method only changes the coordinates and not the underlying triangulation or metric.
    /// Note that this method has no exact Euclidean counterpart as it relies on arbitrary flips
    /// and Euclidean flips are not always valid.
    ///
    /// @param[out] euclidean_flip_sequence: sequence of Euclidean flips used to make the mesh Delaunay
    /// @param[out] hyperbolic_flip_sequence: sequence of Ptolemy flips used to restore the original connectivity
    void convert_to_hyperbolic_surface(
        std::vector<int>& euclidean_flip_sequence,
        std::vector<int>& hyperbolic_flip_sequence);

    /// Convert the mesh to a Euclidean surface without changing the coordinates or triangulation, which may
    /// change the underlying metric if the surface is not Delaunay
    ///
    /// Warning: This method may change the underlying metric and should be used carefully.
    void force_convert_to_euclidean_surface();

    /// Convert the mesh to a Hyperbolic surface without changing the coordinates or triangulation, which may
    /// change the underlying metric if the surface is not Delaunay
    ///
    /// Warning: This method may change the underlying metric and should be used carefully.
    void force_convert_to_hyperbolic_surface();

    /// Change the hyperbolic metric with horocycles of the surface and update the barycentric coordinates.
    ///
    /// If the halfedge lengths are not consistent across edges or the translations
    /// do not correspond to a continuous piece-wise projective interpolation, nothing
    /// is done and the error flag is raised.
    /// TODO: This requires a consistency check across halfedges; a minimal
    /// representation would be ideal.
    ///
    /// This method changes the underlying metric and coordinates but not the triangulation
    ///
    /// @param[in] halfedge_metric_coords: new metric coordinates of the mesh
    /// @param[in] scale_factors: conformal scale factors defining horocycles of the surface
    /// @param[in] halfedge_translations: per halfedge translations for the interpolation
    void change_hyperbolic_surface_metric(
        const VectorX& halfedge_metric_coords,
        const VectorX& scale_factors,
        const VectorX& halfedge_translations);

    /// Flip the edge with the given halfedge counterclockwise.
    ///
    /// The flip will be a Euclidean flip if the mesh is Euclidean and Ptolemy if it is
    /// Hyperbolic. The sign convention for Euclidean and Ptolemy flips is ignored.
    ///
    /// @param[in] flip_index: (possibly signed) index of the halfedge to flip
    void flip_ccw(int flip_index);

    /// Flip the edge with the given halfedge clockwise.
    ///
    /// The flip will be a Euclidean flip if the mesh is Euclidean and Ptolemy if it is
    /// Hyperbolic. The sign convention for Euclidean and Ptolemy flips is ignored.
    ///
    /// @param[in] flip_index: (possibly signed) index of the halfedge to flip
    void flip_clockwise(int flip_index);

    /// Follow the (ccw) flips of a given sequence
    ///
    /// The flips will be Euclidean if the mesh is Euclidean and Ptolemy if it is
    /// Hyperbolic.
    ///
    /// @param[in] flip_sequences: sequence of flips to follow
    void follow_flip_sequence(const std::vector<int>& flip_sequence);

    /// Reverse the (ccw) flips of a given sequence
    ///
    /// The flips will be Euclidean if the mesh is Euclidean and Ptolemy if it is
    /// Hyperbolic.
    ///
    /// @param[in] flip_sequences: sequence of flips to reverse
    void reverse_flip_sequence(const std::vector<int>& flip_sequence);

    /// Make the mesh delaunay and record the flip sequence.
    ///
    /// @param[out] flip_sequence: sequence of flipped halfedges
    void make_delaunay(std::vector<int>& flip_sequence);

    /// Determine if the mesh is Euclidean
    ///
    /// @return true iff the mesh is Euclidean
    bool is_euclidean() const;

    /// Determine if the mesh is Hyperbolic
    ///
    /// @return true iff the mesh is Hyperbolic
    bool is_hyperbolic() const;

    /// Get the number of halfedges in the underlying mesh
    ///
    /// @return number of halfedges
    int num_mesh_halfedges() const;

    /// Get the underlying mesh of the surface.
    ///
    /// @return mesh
    Mesh<Scalar>& get_mesh() { return m_overlay_mesh.cmesh(); }

    /// Get the overlay mesh of the surface.
    ///
    /// @return overlay mesh
    OverlayMesh<Scalar> const& get_overlay_mesh() const { return m_overlay_mesh; }

    /// Get a tufted cover overlay mesh of the surface.
    ///
    /// @param[in] num_vertices: number of vertices of the original mesh
    /// @param[in] indep_vtx: independent vertex indices
    /// @param[in] dep_vtx: dependent vertex indices
    /// @param[in] bnd_loops: boundary loops of the mesh
    /// @return tufted cover overlay mesh
    OverlayMesh<Scalar> get_tufted_overlay_mesh(
        int num_vertices,
        const std::vector<int>& indep_vtx,
        const std::vector<int>& dep_vtx,
        const std::vector<int>& bnd_loops) const;


    /// Get the per halfedge metric coordinates of the surface.
    ///
    /// @return halfedge metric coordinates
    VectorX get_halfedge_metric_coordinates();

    /// Get the per edge metric coordinates of the surface.
    ///
    /// @return edge metric coordinates
    VectorX get_metric_coordinates();

    /// Get the reduced metric coordinates of the surface.
    ///
    /// @return reduced metric coordinates
    VectorX get_reduced_metric_coordinates();

    /// Output all nontrivial barycentric coordinates
    void check_bc_values();

private:
    int get_flip_halfedge_index(int flip_index) const;


    bool is_valid_interpolation_mesh();

    bool is_trivial_symmetric_mesh();

    bool is_valid_halfedge_index(int halfedge_index);

    bool are_valid_halfedge_lengths(const VectorX& halfedge_lengths);

    bool are_valid_halfedge_translations(const VectorX& halfedge_translations) const;

    OverlayMesh<Scalar> m_overlay_mesh;
    VectorX m_scale_factors;
    bool m_is_hyperbolic;
    bool m_is_valid;
};

/// Interpolate the triangulation from the original surface to the surface with the given Penner
/// coordinates on the current triangulation and the reverse interpolation.
///
/// The output forward mesh is hyperbolic while the output reverse mesh is Euclidean.
///
/// @param[in] mesh: input mesh corresponding to the VF mesh
/// @param[in] halfedge_metric_coords: new halfedge Penner coordinates of the mesh
/// @param[in] scale_factors: scale factors for the new metric
/// @param[out] interpolation_mesh: hyperbolic mesh with new metric and interpolated original triangulation
/// @param[out] reverse_interpolation_mesh: euclidean mesh with original metric and interpolated original triangulation
void interpolate_penner_coordinates(
    const Mesh<Scalar>& mesh,
    const VectorX& halfedge_metric_coords,
    const VectorX& scale_factors,
    InterpolationMesh& interpolation_mesh,
    InterpolationMesh& reverse_interpolation_mesh);

/// Interpolate vertex positions from the original surface to the surface with the given Penner coordinates
/// on the current triangulation.
///
/// @param[in] V: input mesh vertices
/// @param[in] vtx_reindex: vertex reindexing for the VF mesh and halfedge mesh
/// @param[in] interpolation_mesh: mesh with new metric and interpolated original triangulation
/// @param[in] reverse_interpolation_mesh: mesh with original metric and interpolated original triangulation
/// @param[out] V_overlay: vertices of the overlay mesh
void interpolate_vertex_positions(
    const Eigen::MatrixXd& V,
    const std::vector<int> vtx_reindex,
    const InterpolationMesh& interpolation_mesh,
    const InterpolationMesh& reverse_interpolation_mesh,
    Eigen::MatrixXd& V_overlay);

bool overlay_has_all_original_halfedges(OverlayMesh<Scalar>& mo);

} // namespace CurvatureMetric
