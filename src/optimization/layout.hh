#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {
  
/// Generate an overlay mesh for the mesh m with given metric coordinates integrated
/// as the mesh metric.
///
/// Note that here the original mesh metric is overwritten, whereas during the optimization
/// the two are kept separate and the mesh maintains the original input length metric.
///
/// @param[in] m: mesh to add overlay to
/// @param[in] reduced_metric_coords: reduced metric coordinates for overlay mesh
/// @return: overlay mesh with new metric coordinates
OverlayMesh<Scalar>
add_overlay(const Mesh<Scalar>& m, const VectorX& reduced_metric_coords);

/// @brief: Make an overlay mesh into a tufted double cover
///
/// @param[in] mo: mesh to make tufted
/// @param[in] V: original vertices used to generate mo
/// @param[in] F: original face array used to generate mo
/// @param[in] Theta_hat: original angles used to generate mo
/// FIXME This should not build the vertex information internally.
void
make_tufted_overlay(OverlayMesh<Scalar>& mo,
                    const Eigen::MatrixXd& V,
                    const Eigen::MatrixXi& F,
                    const std::vector<Scalar>& Theta_hat);

/// Given a VF mesh, check that the face areas are nonzero
///
/// @param[in] V: mesh vertices in 2D or 3D
/// @param[in] F: mesh faces
/// @return true iff the face areas are all nonzero
bool check_areas(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F
);

/// Given a VF mesh with uv coordinates, get the maximum error of the uv lengths
/// across cuts
///
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv coordinates in 2D
/// @param[in] F_uv: mesh uv faces
/// @return maximum uv length error across cuts
Scalar compute_uv_length_error(
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
);

/// Given a VF mesh with uv coordinates, check that it satisfies fundamental uv
/// consistency constraints:
///     - uv lengths match up across cuts
///     - mesh face areas are nonzero
///     - uv face areas are nonzero
///
/// @param[in] V: mesh vertices in 3D
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv coordinates in 2D
/// @param[in] F_uv: mesh uv faces
/// @return true iff the mesh passes all of the tests
bool check_uv(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
);

/// Build a VF mesh for the embedded mesh in the doubled mesh and also extract
/// the mapping from VF mesh corners to opposite halfedge index
///
/// @param[in] m: input mesh
/// @param[in] vtx_reindex: reindexing for the vertices wrt the mesh indexing
/// @param[out] F: mesh faces
/// @param[out] corner_to_halfedge: mesh corner to opposite halfedge indexing
void
extract_embedded_mesh(
  const Mesh<Scalar>& m,
  const std::vector<int>& vtx_reindex,
  Eigen::MatrixXi& F,
  Eigen::MatrixXi& corner_to_halfedge
);

/// Given a metric defined by original edge lengths and scale factor u, do a bfs on dual graph of mesh or 
/// using given cuts to singularities defined in is_cut_h to compute a full layout cut graph
///
/// Note that this only lays out the connected component containing the start halfedge.
/// 
/// @param m, mesh data structure
/// @param is_cut_h, (optional) pre-defined cuts to be included
/// @param start_h, the first halfedge to be laid out, can be used to control the axis-alignment for the whole patch
/// @return is_cut_h #h vector, mark whether the current halfedge is part of cut graph
std::vector<bool>
compute_layout_topology(const Mesh<Scalar> &m, const std::vector<bool>& is_cut_h, int start_h = -1);

/// Given a cut defined on the original or current mesh, pull it back to a cut defined on
/// an overlay mesh for the given mesh (possibly after flips)
/// 
/// @param[in] m_o: overlay mesh data structure
/// @param[in] is_cut_h: cuts on the original or current mesh
/// @param[in] is_original_cut: if true, use cut mask on the original mesh
/// @return cuts on the overlay mesh
std::vector<bool>
pullback_cut_to_overlay(
  OverlayMesh<Scalar> &m_o,
  const std::vector<bool>& is_cut_h,
  bool is_original_cut=true);

/**
 * @brief Given overlay mesh with associated flat metric compute the layout
 * 
 * @tparam Scalar double/mpfr::mpreal
 * @param m_o, overlay mesh
 * @param u_vec, per-vertex scale factor
 * @param singularities, list of singularity vertex ids
 * @return u_o, v_o, is_cut_h (per-corner u/v assignment of overlay mesh and marked cut edges)
 */
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>, std::vector<bool>>
get_consistent_layout(
             OverlayMesh<Scalar> &m_o,
             const std::vector<Scalar> &u_vec,
             std::vector<int> singularities,
             const std::vector<bool>& is_cut_orig,
             const std::vector<bool>& is_cut);

#ifdef PYBIND
std::tuple<
  Eigen::MatrixXi, // F
  Eigen::MatrixXi // corner_to_halfedge
>
extract_embedded_mesh_pybind(
  const Mesh<Scalar>& m,
  const std::vector<int>& vtx_reindex
);
#endif


}
