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
/// @param[in] V: mesh vertices in 3D
/// @param[in] F: mesh faces
/// @return true iff the face areas are all nonzero
bool check_areas(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F
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

std::tuple<std::vector<std::vector<Scalar>>, // V_out
           std::vector<std::vector<int>>,    // F_out
           std::vector<Scalar>,              // layout u (per vertex)
           std::vector<Scalar>,              // layout v (per vertex)
           std::vector<std::vector<int>>,    // FT_out
           std::vector<bool>,                // is_cut
           std::vector<bool>,                // is_cut_o
           std::vector<int>,                 // Fn_to_F
           std::vector<std::pair<int, int>>  // endpoints
           >
parametrize_mesh(const Eigen::MatrixXd& V,
                 const Eigen::MatrixXi& F,
                 const std::vector<Scalar>& Theta_hat,
                 const Mesh<Scalar> &m,
                 const std::vector<int>& vtx_reindex,
                 const VectorX reduced_metric_coords);

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
