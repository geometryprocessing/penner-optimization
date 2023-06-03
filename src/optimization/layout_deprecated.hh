#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {

/// @brief Given an overlay mesh, forget the overlay information.
///
/// @param[in] mo: mesh with overlay information
/// @return mesh without overlay information
Mesh<double>
remove_overlay(const OverlayMesh<double>& mo);

/// @brief: Flip edges in the overlay mesh according to the flip sequence. Ptolemy flips
/// are indexed with nonnegative halfedes indices and Euclidean flips are indexed
/// with negative indices.
///
/// @param[in, out] mo: overlay mesh to flip edges for
/// @param[in] flip_seq: sequence of flips to perform
void
flip_edges_overlay(OverlayMesh<double>& mo, const std::vector<int>& flip_seq);

/// Generate an overlay mesh for the mesh m with given metric coordinates integrated
/// as the mesh metric.
///
/// Note that here the original mesh metric is overwritten, whereas during the optimization
/// the two are kept separate and the mesh maintains the original input length metric.
///
/// @param[in] m: mesh to add overlay to
/// @param[in] reduced_metric_coords: reduced metric coordinates for overlay mesh
/// @return: overlay mesh with new metric coordinates
std::vector<int>
make_delaunay_overlay(OverlayMesh<double>& mo, bool Ptolemy);

/// Overwrite the metric coordinates for an overlay mesh metric directly.
///
/// @param[in, out] mo: overlay mesh to modify
/// @param[in] l: new halfedge lengths
void
change_lengths_overlay(OverlayMesh<double>& mo, const VectorX& l);

/// Map barycentric coordinates of an overlay mesh to the equilateral reference
/// triangle.
///
/// @param[in, out] mo: overlay mesh with barycentric coordinates to change
void
bc_original_to_eq_overlay(OverlayMesh<double>& mo);

/// Reindex face array according to vtx_reindex.
///
/// @param[in] F: face array to reindex
/// @param[in] vtx_reindex: map from old to new vertex indices
/// @return reindexed face array
std::vector<std::vector<int>>
reindex_F(const std::vector<std::vector<int>>& F,
          const std::vector<int> vtx_reindex);

// Get c
std::tuple<std::vector<int>, std::vector<int>>
get_cones_and_bd(const Eigen::MatrixXd& V,
                 const Eigen::MatrixXi& F,
                 const std::vector<double>& Theta_hat,
                 const std::vector<int>& vtx_reindex);

void
change_shear_overlay(OverlayMesh<double>& mo,
                     const VectorX& lambdas_del_he,
                     const VectorX& tau);

std::vector<std::vector<double>>
Interpolate_3d_reparametrized(OverlayMesh<double>& m_o,
                              const VectorX& lambdas_rev_he,
                              const VectorX& tau_rev,
                              const std::vector<int>& flip_seq_init,
                              const std::vector<int>& flip_seq,
                              const std::vector<std::vector<double>>& x);

std::tuple<OverlayMesh<double>,
           std::vector<int>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>,
           std::vector<int>,
           std::vector<std::pair<int, int>>>
generate_optimized_overlay(
  const Eigen::MatrixXd& v,
  const Eigen::MatrixXi& f,
  const std::vector<double>& Theta_hat,
  const VectorX& lambdas,
  const VectorX& tau_init,
  const VectorX& tau,
  const VectorX& tau_post,
  const std::vector<int>& pt_fids_in,
  const std::vector<Eigen::Matrix<double, 3, 1>>& pt_bcs_in,
  bool initial_ptolemy = false,
  bool flip_in_original_metric = true);


#ifdef PYBIND
#endif


}
