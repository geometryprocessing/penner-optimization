#pragma once
#include "common.hh"
#include "cone_metric.hh"

namespace CurvatureMetric
{

  /// Generate a mesh with initial target metric coordinates for optimization
  ///
  /// @param[in] V: initial vertices
  /// @param[in] F: initial faces
  /// @param[in] uv: initial metric vertices
  /// @param[in] F_uv: initial metric faces
  /// @param[in] Th_hat: target angles
  /// @param[out] vtx_reindex: vertices reindexing for new halfedge mesh
  /// @param[in] free_cones: (optional) vertex cones to leave unconstrained
  /// @param[in] fix_boundary: (optional) if true, fix boundary edge lengths
  /// @param[in] use_discrete_metric: (optional) if true, use log edge lengths instead of penner coordinates
  /// @return differentiable mesh with metric
  std::unique_ptr<DifferentiableConeMetric>
  generate_initial_mesh(
      const Eigen::MatrixXd &V,
      const Eigen::MatrixXi &F,
      const Eigen::MatrixXd& uv,
      const Eigen::MatrixXi& F_uv,
      const std::vector<Scalar> &Th_hat,
      std::vector<int>& vtx_reindex,
      std::vector<int> free_cones={},
      bool fix_boundary=false,
      bool use_discrete_metric=false);


  /// Correct cone angles that are multiples of pi/60 to machine precision.
  ///
  /// @param[in] initial_cone_angles: target angles to correct
  /// @return: corrected angles
  std::vector<Scalar>
  correct_cone_angles(
      const std::vector<Scalar> &initial_cone_angles);

  /// Write an obj file with uv coordinates.
  ///
  /// @param[in] filename: obj output file location
  /// @param[in] V: mesh vertices
  /// @param[in] F: mesh faces
  /// @param[in] uv: mesh uv corner coordinates
  /// @param[in] F_uv: mesh uv faces
  void
  write_obj_with_uv(
      const std::string &filename,
      const Eigen::MatrixXd &V,
      const Eigen::MatrixXi &F,
      const Eigen::MatrixXd &uv,
      const Eigen::MatrixXi &F_uv);

  /// Given a mesh with initial target and final optimized metric coordinates, generate a corresponding
  /// overlay VF mesh with parametrization.
  ///
  /// @param[in] V: initial mesh vertices
  /// @param[in] F: initial mesh faces
  /// @param[in] Th_hat: initial target angles
  /// @param[in] m: mesh
  /// @param[in] vtx_reindex: map from old to new vertices
  /// @param[in] reduced_metric_target: initial mesh Penner coordinates
  /// @param[in] reduced_metric_coords: optimized mesh metric
  /// @param[in] do_best_fit_scaling: if true, extract best fit scale factors from the metric
  /// @return parametrized VF mesh with cut and topology mapping information
  std::tuple<
      OverlayMesh<Scalar>,             // m_o
      Eigen::MatrixXd,                 // V_o
      Eigen::MatrixXi,                 // F_o
      Eigen::MatrixXd,                 // uv_o
      Eigen::MatrixXi,                 // FT_o
      std::vector<bool>,               // is_cut_h
      std::vector<bool>,               // is_cut_o
      std::vector<int>,                // Fn_to_F
      std::vector<std::pair<int, int>> // endpoints_o
      >
  generate_VF_mesh_from_metric(
      const Eigen::MatrixXd &V,
      const Eigen::MatrixXi &F,
      const std::vector<Scalar> &Th_hat,
      const DifferentiableConeMetric &initial_cone_metric,
      const VectorX &reduced_metric_coords,
      std::vector<bool> = {},
      bool do_best_fit_scaling = false);

}
