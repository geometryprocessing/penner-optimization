#pragma once
#include "common.hh"


namespace CurvatureMetric {


/// Generate a mesh with initial target metric coordinates for optimization
///
/// @param[in] V: initial vertices
/// @param[in] F: initial faces
/// @param[in] Th_hat: target angles
/// @param[out] m: halfedge closed mesh
/// @param[out] vtx_reindex: map from original to new vertex indices
/// @param[out] reduced_metric_target: reduced per edge Penner coordinates
void
generate_initial_mesh(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
	const std::vector<Scalar>& Th_hat,
	Mesh<Scalar> &m,
	std::vector<int>& vtx_reindex,
	VectorX& reduced_metric_target
);

/// Generate a delaunay mesh with initial target metric coordinates for optimization
///
/// @param[in] V: initial vertices
/// @param[in] F: initial faces
/// @param[in] Th_hat: target angles
/// @param[out] m: halfedge closed mesh
/// @param[out] vtx_reindex: map from original to new vertex indices
/// @param[out] reduced_metric_target: reduced per edge Penner coordinates
void
generate_initial_delaunay_mesh(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
	const std::vector<Scalar>& Th_hat,
	Mesh<Scalar> &m,
	std::vector<int>& vtx_reindex,
	VectorX& reduced_metric_target
);

/// Correct cone angles that are multiples of pi/60 to machine precision.
///
/// @param[in] initial_cone_angles: target angles to correct
/// @param[out] corrected_cone_angles: corrected angles
void
correct_cone_angles(
	const std::vector<Scalar>& initial_cone_angles,
	std::vector<Scalar>& corrected_cone_angles
);


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
	OverlayMesh<Scalar>, // m_o
  Eigen::MatrixXd, // V_o
  Eigen::MatrixXi, // F_o
  Eigen::MatrixXd, // uv_o
  Eigen::MatrixXi, // FT_o
	std::vector<bool>, // is_cut_h
	std::vector<bool>, // is_cut_o
	std::vector<int>, // Fn_to_F
	std::vector<std::pair<int,int>> // endpoints_o
>
generate_VF_mesh_from_metric(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
	const std::vector<Scalar>& Th_hat,
	const Mesh<Scalar>& m,
	const std::vector<int>& vtx_reindex,
	const VectorX& reduced_metric_target,
	const VectorX& reduced_metric_coords,
	bool do_best_fit_scaling=false
);

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
  const Eigen::MatrixXi &F_uv
);

std::tuple<
        std::vector<std::vector<Scalar>>,       // V_out
        std::vector<std::vector<int>>,          // F_out
        std::vector<Scalar>,                    // layout u (per vertex)
        std::vector<Scalar>,                    // layout v (per vertex)
        std::vector<std::vector<int>>,          // FT_out
        std::vector<bool>,                      // is_cut_o
        std::vector<int>,                       // Fn_to_F
        std::vector<std::pair<int,int>>>        // map from new vertices to original endpoints
consistent_overlay_mesh_to_VL(
                   const Eigen::MatrixXi& F,
                   const std::vector<Scalar>& Theta_hat,
                   OverlayMesh<Scalar>& mo,
                   std::vector<Scalar> &u,
                   std::vector<std::vector<Scalar>>& V_overlay,
                   std::vector<int>& vtx_reindex,
                   std::vector<std::pair<int, int>>& endpoints,
									 std::vector<bool>& is_cut);

#ifdef PYBIND
std::vector<Scalar>
correct_cone_angles_pybind(
	const std::vector<Scalar>& initial_cone_angles
);
#endif

}

