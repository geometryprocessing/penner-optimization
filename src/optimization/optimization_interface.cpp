#include "optimization_interface.hh"
#include "targets.hh"
#include "embedding.hh"
#include "energies.hh"
#include "translation.hh"
#include "interpolation.hh"
#include "projection.hh"
#include "optimization_layout.cpp"
#include "conformal_ideal_delaunay/ConformalInterface.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void
generate_initial_mesh(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
	const std::vector<Scalar>& Th_hat,
	Mesh<Scalar> &m,
	std::vector<int>& vtx_reindex,
	VectorX& reduced_metric_target
) {
	// Convert VF mesh to halfedge
	std::vector<int> indep_vtx, dep_vtx, v_rep, bnd_loops;
	std::vector<int> free_cones(0);
	bool fix_boundary = false;
	m = FV_to_double<Scalar>(
		V,
		F,
		V,
		F,
		Th_hat,
		vtx_reindex,
		indep_vtx,
		dep_vtx,
		v_rep,
		bnd_loops,
		free_cones,
		fix_boundary
	);

	// Build initial metric and target metric from edge lengths
	std::vector<int> flip_sequence;
	compute_penner_coordinates(V, F, Th_hat, reduced_metric_target, flip_sequence);
}

void
generate_initial_delaunay_mesh(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
	const std::vector<Scalar>& Th_hat,
	Mesh<Scalar> &m,
	std::vector<int>& vtx_reindex,
	VectorX& reduced_metric_target
) {
	// Convert VF mesh to halfedge
	std::vector<int> indep_vtx, dep_vtx, v_rep, bnd_loops;
	std::vector<int> free_cones(0);
	bool fix_boundary = false;
	m = FV_to_double<Scalar>(
		V,
		F,
		V,
		F,
		Th_hat,
		vtx_reindex,
		indep_vtx,
		dep_vtx,
		v_rep,
		bnd_loops,
		free_cones,
		fix_boundary
	);

	// Make mesh delaunay
  VectorX u;
  u.setZero(m.n_ind_vertices());
  DelaunayStats del_stats;
  SolveStats<Scalar> solve_stats;
  bool use_ptolemy = true;
  ConformalIdealDelaunay<Scalar>::MakeDelaunay(m, u, del_stats, solve_stats, use_ptolemy);

	// Build target metric from edge lengths
	compute_log_edge_lengths(V, F, Th_hat, reduced_metric_target);
}

void
correct_cone_angles(
	const std::vector<Scalar>& initial_cone_angles,
	std::vector<Scalar>& corrected_cone_angles
) {
	// Get precise value of pi
	Scalar pi;
	if (std::is_same<Scalar, mpfr::mpreal>::value)
		pi = Scalar(mpfr::const_pi());
	else
		pi = Scalar(M_PI);

	// Correct angles
	int num_vertices = initial_cone_angles.size();
	corrected_cone_angles.resize(num_vertices);
	for (int i = 0; i < num_vertices; ++i)
	{
		Scalar angle = initial_cone_angles[i];
		int rounded_angle = lround(Scalar(60.0) * angle / pi);
		corrected_cone_angles[i] = (rounded_angle * pi) / Scalar(60.0);
	}
}

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
	bool do_best_fit_scaling
) {
	ReductionMaps reduction_maps(m);

	// Expand metric coordinates to per edge values
	int num_edges = reduction_maps.num_edges;
	VectorX metric_target(num_edges);
	VectorX metric_coords(num_edges);
	for (int e = 0; e < num_edges; ++e)
	{
		int E = reduction_maps.proj[e];
		metric_target[e] = reduced_metric_target[E];
		metric_coords[e] = reduced_metric_coords[E];
	}
	SPDLOG_INFO("Target metric coordinates in range [{}, {}]", reduced_metric_target.minCoeff(), reduced_metric_target.maxCoeff());
	SPDLOG_INFO("Optimized metric coordinates in range [{}, {}]", metric_coords.minCoeff(), metric_coords.maxCoeff());

	// Fit conformal scale factors
	VectorX scale_factors;
	scale_factors.setZero(m.n_ind_vertices());
	if (do_best_fit_scaling)
	{
		scale_factors = best_fit_conformal(m, metric_target, metric_coords);
		MatrixX B = conformal_scaling_matrix(m);
		metric_coords = metric_coords - B * scale_factors;
	}
	VectorX metric_diff = metric_coords - metric_target;
	SPDLOG_INFO("Scale factors in range [{}, {}]", scale_factors.minCoeff(), scale_factors.maxCoeff());
	SPDLOG_INFO("Scaled metric coordinates in range [{}, {}]", metric_coords.minCoeff(), metric_coords.maxCoeff());
	SPDLOG_INFO("Differences from target to optimized metric in range [{}, {}]", metric_diff.minCoeff(), metric_diff.maxCoeff());

	// Expand metric coordinates to per halfedge values
	int num_halfedges = reduction_maps.num_halfedges;
	VectorX he_metric_target(num_halfedges);
	VectorX he_metric_coords(num_halfedges);
	for (int h = 0; h < num_halfedges; ++h)
	{
		int e = reduction_maps.he2e[h];
		he_metric_target[h] = metric_target[e];
		he_metric_coords[h] = metric_coords[e];
	}

	// Compute interpolation overlay mesh
	Eigen::MatrixXd V_overlay;
	InterpolationMesh interpolation_mesh, reverse_interpolation_mesh;
	spdlog::info("Interpolating penner coordinates");
	interpolate_penner_coordinates(m, he_metric_coords, scale_factors, interpolation_mesh, reverse_interpolation_mesh);
	spdlog::info("Interpolating vertex positions");
	interpolate_vertex_positions(V, vtx_reindex, interpolation_mesh, reverse_interpolation_mesh, V_overlay);
	OverlayMesh<Scalar> m_o = interpolation_mesh.get_overlay_mesh();
	make_tufted_overlay(m_o, V, F, Th_hat);

	// Get endpoints
	std::vector<std::pair<int, int>> endpoints;
	find_origin_endpoints(m_o, endpoints);

	// Convert overlay mesh to transposed vector format
	std::vector<std::vector<Scalar>> V_overlay_vec(3);
	for (int i = 0; i < 3; ++i)
	{
		V_overlay_vec[i].resize(V_overlay.rows());
		for (int j = 0; j < V_overlay.rows(); ++j)
		{
			V_overlay_vec[i][j] = V_overlay(j, i);
		}
	}

	// Get layout topology from original mesh
	std::vector<bool> is_cut_place_holder(0);
	std::vector<bool> is_cut = compute_layout_topology(m, is_cut_place_holder);

	// Convert overlay mesh to VL format
	spdlog::info("Getting layout");
	std::vector<int> vtx_reindex_mutable = vtx_reindex;
  std::vector<Scalar> u; // (m_o._m.Th_hat.size(), 0.0);
	convert_eigen_to_std_vector(scale_factors, u);
	// auto parametrize_res = overlay_mesh_to_VL<Scalar>(V, F, Th_hat, m_o, u, V_overlay_vec, vtx_reindex_mutable, endpoints, -1); FIXME
	auto parametrize_res = consistent_overlay_mesh_to_VL(
		F,
		Th_hat,
		m_o,
		u,
		V_overlay_vec,
		vtx_reindex_mutable,
		endpoints,
		is_cut
	);
	std::vector<std::vector<Scalar>> V_o_vec = std::get<0>(parametrize_res);
	std::vector<std::vector<int>> F_o_vec = std::get<1>(parametrize_res);
	std::vector<Scalar> u_o_vec = std::get<2>(parametrize_res);
	std::vector<Scalar> v_o_vec = std::get<3>(parametrize_res);
	std::vector<std::vector<int>> FT_o_vec = std::get<4>(parametrize_res);
	std::vector<bool> is_cut_o = std::get<5>(parametrize_res);
	std::vector<int> Fn_to_F = std::get<6>(parametrize_res);
	std::vector<std::pair<int,int>> endpoints_o = std::get<7>(parametrize_res);

	// Convert vector formats to matrices
	Eigen::MatrixXd V_o, uv_o;
	Eigen::VectorXd u_o, v_o;
	Eigen::MatrixXi F_o, FT_o;
	convert_std_to_eigen_matrix(V_o_vec, V_o);
	convert_std_to_eigen_matrix(F_o_vec, F_o);
	convert_std_to_eigen_matrix(FT_o_vec, FT_o);
	convert_std_to_eigen_vector(u_o_vec, u_o);
	convert_std_to_eigen_vector(v_o_vec, v_o);
	uv_o.resize(u_o.size(), 2);
	uv_o.col(0) = u_o;
	uv_o.col(1) = v_o;

	// Check for validity
	if (!check_uv(V_o, F_o, uv_o, FT_o))
	{
		spdlog::error("Inconsistent uvs");
	}

	return std::make_tuple(
		m_o,
		V_o,
		F_o,
		uv_o,
		FT_o,
		is_cut,
		is_cut_o,
		Fn_to_F,
		endpoints_o
	);
}

void
write_obj_with_uv(
  const std::string &filename,
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv
) {
  Eigen::MatrixXd N;
  Eigen::MatrixXi FN;
  igl::writeOBJ(
    filename,
    V,
    F,
    N,
    FN,
    uv,
    F_uv
  ); 
}

std::tuple<
        std::vector<std::vector<Scalar>>,       // V_out
        std::vector<std::vector<int>>,          // F_out
        std::vector<Scalar>,                    // layout u (per vertex)
        std::vector<Scalar>,                    // layout v (per vertex)
        std::vector<std::vector<int>>,          // FT_out
        std::vector<bool>,                      // is_cut_o
        std::vector<int>,                       // Fn_to_F
        std::vector<std::pair<int,int>>>        // map from new vertices to original endpoints
consistent_overlay_mesh_to_VL(const Eigen::MatrixXi& F,
                   const std::vector<Scalar>& Theta_hat,
                   OverlayMesh<Scalar>& mo,
                   std::vector<Scalar> &u,
                   std::vector<std::vector<Scalar>>& V_overlay,
                   std::vector<int>& vtx_reindex,
                   std::vector<std::pair<int, int>>& endpoints,
									 const std::vector<bool>& is_cut)
{
    // get cones and bd
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F);
    for (size_t i = 0; i < is_bd.size(); i++)
    {
        if (is_bd[i])
        {
            bd.push_back(i);
        }
    }
    for (size_t i = 0; i < Theta_hat.size(); i++)
    {
        if ((!is_bd[i]) && abs(Theta_hat[i] -  2 * M_PI) > 1e-15)
        {
            cones.push_back(i);
        }
    }

    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (size_t i = 0; i < vtx_reindex.size(); i++)
    {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (size_t i = 0; i < cones.size(); i++)
    {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    for (size_t i = 0; i < bd.size(); i++)
    {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    spdlog::info("#bd_vt: {}", bd.size());
    spdlog::info("#cones: {}", cones.size());
    spdlog::info("vtx reindex size: {}", vtx_reindex.size());
    spdlog::info("mc.out size: {}", mo.cmesh().out.size());

    // get layout
    auto layout_res = get_consistent_layout(mo, u, cones, is_cut);
    auto u_o = std::get<0>(layout_res);
    auto v_o = std::get<1>(layout_res);
    auto is_cut_o = std::get<2>(layout_res);

    // get output VF and metric
    auto FVFT_res = get_FV_FTVT(mo, endpoints, is_cut_o, V_overlay, u_o, v_o);
    auto v3d = std::get<0>(FVFT_res); 
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);
    auto Fn_to_F = std::get<5>(FVFT_res);
    auto remapped_endpoints = std::get<6>(FVFT_res);

    // v3d_out = v3d^T
    std::vector<std::vector<Scalar>> v3d_out(v3d[0].size());
    for (size_t i = 0; i < v3d[0].size(); i++)
    {
        v3d_out[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            v3d_out[i][j] = v3d[j][i];
        }
    }

    // reindex back
    auto u_o_out_copy = u_o_out;
    auto v_o_out_copy = v_o_out;
    auto v3d_out_copy = v3d_out;
    auto endpoints_out = remapped_endpoints;
		int num_vertices = vtx_reindex.size();
    for (size_t i = 0; i < F_out.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (F_out[i][j] < num_vertices)
            {
                F_out[i][j] = vtx_reindex[F_out[i][j]];
            }
            if (FT_out[i][j] < num_vertices)
            {
                FT_out[i][j] = vtx_reindex[FT_out[i][j]];
            }
        }
    }
    for (size_t i = 0; i < vtx_reindex.size(); i++)
    {
        u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
        v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
        v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
    }
    for(size_t i = vtx_reindex.size(); i < endpoints_out.size(); i++)
    {
        int a = vtx_reindex[endpoints_out[i].first];
        int b = vtx_reindex[endpoints_out[i].second];
        endpoints_out[i] = std::make_pair(a, b);
    }

    return std::make_tuple(v3d_out, F_out, u_o_out, v_o_out, FT_out, is_cut_o, Fn_to_F, endpoints_out);
}


#ifdef PYBIND

std::vector<Scalar>
correct_cone_angles_pybind(
	const std::vector<Scalar>& initial_cone_angles
) {
	std::vector<Scalar> corrected_cone_angles;
	correct_cone_angles(initial_cone_angles, corrected_cone_angles);
	return corrected_cone_angles;
}

#endif

}

