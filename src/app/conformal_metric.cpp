#include "common.hh"
#include "implicit_optimization.hh"
#include "penner_optimization_interface.hh"
#include "layout.hh"
#include "io.hh"
#include "vector.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
using namespace CurvatureMetric;

int main(int argc, char *argv[])
{
  spdlog::set_level(spdlog::level::info);
  assert(argc > 3);
  std::string input_filename = argv[1];
	std::string Th_hat_filename = argv[2];
  std::string output_dir = argv[3];
  std::filesystem::create_directories(output_dir);

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
	spdlog::info("Optimizing mesh at {}", input_filename);
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);
	
	// Get input angles
	std::vector<Scalar> Th_hat;
	spdlog::info("Using cone angles at {}", Th_hat_filename);
	read_vector_from_file(Th_hat_filename, Th_hat);

	// Get conformal mesh
	std::vector<int> pt_fids(0);
	std::vector<Eigen::Matrix<double, 3, 1>> pt_bcs(0);
	auto conformal_res = conformal_metric<Scalar>(
		V,
		F,
		Th_hat,
		pt_fids,
		pt_bcs
	);
	OverlayMesh<Scalar> m_o = std::get<0>(conformal_res);
	std::vector<Scalar> u = std::get<1>(conformal_res);
	std::vector<int> vtx_reindex = std::get<4>(conformal_res);
  std::vector<std::vector<Scalar>> V_overlay = std::get<5>(conformal_res);
  std::vector<std::pair<int,int>> endpoints = std::get<6>(conformal_res);

	// Build output mesh
	auto parametrize_res = overlay_mesh_to_VL<Scalar>(
		V,
		F,
		Th_hat,
		m_o,
		u,
		V_overlay,
		vtx_reindex,
		endpoints,
		-1
	);
	std::vector<std::vector<Scalar>> V_o_vec = std::get<0>(parametrize_res);
	std::vector<std::vector<int>> F_o_vec = std::get<1>(parametrize_res);
	std::vector<Scalar> u_o_vec = std::get<2>(parametrize_res);
	std::vector<Scalar> v_o_vec = std::get<3>(parametrize_res);
	std::vector<std::vector<int>> FT_o_vec = std::get<4>(parametrize_res);

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


	// Write the output
	std::string output_filename = join_path(output_dir, "conformal_overlay_with_uv.obj");
	write_obj_with_uv(output_filename, V_o, F_o, uv_o, FT_o);
	VectorX scale_factors;
	VectorX l;
	convert_std_to_eigen_vector(u, scale_factors);
	convert_std_to_eigen_vector(m_o.cmesh().l, l);
	output_filename = join_path(output_dir, "conformal_scale_factors.txt");
	write_vector(scale_factors, output_filename);
	output_filename = join_path(output_dir, "conformal_lengths.txt");
	write_vector(l, output_filename);
}

