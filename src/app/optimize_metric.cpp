#include "common.hh"
#include "implicit_optimization.hh"
#include "optimization_interface.hh"
#include "cone_metric.hh"
#include "energy_functor.hh"
#include "logging.hh"
#include "optimization_interface.hh"
#include "targets.hh"
#include "refinement.hh"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
using namespace CurvatureMetric;

int main(int argc, char *argv[])
{
#ifdef MULTIPRECISION
	spdlog::info("Using multiprecision");
	mpfr::mpreal::set_default_prec(200);
	mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
	mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

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
	std::vector<Scalar> Th_hat_init, Th_hat;
	spdlog::info("Using cone angles at {}", Th_hat_filename);
	read_vector_from_file(Th_hat_filename, Th_hat_init);
	correct_cone_angles(Th_hat_init, Th_hat);

	// Get initial mesh for optimization
	std::vector<int> vtx_reindex;
	std::unique_ptr<DifferentiableConeMetric> cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

	// Get energy
	LogLengthEnergy opt_energy(*cone_metric);
	
	// Get initial metric from file or target
	VectorX reduced_metric_init;
	if (argc > 4)
	{
		std::string metric_filename = argv[4];
		std::vector<Scalar> reduced_metric_init_vec;
		read_vector_from_file(metric_filename, reduced_metric_init_vec);
		convert_std_to_eigen_vector(reduced_metric_init_vec, reduced_metric_init);
		cone_metric->set_metric_coordinates(reduced_metric_init);
	}

	// Set all cones as dof
	//std::vector<int> cone_vertices;
	//compute_cone_vertices(m, cone_vertices);
	//std::vector<bool> is_cone_vertex;
	//convert_index_vector_to_boolean_array(cone_vertices, m.n_vertices(), is_cone_vertex);
	//m.fixed_dof = is_cone_vertex;

	// Make default parameters with specified output directory
	auto proj_params = std::make_shared<ProjectionParameters>();
	auto opt_params = std::make_shared<OptimizationParameters>();
	opt_params->output_dir = output_dir;
	opt_params->num_iter = 5;

	// Optimize the metric
	VectorX optimized_reduced_metric_coords = optimize_metric(
		*cone_metric,
		opt_energy,
		proj_params,
		opt_params);

	// Write the output
	std::string output_filename = join_path(output_dir, "reduced_optimized_metric_coords");
	write_vector(optimized_reduced_metric_coords, output_filename, 17);

	// Generate overlay VF mesh with parametrization
	bool do_best_fit_scaling = false;
	auto vf_res = generate_VF_mesh_from_metric(
		V,
		F,
		Th_hat,
		*cone_metric,
		optimized_reduced_metric_coords,
		vtx_reindex,
		do_best_fit_scaling
	);
  OverlayMesh<Scalar> m_o = std::get<0>(vf_res);
  Eigen::MatrixXd V_o = std::get<1>(vf_res);
  Eigen::MatrixXi F_o = std::get<2>(vf_res);
  Eigen::MatrixXd uv_o = std::get<3>(vf_res);
  Eigen::MatrixXi FT_o = std::get<4>(vf_res);
  std::vector<int> fn_to_f_o = std::get<7>(vf_res);
  std::vector<std::pair<int,int>> endpoints_o = std::get<8>(vf_res);

	// Write the overlay output
	output_filename = join_path(output_dir, "overlay_mesh_with_uv.obj");
	write_obj_with_uv(output_filename, V_o, F_o, uv_o, FT_o);

	// Get refinement mesh
	// Build mesh
	Eigen::MatrixXd V_r;
	Eigen::MatrixXi F_r;
	Eigen::MatrixXd uv_r;
	Eigen::MatrixXi FT_r;
	std::vector<int> fn_to_f_r;
	std::vector<std::pair<int,int>> endpoints_r;
	RefinementMesh refinement_mesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o);
	refinement_mesh.get_VF_mesh(V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r);

	// Write the refined output
	output_filename = join_path(output_dir, "refined_mesh_with_uv.obj");
	write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);
}

