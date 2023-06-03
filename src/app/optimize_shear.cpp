#include "common.hh"
#include "explicit_optimization.hh"
#include "optimization_interface.hh"
#include "targets.hh"
#include "optimization_interface.hh"
#include "shear.hh"
#include "logging.hh"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
using namespace CurvatureMetric;

int main(int argc, char *argv[])
{
#ifdef MULTIPRECISION
	spdlog::info("Using multiprecision");
	mpfr::mpreal::set_default_prec(60);
	mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
	mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

  spdlog::set_level(spdlog::level::debug);
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
	Mesh<Scalar> m;
	std::vector<int> vtx_reindex;
	VectorX reduced_metric_target;
	generate_initial_mesh(V, F, Th_hat, m, vtx_reindex, reduced_metric_target);

	// Get initial metric from file or target
	VectorX reduced_metric_init;
	if (argc > 4)
	{
		std::string metric_filename = argv[4];
		std::vector<Scalar> reduced_metric_init_vec;
		read_vector_from_file(metric_filename, reduced_metric_init_vec);
		convert_std_to_eigen_vector(reduced_metric_init_vec, reduced_metric_init);
	} else {
		reduced_metric_init = reduced_metric_target;
	}

	// Make default parameters
	auto proj_params = std::make_shared<ProjectionParameters>();
	auto opt_params = std::make_shared<OptimizationParameters>();
	opt_params->output_dir = output_dir;
	opt_params->num_iter = 500;
	opt_params->min_ratio = 0;
	opt_params->max_grad_range = 10;
	opt_params->direction_choice = "lbfgs";

	// Compute shear dual basis and the corresponding inner product matrix
	MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
	compute_shear_dual_basis(m, shear_basis_matrix, independent_edges);
	//compute_shear_coordinate_basis(m, shear_basis_matrix, independent_edges);

	// Compute the shear dual coordinates for this basis
  VectorX shear_basis_coords_init;
	VectorX scale_factors_init;
	compute_shear_basis_coordinates(m, reduced_metric_init, shear_basis_matrix, shear_basis_coords_init, scale_factors_init);
	spdlog::info("Initial coordinates are {}", shear_basis_coords_init);

	// Optimize the metric
	VectorX reduced_metric_coords;
	optimize_shear_basis_coordinates(m,
									reduced_metric_target,
						    	shear_basis_coords_init,
									scale_factors_init,
									shear_basis_matrix,
									reduced_metric_coords,
									proj_params,
									opt_params);

	// Write the output
	std::string output_filename = join_path(output_dir, "reduced_metric_coords");
	write_vector(reduced_metric_coords, output_filename);
}

