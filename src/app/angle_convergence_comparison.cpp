#include "common.hh"
#include "explicit_optimization.hh"
#include "implicit_optimization.hh"
#include "targets.hh"
#include "energies.hh"
#include "optimization_interface.hh"
#include "constraint.hh"
#include "shear.hh"
#include "projection.hh"
#include "logging.hh"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

/// The convergence has been observed to be slow for several meshes without obvious
/// causes (such as large cones or number of elements). We conjecture this is due to
/// only linear information being used for the constraint approximation and slow 
/// convergence corresponding to an optimum near a high curvature region.

using namespace CurvatureMetric;

void write_vector(
	const std::vector<Scalar>&vec,
	const std::string &filename
) {
	std::ofstream output_file(filename, std::ios::out | std::ios::trunc);
  for (Eigen::Index i = 0; i < vec.size(); ++i)
  {
    output_file << std::setprecision(16) << vec[i] << std::endl;
  }
  output_file.close();
}

/// Build tetrahedron with equilateral base and given fourth vertex height.
void
generate_tet_mesh(
	Scalar height,
	Eigen::MatrixXd& V,
	Eigen::MatrixXi& F
) {
	Scalar alpha = sqrt(3.0) / 2.0; // Third root of unity x coordinate
	V.resize(4, 3);
	V <<
		 sqrt(8.0 / 9.0),                0, -1.0 / 3.0,
		-sqrt(2.0 / 9.0),  sqrt(2.0 / 3.0), -1.0 / 3.0,
		-sqrt(2.0 / 9.0), -sqrt(2.0 / 3.0), -1.0 / 3.0,
		 0, 0, height;

	F.resize(4, 3);
	F <<
		0, 2, 1,
		0, 1, 3,
		1, 2, 3,
		2, 0, 3;
}

/// Build tetrahedron angles with given fourth vertex value and others equal
void
generate_tet_target_angles(
	Scalar angle,
	std::vector<Scalar>& Th_hat
) {
	Scalar other_angles = (4.0 * M_PI - angle) / 3.0;
	Th_hat.resize(4);
	Th_hat[0] = other_angles;
	Th_hat[1] = other_angles;
	Th_hat[2] = other_angles;
	Th_hat[3] = angle;
}

// Build linspace between two values with a given number of nodes
void
generate_linspace(
	Scalar start_val,
	Scalar end_val,
	int num_nodes,
	std::vector<Scalar>& linspace
) {
	// Skip ill defined cases
	if (num_nodes < 2) return;

	// Compute distance between values
	Scalar delta = (end_val - start_val) / (num_nodes - 1);
	
	// Build linspace
	linspace.resize(num_nodes);
	for (int i = 0; i < num_nodes; ++i)
	{
		linspace[i] = start_val + delta * i;
	}
}

// Function to build log linspace if the input values is sorted
void
generate_log_linspace(
	Scalar start_val,
	Scalar end_val,
	int num_nodes,
	std::vector<Scalar>& linspace
) {
	// Skip ill defined cases
	if (num_nodes < 2) return;

	// Compute logarithmic start, end, and difference
	Scalar log_start = log(start_val);
	Scalar log_end = log(end_val);
	Scalar log_diff = log_end - log_start;

	// Compute logarithmic distance between values
	Scalar delta = (log_diff) / (num_nodes - 1);
	
	// Build linspace
	linspace.resize(num_nodes);
	for (int i = 0; i < num_nodes; ++i)
	{
		linspace[i] = exp(log_start + delta * i);
	}
}

void
angle_convergence_comparison(int argc, char *argv[])
{
  spdlog::set_level(spdlog::level::debug);
  assert(argc > 2);
  std::string output_dir = argv[1];
  int num_iterations = std::stod(argv[2]);
  std::filesystem::create_directories(output_dir);
	int num_tests = 100;
	Scalar start_angle = M_PI;
	Scalar end_angle = 1e-8;
	Scalar start_height = 1;
	Scalar end_height = 1e20;
	std::string output_filename;

	// Initialize data structures	
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector<Scalar> Th_hat;
	Mesh<Scalar> m;
	std::vector<int> vtx_reindex;
	VectorX reduced_metric_coords, reduced_metric_target, reduced_metric_init;

	// Set optimization parameters
	auto proj_params = std::make_shared<ProjectionParameters>();
	auto opt_params = std::make_shared<OptimizationParameters>();
	opt_params->num_iter = num_iterations;
	opt_params->min_ratio = 0;
	opt_params->max_beta = 1.0;
#ifdef MULTIPRECISION
	proj_params->error_eps = 1e-55;
#else
	proj_params->error_eps = 1e-10;
#endif

	// Iterate over target angles to compute convergence ratios after optimization
	std::vector<Scalar> angles;
	generate_log_linspace(start_angle, end_angle, num_tests, angles);
	generate_tet_mesh(start_height, V, F);
	std::vector<Scalar> angle_convergence_ratios(num_tests);
	for (int i = 0; i < num_tests; ++i)
	{
		// Get tet angles
		Scalar angle = angles[i];
		generate_tet_target_angles(angle, Th_hat);
		generate_initial_mesh(V, F, Th_hat, m, vtx_reindex, reduced_metric_coords);
		normalize_penner_coordinates(reduced_metric_coords, reduced_metric_target); 
		reduced_metric_init = 0.0 * reduced_metric_target;
		reduced_metric_init[0] = -3.0;
		reduced_metric_init[1] = 3.0;

		// Optimize the metric
		VectorX optimized_reduced_metric_coords;
		OptimizationLog log;
		optimize_metric_log(m,
										reduced_metric_target,
										reduced_metric_init,
										optimized_reduced_metric_coords,
										log,
										proj_params,
										opt_params);
		angle_convergence_ratios[i] = log.convergence_ratio;
	}

	// Iterate over initial heights to compute convergence ratios after optimization
	std::vector<Scalar> heights;
	generate_log_linspace(start_height, end_height, num_tests, heights);
	generate_tet_target_angles(start_angle, Th_hat);
	std::vector<Scalar> height_convergence_ratios(num_tests);
	for (int i = 0; i < num_tests; ++i)
	{
		// Generate mesh
		Scalar height = heights[i];
		generate_tet_mesh(height, V, F);
		generate_initial_mesh(V, F, Th_hat, m, vtx_reindex, reduced_metric_coords);
		normalize_penner_coordinates(reduced_metric_coords, reduced_metric_target); 
		reduced_metric_init = 0.0 * reduced_metric_target;
		reduced_metric_init[0] = -3.0;
		reduced_metric_init[1] = 3.0;

		// Optimize the metric
		VectorX optimized_reduced_metric_coords;
		OptimizationLog log;
		optimize_metric_log(m,
										reduced_metric_target,
										reduced_metric_init,
										optimized_reduced_metric_coords,
										log,
										proj_params,
										opt_params);
		height_convergence_ratios[i] = log.convergence_ratio;
	}

	// Print results
	spdlog::info(
		"Convergence ratios for target angles {} are {}",
		formatted_vector<Scalar>(angles),
		formatted_vector<Scalar>(angle_convergence_ratios)
	);
	spdlog::info(
		"Convergence ratios for initial heights {} are {}",
		formatted_vector<Scalar>(heights),
		formatted_vector<Scalar>(height_convergence_ratios)
	);
	output_filename = join_path(output_dir, "target_angles_"+std::to_string(num_iterations));
	write_vector(angles, output_filename);
	output_filename = join_path(output_dir, "angle_convergence_ratios_"+std::to_string(num_iterations));
	write_vector(angle_convergence_ratios, output_filename);
	output_filename = join_path(output_dir, "initial_heights_"+std::to_string(num_iterations));
	write_vector(heights, output_filename);
	output_filename = join_path(output_dir, "height_convergence_ratios_"+std::to_string(num_iterations));
	write_vector(height_convergence_ratios, output_filename);
}

void
angle_convergence_analysis(int argc, char *argv[])
{
  spdlog::set_level(spdlog::level::debug);
  assert(argc > 2);
  std::string output_dir = argv[1];
  int num_iterations = std::stod(argv[2]);
  std::filesystem::create_directories(output_dir);
	int num_tests = 0;
	Scalar start_angle = M_PI;
	Scalar end_angle = 1e-8;
	Scalar start_height = 1;
	Scalar end_height = 1e20;
	std::string output_filename;

	// Initialize data structures	
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector<Scalar> Th_hat;
	Mesh<Scalar> m;
	std::vector<int> vtx_reindex;
	VectorX reduced_metric_coords, reduced_metric_target, reduced_metric_init;

	// Set optimization parameters
	auto proj_params = std::make_shared<ProjectionParameters>();
	auto opt_params = std::make_shared<OptimizationParameters>();
	opt_params->num_iter = num_iterations;
	opt_params->min_ratio = 0;
	opt_params->max_beta = 1.0;
#ifdef MULTIPRECISION
	proj_params->error_eps = 1e-20;
#else
	proj_params->error_eps = 1e-10;
#endif

	// Generate test mesh for energy plot
	Scalar height = 1.0;
	Scalar angle = 1e-6;
	generate_tet_mesh(height, V, F);
	generate_tet_target_angles(angle, Th_hat);
	generate_initial_mesh(V, F, Th_hat, m, vtx_reindex, reduced_metric_coords);
	normalize_penner_coordinates(reduced_metric_coords, reduced_metric_target); 
	reduced_metric_init = 0.0 * reduced_metric_target;
	reduced_metric_init[0] = -3.0;
	reduced_metric_init[1] = 3.0;

	// Compute shear dual basis
  VectorX shear_basis_coords_init, scale_factors_init;
	MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
	compute_shear_dual_basis(m, shear_basis_matrix, independent_edges);

	// Optimize the metric
	VectorX optimized_reduced_metric_coords = reduced_metric_init;
	OptimizationLog log;
	opt_params->num_iter = 1;
	std::vector<Scalar> x_coords(num_iterations);
	std::vector<Scalar> y_coords(num_iterations);
	for (int i = 0; i < num_iterations; ++i)
	{
		optimize_metric_log(m,
										reduced_metric_target,
										optimized_reduced_metric_coords,
										optimized_reduced_metric_coords,
										log,
										proj_params,
										opt_params);
		spdlog::info("Optimized coordinates are {}", optimized_reduced_metric_coords);
		spdlog::info("Energy is {}", log.energy);
		spdlog::info("Convergence ratio is {}", log.convergence_ratio);
		spdlog::info("Coordinate sum is {}", optimized_reduced_metric_coords.sum());

		// Compute shear basis coordinates
		compute_shear_basis_coordinates(m, optimized_reduced_metric_coords, shear_basis_matrix, shear_basis_coords_init, scale_factors_init);
		spdlog::info("Initial coordinates are {}", shear_basis_coords_init);
		x_coords[i] = shear_basis_coords_init[0];
		y_coords[i] = shear_basis_coords_init[1];
	}
	spdlog::info(
		"Shear coordinates per iteration are {} and {}",
		formatted_vector<Scalar>(x_coords),
		formatted_vector<Scalar>(y_coords)
	);
	opt_params->num_iter = num_iterations;
	output_filename = join_path(output_dir, "x_coords_"+std::to_string(num_iterations));
	write_vector(x_coords, output_filename);
	output_filename = join_path(output_dir, "y_coords_"+std::to_string(num_iterations));
	write_vector(y_coords, output_filename);
	
	// Compute shear basis coordinates
	compute_shear_basis_coordinates(m, optimized_reduced_metric_coords, shear_basis_matrix, shear_basis_coords_init, scale_factors_init);
	spdlog::info("Initial coordinates are {}", shear_basis_coords_init);

  // Get maps for going between halfedge, edge, full, and reduced
  // representations as well as free and fixed edges and vertices
  ReductionMaps reduction_maps(m);

  // Build energy functions for given energy
  VectorX metric_target;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);
  EnergyFunctor opt_energy(m, metric_target, *opt_params);

  // Build matrix to map scale factors to edge coordinates
  MatrixX scale_factor_basis_matrix = conformal_scaling_matrix(m);

  // Build independent and dependent basis vectors by adding a global scaling term
  // to the shear basis and removing and arbitrary basis vector from the scale factors
  VectorX domain_coords;
  MatrixX constraint_domain_matrix, constraint_codomain_matrix;
  compute_optimization_domain(
    m,
    shear_basis_coords_init,
    scale_factors_init,
    shear_basis_matrix,
    scale_factor_basis_matrix,
    domain_coords,
    constraint_domain_matrix,
    constraint_codomain_matrix
  );
  spdlog::info(
    "Plotting {} coordinates with codomain of dimension {}",
    constraint_domain_matrix.cols(),
    constraint_codomain_matrix.cols()
  );
	Scalar x0 = domain_coords[0];
	Scalar y0 = domain_coords[1];

	// Iterate over grid
	Scalar range = 0.05;
  //Scalar range = 0.005;
	int num_grid_steps = 100;
	Scalar delta = 2.0 * range / static_cast<Scalar>(num_grid_steps - 1);
	Eigen::MatrixXd energy_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd average_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd original_average_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd original_min_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd original_max_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd original_energy_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd scale_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd error_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd domain_error_grid(num_grid_steps, num_grid_steps);
	Eigen::MatrixXd flip_grid(num_grid_steps, num_grid_steps);
	for (int i = 0; i < num_grid_steps; ++i)
	{
		for (int j = 0; j < num_grid_steps; ++j)
		{
			// Update metric
			Scalar dx = -range + delta * i;
			Scalar dy = -range + delta * j;
			domain_coords[0] = x0 + dx;
			domain_coords[1] = y0 + dy;

			// Get domain coordinate metric defined by the current coordinates
			VectorX domain_metric_coords = constraint_domain_matrix * domain_coords;

			// Get reduced coordinates from the domain coordinates
			VectorX reduced_domain_metric_coords;
			reduce_symmetric_function(reduction_maps.embed, domain_metric_coords, reduced_domain_metric_coords);

			// Project the domain metric to the constraint
			VectorX reduced_metric_coords;
			VectorX scale_factors;
			scale_factors.setZero(m.n_ind_vertices());
			project_to_constraint(
				m, reduced_domain_metric_coords, reduced_metric_coords, scale_factors, proj_params);

			// Normalize metric coordinates
			VectorX reduced_normalized_metric_coords;
			normalize_penner_coordinates(reduced_metric_coords, reduced_normalized_metric_coords);
			VectorX normalized_metric_coords, metric_coords;
			expand_reduced_function(
				reduction_maps.proj, reduced_metric_coords, metric_coords);
			expand_reduced_function(
				reduction_maps.proj, reduced_normalized_metric_coords, normalized_metric_coords);

			// Compute constraint
			VectorX constraint, domain_constraint;
			MatrixX J_constraint;
			std::vector<int> flip_seq;
			bool need_jacobian = false;
			bool use_edge_lengths = false;
			bool success = constraint_with_jacobian(m,
																							normalized_metric_coords,
																							constraint,
																							J_constraint,
																							flip_seq,
																							need_jacobian,
																							use_edge_lengths);
			constraint_with_jacobian(m,
															 domain_metric_coords,
															 domain_constraint,
															 J_constraint,
															 flip_seq,
															 need_jacobian,
															 use_edge_lengths);

			// Get the initial energy
			energy_grid(i, j) = double(opt_energy.energy(normalized_metric_coords));
			average_grid(i, j) = double(normalized_metric_coords.mean());
			original_energy_grid(i, j) = double(opt_energy.energy(metric_coords));
			original_average_grid(i, j) = double(metric_coords.mean());
			original_min_grid(i, j) = double(metric_coords.minCoeff());
			original_max_grid(i, j) = double(metric_coords.maxCoeff());
			scale_grid(i, j) = double(scale_factors.norm());
			domain_error_grid(i, j) = double(domain_constraint.norm());
			error_grid(i, j) = double(constraint.norm());
			flip_grid(i, j) = flip_seq.size();
		}
	}

	output_filename = join_path(output_dir, "energy_grid");
	write_matrix(energy_grid, output_filename);
	output_filename = join_path(output_dir, "average_grid");
	write_matrix(average_grid, output_filename);
	output_filename = join_path(output_dir, "original_energy_grid");
	write_matrix(original_energy_grid, output_filename);
	output_filename = join_path(output_dir, "original_min_grid");
	write_matrix(original_min_grid, output_filename);
	output_filename = join_path(output_dir, "original_max_grid");
	write_matrix(original_max_grid, output_filename);
	output_filename = join_path(output_dir, "original_average_grid");
	write_matrix(original_average_grid, output_filename);
	output_filename = join_path(output_dir, "error_grid");
	write_matrix(error_grid, output_filename);
	output_filename = join_path(output_dir, "domain_error_grid");
	write_matrix(domain_error_grid, output_filename);
	output_filename = join_path(output_dir, "scale_grid");
	write_matrix(scale_grid, output_filename);
	output_filename = join_path(output_dir, "flip_grid");
	write_matrix(flip_grid, output_filename);
}

int main(int argc, char *argv[])
{
#ifdef MULTIPRECISION
	spdlog::info("Using multiprecision");
	mpfr::mpreal::set_default_prec(200);
	mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
	mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif
	//angle_convergence_comparison(argc, argv);
	angle_convergence_analysis(argc, argv);
}