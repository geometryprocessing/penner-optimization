#include "common.hh"
#include "explicit_optimization.hh"
#include "targets.hh"
#include "energies.hh"
#include "optimization_interface.hh"
#include "constraint.hh"
#include "shear.hh"
#include "projection.hh"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

/// Unlike with Penner coordinates, any choice of shear coordinates gives a valid
/// metric satisfying the constraints with an energy. Thus, we can can plot the energy
/// for any coordinates. We plot the energies of the metrics in a two dimensional grid
/// around the initial metric

using namespace CurvatureMetric;

int main(int argc, char *argv[])
{
  spdlog::set_level(spdlog::level::debug);
  assert(argc > 3);
  std::string input_filename = argv[1];
	std::string Th_hat_filename = argv[2];
  std::string output_dir = argv[3];
  std::string energy_choice = argv[4];
  Scalar range = std::stod(argv[5]);
  std::filesystem::create_directories(output_dir);
	int num_grid_steps = 800;

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
	spdlog::info("Plotting energy for the mesh at {}", input_filename);
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);
	
	// Get input angles
	std::vector<Scalar> Th_hat;
	spdlog::info("Using cone angles at {}", Th_hat_filename);
	read_vector_from_file(Th_hat_filename, Th_hat);

	// Get initial mesh for optimization
	Mesh<Scalar> m;
	std::vector<int> vtx_reindex;
	VectorX reduced_metric_target;
	generate_initial_mesh(V, F, V, F, Th_hat, m, vtx_reindex, reduced_metric_target);
	VectorX reduced_metric_init = reduced_metric_target;
	PennerConeMetric cone_metric(m, reduced_metric_init);

	// Compute shear dual basis and the coordinates
  VectorX shear_basis_coords_init, scale_factors_init;
	MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
	compute_shear_dual_basis(m, shear_basis_matrix, independent_edges);
	compute_shear_basis_coordinates(m, reduced_metric_init, shear_basis_matrix, shear_basis_coords_init, scale_factors_init);
	spdlog::info("Initial coordinates are {}", shear_basis_coords_init);

  // Get maps for going between halfedge, edge, full, and reduced
  // representations as well as free and fixed edges and vertices
  ReductionMaps reduction_maps(m);

  // Build energy functions for given energy
	auto proj_params = std::make_shared<ProjectionParameters>();
	auto opt_params = std::make_shared<OptimizationParameters>();
	opt_params->energy_choice = energy_choice;
  VectorX metric_target;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);
  EnergyFunctor opt_energy(cone_metric, metric_target, *opt_params);

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
	Scalar delta = 2.0 * range / static_cast<Scalar>(num_grid_steps - 1);
	Eigen::MatrixXd energy_grid(num_grid_steps, num_grid_steps);
	for (int i = 0; i < num_grid_steps; ++i)
	{
		for (int j = 0; j < num_grid_steps; ++j)
		{
			// Update metric
			Scalar dx = -range + delta * i;
			Scalar dy = -range + delta * j;
			domain_coords[0] = x0 + dx;
			domain_coords[1] = y0 + dy;

			// Compute the energy for the shear metric coordinates
			Scalar energy = compute_domain_coordinate_energy(
					cone_metric,
					reduction_maps,
					opt_energy,
					domain_coords,
					constraint_domain_matrix,
					proj_params);
			energy_grid(i, j) = double(energy);
		}
	}

	// Write the output
	std::string output_filename = join_path(output_dir, "energy_grid_"+energy_choice + "_range_" + argv[5]);
	write_matrix(energy_grid, output_filename);
}
