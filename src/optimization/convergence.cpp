#include "convergence.hh"

#include "projection.hh"
#include "constraint.hh"
#include "implicit_optimization.hh"
#include "explicit_optimization.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void
compute_descent_directions(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  VectorX &gradient,
  VectorX &descent_direction,
  VectorX &projected_descent_direction 
) {
	// Get reduction maps
  ReductionMaps reduction_maps(m, opt_params.fix_bd_lengths);

	// Get optimization energy
  VectorX metric_target, metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);
  EnergyFunctor opt_energy(m, metric_target, opt_params);

	// Compute unconstrained descent direction
  VectorX prev_gradient(0);
  VectorX prev_descent_direction(0);
	std::string direction_choice = "gradient";
	compute_descent_direction(
		reduced_metric_coords,
		reduced_metric_target,
    prev_gradient,
    prev_descent_direction,
		reduction_maps,
		opt_energy,
		gradient,
		descent_direction,
		direction_choice
	);

	// Constrain descent direction
	constrain_descent_direction(m,
															reduced_metric_coords,
															descent_direction,
															reduction_maps,
															opt_params,
															projected_descent_direction);
  spdlog::info("Unconstrained descent direction found with norm {}", descent_direction.norm());
  spdlog::info("Constrained descent direction found with norm {}", projected_descent_direction.norm());
}

Scalar
compute_metric_convergence_ratio(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params
) {
  // Compute descent directions
  VectorX gradient;
  VectorX descent_direction;
  VectorX projected_descent_direction;
  compute_descent_directions(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    gradient,
    descent_direction,
    projected_descent_direction
  );

  return compute_convergence_ratio(descent_direction, projected_descent_direction);
}

void
compute_direction_energy_values(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& direction,
  const VectorX& step_sizes,
  VectorX& unprojected_energies,
  VectorX& projected_energies
) {
  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Expand the target metric coordinates to the doubled surface
  VectorX metric_target;
  expand_reduced_function(
    proj, reduced_metric_target, metric_target);

  // Build energy functions for given energy
  EnergyFunctor opt_energy(m, metric_target, opt_params);

  // Compute energies at step sizes
  int num_nodes = step_sizes.size();
  unprojected_energies.resize(num_nodes);
  projected_energies.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i)
  {
    // Compute unprojected step
    Scalar step_size = step_sizes[i];
    VectorX reduced_line_step_metric_coords = reduced_metric_coords + step_size * direction;

    // Compute the unprojected energy
    VectorX line_step_metric_coords;
    expand_reduced_function(proj,
                            reduced_line_step_metric_coords,
                            line_step_metric_coords);
    unprojected_energies[i] = opt_energy.energy(line_step_metric_coords);

    // Project to the constraint
    VectorX u;
    u.setZero(m.n_ind_vertices());
    VectorX reduced_projected_metric_coords;
    project_to_constraint(m,
                          reduced_line_step_metric_coords,
                          reduced_projected_metric_coords,
                          u,
                          proj_params);

    // Compute the projected energy
    VectorX projected_metric_coords;
    expand_reduced_function(proj,
                            reduced_projected_metric_coords,
                            projected_metric_coords);
    projected_energies[i] = opt_energy.energy(projected_metric_coords);
  }
}

void
compute_projected_descent_direction_energy_values(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& step_sizes,
  VectorX& unprojected_energies,
  VectorX& projected_energies
) {
  // Compute descent directions
  VectorX gradient;
  VectorX descent_direction;
  VectorX projected_descent_direction;
  compute_descent_directions(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    gradient,
    descent_direction,
    projected_descent_direction
  );

  // Compute energy values along the projected descent direction
  compute_direction_energy_values(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    proj_params,
    projected_descent_direction,
    step_sizes,
    unprojected_energies,
    projected_energies
  );
}

void
compute_projected_descent_direction_stability_values(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& step_sizes,
  VectorX& max_errors,
  VectorX& norm_changes_in_metric_coords,
  VectorX& convergence_ratios,
  VectorX& gradient_signs
) {
  // Compute descent directions
  VectorX gradient;
  VectorX descent_direction;
  VectorX projected_descent_direction;
  compute_descent_directions(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    gradient,
    descent_direction,
    projected_descent_direction
  );

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Expand the target metric coordinates to the doubled surface
  VectorX metric_target;
  expand_reduced_function(
    proj, reduced_metric_target, metric_target);

  // Build energy functions for given energy
  EnergyFunctor opt_energy(m, metric_target, opt_params);

  // Compute energies at step sizes
  int num_nodes = step_sizes.size();
  max_errors.resize(num_nodes);
  norm_changes_in_metric_coords.resize(num_nodes);
  convergence_ratios.resize(num_nodes);
  gradient_signs.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i)
  {
    // Compute unprojected step
    Scalar step_size = step_sizes[i];
    VectorX reduced_line_step_metric_coords = reduced_metric_coords + step_size * projected_descent_direction;

    // Project to the constraint
    VectorX u;
    u.setZero(m.n_ind_vertices());
    VectorX reduced_updated_metric_coords;
    project_to_constraint(m,
                          reduced_line_step_metric_coords,
                          reduced_updated_metric_coords,
                          u,
                          proj_params);
    VectorX updated_metric_coords;
    expand_reduced_function(proj,
                            reduced_updated_metric_coords,
                            updated_metric_coords);

    // Compute constraint values
    VectorX constraint;
    MatrixX J_constraint;
    std::vector<int> flip_seq;
    bool need_jacobian = false;
    bool use_edge_lengths = false;
    constraint_with_jacobian(m,
                            updated_metric_coords,
                            constraint,
                            J_constraint,
                            flip_seq,
                            need_jacobian,
                            use_edge_lengths);

    // Compute change in metric coords
    VectorX change_in_metric_coords = reduced_updated_metric_coords - reduced_metric_coords;

    // Compute descent directions
    VectorX step_gradient;
    VectorX step_descent_direction;
    VectorX step_projected_descent_direction;
    compute_descent_directions(
      m,
      reduced_updated_metric_coords,
      reduced_metric_target,
      opt_params,
      step_gradient,
      step_descent_direction,
      step_projected_descent_direction
    );

    // Compute numerics
    max_errors[i] = constraint.maxCoeff();
    norm_changes_in_metric_coords[i] = change_in_metric_coords.norm();
    convergence_ratios[i] = compute_convergence_ratio(step_descent_direction, step_projected_descent_direction);
    gradient_signs[i] = -step_projected_descent_direction.dot(projected_descent_direction);
  }
}

void
compute_domain_direction_energy_values(
	const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const VectorX& reduced_metric_target,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  const std::shared_ptr<OptimizationParameters> opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& direction,
  const VectorX& step_sizes,
  VectorX& energies
) {
  // Expand the target metric coordinates to the doubled surface
  VectorX metric_target;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);

  // Build energy functions for given energy
  EnergyFunctor opt_energy(m, metric_target, *opt_params);

  // Compute energies along descent direction
  int num_steps = step_sizes.size();
  energies.resize(num_steps);
  for (int i = 0; i < num_steps; ++i)
  {
    Scalar step_size = step_sizes[i];
    VectorX line_step_domain_coords = domain_coords + step_size * direction;

    // Compute the gradient for the shear metric coordinates
    Scalar energy;
    VectorX gradient;
    compute_domain_coordinate_energy_with_gradient(
      m,
      reduction_maps,
      opt_energy,
      line_step_domain_coords,
      constraint_domain_matrix,
      constraint_codomain_matrix,
      proj_params,
      opt_params,
      energy,
      gradient);
    energies[i] = energy;
  }
}

#ifdef PYBIND

std::tuple<
  VectorX, // unprojected_energies
  VectorX // projected_energies
>
compute_direction_energy_values_pybind(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& direction,
  const VectorX& step_sizes
){
  VectorX unprojected_energies;
  VectorX projected_energies;
  compute_direction_energy_values(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    proj_params,
    direction,
    step_sizes,
    unprojected_energies,
    projected_energies
  );
  return std::make_tuple(unprojected_energies, projected_energies);
}

std::tuple<
  VectorX, // unprojected_energies
  VectorX // projected_energies
>
compute_projected_descent_direction_energy_values_pybind(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& step_sizes
) {
  VectorX unprojected_energies;
  VectorX projected_energies;
  compute_projected_descent_direction_energy_values(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    proj_params,
    step_sizes,
    unprojected_energies,
    projected_energies
  );
  return std::make_tuple(unprojected_energies, projected_energies);
}

VectorX
compute_domain_direction_energy_values_pybind(
	const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const VectorX& reduced_metric_target,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  const std::shared_ptr<OptimizationParameters> opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& direction,
  const VectorX& step_sizes
) {
  VectorX energies;
  compute_domain_direction_energy_values(
    m,
    reduction_maps,
    reduced_metric_target,
    domain_coords,
    constraint_domain_matrix,
    constraint_codomain_matrix,
    opt_params,
    proj_params,
    direction,
    step_sizes,
    energies
  );
  return energies;
}

std::tuple<
  VectorX, // max_errors,
  VectorX, // norm_changes_in_metric_coords,
  VectorX, // convergence_ratios,
  VectorX  // gradient_signs
>
compute_projected_descent_direction_stability_values_pybind(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
	const OptimizationParameters& opt_params,
  const std::shared_ptr<ProjectionParameters> proj_params,
  const VectorX& step_sizes
) {
  VectorX max_errors;
  VectorX norm_changes_in_metric_coords;
  VectorX convergence_ratios;
  VectorX gradient_signs;
  compute_projected_descent_direction_stability_values(
    m,
    reduced_metric_coords,
    reduced_metric_target,
    opt_params,
    proj_params,
    step_sizes,
    max_errors,
    norm_changes_in_metric_coords,
    convergence_ratios,
    gradient_signs
  );
  return std::make_tuple(
    max_errors,
    norm_changes_in_metric_coords,
    convergence_ratios,
    gradient_signs
  );
}

#endif

}
