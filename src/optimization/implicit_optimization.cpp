#include "implicit_optimization.hh"

#include "area.hh"
#include "targets.hh"
#include "constraint.hh"
#include "embedding.hh"
#include "energies.hh"
#include "logging.hh"
#include "globals.hh"
#include "logging.hh"
#include "projection.hh"
#include "nonlinear_optimization.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

Scalar compute_convergence_ratio(
  const VectorX& unconstrained_descent_direction,
  const VectorX& constrained_descent_direction
) {
  Scalar unconstrained_descent_norm = unconstrained_descent_direction.norm();
  Scalar constrained_descent_norm = constrained_descent_direction.norm();

  // Compute the convergence ratio, returning just the constrained norm if the unconstrained norm is 0.0
  Scalar convergence_ratio;
  if (unconstrained_descent_norm > 0) {
    convergence_ratio = constrained_descent_norm / unconstrained_descent_norm;
  } else {
    convergence_ratio = constrained_descent_norm;
  }

  return convergence_ratio;
}

void
initialize_data_log(
  const std::filesystem::path& data_log_path
) {
  // Open file
  spdlog::info("Writing data to {}", data_log_path);
  std::ofstream output_file(data_log_path, std::ios::out | std::ios::trunc);

  // Write header
  output_file << "num_iter,";
  output_file << "step_size,";
  output_file << "energy,";
  output_file << "max_error,";
  output_file << "convergence_ratio,";
  output_file << "max_change_in_metric_coords,";
  output_file << "max_total_change_in_metric_coords,";
  output_file << "actual_to_unconstrained_direction_ratio,";
  output_file << "max_constrained_descent_direction,";
  output_file << std::endl;

  // Close file
  output_file.close();
}

void
update_data_log(
  OptimizationLog &log,
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& updated_reduced_metric_coords,
  const VectorX& reduced_metric_coords,
  const VectorX& reduced_metric_target,
  const VectorX& unconstrained_descent_direction,
  const VectorX& constrained_descent_direction,
  Scalar convergence_ratio,
  std::shared_ptr<OptimizationParameters> opt_params
) {
  // Expand reduced functions
  VectorX updated_metric_coords, metric_coords, metric_target;
  expand_reduced_function(
    reduction_maps.proj, updated_reduced_metric_coords, updated_metric_coords);
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);

  // Compute constraint values
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = false;
  constraint_with_jacobian(m,
                           updated_metric_coords,
                           constraint,
                           J_constraint,
                           flip_seq,
                           need_jacobian,
                           opt_params->use_edge_lengths);

  // Compute change in metric coords from target and previous iteration
  VectorX change_in_metric_coords = updated_reduced_metric_coords - reduced_metric_coords;
  VectorX total_change_in_metric_coords = updated_reduced_metric_coords - reduced_metric_target;

  // Compute numerics
  log.energy = opt_energy.energy(updated_metric_coords);
  log.error = sup_norm(constraint);
  log.num_flips = flip_seq.size();
  log.convergence_ratio = convergence_ratio;
  log.max_change_in_metric_coords = sup_norm(change_in_metric_coords);
  log.max_total_change_in_metric_coords = sup_norm(total_change_in_metric_coords);
  log.actual_to_unconstrained_direction_ratio = change_in_metric_coords.norm() / unconstrained_descent_direction.norm();
  log.max_constrained_descent_direction = sup_norm(constrained_descent_direction);
}

void write_checkpoint(
  const std::string& output_dir,
  int iter,
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const VectorX& reduced_metric_target,
  const VectorX& reduced_metric_coords,
  const VectorX& reduced_line_step_metric_coords,
  const VectorX& updated_reduced_metric_coords,
  const VectorX& unconstrained_descent_direction,
  const VectorX& constrained_descent_direction,
  const VectorX& u,
  bool use_edge_lengths=true
) {
  std::string checkpoint_filename;

  checkpoint_filename = join_path(output_dir, "target_lambdas");
  write_vector(reduced_metric_target, checkpoint_filename, 80);

  checkpoint_filename = join_path(output_dir, "initial_lambdas_" + std::to_string(iter));
  write_vector(reduced_metric_coords, checkpoint_filename, 80);

  checkpoint_filename = join_path(output_dir, "unconstrained_descent_direction_" + std::to_string(iter));
  write_vector(unconstrained_descent_direction, checkpoint_filename, 80);

  checkpoint_filename = join_path(output_dir, "constrained_descent_direction_" + std::to_string(iter));
  write_vector(constrained_descent_direction, checkpoint_filename, 80);

  // Compute the constraint function and its Jacobian
  VectorX metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  constraint_with_jacobian(m,
                            metric_coords,
                            constraint,
                            J_constraint,
                            flip_seq,
                            need_jacobian,
                            use_edge_lengths);
  MatrixX J_restricted_constraint;
  compute_submatrix(J_constraint * reduction_maps.projection,
                    reduction_maps.free_v,
                    reduction_maps.free_e,
                    J_restricted_constraint);
  checkpoint_filename = join_path(output_dir, "constraint_" + std::to_string(iter));
  write_vector(constraint, checkpoint_filename);

  checkpoint_filename = join_path(output_dir, "constraint_jacobian_" + std::to_string(iter));
  write_sparse_matrix(J_restricted_constraint, checkpoint_filename, "matlab");

  //MatrixX projection_matrix = compute_descent_direction_projection_matrix(J_restricted_constraint);
  //checkpoint_filename = join_path(output_dir, "projection_matrix_" + std::to_string(iter));
  //write_sparse_matrix(projection_matrix, checkpoint_filename, "matlab");

  checkpoint_filename = join_path(output_dir, "unprojected_lambdas_" + std::to_string(iter));
  write_vector(reduced_line_step_metric_coords, checkpoint_filename, 80);

  checkpoint_filename = join_path(output_dir, "scale_factors_" + std::to_string(iter));
  write_vector(u, checkpoint_filename, 80);

  MatrixX B = conformal_scaling_matrix(m);
  checkpoint_filename = join_path(output_dir, "scaling_matrix");
  write_sparse_matrix(B, checkpoint_filename, "matlab");

  checkpoint_filename = join_path(output_dir, "lambdas_" + std::to_string(iter));
  write_vector(updated_reduced_metric_coords, checkpoint_filename, 80);
}

void
write_data_log_entry(
  const OptimizationLog &log,
  const std::filesystem::path& data_log_path
) {
  // Open file in append mode
  spdlog::info("Writing data iteration to {}", data_log_path);
  std::ofstream output_file(data_log_path, std::ios::out | std::ios::app);

  // Write iteration row
  output_file << log.num_iterations << ",";
  output_file << std::fixed << std::setprecision(17) << log.beta << ",";
  output_file << std::fixed << std::setprecision(17) << log.energy << ",";
  output_file << std::fixed << std::setprecision(17) << log.error << ",";
  output_file << std::fixed << std::setprecision(17) << log.convergence_ratio << ",";
  output_file << std::fixed << std::setprecision(17) << log.max_change_in_metric_coords << ",";
  output_file << std::fixed << std::setprecision(17) << log.max_total_change_in_metric_coords << ",";
  output_file << std::fixed << std::setprecision(17) << log.actual_to_unconstrained_direction_ratio << ",";
  output_file << std::fixed << std::setprecision(17) << log.max_constrained_descent_direction << ",";
  output_file << std::endl;

  // Close file
  output_file.close();
}

void
compute_descent_direction(const VectorX& reduced_metric_coords,
                          const VectorX& reduced_metric_target,
                          const VectorX& prev_gradient,
                          const VectorX& prev_descent_direction,
                          const ReductionMaps& reduction_maps,
                          const EnergyFunctor& opt_energy,
                          VectorX& gradient,
                          VectorX& descent_direction,
                          std::string direction_choice)
{
  // Expand the metric coordinates to the doubled surface
  VectorX metric_coords, metric_target;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);

  // Compute gradient from the functor and zero out the variables that aren't
  // free
  VectorX full_gradient = opt_energy.gradient(metric_coords);
  reduce_symmetric_function(reduction_maps.embed, full_gradient, gradient);
  mask_subset(reduction_maps.fixed_e, gradient);

  // Compute the gradient descent direction
  if (direction_choice == "gradient") {
    descent_direction = -gradient;
  }
  // Compute the conjugate gradient descent direction
  // WARNING: The theory on this choice is dubious
  else if (direction_choice == "conjugate_gradient")
  {
    // Check if the previous gradient and descent direction are trivial
    if ((prev_gradient.size() == 0) ||
        (prev_descent_direction.size() == 0)) {
      descent_direction = -gradient;
    }
    // Compute the conjugate gradient direction from the previous data and the
    // new gradient
    else {
      std::string coefficient = "polak_ribiere"; // Popular choice; could be made a parameter
      compute_conjugate_gradient_direction(
        gradient,
        prev_gradient,
        prev_descent_direction,
        descent_direction,
        coefficient
      );
    }
  }
  // Directly use the minimizer of the energy without concern for the constraint
  // manifold
  // WARNING: The theory on this choice is only sound when paired with the optimal
  // tangent space projection and not the standard orthogonal projection
  else if (direction_choice == "unconstrained_minimizer") {
    descent_direction = -(reduced_metric_coords - reduced_metric_target);
  }
  // By default, use the appropriately sized zero vector
  else {
    descent_direction = VectorX::Constant(gradient.size(), 0.0);
  }
}

void
constrain_descent_direction(const Mesh<Scalar>& m,
                            const VectorX& reduced_metric_coords,
                            const VectorX& descent_direction,
                            const ReductionMaps& reduction_maps,
                            const OptimizationParameters& opt_params,
                            VectorX& projected_descent_direction)
{
  size_t num_reduced_edges = reduction_maps.num_reduced_edges;
  assert(reduced_metric_coords.size() == num_reduced_edges);
  assert(descent_direction.size() == num_reduced_edges);

  // Expand the metric coordinates to the doubled surface
  VectorX metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);

  // Compute the constraint function and its Jacobian
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  bool success = constraint_with_jacobian(m,
                                          metric_coords,
                                          constraint,
                                          J_constraint,
                                          flip_seq,
                                          need_jacobian,
                                          opt_params.use_edge_lengths);
  if (!success)
  {
    spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
  }

  // If using regular length values, apply the change of coordinates from the
  // log values
  if (!opt_params.use_log) {
    MatrixX J_length;
    length_jacobian(reduced_metric_coords, J_length);
    J_constraint = J_constraint * J_length;
  }

  // Remove the fixed degrees of freedom
  VectorX variable_descent_direction, variable_constraint;
  MatrixX restricted_J_constraint = J_constraint * reduction_maps.projection;
  MatrixX variable_J_constraint;
  compute_subset(
    descent_direction, reduction_maps.free_e, variable_descent_direction);
  compute_subset(constraint, reduction_maps.free_v, variable_constraint);
  compute_submatrix(restricted_J_constraint,
                    reduction_maps.free_v,
                    reduction_maps.free_e,
                    variable_J_constraint);

  // Project the descent direction to the constraint tangent plane
  VectorX projected_variable_descent_direction = project_descent_direction(
    variable_descent_direction, variable_constraint, variable_J_constraint);

  // Expand projected descent direction on the free edges to the full edge list
  projected_descent_direction.setZero(num_reduced_edges);
  write_subset(projected_variable_descent_direction,
               reduction_maps.free_e,
               projected_descent_direction);
}

// Generate the lagrangian for optimizing a quadratic energy in a constrained space orthogonal to
// the column space of J_constraint
void
generate_hessian_lagrangian_system(
	const MatrixX& hessian,
	const MatrixX& J_constraint,
	const VectorX& gradient,
	MatrixX& hessian_lagrangian,
	VectorX& rhs
) {
	// Get sizes
	int n = hessian.rows();
	int m = J_constraint.rows();

  // Initialize matrix entry list
  std::vector<T> tripletList;
  tripletList.reserve(hessian.nonZeros() + 2 * J_constraint.nonZeros());

  // Add hessian as upper block
  for (int k = 0; k < hessian.outerSize(); ++k)
  {
    for (MatrixX::InnerIterator it(hessian, k); it; ++it)
    {
      int row = it.row();
      int col = it.col();
      Scalar value = it.value();
      tripletList.push_back(T(row, col, value));
    }
  }

  // Add constraint Jacobian and transpose
  for (int k = 0; k < J_constraint.outerSize(); ++k)
  {
    for (MatrixX::InnerIterator it(J_constraint, k); it; ++it)
    {
      int row = it.row();
      int col = it.col();
      Scalar value = it.value();
      tripletList.push_back(T(n + row, col, value));
      tripletList.push_back(T(col, n + row, value));
		}
  }

  // Build lagrangian matrix
  hessian_lagrangian.resize(n + m, n + m);
  hessian_lagrangian.reserve(tripletList.size());
  hessian_lagrangian.setFromTriplets(tripletList.begin(), tripletList.end());

  // Build right hand side
  rhs.setZero(n + m);
	rhs.head(n) = -gradient;
}

void
compute_optimal_tangent_space_descent_direction(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_coords,
	const VectorX& gradient,
	const VectorX& descent_direction,
	const ReductionMaps& reduction_maps,
  const OptimizationParameters& opt_params,
	const EnergyFunctor& opt_energy,
	VectorX& projected_descent_direction
) {
  // Get projection matrix
	const MatrixX& R = reduction_maps.projection;

  // Expand the metric coordinates to the doubled surface
  VectorX metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);

  // Remove the fixed degrees of freedom from the gradient and descent direction
	VectorX variable_gradient;
	compute_subset(gradient, reduction_maps.free_e, variable_gradient);
	VectorX variable_descent_direction;
	compute_subset(descent_direction, reduction_maps.free_e, variable_descent_direction);

  // Compute hessian with fixed degrees of freedom removed
  MatrixX hessian = opt_energy.hessian();
	MatrixX reduced_hessian = R.transpose() * (hessian * R);
	MatrixX variable_hessian; 
  compute_submatrix(reduced_hessian,
                    reduction_maps.free_e,
                    reduction_maps.free_e,
                    variable_hessian);

  // Compute the constraint function and its Jacobian
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  bool success = constraint_with_jacobian(m,
                                          metric_coords,
                                          constraint,
                                          J_constraint,
                                          flip_seq,
                                          need_jacobian,
                                          opt_params.use_edge_lengths);
  if (!success)
  {
    spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
  }

  // Restrict the constraint Jacobian to the variables
	MatrixX reduced_J_constraint = J_constraint * R;
  MatrixX variable_J_constraint;
  compute_submatrix(reduced_J_constraint,
                    reduction_maps.free_v,
                    reduction_maps.free_e,
                    variable_J_constraint);

  // Solve for optimal descent direction
	int num_variable_edges = variable_descent_direction.size();
	int num_reduced_edges = reduced_metric_coords.size();
	VectorX variable_projected_descent_direction;
	std::string method = "lagrangian";
	if (method == "lagrangian")
	{
    // Generate lagrangian and right hand side for this quadratic programming problem
		MatrixX hessian_lagrangian;
		VectorX rhs;
		generate_hessian_lagrangian_system(
			variable_hessian,
			variable_J_constraint,
			variable_gradient,
			hessian_lagrangian,
			rhs
		);

    // Solve for the optimal descent direction
		Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
		solver.compute(hessian_lagrangian);
		VectorX solution = solver.solve(rhs);
		variable_projected_descent_direction = solution.head(num_variable_edges);
	}
	if (method == "explicit_inverse")
	{
		// Get the Hessian inverse restricted to the variable degrees of freedom
		MatrixX hessian_inverse = opt_energy.hessian_inverse();
		MatrixX reduced_hessian_inverse = R.transpose() * (hessian_inverse * R);
		MatrixX variable_hessian_inverse; 
		compute_submatrix(reduced_hessian_inverse,
											reduction_maps.free_e,
											reduction_maps.free_e,
											variable_hessian_inverse);

		// Solve for the explicit correction to the descent direction
		MatrixX L = variable_J_constraint * (variable_hessian_inverse * variable_J_constraint.transpose());
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
		solver.compute(L);
		VectorX w = variable_J_constraint * variable_descent_direction;
		VectorX mu = solver.solve(w);
		VectorX nu = variable_J_constraint.transpose() * mu;
		VectorX correction = variable_hessian_inverse * nu;

    // Compute the optimal descent direction
		variable_projected_descent_direction = variable_descent_direction - correction;
	}
	if (method == "implicit_inverse")
	{
		// Solve for correction to the descent direction using a solver for the Hessian
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> hessian_solver;
		hessian_solver.compute(variable_hessian);
		Eigen::SparseMatrix<Scalar, Eigen::ColMajor> rhs = variable_J_constraint.transpose();
		Eigen::SparseMatrix<Scalar, Eigen::ColMajor> temp_matrix = hessian_solver.solve(rhs);
		MatrixX L = variable_J_constraint * temp_matrix;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
		solver.compute(L);
		VectorX w = variable_J_constraint * variable_descent_direction;
		VectorX mu = solver.solve(w);
		VectorX nu = variable_J_constraint.transpose() * mu;
		VectorX correction = hessian_solver.solve(nu);

    // Compute the optimal descent direction
		variable_projected_descent_direction = variable_descent_direction - correction;
	}

  // Expand projected descent direction on the free edges to the full edge list
  projected_descent_direction.setZero(num_reduced_edges);
  write_subset(variable_projected_descent_direction,
               reduction_maps.free_e,
               projected_descent_direction);
}

// Check if the line step is valid and stable
bool
check_line_step_success(
  const Mesh<Scalar>& m,
  const VectorX &reduced_line_step_metric_coords,
  const ReductionMaps& reduction_maps,
  std::shared_ptr<OptimizationParameters> opt_params,
  const EnergyFunctor& opt_energy
) {
  // Expand the new metric coordinates to the doubled surface
  VectorX line_step_metric_coords;
  expand_reduced_function(reduction_maps.proj,
                          reduced_line_step_metric_coords,
                          line_step_metric_coords);
  Scalar opt_energy_step = opt_energy.energy(line_step_metric_coords);
  spdlog::get("optimize_metric")->info("Energy after line step is {}", opt_energy_step);

  // Compute the contraint values after the line step
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = false;
  bool success = constraint_with_jacobian(m,
                                          line_step_metric_coords,
                                          constraint,
                                          J_constraint,
                                          flip_seq,
                                          need_jacobian,
                                          opt_params->use_edge_lengths);

  // Shrink step size if the constraint is not computed correctly due to a
  // triangle inequality violation
  if (!success) {
    spdlog::get("optimize_metric")->info("Reducing step size due to triangle inequality violation while computing the constraint function");
    return false;
  }

  // Check if the maximum constraint error is too large
  Scalar max_line_search_constraint = sup_norm(constraint);
  spdlog::get("optimize_metric")->info("Maximum constraint error after line step is {}", max_line_search_constraint);
  if (max_line_search_constraint > opt_params->max_angle) {
    spdlog::get("optimize_metric")->info("Max angle error {} larger than bound {}",
      max_line_search_constraint,
                  opt_params->max_angle);
    return false;
  }

  // Success otherwise
  return true;
}

// Check if the projection to the constraint manifold is valid and stable
bool
check_projection_success(
  const Mesh<Scalar>& m,
  const VectorX &reduced_metric_coords,
  const ReductionMaps& reduction_maps,
  std::shared_ptr<OptimizationParameters> opt_params,
  const Scalar& max_initial_constraint
) {
  // Expand the new metric coordinates to the doubled surface
  VectorX metric_coords;
  expand_reduced_function(reduction_maps.proj,
                          reduced_metric_coords,
                          metric_coords);

  // Compute the constraint values after the projection
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = false;
  bool success;
  success = constraint_with_jacobian(m,
                                      metric_coords,
                                      constraint,
                                      J_constraint,
                                      flip_seq,
                                      need_jacobian,
                                      opt_params->use_edge_lengths);

  // Shrink step size if the constraint is not computed correctly due to a
  // triangle inequality violation
  if (!success) {
    spdlog::get("optimize_metric")->info("Reducing step size due to triangle inequality violation while computing the constraint function");
    return false;
  }

  // Check the angle decrease condition
  Scalar max_constraint = constraint.maxCoeff();
  spdlog::get("optimize_metric")->info("Maximum constraint error after projection is {}", max_constraint);
  Scalar constraint_threshold = max_initial_constraint + opt_params->max_angle_incr;
  if (max_constraint > constraint_threshold)
  {
    spdlog::get("optimize_metric")->info("Reducing step size as the angle error {} > {} did not decrease from {} with tolerance {}",
                  max_constraint,
                  constraint_threshold,
                  max_initial_constraint,
                  opt_params->max_angle_incr);
    return false;
  }

  // Success otherwise
  return true;
}

// Check if the full step is sufficiently small for convergence
bool
check_convergence_progress(
  const VectorX &reduced_metric_coords,
  const ReductionMaps& reduction_maps,
  std::shared_ptr<OptimizationParameters> opt_params,
  const VectorX &descent_direction,
  const VectorX& gradient,
  const VectorX &constrained_gradient,
  const EnergyFunctor& opt_energy,
  const Scalar& opt_energy_init,
  const Scalar& convergence_ratio_init)
{
  // Expand the new metric coordinates to the doubled surface
  VectorX metric_coords;
  expand_reduced_function(reduction_maps.proj,
                          reduced_metric_coords,
                          metric_coords);

  // Check the energy decreases enough
  Scalar opt_energy_proj = opt_energy.energy(metric_coords);
  spdlog::get("optimize_metric")->info("Energy after projection is {}", opt_energy_proj);
  if ((opt_params->require_energy_decr) &&
      (opt_energy_proj > opt_energy_init * (1.0 + opt_params->max_energy_incr))) {
    spdlog::get("optimize_metric")->info("Reducing step size as the energy did not decrease");
    return false;
  }

  // Check the gradient projection on the descent direction is negative
  if ((opt_params->require_gradient_proj_negative) && (descent_direction.dot(constrained_gradient) > 0))
  {
    spdlog::get("optimize_metric")->info("Reducing step size as the gradient projection is positive");
    return false;
  }

  // Check the convergence ratio of the gradient at the step decreases enough
  Scalar convergence_ratio_proj = compute_convergence_ratio(gradient, constrained_gradient);
  if (convergence_ratio_proj > convergence_ratio_init * (1 + opt_params->max_ratio_incr))
  {
    spdlog::get("optimize_metric")->info("Reducing step size as the convergence ratio {} did not decrease from {} with tolerance {}",
                  convergence_ratio_proj,
                  convergence_ratio_init,
                  opt_params->max_ratio_incr);
    return false;
  }

  return true;
}

void
line_search_with_projection(const Mesh<Scalar>& m,
                            const VectorX& initial_reduced_metric_coords,
                            const VectorX& descent_direction,
                            const ReductionMaps& reduction_maps,
                            std::shared_ptr<ProjectionParameters> proj_params,
                            std::shared_ptr<OptimizationParameters> opt_params,
                            const EnergyFunctor& opt_energy,
                            Scalar& beta,
                            Scalar& convergence_ratio,
                            VectorX& reduced_line_step_metric_coords,
                            VectorX& u,
                            VectorX& reduced_metric_coords)
{
  // Expand the initial metric coordinates to the doubled surface
  VectorX initial_metric_coords;
  expand_reduced_function(
    reduction_maps.proj, initial_reduced_metric_coords, initial_metric_coords);

  // Get the initial optimization energy
  Scalar opt_energy_init = opt_energy.energy(initial_metric_coords);
  spdlog::get("optimize_metric")->info(
    "Attempting line search with step size {} from initial energy {}",
    beta,
    opt_energy_init
  );

  // Compute the initial constraint function and its Jacobian
  VectorX initial_constraint;
  MatrixX J_initial_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = false;
  bool success = constraint_with_jacobian(m,
                                          initial_metric_coords,
                                          initial_constraint,
                                          J_initial_constraint,
                                          flip_seq,
                                          need_jacobian,
                                          opt_params->use_edge_lengths);
  if (!success)
  {
    spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
  }
  Scalar max_initial_constraint = sup_norm(initial_constraint);

  // Save initial convergence ratio
  Scalar convergence_ratio_init = convergence_ratio;

  while (true) {
    // Break if the step size is too small
    if (beta < 1e-16) {
      spdlog::get("optimize_metric")->info("Beta {} too small to continue", beta);
      break;
    }

    // Make a line step in log space
    if (opt_params->use_log) {
      reduced_line_step_metric_coords =
        initial_reduced_metric_coords + beta * descent_direction;
    }
    // Make a line step in length space
    else 
    {
      size_t num_reduced_edges = reduction_maps.num_reduced_edges;
      reduced_line_step_metric_coords.resize(num_reduced_edges);

      // For each edge, go to length space, take a step, and then return to log space
      for (size_t Ei = 0; Ei < num_reduced_edges; ++Ei) {
        Scalar initial_length_coord =
          exp(initial_reduced_metric_coords[Ei] / 2.0);
        Scalar length_coord =
          initial_length_coord + beta * descent_direction[Ei];
        reduced_line_step_metric_coords[Ei] = 2 * log(length_coord);
      }
    }

    // Check if the line step is valid and reduce beta if not
    bool line_step_success = check_line_step_success(
      m,
      reduced_line_step_metric_coords,
      reduction_maps,
      opt_params,
      opt_energy
    );
    if (!line_step_success)
    {
      beta /= 2.0;
      continue;
    }

    // Project to the constraint
    u.setZero(m.n_ind_vertices());
    VectorX metric_coords;
    project_to_constraint(m,
                          reduced_line_step_metric_coords,
                          reduced_metric_coords,
                          u,
                          proj_params);
    expand_reduced_function(
      reduction_maps.proj, reduced_metric_coords, metric_coords);

    // Check if the projection was successful and reduce beta if not
    bool projection_success = check_projection_success(
      m,
      reduced_metric_coords,
      reduction_maps,
      opt_params,
      max_initial_constraint
    );
    if (!projection_success)
    {
      beta /= 2.0;
      continue;
    }

    // Compute the gradient and its orthogonal projection 
    VectorX full_gradient = opt_energy.gradient(metric_coords);
    VectorX gradient;
    reduce_symmetric_function(reduction_maps.embed, full_gradient, gradient);
    VectorX constrained_gradient;
    constrain_descent_direction(m,
                                reduced_metric_coords,
                                gradient,
                                reduction_maps,
                                *opt_params,
                                constrained_gradient);
    spdlog::info("Projected gradient is {}", descent_direction.dot(constrained_gradient));

    // Update the convergence ratio 
    convergence_ratio = compute_convergence_ratio(gradient, constrained_gradient);

    // Check if the convergence conditions were satisfied and reduce beta if not
    bool convergence_success = check_convergence_progress(
      reduced_metric_coords,
      reduction_maps,
      opt_params,
      descent_direction,
      gradient,
      constrained_gradient,
      opt_energy,
      opt_energy_init,
      convergence_ratio_init
    );
    if (!convergence_success)
    {
      beta /= 2.0;
      continue;
    }

    // Break the loop if no flag to reduce beta was hit
    break;
  }
}

bool
check_if_converged(const OptimizationParameters& opt_params,
                   Scalar convergence_ratio,
                   Scalar beta)
{
  // Check if the convergence ratio is suitably small
  if (convergence_ratio < opt_params.min_ratio) {
    spdlog::get("optimize_metric")
      ->info("Convergence ratio below termination threshold of {}",
             opt_params.min_ratio);
    return true;
  }

  // Check if beta is too small to continue
  if (beta < 1e-16) {
    spdlog::get("optimize_metric")
      ->info("Beta {} is below termination threshold of {}", beta, 1e-16);
    return true;
  }

  // Not converged otherwise
  return false;
}

void
optimize_metric_log(const Mesh<Scalar>& m,
                const VectorX& reduced_metric_target,
                const VectorX& reduced_metric_init,
                VectorX& optimized_reduced_metric_coords,
                OptimizationLog& log,
                std::shared_ptr<ProjectionParameters> proj_params,
                std::shared_ptr<OptimizationParameters> opt_params)
{
  // Build default parameters if none given
  if (proj_params == nullptr)
    proj_params = std::make_shared<ProjectionParameters>();
  if (opt_params == nullptr)
    opt_params = std::make_shared<OptimizationParameters>();

  // Extract relevant parameters for main optimization method
  int num_iter = opt_params->num_iter;
  Scalar beta_0 = opt_params->beta_0;
  std::string energy_choice = opt_params->energy_choice;
  std::string direction_choice = opt_params->direction_choice;
  std::string output_dir = opt_params->output_dir;
  bool use_optimal_projection = opt_params->use_optimal_projection;
  bool use_checkpoints = opt_params->use_checkpoints;
  bool use_edge_lengths = opt_params->use_edge_lengths;
  Scalar max_grad_range = opt_params->max_grad_range;

  // Log mesh data
  create_log(output_dir, "mesh_data");
  spdlog::get("mesh_data")->set_level(spdlog::level::trace);
  log_mesh_information(m, "mesh_data");

  // Creat log for diagnostics if an output directory is specified
  create_log(output_dir, "optimize_metric");
  spdlog::get("optimize_metric")->set_level(spdlog::level::debug);
  spdlog::get("optimize_metric")->info("Beginning implicit optimization");

  // Create per iteration data log if an output directory is specified
  std::filesystem::path data_log_path;
  if (!output_dir.empty()) {
    data_log_path = join_path(output_dir, "iteration_data_log.csv");
    initialize_data_log(data_log_path);
  }

  // Get maps for going between halfedge, edge, full, and reduced
  // representations as well as free and fixed edges and vertices
  ReductionMaps reduction_maps(m, opt_params->fix_bd_lengths);

  // Expand the target metric coordinates to the doubled surface
  VectorX metric_target, metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);

  // Build energy functions for given energy
  EnergyFunctor opt_energy(m, metric_target, *opt_params);

  // Perform initial conformal projection to get initial metric coordinates
  spdlog::get("optimize_metric")
    ->info("Performing initial projection to the constraint");
  VectorX reduced_metric_coords;
  VectorX u;
  u.setZero(m.n_ind_vertices());
  project_to_constraint(
    m, reduced_metric_init, reduced_metric_coords, u, proj_params);
  SPDLOG_TRACE("Initial projected metric is {}", reduced_metric_coords);

  // Log the initial energy
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);
  Scalar initial_energy = opt_energy.energy(metric_coords);
  spdlog::get("optimize_metric")
    ->info("Initial energy is {}", initial_energy);

  // Main optimization loop
  Scalar beta = beta_0;
  Scalar convergence_ratio = 1.0;
  VectorX gradient, unconstrained_descent_direction,
      constrained_descent_direction;
  VectorX prev_gradient(0);
  VectorX prev_descent_direction(0);
  for (int iter = 0; iter < num_iter; ++iter) {
    spdlog::get("optimize_metric")->info("\nStarting Iteration {}", iter);

    // Get the initial descent direction
    compute_descent_direction(reduced_metric_coords,
                              reduced_metric_target,
                              prev_gradient,
                              prev_descent_direction,
                              reduction_maps,
                              opt_energy,
                              gradient,
                              unconstrained_descent_direction,
                              direction_choice);
    spdlog::get("optimize_metric")->info("Unconstrained descent direction found with norm {}", unconstrained_descent_direction.norm());

    // Project the descent direction so it is tangent to the constraint manifold
    if (use_optimal_projection)
    {
      compute_optimal_tangent_space_descent_direction(
        m,
        reduced_metric_coords,
        gradient,
        unconstrained_descent_direction,
        reduction_maps,
        *opt_params,
        opt_energy,
        constrained_descent_direction
      );
    }
    else
    {
      constrain_descent_direction(m,
                                  reduced_metric_coords,
                                  unconstrained_descent_direction,
                                  reduction_maps,
                                  *opt_params,
                                  constrained_descent_direction);
    }
    spdlog::get("optimize_metric")->info("Constrained descent direction found with norm {}", constrained_descent_direction.norm());

    // Optionally reduce the descent direction to a given range
    Scalar grad_range =
      beta * (constrained_descent_direction.maxCoeff() - constrained_descent_direction.minCoeff());
    if ((max_grad_range > 0) && (grad_range >= max_grad_range)) {
      beta *= (max_grad_range / grad_range);
      spdlog::get("optimize_metric")->info("Reducing beta to {} for stability", beta);
    }

    // Perform the line search with projection to the constraint
    VectorX reduced_line_step_metric_coords;
    VectorX u;
    VectorX updated_reduced_metric_coords;
    line_search_with_projection(m,
                                reduced_metric_coords,
                                constrained_descent_direction,
                                reduction_maps,
                                proj_params,
                                opt_params,
                                opt_energy,
                                beta,
                                convergence_ratio,
                                reduced_line_step_metric_coords,
                                u,
                                updated_reduced_metric_coords);

    // Write iteration data if output directory specified
    log.num_iterations = iter;
    log.beta = beta;
    update_data_log(log,
                    m,
                    reduction_maps,
                    opt_energy,
                    updated_reduced_metric_coords,
                    reduced_metric_coords,
                    reduced_metric_target,
                    unconstrained_descent_direction,
                    constrained_descent_direction,
                    convergence_ratio,
                    opt_params);
    if (!output_dir.empty()) {
      write_data_log_entry(log, data_log_path);
    }

    // Optionally write data per iteration
    if ((use_checkpoints) && ((iter % 100) == 0))
    {
      write_checkpoint(
        output_dir,
        iter,
        m,
        reduction_maps,
        reduced_metric_target,
        reduced_metric_coords,
        reduced_line_step_metric_coords,
        updated_reduced_metric_coords,
        unconstrained_descent_direction,
        constrained_descent_direction,
        u,
        use_edge_lengths
      );
    }

    // Check for convergence
    reduced_metric_coords = updated_reduced_metric_coords;
    bool converged = check_if_converged(*opt_params,
                                        convergence_ratio,
                                        beta);
    if (converged)
      break;

    // Update beta and descent direction data
    beta = std::min<Scalar>(2.0 * beta, opt_params->max_beta);
    prev_gradient = gradient;
    prev_descent_direction = constrained_descent_direction; // Could also use unconstrained
  }

  // Deregister loggers
  spdlog::drop("mesh_data");
  spdlog::drop("optimize_metric");

  // Update final optimized metric coordinates
  optimized_reduced_metric_coords = reduced_metric_coords;
}

void
optimize_metric(const Mesh<Scalar>& m,
                const VectorX& reduced_metric_target,
                const VectorX& reduced_metric_init,
                VectorX& optimized_reduced_metric_coords,
                std::shared_ptr<ProjectionParameters> proj_params,
                std::shared_ptr<OptimizationParameters> opt_params)
{
  OptimizationLog log;
  optimize_metric_log(
    m,
    reduced_metric_target,
    reduced_metric_init,
    optimized_reduced_metric_coords,
    log,
    proj_params,
    opt_params
  );
}

#ifdef PYBIND
VectorX
optimize_metric_pybind(const Mesh<Scalar>& m,
                const VectorX& reduced_metric_target,
                const VectorX& reduced_metric_init,
                std::shared_ptr<ProjectionParameters> proj_params,
                std::shared_ptr<OptimizationParameters> opt_params)
{
    VectorX optimized_reduced_metric_coords;
    optimize_metric(
        m,
        reduced_metric_target,
        reduced_metric_init,
        optimized_reduced_metric_coords,
        proj_params,
        opt_params
    );

    return optimized_reduced_metric_coords;
}

#endif

}
