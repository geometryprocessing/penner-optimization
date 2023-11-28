#include "explicit_optimization.hh"

#include "targets.hh"
#include "constraint.hh"
#include "embedding.hh"
#include "energies.hh"
#include "globals.hh"
#include "shear.hh"
#include "logging.hh"
#include "projection.hh"
#include "nonlinear_optimization.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void
initialize_explicit_data_log(
  const std::filesystem::path& data_log_path
) {
  spdlog::trace("Writing data to {}", data_log_path);
  std::ofstream output_file(data_log_path, std::ios::out | std::ios::trunc);
  output_file << "step_size,";
  output_file << "energy,";
  output_file << "max_error,";
  output_file << "gradient_norm,";
  output_file << "max_change_in_domain_coords,";
  output_file << std::endl;
  output_file.close();
}

void
write_explicit_data_log_entry(
  const std::filesystem::path& data_log_path,
  const DifferentiableConeMetric& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& optimized_domain_coords,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const VectorX& gradient,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params,
  Scalar beta
) {
  spdlog::trace("Writing data iteration to {}", data_log_path);
  std::ofstream output_file(data_log_path, std::ios::out | std::ios::app);

  // Compute metric coordinates
  VectorX reduced_optimized_metric_coords;
  compute_domain_coordinate_metric(
    m,
    reduction_maps,
    optimized_domain_coords,
    constraint_domain_matrix,
    proj_params,
    reduced_optimized_metric_coords
  );

  // Expand reduced metric coordinates
  VectorX optimized_metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_optimized_metric_coords, optimized_metric_coords);

  // Get the full per vertex constraint Jacobian with respect to Penner coordinates
  VectorX constraint;
  MatrixX full_constraint_penner_jacobian;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  constraint_with_jacobian(m,
                           optimized_metric_coords,
                           constraint,
                           full_constraint_penner_jacobian,
                           flip_seq,
                           need_jacobian,
                           opt_params->use_edge_lengths);

  // Compute change in domain coords
  VectorX change_in_domain_coords = optimized_domain_coords - domain_coords;

  // Compute numerics
  Scalar energy = opt_energy.energy(optimized_metric_coords);
  Scalar max_error = constraint.maxCoeff();
  Scalar gradient_norm = gradient.norm();
  Scalar max_change_in_domain_coords = change_in_domain_coords.cwiseAbs().maxCoeff();

  output_file << std::fixed << std::setprecision(17) << beta << ",";
  output_file << std::fixed << std::setprecision(17) << energy << ",";
  output_file << std::fixed << std::setprecision(17) << max_error << ",";
  output_file << std::fixed << std::setprecision(17) << gradient_norm << ",";
  output_file << std::fixed << std::setprecision(17) << max_change_in_domain_coords << ",";
  output_file << std::endl;

  // Close file
  output_file.close();
}

void
compute_optimization_domain(
  const Mesh<Scalar>& m,
  const VectorX& shear_basis_coords,
  const VectorX& scale_factors,
  const MatrixX& shear_basis_matrix,
  const MatrixX& scale_factor_basis_matrix,
  VectorX& domain_coords,
  MatrixX& constraint_domain_matrix,
  MatrixX& constraint_codomain_matrix
) {
  std::vector<T> domain_triplet_list;
  std::vector<T> codomain_triplet_list;
  domain_triplet_list.reserve(shear_basis_matrix.nonZeros() + shear_basis_matrix.rows());
  codomain_triplet_list.reserve(scale_factor_basis_matrix.nonZeros());
  
  // Copy the shear basis matrix to to domain matrix
  int num_edges = shear_basis_matrix.rows();
  int num_independent_edges = shear_basis_matrix.cols();
  for (int k = 0; k < shear_basis_matrix.outerSize(); ++k)
  {
    for (MatrixX::InnerIterator it(shear_basis_matrix, k); it; ++it)
    {
      int row = it.row();
      int col = it.col();
      Scalar value = it.value();
      domain_triplet_list.push_back(T(row, col, value));
    }
  }

  // Enumerate the variable and fixed vertex degrees of freedom
  std::vector<int> fixed_dof, variable_dof, vertices_to_index_map; 
  enumerate_boolean_array(m.fixed_dof, fixed_dof, variable_dof, vertices_to_index_map);

  // Add the fixed scale factor dof to the domain and the variable to the codomain
  for (int k = 0; k < scale_factor_basis_matrix.outerSize(); ++k)
  {
    for (MatrixX::InnerIterator it(scale_factor_basis_matrix, k); it; ++it)
    {
      int row = it.row();
      int col = it.col();
      Scalar value = it.value();

      // Add fixed dof to domain
      if (m.fixed_dof[m.v_rep[col]])
      {
        int remapped_col = num_independent_edges + vertices_to_index_map[col];
        domain_triplet_list.push_back(T(row, remapped_col, value));
      }
      // Add variable dof to codomain
      else
      {
        int remapped_col = vertices_to_index_map[col];
        codomain_triplet_list.push_back(T(row, remapped_col, value));
      }
    }

    // Get initial domain coordinates
    int num_shear_basis_coords = shear_basis_coords.size();
    domain_coords.resize(num_shear_basis_coords + fixed_dof.size());
    domain_coords.head(num_shear_basis_coords) = shear_basis_coords;
    for (size_t i = 0; i < fixed_dof.size(); ++i)
    {
      domain_coords[num_shear_basis_coords + i] = scale_factors[fixed_dof[i]];
    }
  }

  // Build the domain matrix
  constraint_domain_matrix.resize(num_edges, shear_basis_matrix.cols() + fixed_dof.size());
  constraint_domain_matrix.reserve(domain_triplet_list.size());
  constraint_domain_matrix.setFromTriplets(domain_triplet_list.begin(), domain_triplet_list.end());
  SPDLOG_TRACE("Domain matrix is {}", constraint_domain_matrix);

  // Build the codomain matrix
  constraint_codomain_matrix.resize(num_edges, variable_dof.size());
  constraint_codomain_matrix.reserve(codomain_triplet_list.size());
  constraint_codomain_matrix.setFromTriplets(codomain_triplet_list.begin(), codomain_triplet_list.end());
  SPDLOG_TRACE("Codomain matrix is {}", constraint_codomain_matrix);
}

void
compute_domain_coordinate_metric(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  VectorX& reduced_metric_coords
) {
  spdlog::trace("Making domain coordinate metric");
  SPDLOG_TRACE("Domain coordinates in range [{}, {}]", domain_coords.maxCoeff(), domain_coords.minCoeff());

  // Get domain coordinate metric defined by the current coordinates
  VectorX domain_metric_coords = constraint_domain_matrix * domain_coords;

  // Get reduced coordinates from the domain coordinates
  VectorX reduced_domain_metric_coords;
  reduce_symmetric_function(reduction_maps.embed, domain_metric_coords, reduced_domain_metric_coords);
  SPDLOG_TRACE("Domain metric in range [{}, {}]", reduced_domain_metric_coords.minCoeff(), reduced_domain_metric_coords.maxCoeff());

  // Project the domain metric to the constraint
  VectorX scale_factors;
  scale_factors.setZero(m.n_ind_vertices());
  project_to_constraint(
    m, reduced_domain_metric_coords, reduced_metric_coords, scale_factors, proj_params);
  SPDLOG_TRACE("Domain metric is\n{}", reduced_domain_metric_coords);
  SPDLOG_TRACE("Projected metric is\n{}", reduced_metric_coords);
  SPDLOG_TRACE("Projected metric in range [{}, {}]", reduced_metric_coords.minCoeff(), reduced_metric_coords.maxCoeff());
}

Scalar
compute_domain_coordinate_energy(
  const Mesh<Scalar>& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params
) {
  // Compute penner coordinates from the domain coordinates
  VectorX metric_coords, reduced_metric_coords;
  compute_domain_coordinate_metric(
    m,
    reduction_maps,
    domain_coords,
    constraint_domain_matrix,
    proj_params,
    reduced_metric_coords
  );

  // Expand reduced metric coordinates
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);

  // Get the initial energy
  return opt_energy.energy(metric_coords);
}

bool
compute_domain_coordinate_energy_with_gradient(
  const DifferentiableConeMetric& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params,
  Scalar& energy,
  VectorX& gradient
) {
  // Compute penner coordinates from the domain coordinates
  VectorX metric_coords, reduced_metric_coords;
  compute_domain_coordinate_metric(
    m,
    reduction_maps,
    domain_coords,
    constraint_domain_matrix,
    proj_params,
    reduced_metric_coords
  );

  // Expand reduced metric coordinates
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);

  // Get the initial energy
  energy = opt_energy.energy(metric_coords);

  // Get the gradients of the energy with respect to the domain and codomain coordinates
  VectorX energy_penner_gradient = opt_energy.gradient(metric_coords);
  VectorX energy_domain_gradient = constraint_domain_matrix.transpose() * energy_penner_gradient;
  VectorX energy_codomain_gradient = constraint_codomain_matrix.transpose() * energy_penner_gradient;

  // Get the full per vertex constraint Jacobian with respect to Penner coordinates
  VectorX constraint;
  MatrixX full_constraint_penner_jacobian;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  bool success = constraint_with_jacobian(m,
                           metric_coords,
                           constraint,
                           full_constraint_penner_jacobian,
                           flip_seq,
                           need_jacobian,
                           opt_params->use_edge_lengths);
  if (!success)
  {
    spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
    return false;
  }
                                          
  // Remove the redundant constraints
  MatrixX constraint_penner_jacobian;
  int num_edges = full_constraint_penner_jacobian.cols();
  std::vector<int> all_edges;
  arange(num_edges, all_edges);
  compute_submatrix(full_constraint_penner_jacobian,
                    reduction_maps.free_v,
                    all_edges,
                    constraint_penner_jacobian);

  // Get the Jacobians of the constraint with respect to the domain and codomain coordinates
  MatrixX constraint_domain_jacobian = constraint_domain_matrix.transpose() * constraint_penner_jacobian.transpose();
  MatrixX constraint_codomain_jacobian = constraint_codomain_matrix.transpose() * constraint_penner_jacobian.transpose();

  // Solve for the component of the gradient corresponding to the implicit metric coordinates
  Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(constraint_codomain_jacobian);
  VectorX energy_implicit_gradient = -constraint_domain_jacobian * solver.solve(energy_codomain_gradient);

  // Construct the final gradient
  gradient = energy_domain_gradient + energy_implicit_gradient;

  return true;
}

void
compute_descent_direction(
  const VectorX& prev_gradient,
  const VectorX& prev_descent_direction,
  const std::deque<VectorX>& delta_variables,
  const std::deque<VectorX>& delta_gradients,
  const MatrixX& approximate_hessian_inverse,
  const VectorX& gradient,
  std::shared_ptr<OptimizationParameters> opt_params,
  VectorX& descent_direction
) {
  std::string direction_choice = opt_params->direction_choice;

  // Compute descent direction
  if (direction_choice == "gradient")
  {
    descent_direction = -gradient;
  }
  else if (direction_choice == "conjugate_gradient")
  {
    // Check if the previous gradient and descent direction are trivial
    if ((prev_gradient.size() == 0) ||
        (prev_descent_direction.size() == 0)) {
      descent_direction = -gradient;
    }
    else
    {
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
  else if (direction_choice == "bfgs")
  {
    descent_direction = -(approximate_hessian_inverse * gradient);
  }
  else if (direction_choice == "lbfgs")
  {
    // Check if the previous gradient and descent direction are trivial
    if ((delta_variables.size() == 0) || (delta_gradients.size() == 0))
    {
      descent_direction = -gradient;
    }
    else
    {
      compute_lbfgs_direction(
        delta_variables,
        delta_gradients,
        gradient,
        descent_direction
      );
    }
  }
}

void
backtracking_domain_line_search(
  const DifferentiableConeMetric& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  const VectorX& gradient,
  const VectorX& descent_direction,
  VectorX& optimized_domain_coords,
  Scalar& beta,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params
) {
  spdlog::get("optimize_metric")->info("Beginning line search");

  // Parameters for the line search
  Scalar shrink_factor = 0.5;

  // Get initial energy
  Scalar initial_energy = compute_domain_coordinate_energy(
      m,
      reduction_maps,
      opt_energy,
      domain_coords,
      constraint_domain_matrix,
      proj_params);
  spdlog::get("optimize_metric")->info("Initial energy is {}", initial_energy);

  // Get the slope along the descent direction
  Scalar descent_slope = gradient.dot(descent_direction);
  spdlog::get("optimize_metric")->info("Descent direction slope is {}", descent_slope);

  // Make an initial line step and compute the energy and gradient
  spdlog::get("optimize_metric")->info("Making step of size {}", beta);
  optimized_domain_coords = domain_coords + beta * descent_direction;
  SPDLOG_TRACE("Optimized domain coordinates in range [{}, {}]", optimized_domain_coords.minCoeff(), optimized_domain_coords.maxCoeff());
  VectorX domain_metric_coords = constraint_domain_matrix * domain_coords;
  SPDLOG_TRACE("Optimized domain metric in range [{}, {}]", domain_metric_coords.minCoeff(), domain_metric_coords.maxCoeff());
  Scalar energy;
  VectorX step_gradient;
  bool success = compute_domain_coordinate_energy_with_gradient(
    m,
    reduction_maps,
    opt_energy,
    optimized_domain_coords,
    constraint_domain_matrix,
    constraint_codomain_matrix,
    proj_params,
    opt_params,
    energy,
    step_gradient);
  spdlog::get("optimize_metric")->info("Step energy is {}", energy);

  // TODO Backtrack until the Armijo condition is satisfied
  // Scalar control_parameter = 1e-4;
  //while (energy > initial_energy + beta * control_parameter * descent_slope)
  // Backtrack until the energy decreases sufficiently and the gradient sign condition
  // is satisfied
  while (
       (!success)
    || (energy > initial_energy * (1 + 1e-8))
    || (step_gradient.dot(descent_direction) > 0)
  ) {
    if (beta < 1e-16)
    {
      spdlog::get("optimize_metric")->warn("Terminating line step as beta too small");
      return;
    }

    // Reduce beta
    beta *= shrink_factor;

    // Make a line step
    optimized_domain_coords = domain_coords + beta * descent_direction;

    // Compute the new energy for the line step
    spdlog::get("optimize_metric")->info("Making step of size {}", beta);
    success = compute_domain_coordinate_energy_with_gradient(
      m,
      reduction_maps,
      opt_energy,
      optimized_domain_coords,
      constraint_domain_matrix,
      constraint_codomain_matrix,
      proj_params,
      opt_params,
      energy,
      step_gradient);
    spdlog::get("optimize_metric")->info("Step energy is {}", energy);
    if (!success)
    {
      spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
      
    }
  }
}

void
optimize_shear_basis_coordinates(
  const DifferentiableConeMetric& m,
  const VectorX& reduced_metric_target,
  const VectorX& shear_basis_coords_init,
  const VectorX& scale_factors_init,
  const MatrixX& shear_basis_matrix,
  VectorX& reduced_metric_coords,
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
  Scalar max_beta = opt_params->max_beta;
  Scalar max_grad_range = opt_params->max_grad_range;
  std::string energy_choice = opt_params->energy_choice;
  std::string output_dir = opt_params->output_dir;

  // Log mesh data
  create_log(output_dir, "mesh_data");
  spdlog::get("mesh_data")->set_level(spdlog::level::trace);
  log_mesh_information(m, "mesh_data");

  // Creat log for diagnostics if an output directory is specified
  create_log(output_dir, "optimize_metric");
  spdlog::get("optimize_metric")->set_level(spdlog::level::debug);
  spdlog::get("optimize_metric")->info("Beginning explicit optimization");

  // Create per iteration data log if an output directory is specified
  std::filesystem::path data_log_path;
  if (!output_dir.empty()) {
    data_log_path = join_path(output_dir, "iteration_data_log.csv");
    initialize_explicit_data_log(data_log_path);
  }

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
  spdlog::get("optimize_metric")->info(
    "Optimizing {} coordinates with codomain of dimension {}",
    constraint_domain_matrix.cols(),
    constraint_codomain_matrix.cols()
  );

  // Keep track of various data for the given descent direction method
  // Note that only the necessary data will be updated
  Scalar beta = beta_0;
  VectorX prev_gradient(0);
  VectorX prev_descent_direction(0);
  VectorX gradient(0);
  VectorX descent_direction(0);
  VectorX prev_domain_coords(0);
  std::deque<VectorX> delta_variables(0);
  std::deque<VectorX> delta_gradients(0);
  MatrixX approximate_hessian_inverse = id_matrix(domain_coords.size());

  for (int iter = 0; iter < num_iter; ++iter) {
    spdlog::get("optimize_metric")->info("Beginning iteration {}", iter);

    // Compute the gradient for the shear metric coordinates
    Scalar energy;
    prev_gradient = gradient;
    compute_domain_coordinate_energy_with_gradient(
      m,
      reduction_maps,
      opt_energy,
      domain_coords,
      constraint_domain_matrix,
      constraint_codomain_matrix,
      proj_params,
      opt_params,
      energy,
      gradient);
    spdlog::get("optimize_metric")->info("Energy at start of iteration is {}", energy);
    SPDLOG_TRACE("Domain coordinates in range [{}, {}]", domain_coords.minCoeff(), domain_coords.maxCoeff());
    SPDLOG_TRACE("Gradient coefficients in range [{}, {}]", gradient.minCoeff(), gradient.maxCoeff());

    // Update hessian inverse information if necessary
    if (iter > 0)
    {
      if (opt_params->direction_choice == "bfgs")
      {
        VectorX current_delta_variables = domain_coords - prev_domain_coords;
        update_bfgs_hessian_inverse(
          gradient,
          prev_gradient,
          current_delta_variables,
          approximate_hessian_inverse
        );
      }
      else if (opt_params->direction_choice == "lbfgs")
      {
        delta_variables.push_front(domain_coords - prev_domain_coords);
        delta_gradients.push_front(gradient - prev_gradient);
        if (delta_variables.size() > 10) delta_variables.pop_back();
        if (delta_gradients.size() > 10) delta_gradients.pop_back();
      }
    }

    // Compute descent directions
    prev_descent_direction = descent_direction;
    compute_descent_direction(
      prev_gradient,
      prev_descent_direction,
      delta_variables,
      delta_gradients,
      approximate_hessian_inverse,
      gradient,
      opt_params,
      descent_direction
    );

    // Ensure the descent direction range is stable
    Scalar grad_range =
      beta * (descent_direction.maxCoeff() - descent_direction.minCoeff());
    if ((max_grad_range > 0) && (grad_range >= max_grad_range)) {
      beta *= (max_grad_range / grad_range);
      spdlog::get("optimize_metric")->info("Reducing beta to {} for stability", beta);
    }

    // Perform backtracking gradient descent
    VectorX optimized_domain_coords;
    backtracking_domain_line_search(
      m,
      reduction_maps,
      opt_energy,
      domain_coords,
      constraint_domain_matrix,
      constraint_codomain_matrix,
      gradient,
      descent_direction,
      optimized_domain_coords,
      beta,
      proj_params,
      opt_params
    );

    // Write iteration data if output directory specified
    if (!output_dir.empty()) {
      write_explicit_data_log_entry(data_log_path,
                          m,
                          reduction_maps,
                          opt_energy,
                          optimized_domain_coords,
                          domain_coords,
                          constraint_domain_matrix,
                          gradient,
                          proj_params,
                          opt_params,
                          beta);
    }

    // Update for next iteration
    prev_domain_coords = domain_coords;
    domain_coords = optimized_domain_coords;
    beta = std::min(2.0 * beta, max_beta);
  }

  // Compute final projection
  compute_domain_coordinate_metric(
    m,
    reduction_maps,
    domain_coords,
    constraint_domain_matrix,
    proj_params,
    reduced_metric_coords);

  // Get final energy
  VectorX metric_coords;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_coords, metric_coords);
  Scalar energy = compute_domain_coordinate_energy(
      m,
      reduction_maps,
      opt_energy,
      domain_coords,
      constraint_domain_matrix,
      proj_params);
  spdlog::get("optimize_metric")->info("Final energy is {}", energy);

  // Close loggers
  spdlog::drop("mesh_data");
  spdlog::drop("optimize_metric");
}


#ifdef PYBIND

std::tuple<
  VectorX, // domain_coords
  MatrixX, // constraint_domain_matrix
  MatrixX // constraint_codomain_matrix
>
compute_optimization_domain_pybind(
  const DifferentiableConeMetric& m,
  const VectorX& shear_basis_coords,
  const VectorX& scale_factors,
  const MatrixX& shear_basis_matrix,
  const MatrixX& scale_factor_basis_matrix
) {
  VectorX domain_coords;
  MatrixX constraint_domain_matrix;
  MatrixX constraint_codomain_matrix;
  compute_optimization_domain(
    m,
    shear_basis_coords,
    scale_factors,
    shear_basis_matrix,
    scale_factor_basis_matrix,
    domain_coords,
    constraint_domain_matrix,
    constraint_codomain_matrix
  );
  return std::make_tuple(domain_coords, constraint_domain_matrix, constraint_codomain_matrix);
}

VectorX
optimize_shear_basis_coordinates_pybind(
  const DifferentiableConeMetric& m,
  const VectorX& reduced_metric_target,
  const VectorX& shear_basis_coords_init,
  const VectorX& scale_factors_init,
  const MatrixX& shear_basis_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params)
{
  VectorX reduced_metric_coords;
  optimize_shear_basis_coordinates(m, reduced_metric_target, shear_basis_coords_init, scale_factors_init, shear_basis_matrix, reduced_metric_coords, proj_params, opt_params);
  return reduced_metric_coords;
}

std::tuple<Scalar, VectorX>
compute_domain_coordinate_energy_with_gradient_pybind(
  const DifferentiableConeMetric& m,
  const ReductionMaps& reduction_maps,
  const EnergyFunctor& opt_energy,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  const MatrixX& constraint_codomain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params,
  std::shared_ptr<OptimizationParameters> opt_params
) {
  Scalar energy;
  VectorX gradient;
  compute_domain_coordinate_energy_with_gradient(
    m,
    reduction_maps,
    opt_energy,
    domain_coords,
    constraint_domain_matrix,
    constraint_codomain_matrix,
    proj_params,
    opt_params,
    energy,
    gradient
  );
  return std::make_tuple(energy, gradient);
}

VectorX
compute_domain_coordinate_metric_pybind(
  const DifferentiableConeMetric& m,
  const ReductionMaps& reduction_maps,
  const VectorX& domain_coords,
  const MatrixX& constraint_domain_matrix,
  std::shared_ptr<ProjectionParameters> proj_params)
{
  VectorX reduced_metric_coords;
  compute_domain_coordinate_metric(
    m,
    reduction_maps,
    domain_coords,
    constraint_domain_matrix,
    proj_params,
    reduced_metric_coords
  );
  return reduced_metric_coords;
}

#endif

}
