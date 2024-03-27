/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#include "implicit_optimization.hh"

#include <igl/Timer.h>
#include <Eigen/SparseQR>
#include "area.hh"
#include "constraint.hh"
#include "embedding.hh"
#include "energies.hh"
#include "globals.hh"
#include "io.hh"
#include "nonlinear_optimization.hh"
#include "projection.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {
Scalar compute_convergence_ratio(
    const VectorX& unconstrained_descent_direction,
    const VectorX& constrained_descent_direction)
{
    Scalar unconstrained_descent_norm = unconstrained_descent_direction.norm();
    Scalar constrained_descent_norm = constrained_descent_direction.norm();

    // Compute the convergence ratio, returning just the constrained norm if the unconstrained norm
    // is 0.0
    Scalar convergence_ratio;
    if (unconstrained_descent_norm > 0) {
        convergence_ratio = constrained_descent_norm / unconstrained_descent_norm;
    } else {
        convergence_ratio = constrained_descent_norm;
    }

    return convergence_ratio;
}

void initialize_data_log(const std::filesystem::path& data_log_path)
{
    // Open file
    spdlog::trace("Writing data to {}", data_log_path);
    std::ofstream output_file(data_log_path, std::ios::out | std::ios::trunc);

    // Write header
    output_file << "num_iter,";
    output_file << "time,";
    output_file << "step_size,";
    output_file << "energy,";
    output_file << "max_error,";
    output_file << "rmse,";
    output_file << "rrmse,";
    output_file << "rmsre,";
    output_file << "convergence_ratio,";
    output_file << "max_change_in_metric_coords,";
    output_file << "actual_to_unconstrained_direction_ratio,";
    output_file << "max_constrained_descent_direction,";
    output_file << "num_linear_solves,";
    output_file << std::endl;

    // Close file
    output_file.close();
}

void update_data_log(
    OptimizationLog& log,
    DifferentiableConeMetric& initial_cone_metric,
    const EnergyFunctor& opt_energy,
    const VectorX& reduced_metric_coords,
    const VectorX& prev_reduced_metric_coords,
    const VectorX& gradient,
    const VectorX& descent_direction,
    Scalar convergence_ratio)
{
    // Get initial metric coordinates
    VectorX metric_init = initial_cone_metric.get_reduced_metric_coordinates();

    // Get edge lengths
    int num_reduced_edges = metric_init.size();
    VectorX l_init(num_reduced_edges);
    VectorX l(num_reduced_edges);
    for (int E = 0; E < num_reduced_edges; ++E) {
        l_init[E] = exp(metric_init[E] / 2.0);
        l[E] = exp(reduced_metric_coords[E] / 2.0);
    }

    // Copy metric and make discrete
    std::unique_ptr<DifferentiableConeMetric> cone_metric =
        initial_cone_metric.set_metric_coordinates(reduced_metric_coords);
    cone_metric->make_discrete_metric(); // Need to flip here to get number of flips

    // Compute constraint values
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = false;
    cone_metric->constraint(constraint, J_constraint, need_jacobian, only_free_vertices);

    // Compute change in metric coords from target and previous iteration
    VectorX change_in_metric_coords = reduced_metric_coords - prev_reduced_metric_coords;

    // Compute numerics
    log.energy = opt_energy.energy(initial_cone_metric);
    log.error = sup_norm(constraint);
    log.rmse = root_mean_square_error(l, l_init);
    log.rrmse = relative_root_mean_square_error(l, l_init);
    log.rmsre = root_mean_square_relative_error(l, l_init);
    log.num_flips = cone_metric->num_flips();
    log.convergence_ratio = convergence_ratio;
    log.max_change_in_metric_coords = sup_norm(change_in_metric_coords);
    log.actual_to_unconstrained_direction_ratio = change_in_metric_coords.norm() / gradient.norm();
    log.max_constrained_descent_direction = sup_norm(descent_direction);
}

//  void write_checkpoint(
//      const std::string &output_dir,
//      int iter,
//      const DifferentiableConeMetric &cone_metric,
//      const ReductionMaps &reduction_maps,
//      const VectorX &reduced_metric_target,
//      const VectorX &reduced_metric_coords,
//      const VectorX &reduced_line_step_metric_coords,
//      const VectorX &updated_reduced_metric_coords,
//      const VectorX &unconstrained_descent_direction,
//      const VectorX &constrained_descent_direction,
//      const VectorX &u,
//      bool use_edge_lengths = true)
//  {
//    std::string checkpoint_filename;
//
//    checkpoint_filename = join_path(output_dir, "target_lambdas");
//    write_vector(reduced_metric_target, checkpoint_filename, 80);
//
//    checkpoint_filename = join_path(output_dir, "initial_lambdas_" + std::to_string(iter));
//    write_vector(reduced_metric_coords, checkpoint_filename, 80);
//
//    checkpoint_filename = join_path(output_dir, "unconstrained_descent_direction_" +
//    std::to_string(iter)); write_vector(unconstrained_descent_direction, checkpoint_filename, 80);
//
//    checkpoint_filename = join_path(output_dir, "constrained_descent_direction_" +
//    std::to_string(iter)); write_vector(constrained_descent_direction, checkpoint_filename, 80);
//
//    // Compute the constraint function and its Jacobian
//    VectorX metric_coords;
//    expand_reduced_function(
//        reduction_maps.proj, reduced_metric_coords, metric_coords);
//    VectorX constraint;
//    MatrixX J_constraint;
//    std::vector<int> flip_seq;
//    bool need_jacobian = true;
//    constraint(cone_metric,
//                             metric_coords,
//                             constraint,
//                             J_constraint,
//                             flip_seq,
//                             need_jacobian,
//                             use_edge_lengths);
//    MatrixX J_restricted_constraint;
//    compute_submatrix(J_constraint * reduction_maps.projection,
//                      reduction_maps.free_v,
//                      reduction_maps.free_e,
//                      J_restricted_constraint);
//    checkpoint_filename = join_path(output_dir, "constraint_" + std::to_string(iter));
//    write_vector(constraint, checkpoint_filename);
//
//    checkpoint_filename = join_path(output_dir, "constraint_jacobian_" + std::to_string(iter));
//    write_sparse_matrix(J_restricted_constraint, checkpoint_filename, "matlab");
//
//    // MatrixX projection_matrix = compute_descent_direction_projection_matrix(J_restricted_constraint);
//    // checkpoint_filename = join_path(output_dir, "projection_matrix_" + std::to_string(iter));
//    // write_sparse_matrix(projection_matrix, checkpoint_filename, "matlab");
//
//    checkpoint_filename = join_path(output_dir, "unprojected_lambdas_" + std::to_string(iter));
//    write_vector(reduced_line_step_metric_coords, checkpoint_filename, 80);
//
//    checkpoint_filename = join_path(output_dir, "scale_factors_" + std::to_string(iter));
//    write_vector(u, checkpoint_filename, 80);
//
//    MatrixX B = conformal_scaling_matrix(cone_metric);
//    checkpoint_filename = join_path(output_dir, "scaling_matrix");
//    write_sparse_matrix(B, checkpoint_filename, "matlab");
//
//    checkpoint_filename = join_path(output_dir, "lambdas_" + std::to_string(iter));
//    write_vector(updated_reduced_metric_coords, checkpoint_filename, 80);
//  }

void write_data_log_entry(const OptimizationLog& log, const std::filesystem::path& data_log_path)
{
    // Open file in append mode
    spdlog::trace("Writing data iteration to {}", data_log_path);
    std::ofstream output_file(data_log_path, std::ios::out | std::ios::app);

    // Write iteration row
    output_file << log.num_iterations << ",";
    output_file << std::fixed << std::setprecision(17) << log.time << ",";
    output_file << std::fixed << std::setprecision(17) << log.beta << ",";
    output_file << std::fixed << std::setprecision(17) << log.energy << ",";
    output_file << std::fixed << std::setprecision(17) << log.error << ",";
    output_file << std::fixed << std::setprecision(17) << log.rmse << ",";
    output_file << std::fixed << std::setprecision(17) << log.rrmse << ",";
    output_file << std::fixed << std::setprecision(17) << log.rmsre << ",";
    output_file << std::fixed << std::setprecision(17) << log.convergence_ratio << ",";
    output_file << std::fixed << std::setprecision(17) << log.max_change_in_metric_coords << ",";
    output_file << std::fixed << std::setprecision(17)
                << log.actual_to_unconstrained_direction_ratio << ",";
    output_file << std::fixed << std::setprecision(17) << log.max_constrained_descent_direction
                << ",";
    output_file << log.num_linear_solves << ",";
    output_file << std::endl;

    // Close file
    output_file.close();
}

/// Given a metric with reduction maps and constraints and an energy gradient
/// functor, compute a gradient and descent direction for the energy.
///
/// @param[in] cone_metric: mesh with metric
/// @param[in] opt_energy: optimization energy
/// @param[in] prev_gradient: previous gradient of the energy
/// @param[in] prev_descent_direction: previous descent direction
/// @param[out] gradient: current gradient of the energy
/// @param[out] descent_direction: descent direction
/// @param[in] direction_choice: (optional) type of descent direction to use
void compute_descent_direction(
    const DifferentiableConeMetric& cone_metric,
    const EnergyFunctor& opt_energy,
    const VectorX& prev_gradient,
    const VectorX& prev_descent_direction,
    VectorX& gradient,
    VectorX& descent_direction,
    std::string direction_choice)
{
    // Compute gradient from the functor
    gradient = opt_energy.gradient(cone_metric);

    // Compute the gradient descent direction
    if (direction_choice == "gradient") {
        descent_direction = -gradient;
    }
    // Compute the conjugate gradient descent direction
    // WARNING: The theory on this choice is dubious with projection
    else if (direction_choice == "conjugate_gradient") {
        // Check if the previous gradient and descent direction are trivial
        if ((prev_gradient.size() == 0) || (prev_descent_direction.size() == 0)) {
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
                coefficient);
        }
    }
    // By default, use the zero vector
    else {
        descent_direction = VectorX::Constant(gradient.size(), 0.0);
    }
}

// Generate the lagrangian for optimizing a quadratic energy in a constrained space orthogonal to
// the column space of J_constraint
void generate_hessian_lagrangian_system(
    const MatrixX& hessian,
    const MatrixX& J_constraint,
    const VectorX& gradient,
    MatrixX& hessian_lagrangian,
    VectorX& rhs)
{
    // Get sizes
    int n = hessian.rows();
    int m = J_constraint.rows();
    assert(J_constraint.cols() == n);
    assert(gradient.size() == n);

    // Initialize matrix entry list
    std::vector<T> tripletList;
    tripletList.reserve(hessian.nonZeros() + 2 * J_constraint.nonZeros());

    // Add hessian as upper block
    for (int k = 0; k < hessian.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(hessian, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            Scalar value = it.value();
            tripletList.push_back(T(row, col, value));
        }
    }

    // Add constraint Jacobian and transpose
    for (int k = 0; k < J_constraint.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(J_constraint, k); it; ++it) {
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
    // TODO Add constraint to tail
}

/// Given a metric with reduction maps and constraints and an energy gradient
/// functor, compute a constrained descent direction for the energy that is optimal
/// in the tangent space to the constraint manifold.
///
/// @param[in] m: surface
/// @param[in] opt_energy: optimization energy
/// @param[out] gradient: current gradient of the energy
/// @param[out] descent_direction: quadratically optimal descent direction in the tangent space
void compute_projected_newton_descent_direction(
    const DifferentiableConeMetric& cone_metric,
    const EnergyFunctor& opt_energy,
    VectorX& gradient,
    VectorX& descent_direction)
{
    // Compute the constraint function and its Jacobian
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = true;
    bool success = cone_metric.constraint(constraint, J_constraint, need_jacobian);
    if (!success) {
        spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
    }

    // Get the reduced energy gradient and hessian
    gradient = opt_energy.gradient(cone_metric);
    MatrixX hessian = opt_energy.hessian(cone_metric);

    // Generate lagrangian and right hand side for this quadratic programming problem
    MatrixX hessian_lagrangian;
    VectorX rhs;
    generate_hessian_lagrangian_system(hessian, J_constraint, gradient, hessian_lagrangian, rhs);

    // Solve for the optimal descent direction
    igl::Timer timer;
    timer.start();
    hessian_lagrangian.makeCompressed();
    VectorX solution = solve_linear_system(hessian_lagrangian, rhs);
    double time = timer.getElapsedTime();
    spdlog::trace("Lagrangian descent direction solve took {} s", time);

    // Get projected descent direction from lagrangian solution
    int num_variables = gradient.size();
    descent_direction = solution.head(num_variables);
}

// Check if the line step is valid and stable
bool check_line_step_success(
    const DifferentiableConeMetric& cone_metric,
    std::shared_ptr<OptimizationParameters> opt_params)
{
    // Compute the contraint values after the line step
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = false;
    bool success =
        cone_metric.constraint(constraint, J_constraint, need_jacobian, only_free_vertices);

    // Shrink step size if the constraint is not computed correctly due to a triangle inequality
    // violation
    if (!success) {
        spdlog::get("optimize_metric")
            ->info("Reducing step size due to triangle inequality violation while computing the "
                   "constraint function");
        return false;
    }

    // Check if the maximum constraint error is too large
    Scalar max_line_search_constraint = sup_norm(constraint);
    spdlog::get("optimize_metric")
        ->info("Maximum constraint error after line step is {}", max_line_search_constraint);
    if (max_line_search_constraint > opt_params->max_angle) {
        spdlog::get("optimize_metric")
            ->info(
                "Max angle error {} larger than bound {}",
                max_line_search_constraint,
                opt_params->max_angle);
        return false;
    }

    // Success otherwise
    return true;
}

// Check if the projection to the constraint manifold is valid and stable
bool check_projection_success(
    const DifferentiableConeMetric& cone_metric,
    std::shared_ptr<OptimizationParameters> opt_params,
    const Scalar& max_initial_constraint)
{
    // Compute the constraint values after the projection
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = true;
    bool success =
        cone_metric.constraint(constraint, J_constraint, need_jacobian, only_free_vertices);

    // Shrink step size if the constraint is not computed correctly due to a triangle inequality
    // violation
    if (!success) {
        spdlog::get("optimize_metric")
            ->info("Reducing step size due to triangle inequality violation while computing the "
                   "constraint function");
        return false;
    }

    // Check the angle decrease condition
    Scalar max_constraint = constraint.maxCoeff();
    spdlog::get("optimize_metric")
        ->info("Maximum constraint error after projection is {}", max_constraint);
    Scalar constraint_threshold = max_initial_constraint + opt_params->max_angle_incr;
    if (max_constraint > 1e-8) {
        spdlog::get("optimize_metric")->info("Reducing step size as the angle error is too large");
        return false;
    }
    if (max_constraint > constraint_threshold) {
        spdlog::get("optimize_metric")
            ->info(
                "Reducing step size as the angle error {} > {} did not decrease from {} with "
                "tolerance {}",
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
bool check_convergence_progress(
    const DifferentiableConeMetric& cone_metric,
    const EnergyFunctor& opt_energy,
    std::shared_ptr<OptimizationParameters> opt_params,
    const VectorX& descent_direction,
    const VectorX& constrained_gradient,
    Scalar opt_energy_init)
{
    // Check the energy decreases enough
    Scalar opt_energy_proj = opt_energy.energy(cone_metric);
    spdlog::get("optimize_metric")->info("Energy after projection is {}", opt_energy_proj);
    if ((opt_params->require_energy_decr) &&
        (opt_energy_proj >= opt_energy_init * (1.0 + opt_params->max_energy_incr))) {
        spdlog::get("optimize_metric")->info("Reducing step size as the energy did not decrease");
        return false;
    }

    // Check the gradient projection on the descent direction is negative
    if ((opt_params->require_gradient_proj_negative) &&
        (descent_direction.dot(constrained_gradient) > 0)) {
        spdlog::get("optimize_metric")
            ->info("Reducing step size as the gradient projection is positive");
        return false;
    }

    return true;
}

/// Perform a line search with projection to the constraint after the step.
///
/// @param[in] cone_metric: surface
/// @param[in] opt_energy: optimization energy
/// @param[in] descent_direction: descent direction for the line search
/// @param[in] proj_params: projection parameters
/// @param[in] opt_params: optimization parameters
/// @param[in, out] beta: adaptive line step size
/// @param[in, out] num_linear_solves: number of linear solves
VectorX line_search_with_projection(
    const DifferentiableConeMetric& initial_cone_metric,
    const EnergyFunctor& opt_energy,
    const VectorX& descent_direction,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::shared_ptr<OptimizationParameters> opt_params,
    Scalar& beta,
    int& num_linear_solves)
{
    // Expand the initial metric coordinates to the doubled surface
    VectorX initial_reduced_metric_coords = initial_cone_metric.get_reduced_metric_coordinates();
    VectorX reduced_metric_coords = initial_reduced_metric_coords;

    // Get the initial optimization energy
    Scalar opt_energy_init = opt_energy.energy(initial_cone_metric);
    spdlog::get("optimize_metric")
        ->info(
            "Attempting line search with step size {} from initial energy {}",
            beta,
            opt_energy_init);

    // Compute the initial constraint value and convergence ratio
    Scalar max_initial_constraint = compute_max_constraint(initial_cone_metric);

    while (true) {
        // Break if the step size is too small
        if (beta < 1e-16) {
            spdlog::get("optimize_metric")->info("Beta {} too small to continue", beta);
            break;
        }

        // Make a line step in log space
        VectorX reduced_line_step_metric_coords =
            initial_reduced_metric_coords + beta * descent_direction;
        std::unique_ptr<DifferentiableConeMetric> cone_metric =
            initial_cone_metric.set_metric_coordinates(reduced_line_step_metric_coords);

        // Check if the line step is valid and reduce beta if not
        if (!check_line_step_success(*cone_metric, opt_params)) {
            beta /= 2.0;
            continue;
        }

        // Project to the constraint
        SolveStats<Scalar> solve_stats;
        std::unique_ptr<DifferentiableConeMetric> constrained_cone_metric =
            cone_metric->project_to_constraint(solve_stats, proj_params);
        num_linear_solves += solve_stats.n_solves;
        reduced_metric_coords = constrained_cone_metric->get_reduced_metric_coordinates();
        spdlog::get("optimize_metric")->set_level(spdlog::level::info);

        // Check if the projection was successful and reduce beta if not
        if (!check_projection_success(
                *constrained_cone_metric,
                opt_params,
                max_initial_constraint)) {
            beta /= 2.0;
            continue;
        }

        // Compute the gradient and its orthogonal projection
        VectorX gradient = opt_energy.gradient(*constrained_cone_metric);
        VectorX constrained_gradient =
            project_descent_direction(*constrained_cone_metric, gradient);
        num_linear_solves++;
        SPDLOG_TRACE("Projected gradient is {}", descent_direction.dot(constrained_gradient));

        // Check if the convergence conditions were satisfied and reduce beta if not
        bool convergence_success = check_convergence_progress(
            *constrained_cone_metric,
            opt_energy,
            opt_params,
            descent_direction,
            constrained_gradient,
            opt_energy_init);
        if (!convergence_success) {
            beta /= 2.0;
            continue;
        }

        // Break the loop if no flag to reduce beta was hit
        break;
    }

    return reduced_metric_coords;
}

bool check_if_converged(
    const OptimizationParameters& opt_params,
    Scalar convergence_ratio,
    Scalar beta)
{
    // Check if the convergence ratio is suitably small
    if (convergence_ratio < opt_params.min_ratio) {
        spdlog::get("optimize_metric")
            ->info("Convergence ratio below termination threshold of {}", opt_params.min_ratio);
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

std::unique_ptr<DifferentiableConeMetric> optimize_metric_log(
    const DifferentiableConeMetric& initial_cone_metric,
    const EnergyFunctor& opt_energy,
    OptimizationLog& log,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::shared_ptr<OptimizationParameters> opt_params)
{
    // Build default parameters if none given
    if (proj_params == nullptr) proj_params = std::make_shared<ProjectionParameters>();
    if (opt_params == nullptr) opt_params = std::make_shared<OptimizationParameters>();

    // Extract relevant parameters for main optimization method
    int num_iter = opt_params->num_iter;
    Scalar beta_0 = opt_params->beta_0;
    std::string direction_choice = opt_params->direction_choice;
    std::string output_dir = opt_params->output_dir;
    Scalar max_grad_range = opt_params->max_grad_range;

    // Log mesh data TODO
    // create_log(output_dir, "mesh_data");
    // spdlog::get("mesh_data")->set_level(spdlog::level::trace);
    // log_mesh_information(initial_cone_metric, "mesh_data");

    // Creat log for diagnostics if an output directory is specified
    create_log(output_dir, "optimize_metric");
    spdlog::get("optimize_metric")->set_level(spdlog::level::info);
    spdlog::get("optimize_metric")->info("Beginning implicit optimization");

    // Create per iteration data log if an output directory is specified
    std::filesystem::path data_log_path;
    if (!output_dir.empty()) {
        data_log_path = join_path(output_dir, "iteration_data_log.csv");
        initialize_data_log(data_log_path);

        // Clear other data logs
        std::ofstream error_output_file(
            join_path(output_dir, "conformal_iteration_error.csv"),
            std::ios::out | std::ios::trunc);
        error_output_file.close();
        std::ofstream time_output_file(
            join_path(output_dir, "conformal_iteration_times.csv"),
            std::ios::out | std::ios::trunc);
        time_output_file.close();
    }

    // Perform initial conformal projection to get initial metric coordinates
    spdlog::get("optimize_metric")->info("Performing initial projection to the constraint");
    std::unique_ptr<DifferentiableConeMetric> initial_constrained_cone_metric =
        initial_cone_metric.project_to_constraint(proj_params);
    assert(float_equal(compute_max_constraint(*initial_constrained_cone_metric), 0.0, 1e-6));
    spdlog::get("optimize_metric")->set_level(spdlog::level::info);

    // Log the initial energy
    Scalar initial_energy = opt_energy.energy(*initial_constrained_cone_metric);
    spdlog::get("optimize_metric")->info("Initial energy is {}", initial_energy);

    // Main optimization loop
    Scalar beta = beta_0;
    Scalar convergence_ratio = 1.0;
    VectorX reduced_metric_coords =
        initial_constrained_cone_metric->get_reduced_metric_coordinates();
    VectorX prev_reduced_metric_coords = reduced_metric_coords;
    VectorX gradient, descent_direction;
    VectorX prev_gradient(0);
    VectorX prev_descent_direction(0);
    igl::Timer timer;
    timer.start();
    int num_linear_solves = 0;
    for (int iter = 0; iter < num_iter; ++iter) {
        spdlog::get("optimize_metric")->info("\nStarting Iteration {}", iter);

        // Initialize clean copy of the cone metric with the original connectivity and current
        // coordinates
        std::unique_ptr<DifferentiableConeMetric> cone_metric =
            initial_cone_metric.set_metric_coordinates(reduced_metric_coords);

        // Get descent direction
        if (direction_choice == "projected_newton") {
            compute_projected_newton_descent_direction(
                *cone_metric,
                opt_energy,
                gradient,
                descent_direction);
            num_linear_solves++; // TODO Would be better to have near solver code
        } else if (direction_choice == "projected_gradient") {
            gradient = opt_energy.gradient(*cone_metric);
            descent_direction = project_descent_direction(*cone_metric, -gradient);
            num_linear_solves++; // TODO Would be better to have near solver code
        }
        convergence_ratio = compute_convergence_ratio(gradient, descent_direction);
        spdlog::get("optimize_metric")
            ->info("Descent direction found with norm {}", descent_direction.norm());

        // Optionally reduce the descent direction to a given range
        Scalar grad_range = beta * (descent_direction.maxCoeff() - descent_direction.minCoeff());
        if ((max_grad_range > 0) && (grad_range >= max_grad_range)) {
            beta *= (max_grad_range / grad_range);
            spdlog::get("optimize_metric")->info("Reducing beta to {} for stability", beta);
        }

        // Perform the line search with projection to the constraint
        prev_reduced_metric_coords = reduced_metric_coords;
        reduced_metric_coords = line_search_with_projection(
            *cone_metric,
            opt_energy,
            descent_direction,
            proj_params,
            opt_params,
            beta,
            num_linear_solves);

        // Write iteration data if output directory specified
        log.num_iterations = iter;
        log.beta = beta;
        log.time = timer.getElapsedTime();
        log.num_linear_solves = num_linear_solves;
        update_data_log(
            log,
            *cone_metric,
            opt_energy,
            reduced_metric_coords,
            prev_reduced_metric_coords,
            gradient,
            descent_direction,
            convergence_ratio);
        if (!output_dir.empty()) {
            write_data_log_entry(log, data_log_path);
        }

        // Optionally write data per iteration
        // TODO

        // Check for convergence
        bool converged = check_if_converged(*opt_params, convergence_ratio, beta);
        if (converged) break;

        // Update beta
        beta = std::min<Scalar>(2.0 * beta, opt_params->max_beta);
        prev_gradient = gradient;
        prev_descent_direction = descent_direction; // Could also use unconstrained
    }

    // Deregister loggers
    spdlog::drop("mesh_data");
    spdlog::drop("optimize_metric");

    // Update final optimized metric coordinates
    return initial_cone_metric.set_metric_coordinates(reduced_metric_coords);
}

std::unique_ptr<DifferentiableConeMetric> optimize_metric(
    const DifferentiableConeMetric& initial_cone_metric,
    const EnergyFunctor& opt_energy,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::shared_ptr<OptimizationParameters> opt_params)
{
    OptimizationLog log;
    return optimize_metric_log(initial_cone_metric, opt_energy, log, proj_params, opt_params);
}

} // namespace CurvatureMetric
