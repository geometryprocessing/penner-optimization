#include "holonomy/holonomy/newton.h"

#include "holonomy/holonomy/constraint.h"
#include "holonomy/holonomy/holonomy.h"
#include "holonomy/core/viewer.h"
#include "holonomy/core/vector.h"

#include <nlohmann/json.hpp>
#include "optimization/metric_optimization/energies.h"
#include "optimization/metric_optimization/energy_functor.h"
#include "optimization/core/projection.h"
#include "optimization/core/shear.h"
#include "optimization/core/io.h"

#ifdef USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#include <Eigen/SPQRSupport>
#endif

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace PennerHolonomy {

// Initialize logging level
void OptimizeHolonomyNewton::initialize_logging() {
    switch (alg_params.log_level) {
        case 6: spdlog::set_level(spdlog::level::trace); break;
        case 5: spdlog::set_level(spdlog::level::debug); break;
        case 4: spdlog::set_level(spdlog::level::info); break;
        case 3: spdlog::set_level(spdlog::level::warn); break;
        case 2: spdlog::set_level(spdlog::level::err); break;
        case 1: spdlog::set_level(spdlog::level::critical); break;
        case 0: spdlog::set_level(spdlog::level::off); break;
    }
}

void OptimizeHolonomyNewton::initialize_metric_status_log(MarkedPennerConeMetric& marked_metric)
{
    // Open main logging file
    std::string data_log_path = CurvatureMetric::join_path(alg_params.output_dir, "metric_status_log.csv");
    spdlog::info("Writing data to {}", data_log_path);
    metric_status_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
    marked_metric.write_status_log(metric_status_file, true);
}

// Open a per iteration data log and write a header
void OptimizeHolonomyNewton::initialize_data_log()
{
    // Do nothing if error logging disabled
    if (!alg_params.error_log) return;

    // Generate data log path
    std::filesystem::create_directory(alg_params.output_dir);
    std::string data_log_path;

    // Open main logging file
    data_log_path = CurvatureMetric::join_path(alg_params.output_dir, "iteration_data_log.csv");
    spdlog::info("Writing data to {}", data_log_path);
    log_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
    log_file << "num_iter,";
    log_file << "max_error,";
    log_file << "step_size,";
    log_file << "rmsre,";
    log_file << "time,";
    log_file << "solve_time,";
    log_file << std::endl;
}

// Write newton log iteration data to file
void OptimizeHolonomyNewton::write_data_log_entry()
{
    // Do nothing if error logging disabled
    if (!alg_params.error_log) return;

    // Write iteration row
    log_file << log.num_iter << ",";
    log_file << std::fixed << std::setprecision(17) << log.max_error << ",";
    log_file << std::fixed << std::setprecision(17) << log.step_size << ",";
    log_file << std::fixed << std::setprecision(17) << log.rmsre << ",";
    log_file << std::fixed << std::setprecision(17) << log.time << ",";
    log_file << std::fixed << std::setprecision(17) << log.solve_time << ",";
    log_file << std::endl;
}

void OptimizeHolonomyNewton::initialize_timing_log()
{
    // Open timing logging file
    std::string data_log_path = CurvatureMetric::join_path(alg_params.output_dir, "iteration_timing_log.csv");
    spdlog::info("Writing timing data to {}", data_log_path);
    timing_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
    timing_file << "time,";
    timing_file << "solve_time,";
    timing_file << "constraint_time,";
    timing_file << "direction_time,";
    timing_file << "line_search_time,";
    timing_file << std::endl;
}

// Write newton log iteration data to file
void OptimizeHolonomyNewton::write_timing_log_entry()
{
    // Do nothing if error logging disabled
    if (!alg_params.error_log) return;

    // Write iteration row
    timing_file << std::fixed << std::setprecision(8) << log.time << ",";
    timing_file << std::fixed << std::setprecision(8) << log.solve_time << ",";
    timing_file << std::fixed << std::setprecision(8) << log.constraint_time << ",";
    timing_file << std::fixed << std::setprecision(8) << log.direction_time << ",";
    timing_file << std::fixed << std::setprecision(8) << log.line_search_time << ",";
    timing_file << std::endl;
}


void OptimizeHolonomyNewton::initialize_energy_log()
{
    std::string data_log_path = CurvatureMetric::join_path(alg_params.output_dir, "iteration_energy_log.csv");
    spdlog::info("Writing energy data to {}", data_log_path);
    energy_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
    energy_file << "l2_energy,";
    energy_file << "rmsre,";
    energy_file << "rmse,";
    energy_file << "rrmse,";
    energy_file << std::endl;
}

// Write newton log iteration data to file
void OptimizeHolonomyNewton::write_energy_log_entry()
{
    // Do nothing if error logging disabled
    if (!alg_params.error_log) return;

    energy_file << std::fixed << std::setprecision(8) << log.l2_energy << ",";
    energy_file << std::fixed << std::setprecision(8) << log.rmsre << ",";
    energy_file << std::fixed << std::setprecision(8) << log.rmse << ",";
    energy_file << std::fixed << std::setprecision(8) << log.rrmse << ",";
    energy_file << std::endl;
}

void OptimizeHolonomyNewton::initialize_stability_log()
{
    std::string data_log_path = CurvatureMetric::join_path(alg_params.output_dir, "iteration_stability_log.csv");
    spdlog::info("Writing stability data to {}", data_log_path);
    stability_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
    stability_file << "max_error,";
    stability_file << "step_size,";
    stability_file << "num_flips,";
    stability_file << "min_corner_angle,";
    stability_file << "max_corner_angle,";
    stability_file << "direction_angle_change,";
    stability_file << "direction_norm,";
    stability_file << "direction_residual,";
    stability_file << std::endl;

}

// Write newton log iteration data to file
void OptimizeHolonomyNewton::write_stability_log_entry()
{
    // Do nothing if error logging disabled
    if (!alg_params.error_log) return;

    // Write iteration row
    stability_file << std::scientific << std::setprecision(8) << log.max_error << ",";
    stability_file << std::scientific << std::setprecision(8) << log.step_size << ",";
    stability_file << log.num_flips << ",";
    stability_file << std::scientific << std::setprecision(8) << log.min_corner_angle << ",";
    stability_file << std::scientific << std::setprecision(8) << log.max_corner_angle << ",";
    stability_file << std::scientific << std::setprecision(8) << log.direction_angle_change << ",";
    stability_file << std::scientific << std::setprecision(8) << log.direction_norm << ",";
    stability_file << std::scientific << std::setprecision(8) << log.direction_residual << ",";
    stability_file << std::endl;
}

// Open all logs
void OptimizeHolonomyNewton::initialize_logs()
{
    initialize_data_log();
    initialize_timing_log();
    initialize_energy_log();
    initialize_stability_log();
}

void OptimizeHolonomyNewton::write_log_entries()
{
    write_data_log_entry();
    write_timing_log_entry();
    write_energy_log_entry();
    write_stability_log_entry();
}

// Close the error log file
void OptimizeHolonomyNewton::close_logs()
{
    // Do nothing if error logging disabled
    if (!alg_params.error_log) return;

    log_file.close();
    timing_file.close();
    energy_file.close();
    stability_file.close();
}

// Prepare output directory for checkpoints
void OptimizeHolonomyNewton::initialize_checkpoints()
{
    // Do nothing if checkpointing disabled
    if (alg_params.checkpoint_frequency <= 0) return;

    // Create output directory for checkpoints
    checkpoint_dir = CurvatureMetric::join_path(alg_params.output_dir, "checkpoint");
    std::filesystem::create_directory(checkpoint_dir);
}

// Write metric and descent direction data to file
// WARNING: Assumes the written data is updated and consistent
void OptimizeHolonomyNewton::checkpoint_direction()
{
    // Do nothing if this is not a checkpointing iteration
    if (alg_params.checkpoint_frequency <= 0) return;
    if ((log.num_iter % alg_params.checkpoint_frequency) != 0) return;
    std::string checkpoint_path;
    std::string suffix = std::to_string(log.num_iter);

    // Write metric coordinates
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "metric_coords_" + suffix);
    write_vector(reduced_metric_coords, checkpoint_path);

    // Write corner angles
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "angles_" + suffix);
    write_vector(alpha, checkpoint_path);

    // Write constraint vector
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "constraint_" + suffix);
    write_vector(constraint, checkpoint_path);

    // Write descent direction
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "direction_" + suffix);
    write_vector(descent_direction, checkpoint_path);

    // Write Jacobian
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "jacobian_" + suffix);
    CurvatureMetric::write_sparse_matrix(J, checkpoint_path);
}

void OptimizeHolonomyNewton::checkpoint_metric(const MarkedPennerConeMetric& marked_metric) {
    // Do nothing if this is not a checkpointing iteration
    if (alg_params.checkpoint_frequency <= 0) return;
    if ((log.num_iter % alg_params.checkpoint_frequency) != 0) return;
    std::string checkpoint_path;
    std::string suffix = std::to_string(log.num_iter);

    // Write best fit scale factors
    int num_halfedges = marked_metric.n_halfedges();
    VectorX scale_factors = CurvatureMetric::best_fit_conformal(marked_metric, VectorX::Zero(num_halfedges));
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "scale_factors_" + suffix);
    write_vector(scale_factors, checkpoint_path);

    // Write edge shears
    int num_edges = marked_metric.n_edges();
    MatrixX shear_dual_matrix;
    std::vector<int> edges;
    CurvatureMetric::arange(num_edges, edges);
    CurvatureMetric::compute_shear_dual_matrix(marked_metric, edges, shear_dual_matrix);
    VectorX metric_coords = marked_metric.get_metric_coordinates();
    VectorX shears = shear_dual_matrix.transpose() * metric_coords;
    checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, "shears_" + suffix);
    write_vector(shears, checkpoint_path);

    // Write dual loop face sequences
    for (int i = 0; i < marked_metric.n_homology_basis_loops(); ++i) {
        std::string checkpoint_file = "dual_loop_" + std::to_string(i) + "_" + suffix;
        checkpoint_path = CurvatureMetric::join_path(checkpoint_dir, checkpoint_file);
        write_vector(
            marked_metric.get_homology_basis_loops()[i]->generate_face_sequence(marked_metric),
            checkpoint_path);
    }
}

// Update the holonomy and length error log data
void OptimizeHolonomyNewton::update_log_error(const MarkedPennerConeMetric& marked_metric)
{
    // Get edge lengths
    int num_edges = reduced_metric_coords.size();
    VectorX l_init(num_edges);
    VectorX l(num_edges);
    for (int E = 0; E < num_edges; ++E) {
        l_init[E] = exp(reduced_metric_init[E] / 2.0);
        l[E] = exp(reduced_metric_coords[E] / 2.0);
    }

    // Update holonomy error
    log.max_error = constraint.cwiseAbs().maxCoeff();

    // Update metric error
    log.l2_energy = l2_energy->EnergyFunctor::energy(marked_metric);
    log.rmse = CurvatureMetric::root_mean_square_error(l, l_init);
    log.rrmse = CurvatureMetric::relative_root_mean_square_error(l, l_init);
    log.rmsre = CurvatureMetric::root_mean_square_relative_error(l, l_init);

    // Update corner angle measurements
    log.min_corner_angle = alpha.minCoeff();
    log.max_corner_angle = alpha.maxCoeff();

    // Update changes in angle for the gradient and direction
    auto cos_angle = [](const VectorX& v, const VectorX& w)
    {
        return acos(v.dot(w) / (v.norm() * w.norm()));
    };
    if (log.num_iter > 1)
    {
        log.direction_angle_change = cos_angle(descent_direction, prev_descent_direction);
    }
}

void OptimizeHolonomyNewton::solve_linear_system(const MatrixX& metric_basis_matrix)
{
    // Make matrix for optimization

    // TODO: Split into individual methods
    // Generally unstable and slow dense matrix approach
    // WARNING: Only use for debugging
    spdlog::debug("Using {} solver", alg_params.solver);
    if (alg_params.solver == "dense_qr") {
#ifndef MULTIPRECISION
        Eigen::MatrixXd A = J * metric_basis_matrix;
        double t_solve_start = timer.getElapsedTime();
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(A);
        VectorX rhs = solver.solve(constraint);
        log.solve_time = timer.getElapsedTime() - t_solve_start;

        descent_direction = -metric_basis_matrix * rhs;
#else
        spdlog::error("Dense QR solver not supported for multiprecision");
#endif
    }
    // QR based minimum norm computation: more numerically stable than direct pseudo-inverse
    else if (alg_params.solver == "qr") {
#ifdef USE_SUITESPARSE
        typedef int32_t Int;
        typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Int> PinvMatrix;

        // Make cholmod views of the linear system
        PinvMatrix A = J * metric_basis_matrix;
        A.makeCompressed();
        PinvMatrix g = constraint.sparseView();
        cholmod_sparse M =
            Eigen::viewAsCholmod<Scalar, Eigen::ColMajor, Int>(Eigen::Ref<PinvMatrix>(A));
        cholmod_sparse b =
            Eigen::viewAsCholmod<Scalar, Eigen::ColMajor, Int>(Eigen::Ref<PinvMatrix>(g));

        // Run SuiteSparse QR method
        double t_solve_start = timer.getElapsedTime();
        int ordering = 7;
        Scalar pivotThreshold = -2;
        cholmod_common m_cc; // Workspace and parameters
        cholmod_start(&m_cc);
        cholmod_sparse* cholmod_descent_direction =
            SuiteSparseQR_min2norm<Scalar, Int>(ordering, pivotThreshold, &M, &b, &m_cc);
        log.solve_time = timer.getElapsedTime() - t_solve_start;

        // Copy descent direction to Eigen vector
        descent_direction =
            -metric_basis_matrix *
            Eigen::viewAsEigen<Scalar, Eigen::ColMajor, Int>(*cholmod_descent_direction);
#else
        spdlog::error("QR with SuiteSparse not available. Set USE_SUITESPARSE to use.");
#endif
    } else if (alg_params.solver == "cholmod") {
#ifdef USE_SUITESPARSE
        // Build pseudo-inverse
        MatrixX A = J * metric_basis_matrix;
        MatrixX AAt = A * A.transpose();
        AAt.makeCompressed();

        // Solve for descent direction (timer consistent with conformal)
        double t_solve_start = timer.getElapsedTime();
        Eigen::CholmodSupernodalLLT<MatrixX> solver;
        solver.compute(AAt);
        VectorX rhs = solver.solve(constraint);
        log.solve_time = timer.getElapsedTime() - t_solve_start;
        if (solver.info() != Eigen::Success) spdlog::error("Solve failed");

        descent_direction = -metric_basis_matrix * (A.transpose() * rhs);
#else
        spdlog::error("Cholmod with SuiteSparse not available. Set USE_SUITESPARSE to use.");
#endif
    } else {
        // Build pseudo-inverse
        MatrixX A = J * metric_basis_matrix;
        MatrixX AAt = A * A.transpose();

        // Solve for descent direction (timer consistent with conformal)
        double t_solve_start = timer.getElapsedTime();
        Eigen::SimplicialLDLT<MatrixX> solver;
        solver.compute(AAt);
        VectorX rhs = solver.solve(constraint);
        log.solve_time = timer.getElapsedTime() - t_solve_start;

        descent_direction = -metric_basis_matrix * (A.transpose() * rhs);
    }
}

// Determine initial lambda for next line search based on method parameters
void OptimizeHolonomyNewton::update_lambda()
{
    if (alg_params.reset_lambda) {
        lambda = alg_params.lambda0;
    } else {
        lambda = std::min<Scalar>(1, 2 * lambda); // adaptive step length
    }
}

// Update the corner angles and constraint given the marked metric
void OptimizeHolonomyNewton::update_holonomy_constraint(MarkedPennerConeMetric& marked_metric)
{
    // TODO Make method
    // Check current metric coordinates for validity
    if (vector_contains_nan(reduced_metric_coords)) {
        spdlog::error("Coordinates contain NaN");
    }
    assert(!vector_contains_nan(reduced_metric_coords));
    SPDLOG_DEBUG(
        "Coordinates in range [{}, {}]",
        reduced_metric_coords.minCoeff(),
        reduced_metric_coords.maxCoeff());
    SPDLOG_DEBUG("Coordinates have average {}", reduced_metric_coords.mean());

    // Get corner angles and metric constraints
    marked_metric.make_discrete_metric();
    log.num_flips = marked_metric.num_flips();
    marked_metric.get_corner_angles(alpha, cot_alpha);
    constraint = marked_metric.constraint(alpha);

    // TODO Make method
    // Check angles for validity
    assert(!vector_contains_nan(alpha));
    SPDLOG_DEBUG("Angles in range [{}, {}]", alpha.minCoeff(), alpha.maxCoeff());
    SPDLOG_DEBUG("Angles have average {}", alpha.mean());
}


// Update the corner angles, constraint, constraint jacobian, and descent direction given the
// marked metric
void OptimizeHolonomyNewton::update_descent_direction(
    MarkedPennerConeMetric& marked_metric,
    const MatrixX& metric_basis_matrix)
{
    double t_start;
    prev_descent_direction = descent_direction;

    // Compute corner angles and the constraint with its jacobian
    // TODO Add safety checks
    t_start = timer.getElapsedTime();
    marked_metric.make_discrete_metric();
    marked_metric.get_corner_angles(alpha, cot_alpha);
    constraint = marked_metric.constraint(alpha);
    J = marked_metric.constraint_jacobian(cot_alpha);
    log.constraint_time = timer.getElapsedTime() - t_start;
    marked_metric.write_status_log(metric_status_file);

    // Compute Newton descent direction from the constraint and jacobian
    t_start = timer.getElapsedTime();

    SPDLOG_DEBUG("Jacobian with maximum value {}", J.coeffs().maxCoeff());
    SPDLOG_DEBUG("Jacobian with mean {}", J.coeffs().mean());
    SPDLOG_DEBUG("Metric basis with average {}", metric_basis_matrix.coeffs().mean());
    SPDLOG_DEBUG("Constraint with average {}", constraint.mean());

    solve_linear_system(metric_basis_matrix);

    SPDLOG_TRACE("Descent direction found with norm {}", descent_direction.norm());
    SPDLOG_TRACE("Descent direction error is {}", (J * descent_direction + constraint).norm());
    SPDLOG_TRACE("Projected constraint is {}", constraint.dot(J * descent_direction));
    log.direction_time = timer.getElapsedTime() - t_start;
    log.direction_norm = descent_direction.norm();
    log.direction_residual = (J * descent_direction + constraint).norm();
}

// Perform a backtracking line search along the current descent direction from the current
// metric coordinates using the initial metric connectivity
void OptimizeHolonomyNewton::perform_line_search(
    const MarkedPennerConeMetric& initial_marked_metric,
    MarkedPennerConeMetric& marked_metric)
{
    double t_start = timer.getElapsedTime();

    // Get starting  metric coordinates
    VectorX reduced_metric_start = reduced_metric_coords;

    // Get the constraint norm and its dot product with the jacobian-projected descent direction
    // Note: the product of the jacobian and descent direction should be the negative
    // constraint, but this may fail due to numerical instability or regularization
    Scalar l2_c0_sq = constraint.squaredNorm();
    Scalar proj_g0 = constraint.dot(J * descent_direction);
    spdlog::debug("Initial squared error norm is {}", l2_c0_sq);
    spdlog::debug("Initial projected constraint is {}", proj_g0);

    // Reduce descent direction range to avoid nans/infs
    if (alg_params.do_reduction) {
        while (lambda * (descent_direction.maxCoeff() - descent_direction.minCoeff()) > 2.5) {
            lambda /= 2;
            spdlog::debug("Reducing lambda to {} for stability", lambda);
        }
    }

    // Make initial line step with updated constraint
    reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
    marked_metric.change_metric(initial_marked_metric, reduced_metric_coords, false, false);
    update_holonomy_constraint(marked_metric);

    // Line search until the constraint norm decreases and the projected constraint is
    // nonpositive We also allow the norm bound to be dropped or made approximate with some
    // relative term alpha
    spdlog::debug("Starting line search");
    Scalar l2_c_sq = constraint.squaredNorm();
    Scalar proj_cons = constraint.dot(J * descent_direction);
    Scalar gamma = 1.; // require ratio of current to initial constraint norm to be below alpha
    bool bound_norm = true;
    while ((bound_norm && (l2_c_sq > (gamma * l2_c0_sq))) || (proj_cons > 0)) {
        // Backtrack one step
        lambda /= 2;

        // If lambda low enough, stop bounding the norm
        if (lambda <= alg_params.bound_norm_thres) {
            bound_norm = false;
            spdlog::debug("Dropping norm bound.");
        }

        // Change metric and update constraint
        reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
        marked_metric.change_metric(initial_marked_metric, reduced_metric_coords, false, false);
        update_holonomy_constraint(marked_metric);

        // Update squared constraint norm and projected constraint
        l2_c_sq = constraint.squaredNorm();
        proj_cons = constraint.dot(J * descent_direction);
        spdlog::debug("Squared error norm is {}", l2_c_sq);
        spdlog::debug("Projected constraint is {}", proj_cons);

        // Check if lambda is below the termination threshold
        if (lambda < alg_params.min_lambda) break;
    }

    // Make final line step in original connectivity
    spdlog::debug("Updating metric");
    reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
    marked_metric.change_metric(initial_marked_metric, reduced_metric_coords, true, false);

    log.line_search_time = timer.getElapsedTime() - t_start;
}

// Determine if the optimization has converged or maximum time/iteration is reached
bool OptimizeHolonomyNewton::is_converged()
{
    if (constraint.cwiseAbs().maxCoeff() < alg_params.error_eps) {
        spdlog::info("Stopping optimization as max error {} reached", alg_params.error_eps);
        return true;
    }
    if (lambda < alg_params.min_lambda) {
        spdlog::info("Stopping optimization as step size {} too small", lambda);
        return true;
    }
    if (log.num_iter >= alg_params.max_itr) {
        spdlog::trace(
            "Stopping optimization as reached maximum iteration {}",
            alg_params.max_itr);
        return true;
    }
    if (timer.getElapsedTime() >= alg_params.max_time) {
        spdlog::trace("Stopping optimization as reached maximum time {}", alg_params.max_time);
        return true;
    }

    return false;
}

// Main method to run the optimization with a given metric, metric basis, and parameters
// Each run completely resets the state of the optimization
MarkedPennerConeMetric OptimizeHolonomyNewton::run(
    const MarkedPennerConeMetric& initial_marked_metric,
    const MatrixX& metric_basis_matrix,
    const NewtonParameters& input_alg_params)
{
    // Initialize logging methods
    timer.start();
    alg_params = input_alg_params;
    lambda = alg_params.lambda0;
    l2_energy = std::make_unique<CurvatureMetric::LogLengthEnergy>(
        CurvatureMetric::LogLengthEnergy(initial_marked_metric));
    initialize_logging();
    initialize_logs();
    initialize_checkpoints();
    checkpoint_metric(initial_marked_metric);

    // Get initial metric
    std::unique_ptr<MarkedPennerConeMetric> marked_metric = initial_marked_metric.clone_marked_metric();
    reduced_metric_init = marked_metric->get_reduced_metric_coordinates();
    reduced_metric_coords = reduced_metric_init;
    initialize_metric_status_log(*marked_metric);

    // Get initial constraint
    update_holonomy_constraint(*marked_metric);

    // Get before-first-iteration information
    update_log_error(*marked_metric);
    write_log_entries();
    spdlog::info("itr(0) lm({}) max_error({}))", lambda, log.max_error);

    int itr = 0;
    while (true) {
        // Check termination conditions
        if (is_converged()) break;

        // Increment iteration
        itr++;
        log.num_iter = itr;

        // Compute Newton descent direction
        update_descent_direction(*marked_metric, metric_basis_matrix);

        // Checkpoint current state
        // WARNING: Must be done after updating descent direction and before line search
        checkpoint_direction();

        // Search for updated metric, constraint, and angles
        perform_line_search(initial_marked_metric, *marked_metric);

        // Log current step
        log.step_size = lambda;
        log.time = timer.getElapsedTime();
        update_log_error(*marked_metric);
        write_log_entries();
        checkpoint_metric(*marked_metric);
        spdlog::info("itr({}) lm({}) max_error({}))", itr, lambda, log.max_error);

        // Update lambda
        update_lambda();
    }

    // Close logging
    close_logs();
    metric_status_file.close();

    // Change metric to final values and restore the original connectivity
    marked_metric->change_metric(initial_marked_metric, reduced_metric_coords, true, false);

    return *marked_metric;
}


MarkedPennerConeMetric optimize_metric_angles(
    const MarkedPennerConeMetric& initial_marked_metric,
    const NewtonParameters& alg_params)
{
    // Optimize metric with full metric space (basis is identity)
    MatrixX identity = CurvatureMetric::id_matrix(initial_marked_metric.n_reduced_coordinates());
    OptimizeHolonomyNewton solver;
    return solver.run(initial_marked_metric, identity, alg_params);
}

MarkedPennerConeMetric optimize_subspace_metric_angles(
    const MarkedPennerConeMetric& initial_marked_metric,
    const MatrixX& metric_basis_matrix,
    const NewtonParameters& alg_params)
{
    OptimizeHolonomyNewton solver;
    return solver.run(initial_marked_metric, metric_basis_matrix, alg_params);
}

MarkedPennerConeMetric optimize_subspace_metric_angles_log(
    const MarkedPennerConeMetric& initial_marked_metric,
    const MatrixX& metric_basis_matrix,
    const NewtonParameters& alg_params,
    NewtonLog& log)
{
    OptimizeHolonomyNewton solver;
    MarkedPennerConeMetric marked_metric =
        solver.run(initial_marked_metric, metric_basis_matrix, alg_params);
    log = solver.get_log();
    return marked_metric;
}

void view_optimization_state(
    const MarkedPennerConeMetric& init_marked_metric,
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle,
    bool show)
{
    int num_ind_vertices = V.rows();
    if (show) {
        spdlog::info("Viewing mesh {} with {} vertices", mesh_handle, num_ind_vertices);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, marked_metric, vtx_reindex);
    
    // get constraint errors
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = false;
    marked_metric.constraint(constraint, J_constraint, need_jacobian, only_free_vertices);
    VectorX scale_coords = best_fit_conformal(init_marked_metric, marked_metric.get_metric_coordinates());

    // extend vertex angles to full angle map
    int num_vertices = marked_metric.n_vertices();
    VectorX angle_constraint(num_vertices);
    VectorX scale_distortion(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        angle_constraint[vi] = constraint[marked_metric.v_rep[vi]];
        scale_distortion[vi] = scale_coords[marked_metric.v_rep[vi]];
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    if (mesh_handle == "") {
        mesh_handle = "optimization state";
    }
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity(
            "angle error",
            CurvatureMetric::convert_scalar_to_double_vector(angle_constraint))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity(
            "scale distortion",
            CurvatureMetric::convert_scalar_to_double_vector(scale_distortion))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    if (show) polyscope::show();
#endif
}

} // namespace PennerHolonomy
