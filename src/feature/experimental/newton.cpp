#include "feature/experimental/newton.h"

#include "holonomy/holonomy/holonomy.h"
#include "holonomy/holonomy/newton.h"
#include "holonomy/holonomy/constraint.h"
#include "feature/dirichlet/constraint.h"

#include <igl/Timer.h>
#include <nlohmann/json.hpp>
#include "optimization/metric_optimization/energies.h"
#include "optimization/metric_optimization/energy_functor.h"
#include "optimization/core/projection.h"
#include "optimization/core/shear.h"
#include "util/io.h"

#ifdef USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#include <Eigen/SPQRSupport>
#endif

namespace Penner {
namespace Feature {

// TODO: Deprecated method; should avoid unless substantial changes needed for logging
// or methodology changes. The override approach used here avoids code duplication, but may
// cause undesireable class hierarchies if abused

class OptimizeDirichletNewton : public Holonomy::OptimizeHolonomyNewton
{
public:
    // Update the corner angles and holonomy constraint given the marked metric
    void update_holonomy_constraint(DirichletPennerConeMetric& dirichlet_metric)
    {
        // Get corner angles and metric constraints
        dirichlet_metric.make_discrete_metric();
        log.num_flips = dirichlet_metric.num_flips();
        dirichlet_metric.get_corner_angles(alpha, cot_alpha);
        holonomy_constraint = compute_metric_constraint(dirichlet_metric, alpha);
    }

    // Update the corner angles and boundary constraint given the marked metric
    void update_boundary_constraint(DirichletPennerConeMetric& dirichlet_metric)
    {
        // Get corner angles and metric constraints
        dirichlet_metric.make_discrete_metric();
        log.num_flips = dirichlet_metric.num_flips();
        dirichlet_metric.get_corner_angles(alpha, cot_alpha);
        boundary_constraint = compute_boundary_constraint(dirichlet_metric);
    }

    void update_all_constraint(DirichletPennerConeMetric& dirichlet_metric)
    {
        dirichlet_metric.make_discrete_metric();
        log.num_flips = dirichlet_metric.num_flips();
        dirichlet_metric.get_corner_angles(alpha, cot_alpha);
        constraint = dirichlet_metric.constraint(alpha);
    }

    void update_dirichlet_constraint(DirichletPennerConeMetric& dirichlet_metric)
    {
        if (constraint_type == ConstraintType::holonomy) 
        {
            update_holonomy_constraint(dirichlet_metric);
            constraint = holonomy_constraint;
        }
        else if (constraint_type == ConstraintType::boundary) 
        {
            update_boundary_constraint(dirichlet_metric);
            constraint = boundary_constraint;
        }
        else if (constraint_type == ConstraintType::all) 
        {
            update_all_constraint(dirichlet_metric);
        }
    }


    // Update the corner angles, constraint, constraint jacobian, and descent direction given the
    // marked metric
    void update_holonomy_descent_direction(
        DirichletPennerConeMetric& dirichlet_metric,
        const MatrixX& metric_basis_matrix)
    {
        double t_start;
        prev_descent_direction = descent_direction;

        // Compute corner angles and the constraint with its jacobian
        // TODO Add safety checks
        t_start = timer.getElapsedTime();
        dirichlet_metric.make_discrete_metric();
        dirichlet_metric.get_corner_angles(alpha, cot_alpha);
        constraint = compute_metric_constraint(dirichlet_metric, alpha);
        J = compute_metric_constraint_jacobian(dirichlet_metric, cot_alpha);
        log.constraint_time = timer.getElapsedTime() - t_start;
        dirichlet_metric.write_status_log(metric_status_file);

        // Compute Newton descent direction from the constraint and jacobian
        t_start = timer.getElapsedTime();
        solve_linear_system(metric_basis_matrix);
        log.direction_time = timer.getElapsedTime() - t_start;
        log.direction_norm = descent_direction.norm();
        log.direction_residual = (J * descent_direction + constraint).norm();
    }

    // Update the corner angles, constraint, constraint jacobian, and descent direction given the
    // marked metric
    void update_boundary_descent_direction(
        DirichletPennerConeMetric& dirichlet_metric,
        const MatrixX& metric_basis_matrix)
    {
        double t_start;
        prev_descent_direction = descent_direction;

        // Compute corner angles and the constraint with its jacobian
        // TODO Add safety checks
        t_start = timer.getElapsedTime();
        dirichlet_metric.make_discrete_metric();
        constraint = compute_boundary_constraint(dirichlet_metric);
        J = compute_boundary_constraint_jacobian(dirichlet_metric);
        log.constraint_time = timer.getElapsedTime() - t_start;
        dirichlet_metric.write_status_log(metric_status_file);

        // Compute Newton descent direction from the constraint and jacobian
        t_start = timer.getElapsedTime();
        solve_linear_system(metric_basis_matrix);
        log.direction_time = timer.getElapsedTime() - t_start;
        log.direction_norm = descent_direction.norm();
        log.direction_residual = (J * descent_direction + constraint).norm();
    }

    // Update the corner angles, constraint, constraint jacobian, and descent direction given the
    // marked metric
    void update_all_descent_direction(
        DirichletPennerConeMetric& dirichlet_metric,
        const MatrixX& metric_basis_matrix)
    {
        double t_start;
        prev_descent_direction = descent_direction;

        // Compute corner angles and the constraint with its jacobian
        // TODO Add safety checks
        t_start = timer.getElapsedTime();
        dirichlet_metric.make_discrete_metric();
        dirichlet_metric.get_corner_angles(alpha, cot_alpha);
        constraint = dirichlet_metric.constraint(alpha);
        J = dirichlet_metric.constraint_jacobian(cot_alpha);
        log.constraint_time = timer.getElapsedTime() - t_start;
        dirichlet_metric.write_status_log(metric_status_file);

        // Compute Newton descent direction from the constraint and jacobian
        t_start = timer.getElapsedTime();
        solve_linear_system(metric_basis_matrix);
        log.direction_time = timer.getElapsedTime() - t_start;
        log.direction_norm = descent_direction.norm();
        log.direction_residual = (J * descent_direction + constraint).norm();
    }
  
    void update_dirichlet_descent_direction(
        DirichletPennerConeMetric& dirichlet_metric,
        const MatrixX& metric_basis_matrix)
    {
        if (constraint_type == ConstraintType::holonomy) 
        {
            update_holonomy_descent_direction(dirichlet_metric, metric_basis_matrix);
        }
        else if (constraint_type == ConstraintType::boundary) 
        {
            update_boundary_descent_direction(dirichlet_metric, metric_basis_matrix);
        }
        else if (constraint_type == ConstraintType::all) 
        {
            update_all_descent_direction(dirichlet_metric, metric_basis_matrix);
        }
    }

    // Perform a backtracking line search along the current descent direction from the current
    // metric coordinates using the initial metric connectivity
    void perform_dirichlet_line_search(
        const DirichletPennerConeMetric& initial_dirichlet_metric,
        DirichletPennerConeMetric& dirichlet_metric,
        const MatrixX& metric_basis_matrix)
    {
        double t_start = timer.getElapsedTime();

        // Get starting  metric coordinates
        VectorX reduced_metric_start = reduced_metric_coords;
        VectorX d = descent_direction;

        // Get the constraint norm and its dot product with the jacobian-projected descent direction
        // Note: the product of the jacobian and descent direction should be the negative
        // constraint, but this may fail due to numerical instability or regularization
        Scalar l2_c0_sq = constraint.squaredNorm();
        Scalar proj_g0 = constraint.dot(J * descent_direction);
        Scalar Jg0_sq = (J.transpose() * constraint).squaredNorm();
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
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords, false, false);
        update_dirichlet_constraint(dirichlet_metric);

        // Ideas for line search
        // - Use descent direction norm (with linearized or actual jacobian)
        // - Use energy slack

        // Line search until the constraint norm decreases and the projected constraint is
        // nonpositive We also allow the norm bound to be dropped or made approximate with some
        // relative term alpha
        spdlog::debug("Starting line search");
        spdlog::debug("{}", metric_basis_matrix.norm());
        Scalar l2_c_sq = constraint.squaredNorm();
        Scalar proj_cons = constraint.dot(J * d);
        Scalar Jg_sq = (J.transpose() * constraint).squaredNorm();
        Scalar gamma = 1.; // require ratio of current to initial constraint norm to be below alpha
        bool bound_norm = true;
        while (
            (proj_cons > 0)
            || (bound_norm && (l2_c_sq > (gamma * l2_c0_sq)))
            || (0. * Jg_sq > Jg0_sq)
        ) {
            // Backtrack one step
            lambda /= 2;

            // If lambda low enough, stop bounding the norm
            if (lambda <= alg_params.bound_norm_thres) {
                bound_norm = false;
                spdlog::debug("Dropping norm bound.");
            }

            // Change metric and update constraint
            reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
            dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords, false, false);
            update_dirichlet_constraint(dirichlet_metric);

            // Update squared constraint norm and projected constraint
            l2_c_sq = constraint.squaredNorm();
            proj_cons = constraint.dot(J * d);
            Jg_sq = (J.transpose() * constraint).squaredNorm();
            spdlog::debug("Squared error norm is {}", l2_c_sq);
            spdlog::debug("Projected constraint is {}", proj_cons);

            // Check if lambda is below the termination threshold
            if (lambda < alg_params.min_lambda) break;
        }

        // Make final line step in original connectivity
        spdlog::debug("Updating metric");
        reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords, true, false);

        log.line_search_time = timer.getElapsedTime() - t_start;
    }

    // Determine if the optimization has converged or maximum time/iteration is reached
    bool is_converged()
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
    DirichletPennerConeMetric run(
        const DirichletPennerConeMetric& initial_dirichlet_metric,
        const MatrixX& metric_basis_matrix,
        const NewtonParameters& input_alg_params)
    {
        // Initialize logging methods
        timer.start();
        alg_params = input_alg_params;
        lambda = alg_params.lambda0;
        l2_energy = std::make_unique<Optimization::LogLengthEnergy>(
            Optimization::LogLengthEnergy(initial_dirichlet_metric));
        initialize_logging();
        initialize_logs();
        initialize_checkpoints();
        checkpoint_metric(initial_dirichlet_metric);

        // Get initial metric
        DirichletPennerConeMetric dirichlet_metric = initial_dirichlet_metric;
        reduced_metric_init = dirichlet_metric.get_reduced_metric_coordinates();
        reduced_metric_coords = reduced_metric_init;
        initialize_metric_status_log(dirichlet_metric);

        // Get initial constraint
        constraint_type = ConstraintType::all;
        update_dirichlet_constraint(dirichlet_metric);

        // Get before-first-iteration information
        update_log_error(dirichlet_metric);
        write_log_entries();
        spdlog::info("itr(0) lm({}) max_error({}))", lambda, log.max_error);

        int itr = 0;
        constraint_type = ConstraintType::all;
        while (true) {
            // Check termination conditions
            if ((is_converged()) && (constraint_type == ConstraintType::holonomy))
            {
                spdlog::info("Adding boundary constraints");
                constraint_type = ConstraintType::all;
            }
            else if ((is_converged()) && (constraint_type == ConstraintType::all))
            {
                break;
            }

            // Increment iteration
            itr++;
            log.num_iter = itr;

            // Do descent for holonomy constraint and boundary constraint separately
            bool alternate = false;
            if (alternate) {
                constraint_type = ConstraintType::holonomy;
                update_holonomy_descent_direction(dirichlet_metric, metric_basis_matrix);
                perform_dirichlet_line_search(initial_dirichlet_metric, dirichlet_metric, metric_basis_matrix);

                // Do descent for boundary constraint
                constraint_type = ConstraintType::boundary;
                update_boundary_descent_direction(dirichlet_metric, metric_basis_matrix);
                perform_dirichlet_line_search(initial_dirichlet_metric, dirichlet_metric, metric_basis_matrix);
            }

            // Do descent for all constraints
            update_dirichlet_descent_direction(dirichlet_metric, metric_basis_matrix);
            perform_dirichlet_line_search(initial_dirichlet_metric, dirichlet_metric, metric_basis_matrix);

            // Log current step
            log.step_size = lambda;
            log.time = timer.getElapsedTime();
            update_log_error(dirichlet_metric);
            write_log_entries();
            checkpoint_metric(dirichlet_metric);
            spdlog::info("itr({}) lm({}) max_error({}))", itr, lambda, log.max_error);

            // Update lambda
            update_lambda();
        }

        // Close logging
        close_logs();
        metric_status_file.close();

        // Change metric to final values and restore the original connectivity
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords, true, false);

        return dirichlet_metric;
    }


    // Trivial constructor (stateless optimization method)
    OptimizeDirichletNewton() {}

private:
    VectorX holonomy_constraint;
    VectorX boundary_constraint;
    enum ConstraintType {
        boundary,
        holonomy,
        all };
    ConstraintType constraint_type;
};

DirichletPennerConeMetric optimize_dirichlet_metric(
    const DirichletPennerConeMetric& initial_dirichlet_metric,
    const NewtonParameters& alg_params)
{
    // Optimize metric with full metric space (basis is identity)
    MatrixX identity = id_matrix(initial_dirichlet_metric.n_reduced_coordinates());
    OptimizeDirichletNewton solver;
    return solver.run(initial_dirichlet_metric, identity, alg_params);
}

} // namespace Feature
} // namespace Penner