#pragma once

#include "resolution/core/common.h"
#include "holonomy/core/common.h"
#include "holonomy/holonomy/newton.h"

namespace Penner {
namespace Resolution {

class OptimizeCorrectionNewton : public Holonomy::OptimizeNewton
{
public:
OptimizeCorrectionNewton() {}

protected:
    virtual void update_descent_direction(MarkedPennerConeMetric& marked_metric) override
    {
        double t_start;

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

        solve_linear_system();

        // add regularization term
        VectorX delta_metric = reduced_metric_coords - reduced_metric_init;
        MatrixX L = J * J.transpose();
        VectorX w = J * delta_metric;
        VectorX projection = J.transpose() * solve_psd_system(L, w);
        VectorX complement = delta_metric - projection;
        descent_direction -= complement;

        SPDLOG_TRACE("Descent direction found with norm {}", descent_direction.norm());
        SPDLOG_TRACE("Descent direction error is {}", (J * descent_direction + constraint).norm());
        SPDLOG_TRACE("Projected constraint is {}", constraint.dot(J * descent_direction));
        log.direction_time = timer.getElapsedTime() - t_start;
        log.direction_norm = descent_direction.norm();
        log.direction_residual = (J * descent_direction + constraint).norm();
    }

    virtual void perform_line_search(
        const MarkedPennerConeMetric& initial_marked_metric,
        MarkedPennerConeMetric& marked_metric) override
    {
        double t_start = timer.getElapsedTime();
        Scalar epsilon = 1. / (log.num_iter + 1);
        epsilon = 0.;

        // Get starting  metric coordinates
        VectorX reduced_metric_start = reduced_metric_coords;

        // Get the constraint norm and its dot product with the jacobian-projected descent direction
        // Note: the product of the jacobian and descent direction should be the negative
        // constraint, but this may fail due to numerical instability or regularization
        VectorX metric_delta = reduced_metric_coords - reduced_metric_init;
        Scalar l2_c0_sq = constraint.squaredNorm() + epsilon * metric_delta.squaredNorm();
        Scalar proj_g0 = descent_direction.dot(J.transpose() * constraint);
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
        update_constraint(marked_metric);

        // Line search until the constraint norm decreases and the projected constraint is
        // nonpositive We also allow the norm bound to be dropped or made approximate with some
        // relative term alpha
        spdlog::debug("Starting line search");
        metric_delta = reduced_metric_coords - reduced_metric_init;
        Scalar l2_c_sq = constraint.squaredNorm() + epsilon * metric_delta.squaredNorm();
        Scalar proj_cons = descent_direction.dot(J.transpose() * constraint);
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
            update_constraint(marked_metric);

            // Update squared constraint norm and projected constraint
            metric_delta = reduced_metric_coords - reduced_metric_init;
            l2_c_sq = constraint.squaredNorm() + epsilon * metric_delta.squaredNorm();
            proj_cons = descent_direction.dot(J.transpose() * constraint);
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
};

class OptimizeRegularizedNewton : public Holonomy::OptimizeNewton
{
public:
    OptimizeRegularizedNewton() {}
    Scalar epsilon = 0.01;
    Scalar max_grad = 0.;
    VectorX reduced_metric_base;

    void set_base_coordinates(const MarkedPennerConeMetric& marked_metric)
    {
        reduced_metric_base = marked_metric.get_reduced_metric_coordinates();
    }

protected:
    void initialize_data_log() override
    {
        // Do nothing if error logging disabled
        if (!alg_params.error_log) return;

        // Generate data log path
        std::filesystem::create_directory(alg_params.output_dir);
        std::string data_log_path;

        // Open main logging file
        data_log_path = join_path(alg_params.output_dir, "iteration_data_log.csv");
        spdlog::info("Writing data to {}", data_log_path);
        log_file = std::ofstream(data_log_path, std::ios::out | std::ios::app);
        log_file << "num_iter,";
        log_file << "max_error,";
        log_file << "max_grad,";
        log_file << "step_size,";
        log_file << "rmsre,";
        log_file << "time,";
        log_file << "solve_time,";
        log_file << std::endl;
    }

    // Write newton log iteration data to file
    void write_data_log_entry() override
    {
        // Do nothing if error logging disabled
        if (!alg_params.error_log) return;

        // Write iteration row
        log_file << log.num_iter << ",";
        log_file << std::fixed << std::setprecision(17) << log.max_error << ",";
        log_file << std::fixed << std::setprecision(17) << max_grad << ",";
        log_file << std::fixed << std::setprecision(17) << log.step_size << ",";
        log_file << std::fixed << std::setprecision(17) << log.rmsre << ",";
        log_file << std::fixed << std::setprecision(17) << log.time << ",";
        log_file << std::fixed << std::setprecision(17) << log.solve_time << ",";
        log_file << std::endl;
    }

    virtual void update_descent_direction(MarkedPennerConeMetric& marked_metric) override
    {
        double t_start;

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

        solve_linear_system();

        // add regularization term
        MatrixX identity = id_matrix(marked_metric.n_reduced_coordinates());
        MatrixX L = (J.transpose() * J) + (epsilon * identity);
        VectorX delta_metric = reduced_metric_coords - reduced_metric_base;
        VectorX w = (J.transpose() * constraint) + (epsilon * delta_metric);
        max_grad = sup_norm(w);
        spdlog::info("Gradient has 2-norm {}", w.norm());
        spdlog::info("Gradient has sup-norm {}", max_grad);
        descent_direction = -solve_psd_system(L, w);

        SPDLOG_TRACE("Descent direction found with norm {}", descent_direction.norm());
        SPDLOG_TRACE("Descent direction error is {}", (J * descent_direction + constraint).norm());
        SPDLOG_TRACE("Projected constraint is {}", constraint.dot(J * descent_direction));
        log.direction_time = timer.getElapsedTime() - t_start;
        log.direction_norm = descent_direction.norm();
        log.direction_residual = (J * descent_direction + constraint).norm();
    }

    virtual void perform_line_search(
        const MarkedPennerConeMetric& initial_marked_metric,
        MarkedPennerConeMetric& marked_metric) override
    {
        double t_start = timer.getElapsedTime();

        // Get starting  metric coordinates
        VectorX reduced_metric_start = reduced_metric_coords;

        // Get the constraint norm and its dot product with the jacobian-projected descent direction
        // Note: the product of the jacobian and descent direction should be the negative
        // constraint, but this may fail due to numerical instability or regularization
        VectorX metric_delta = reduced_metric_coords - reduced_metric_base;
        Scalar l2_c0_sq = constraint.squaredNorm() + epsilon * metric_delta.squaredNorm();
        Scalar proj_g0 = descent_direction.dot(J.transpose() * constraint + epsilon * metric_delta);
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
        update_constraint(marked_metric);

        // Line search until the constraint norm decreases and the projected constraint is
        // nonpositive We also allow the norm bound to be dropped or made approximate with some
        // relative term alpha
        spdlog::debug("Starting line search");
        metric_delta = reduced_metric_coords - reduced_metric_base;
        Scalar l2_c_sq = constraint.squaredNorm() + epsilon * metric_delta.squaredNorm();
        Scalar proj_cons = descent_direction.dot(J.transpose() * constraint + epsilon * metric_delta);
        Scalar gamma = 1.; // require ratio of current to initial constraint norm to be below alpha
        bool bound_norm = (lambda > alg_params.bound_norm_thres);
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
            update_constraint(marked_metric);

            // Update squared constraint norm and projected constraint
            metric_delta = reduced_metric_coords - reduced_metric_base;
            l2_c_sq = constraint.squaredNorm() + epsilon * metric_delta.squaredNorm();
            proj_cons = descent_direction.dot(J.transpose() * constraint + epsilon * metric_delta);
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

};


}
}