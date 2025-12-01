#include "feature/dirichlet/optimization.h"

#ifdef USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#endif

#include "feature/dirichlet/angle_constraint_relaxer.h"
#include "feature/dirichlet/constraint.h"
#include "holonomy/holonomy/constraint.h"
#include "optimization/core/constraint.h"
#include <queue>
#include <igl/Timer.h>

namespace Penner {
namespace Feature {

// utility class with state to run the relaxation optimization
class OptimizeDirichletRelaxation
{
public:
    /**
     * @brief Solve for a metric with angle and feature alignment constraints with
     * relaxed aligned edges that are only sofly enforced.
     * 
     * This method first ensures the relaxed constraints are satisfied to machine precision,
     * and then attempts to reduce for the full constraint error as much as possible without
     * degenerating the metric.
     * 
     * @param initial_dirichlet_metric: metric with full alignment constraints
     * @param input_alg_params: parameters for the optimization
     * @return output aligned metric
     */
    DirichletPennerConeMetric run(
        const DirichletPennerConeMetric& initial_dirichlet_metric,
        const Holonomy::NewtonParameters& input_alg_params)
    {
        alg_params = input_alg_params;
        num_solves = 0;

        switch (alg_params.log_level) {
            case 6: spdlog::set_level(spdlog::level::trace); break;
            case 5: spdlog::set_level(spdlog::level::debug); break;
            case 4: spdlog::set_level(spdlog::level::info); break;
            case 3: spdlog::set_level(spdlog::level::warn); break;
            case 2: spdlog::set_level(spdlog::level::err); break;
            case 1: spdlog::set_level(spdlog::level::critical); break;
            case 0: spdlog::set_level(spdlog::level::off); break;
        }

        // build projection parameters for the subroutines
        SolveStats<Scalar> solve_stats;
        proj_params = std::make_shared<Optimization::ProjectionParameters>();
        proj_params->do_reduction = true;
        proj_params->output_dir = alg_params.output_dir;
        proj_params->error_eps = 1e-12;
        proj_params->max_itr = std::max(200, input_alg_params.max_itr);

        // initialize state
        lambda = 1;
        log.num_iter = 0;
        dirichlet_metric = initial_dirichlet_metric;
        dirichlet_metric.use_relaxed_system = true; // make sure using relaxed system for projection
        reduced_metric_init = dirichlet_metric.get_reduced_metric_coordinates();
        update_metric(initial_dirichlet_metric, reduced_metric_init);
        num_solves = 0;
        log.max_error = constraint.cwiseAbs().maxCoeff();

        // write first data log
        initialize_data_log();
        write_data_log_entry();

        // perform initial projection
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_init);
        std::unique_ptr<Optimization::DifferentiableConeMetric> projected_cone_metric =
            dirichlet_metric.project_to_constraint(solve_stats, proj_params);
        reduced_metric_coords = projected_cone_metric->get_reduced_metric_coordinates();
        update_metric(initial_dirichlet_metric, reduced_metric_coords);
        num_solves += solve_stats.n_solves;

        // iterate line search with constraint projection until converged
        while (true) {
            spdlog::info("Beginning iteration {}: {} solves so far", log.num_iter, num_solves);

            // Check termination conditions
            log.max_error = constraint.cwiseAbs().maxCoeff();
            write_data_log_entry();
            if (is_converged()) break;

            // Increment iteration and lambda
            log.num_iter++;
            lambda = min(2. * lambda, 1.);

            // Compute Newton descent direction
            update_metric(initial_dirichlet_metric, reduced_metric_coords);
            descent_direction = compute_descent_direction();

            // Search for updated metric, constraint, and angles
            perform_line_search(initial_dirichlet_metric);
            spdlog::info(
                "Coordinates in range [{}, {}]",
                reduced_metric_coords.minCoeff(),
                reduced_metric_coords.maxCoeff());
        }

        // build final metric and close data files
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords);
        close_logs();

        return dirichlet_metric;
    }

    /**
     * @brief Trivial constructor
     * 
     */
    OptimizeDirichletRelaxation() {}

protected:
    // Metric data
    DirichletPennerConeMetric dirichlet_metric;
    VectorX reduced_metric_init;
    VectorX reduced_metric_coords;

    // Constraint and descent direction data
    VectorX hard_constraint;
    VectorX constraint;
    MatrixX J;
    VectorX descent_direction;

    // Algorithm data
    Scalar lambda;
    Holonomy::NewtonParameters alg_params;

    // log data
    std::ofstream log_file;
    Holonomy::NewtonLog log;
    int num_solves;
    Scalar max_triangle_quality;
    Scalar max_hard_error;
    std::shared_ptr<Optimization::ProjectionParameters> proj_params;
    VectorX triangle_quality, he2angle, he2cot;

    // Open a per iteration data log and write a header
    void initialize_data_log()
    {
        // Generate data log path
        std::filesystem::create_directory(alg_params.output_dir);
        std::string data_log_path;

        // Open main logging file
        data_log_path = join_path(alg_params.output_dir, "feature_iteration_log.csv");
        spdlog::info("Writing relaxation data to {}", data_log_path);
        log_file = std::ofstream(data_log_path, std::ios::out | std::ios::trunc);
        log_file << "num_iter,";
        log_file << "max_error,";
        log_file << "max_hard_error,";
        log_file << "solves,";
        log_file << "solve_time,";
        log_file << "lambda,";
        log_file << "triangle_quality,";
        log_file << std::endl;
    }

    // Write newton log iteration data to file
    void write_data_log_entry()
    {
        log_file << log.num_iter << ",";
        log_file << std::fixed << std::setprecision(17) << log.max_error << ",";
        log_file << std::fixed << std::setprecision(17) << max_hard_error << ",";
        log_file << num_solves << ",";
        log_file << log.solve_time << ",";
        log_file << std::fixed << std::setprecision(17) << lambda << ",";
        log_file << std::fixed << std::setprecision(17) << max_triangle_quality << ",";
        log_file << std::endl;
    }

    // close the log file
    void close_logs()
    {
        log_file.close();
    }

    // change metric coordinates (and flip to Delaunay) and compute constraint and triangle quality
    void update_metric(
        const DirichletPennerConeMetric& initial_dirichlet_metric,
        const VectorX& metric_coords)
    {
        // change metric
        dirichlet_metric.change_metric(initial_dirichlet_metric, metric_coords);
        dirichlet_metric.make_discrete_metric();

        // update angles and full constraint
        dirichlet_metric.use_relaxed_system = false;
        dirichlet_metric.get_corner_angles(he2angle, he2cot);
        constraint = dirichlet_metric.constraint(he2angle);

        // update hard constraints and set systme to relaxed
        dirichlet_metric.use_relaxed_system = true;
        hard_constraint = dirichlet_metric.constraint(he2angle);
        max_hard_error = hard_constraint.cwiseAbs().maxCoeff();
        
        // update triangle quality data
        int num_faces = dirichlet_metric.n_faces();
        triangle_quality.resize(num_faces);
        for (int f = 0; f < num_faces; ++f) {
            // Get face halfedges
            const auto& l = dirichlet_metric.l;
            int hij = dirichlet_metric.h[f];
            int hjk = dirichlet_metric.n[hij];
            int hki = dirichlet_metric.n[hjk];

            // Compute ratio of inradius to outradius for face
            Scalar numer = 2*l[hij]*l[hjk]*l[hki];
            Scalar denom = ((-l[hij] + l[hjk] + l[hki])*(l[hij] - l[hjk] + l[hki])*(l[hij] + l[hjk] - l[hki]));
            triangle_quality[f] = numer / denom;
        }
        max_triangle_quality = triangle_quality.maxCoeff();

        spdlog::info("Constraint computed with norm {}", constraint.squaredNorm());
        spdlog::info("triangle quality in range [{}, {}]", triangle_quality.minCoeff(), max_triangle_quality);
        spdlog::info("corner angles in range [{}, {}]", he2angle.minCoeff(), he2angle.maxCoeff());
    }

    // find descent direction for full constraints
    // WARNING: assumes metric is updated
    VectorX compute_descent_direction()
    {
        dirichlet_metric.use_relaxed_system = false;
        J = dirichlet_metric.constraint_jacobian(he2cot);
        dirichlet_metric.use_relaxed_system = true;
        igl::Timer timer;
        timer.start();

        // build Gram matrix
        MatrixX JJt = J * J.transpose();
        JJt.makeCompressed();
        VectorX rhs;

        // use cholmod solver (only possible if suitesparse enabled)
        if (alg_params.solver == "cholmod") {
#ifdef USE_SUITESPARSE
            double t_solve_start = timer.getElapsedTime();
            Eigen::CholmodSupernodalLLT<MatrixX> solver;
            solver.compute(JJt);
            rhs = solver.solve(constraint);
            log.solve_time = timer.getElapsedTime() - t_solve_start;
#else
            spdlog::error("Cholmod with SuiteSparse not available. Set USE_SUITESPARSE to use.");
#endif
        }
        // use simplicial LDLT solver
        else {
            double t_solve_start = timer.getElapsedTime();
            Eigen::SimplicialLDLT<MatrixX> solver;
            solver.compute(JJt);
            rhs = solver.solve(constraint);
            log.solve_time = timer.getElapsedTime() - t_solve_start;
        }

        // compute final descent direction
        VectorX descent_direction = (-J.transpose() * rhs);
        spdlog::info(
            "Descent direction with norm {} for constraint with norm {}",
            descent_direction.squaredNorm(),
            constraint.squaredNorm());

        // record solve in log
        num_solves++;

        return descent_direction;
    }

    void perform_line_search(const DirichletPennerConeMetric& initial_dirichlet_metric)
    {
        // Get starting  metric coordinates
        VectorX reduced_metric_start = reduced_metric_coords;

        // Get the constraint norm and its dot product with the jacobian-projected descent direction
        // Note: the product of the jacobian and descent direction should be the negative
        // constraint, but this may fail due to numerical instability or regularization
        Scalar l2_c0_sq = constraint.squaredNorm();
        spdlog::info("Initial squared error norm is {}", l2_c0_sq);

        // Reduce descent direction range to avoid nans/infs
        if (alg_params.do_reduction) {
            while (lambda * (descent_direction.maxCoeff() - descent_direction.minCoeff()) > 2.5) {
                lambda /= 2;
                spdlog::debug("Reducing lambda to {} for stability", lambda);
            }
        }

        // Make initial line step with updated constraint
        SolveStats<Scalar> solve_stats;
        reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords);
        std::unique_ptr<Optimization::DifferentiableConeMetric> line_step_cone_metric =
            dirichlet_metric.project_to_constraint(solve_stats, proj_params);
        reduced_metric_coords = line_step_cone_metric->get_reduced_metric_coordinates();
        update_metric(initial_dirichlet_metric, reduced_metric_coords);
        num_solves += solve_stats.n_solves;

        // Line search until the constraint norm decreases and the projected constraint is
        // nonpositive We also allow the norm bound to be dropped or made approximate with some
        // relative term alpha
        Scalar l2_c_sq = constraint.squaredNorm();
        spdlog::info("projected squared error norm is {}", l2_c_sq);
        spdlog::info("hard constraint error is {}", max_hard_error);
        while ((l2_c_sq > l2_c0_sq) || (max_hard_error > proj_params->error_eps)) {
            // Backtrack one step
            lambda /= 2;
            spdlog::info("Starting line search with lambda {}", lambda);

            // Change metric and update constraint
            reduced_metric_coords = reduced_metric_start + lambda * descent_direction;
            dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords);
            line_step_cone_metric = dirichlet_metric.project_to_constraint(solve_stats, proj_params);
            reduced_metric_coords = line_step_cone_metric->get_reduced_metric_coordinates();
            update_metric(initial_dirichlet_metric, reduced_metric_coords);
            num_solves += solve_stats.n_solves;

            // Update squared constraint norm and projected constraint
            l2_c_sq = constraint.squaredNorm();
            spdlog::info("squared error norm is {}", l2_c_sq);
            spdlog::info("hard constraint error is {}", max_hard_error);

            // Check if lambda is below the termination threshold
            if (lambda < alg_params.min_lambda) break;
            if (num_solves >= alg_params.max_itr) break;
        }

        // if fail to make progress, regress to original metric
        if (max_hard_error > proj_params->error_eps)
        {
            reduced_metric_coords = reduced_metric_start;
        }

        // Make final line step in original connectivity
        spdlog::info("Updating metric");
        dirichlet_metric.change_metric(initial_dirichlet_metric, reduced_metric_coords);
    }

    bool is_converged()
    {
        // check maximum error
        if (constraint.cwiseAbs().maxCoeff() < alg_params.error_eps) {
            spdlog::info("Stopping optimization as max error {} reached", alg_params.error_eps);
            return true;
        }

        // check how many linear solves performed
        if (num_solves >= alg_params.max_itr) {
            spdlog::info(
                "Stopping optimization as {} is over maximum number of solves {}",
                num_solves, alg_params.max_itr);
            return true;
        }

        // check if descent direction too small
        if ((log.num_iter > 0) && (descent_direction.cwiseAbs().maxCoeff() < alg_params.error_eps)) {
            spdlog::info("Stopping optimization as descent direction too small", alg_params.error_eps);
            return true;
        }

        // check if line step too small
        if (lambda < alg_params.min_lambda){
            spdlog::info("Stopping optimization as lambda {} too small", alg_params.min_lambda);
            return true;
        }

        // check if triangle quality too poor
        if (max_triangle_quality > 1e3)
        {
            spdlog::info("Stopping optimization as triangle quality 1/{} too small", max_triangle_quality);
            return true;
        }

        return false;
    }
};

DirichletPennerConeMetric optimize_relaxed_angles(
    const DirichletPennerConeMetric& initial_dirichlet_metric,
    const Holonomy::NewtonParameters& alg_params)
{
    // Optimize metric with full metric space (basis is identity)
    OptimizeDirichletRelaxation solver;
    return solver.run(initial_dirichlet_metric, alg_params);
}

// TODO
std::vector<std::pair<int, int>> reduce_relaxed_edges(
    DirichletPennerConeMetric& relaxed_dirichlet_metric,
    const std::vector<std::pair<int, int>>& initial_relaxed_edges,
    Holonomy::NewtonParameters alg_params,
    int num_reductions)
{
    // comparison for weighted relaxed edges
    typedef std::pair<int, Scalar> WeightedEdge;
    auto edge_compare = [](const WeightedEdge& left, const WeightedEdge& right) {
        return left.second < right.second;
    };

    // get initial relaxed system
    MatrixX relaxed_angle_constraint_system = relaxed_dirichlet_metric.get_angle_constraint_system();
    VectorX prev_metric = relaxed_dirichlet_metric.get_metric_coordinates();
    std::vector<std::pair<int, int>> relaxed_edges = initial_relaxed_edges;

    for (int i = 0; i < num_reductions; ++i)
    {
        // try to halve number of relaxed edges
        int num_relaxed = relaxed_edges.size();
        int num_to_relax = 0.5 * num_relaxed;

        // get the initial cone angle error and propogate to adjacent relaxed edges
        const auto& m = relaxed_dirichlet_metric;
        VectorX constraint;
        MatrixX J_constraint;
        constraint_with_jacobian(m, constraint, J_constraint, false, false);
        std::vector<Scalar> error(num_relaxed, 0.);
        for (int i = 0; i < num_relaxed; ++i)
        {
            auto [hij, hji] = relaxed_edges[i];
            Scalar error_0 = constraint[m.v_rep[m.to[hij]]];
            Scalar error_1 = constraint[m.v_rep[m.to[hji]]];
            error[i] = max(abs(error_0), abs(error_1));
        }

        // Initialize the stack of vertices to process with all vertices
        std::priority_queue<WeightedEdge, std::vector<WeightedEdge>, decltype(edge_compare)>
            failures(edge_compare);
        for (int i = 0; i < num_relaxed; ++i)
        {
            failures.push(std::make_pair(i, error[i]));
        }

        std::vector<std::pair<int, int>> reduced_relaxed_edges = {};
        for (int i = 0; i < num_to_relax; ++i)
        {
            auto [eij, edge_error] = failures.top();
            failures.pop();
            spdlog::info("Relaxing {} with error {}", eij, edge_error);
            reduced_relaxed_edges.push_back(relaxed_edges[eij]);
        }

        // get angle system for reduced relaxed edges
        AngleConstraintMatrixRelaxer reduced_relaxer;
        MatrixX reduced_relaxed_angle_constraint_system = reduced_relaxer.run(relaxed_dirichlet_metric, reduced_relaxed_edges);
        relaxed_dirichlet_metric.set_angle_constraint_system(reduced_relaxed_angle_constraint_system);

        // check if converged
        auto reduced_marked_metric = optimize_metric_angles(relaxed_dirichlet_metric, alg_params);
        relaxed_dirichlet_metric.change_metric(reduced_marked_metric, reduced_marked_metric.get_metric_coordinates());
        relaxed_dirichlet_metric.constraint(constraint, J_constraint, false, true);
        if (constraint.maxCoeff() < alg_params.error_eps)
        {
            spdlog::info("Reduction succesful: removing relaxed constraints");
            relaxed_edges = reduced_relaxed_edges;
            relaxed_angle_constraint_system = reduced_relaxed_angle_constraint_system;
            prev_metric = relaxed_dirichlet_metric.get_metric_coordinates();
        }
        else {
            spdlog::info("Reduction failed: aborting");
            relaxed_dirichlet_metric.set_angle_constraint_system(relaxed_angle_constraint_system);
            relaxed_dirichlet_metric.change_metric(relaxed_dirichlet_metric, prev_metric);
            return relaxed_edges;
        }
    }

    return relaxed_edges;
}

} // namespace Feature
} // namespace Penner
