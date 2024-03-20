#include "explicit_optimization.hh"

#include "constraint.hh"
#include "embedding.hh"
#include "energies.hh"
#include "globals.hh"
#include "nonlinear_optimization.hh"
#include "projection.hh"
#include "shear.hh"
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include "vector.hh"
#include "io.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric
{

    void initialize_explicit_data_log(const std::filesystem::path &data_log_path)
    {
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

    void write_explicit_data_log_entry(
        const std::filesystem::path &data_log_path,
        const DifferentiableConeMetric &m,
        const EnergyFunctor &opt_energy,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &optimized_domain_coords,
        const VectorX &domain_coords,
        const VectorX &init_codomain_coords,
        const VectorX &gradient,
        std::shared_ptr<ProjectionParameters> proj_params,
        Scalar beta)
    {
        spdlog::trace("Writing data iteration to {}", data_log_path);
        std::ofstream output_file(data_log_path, std::ios::out | std::ios::app);

        // Compute metric coordinates
        VectorX reduced_optimized_metric_coords;
        std::unique_ptr<DifferentiableConeMetric> cone_metric = compute_domain_coordinate_metric(
            m,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            optimized_domain_coords,
            init_codomain_coords,
            proj_params);

        // Get the full per vertex constraint Jacobian with respect to Penner coordinates
        VectorX constraint;
        MatrixX constraint_penner_jacobian;
        bool need_jacobian = true;
        bool only_free_vertices = false;
        cone_metric
            ->constraint(constraint, constraint_penner_jacobian, need_jacobian, only_free_vertices);

        // Compute change in domain coords
        VectorX change_in_domain_coords = optimized_domain_coords - domain_coords;

        // Compute numerics
        Scalar energy = opt_energy.energy(*cone_metric);
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

    void compute_optimization_domain(
        const DifferentiableConeMetric &m,
        const MatrixX &shear_basis_matrix,
        MatrixX &constraint_domain_matrix,
        MatrixX &constraint_codomain_matrix,
        VectorX &domain_coords,
        VectorX &codomain_coords)
    {
        // Build matrix to map scale factors to edge coordinates
        MatrixX scale_factor_basis_matrix = conformal_scaling_matrix(m);

        // Initialize domain and codomain matrix matrices
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
        }

        // Get initial domain coordinates
        VectorX shear_basis_coords, scale_factors;
        compute_shear_basis_coordinates(m, shear_basis_matrix, shear_basis_coords, scale_factors);
        int num_shear_basis_coords = shear_basis_coords.size();
        domain_coords.resize(num_shear_basis_coords + fixed_dof.size());
        codomain_coords.resize(variable_dof.size());
        domain_coords.head(num_shear_basis_coords) = shear_basis_coords;
        for (size_t i = 0; i < fixed_dof.size(); ++i)
        {
            domain_coords[num_shear_basis_coords + i] = scale_factors[fixed_dof[i]];
        }
        for (size_t i = 0; i < variable_dof.size(); ++i)
        {
            codomain_coords[i] = scale_factors[variable_dof[i]];
        }

        // Build the domain matrix
        constraint_domain_matrix.resize(num_edges, shear_basis_matrix.cols() + fixed_dof.size());
        constraint_domain_matrix.reserve(domain_triplet_list.size());
        constraint_domain_matrix.setFromTriplets(
            domain_triplet_list.begin(),
            domain_triplet_list.end());
        constraint_domain_matrix =
            m.change_metric_to_reduced_coordinates(constraint_domain_matrix.transpose()).transpose();
        SPDLOG_TRACE("Domain matrix is {}", constraint_domain_matrix);

        // Build the codomain matrix
        constraint_codomain_matrix.resize(num_edges, variable_dof.size());
        constraint_codomain_matrix.reserve(codomain_triplet_list.size());
        constraint_codomain_matrix.setFromTriplets(
            codomain_triplet_list.begin(),
            codomain_triplet_list.end());
        constraint_codomain_matrix =
            m.change_metric_to_reduced_coordinates(constraint_codomain_matrix.transpose()).transpose();
        SPDLOG_TRACE("Codomain matrix is {}", constraint_codomain_matrix);
    }

    std::unique_ptr<DifferentiableConeMetric> compute_domain_coordinate_metric(
        const DifferentiableConeMetric &m,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &domain_coords,
        const VectorX &init_codomain_coords,
        std::shared_ptr<ProjectionParameters> proj_params)
    {
        spdlog::trace("Making domain coordinate metric");
        SPDLOG_TRACE(
            "Domain coordinates in range [{}, {}]",
            domain_coords.minCoeff(),
            domain_coords.maxCoeff());
        SPDLOG_TRACE(
            "Codomain coordinates in range [{}, {}]",
            init_codomain_coords.minCoeff(),
            init_codomain_coords.maxCoeff());

        // Get domain coordinate metric defined by the current coordinates
        VectorX domain_metric_coords = constraint_domain_matrix * domain_coords;
        VectorX codomain_metric_coords = constraint_codomain_matrix * init_codomain_coords;
        std::unique_ptr<DifferentiableConeMetric> cone_metric =
            m.set_metric_coordinates(domain_metric_coords + codomain_metric_coords);
        SPDLOG_TRACE(
            "Domain metric in range [{}, {}]",
            domain_metric_coords.minCoeff(),
            domain_metric_coords.maxCoeff());
        SPDLOG_TRACE(
            "Codomain metric in range [{}, {}]",
            codomain_metric_coords.minCoeff(),
            codomain_metric_coords.maxCoeff());

        // Project the domain metric to the constraint
        SolveStats<Scalar> solve_stats;
        return cone_metric->project_to_constraint(solve_stats, proj_params);
    }

    Scalar compute_domain_coordinate_energy(
        const DifferentiableConeMetric &m,
        const EnergyFunctor &opt_energy,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &domain_coords,
        const VectorX &init_codomain_coords,
        std::shared_ptr<ProjectionParameters> proj_params)
    {
        // Compute penner coordinates from the domain coordinates
        std::unique_ptr<DifferentiableConeMetric> cone_metric = compute_domain_coordinate_metric(
            m,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            init_codomain_coords,
            proj_params);

        // Get the initial energy
        return opt_energy.energy(*cone_metric);
    }

    bool compute_domain_coordinate_energy_with_gradient(
        const DifferentiableConeMetric &m,
        const EnergyFunctor &opt_energy,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &domain_coords,
        const VectorX &init_codomain_coords,
        std::shared_ptr<ProjectionParameters> proj_params,
        Scalar &energy,
        VectorX &gradient)
    {
        // Compute penner coordinates from the domain coordinates
        std::unique_ptr<DifferentiableConeMetric> cone_metric = compute_domain_coordinate_metric(
            m,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            init_codomain_coords,
            proj_params);

        // Get the initial energy
        energy = opt_energy.energy(*cone_metric);

        // Get the gradients of the energy with respect to the domain and codomain coordinates
        VectorX energy_penner_gradient = opt_energy.gradient(*cone_metric);
        VectorX energy_domain_gradient = constraint_domain_matrix.transpose() * energy_penner_gradient;
        VectorX energy_codomain_gradient =
            constraint_codomain_matrix.transpose() * energy_penner_gradient;

        // Get the per vertex constraint Jacobian with respect to Penner coordinates
        VectorX constraint;
        MatrixX constraint_penner_jacobian;
        std::vector<int> flip_seq;
        bool need_jacobian = true;
        bool only_free_vertices = true;
        bool success = cone_metric->constraint(
            constraint,
            constraint_penner_jacobian,
            need_jacobian,
            only_free_vertices);
        if (!success)
        {
            spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
            return false;
        }

        // Get the Jacobians of the constraint with respect to the domain and codomain coordinates
        MatrixX constraint_domain_jacobian =
            constraint_domain_matrix.transpose() * constraint_penner_jacobian.transpose();
        MatrixX constraint_codomain_jacobian =
            constraint_codomain_matrix.transpose() * constraint_penner_jacobian.transpose();

        // Solve for the component of the gradient corresponding to the implicit metric coordinates
        VectorX solution = solve_linear_system(constraint_codomain_jacobian, energy_codomain_gradient);
        VectorX energy_implicit_gradient = -constraint_domain_jacobian * solution;

        // Construct the final gradient
        gradient = energy_domain_gradient + energy_implicit_gradient;

        return true;
    }

    VectorX compute_codomain_coordinates(
        const DifferentiableConeMetric &m,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &domain_coords)
    {
        MatrixX inner_product_matrix =
            constraint_codomain_matrix.transpose() * constraint_codomain_matrix;
        VectorX metric_coords = m.get_reduced_metric_coordinates();
        VectorX metric_residual = metric_coords - (constraint_domain_matrix * domain_coords);
        VectorX rhs = constraint_codomain_matrix.transpose() * metric_residual;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(inner_product_matrix);
        VectorX codomain_coords = solver.solve(rhs);
        SPDLOG_DEBUG(vector_equal(
            constraint_domain_matrix * domain_coords + constraint_codomain_matrix * codomain_coords,
            metric_coords));
        return codomain_coords;
    }

    void compute_descent_direction(
        const VectorX &prev_gradient,
        const VectorX &prev_descent_direction,
        const std::deque<VectorX> &delta_variables,
        const std::deque<VectorX> &delta_gradients,
        const MatrixX &approximate_hessian_inverse,
        const VectorX &gradient,
        std::shared_ptr<OptimizationParameters> opt_params,
        VectorX &descent_direction)
    {
        std::string direction_choice = opt_params->direction_choice;

        // Compute descent direction
        if (direction_choice == "gradient")
        {
            descent_direction = -gradient;
        }
        else if (direction_choice == "conjugate_gradient")
        {
            // Check if the previous gradient and descent direction are trivial
            if ((prev_gradient.size() == 0) || (prev_descent_direction.size() == 0))
            {
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
                    coefficient);
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
                compute_lbfgs_direction(delta_variables, delta_gradients, gradient, descent_direction);
            }
        }
    }

    std::unique_ptr<DifferentiableConeMetric> backtracking_domain_line_search(
        const DifferentiableConeMetric &m,
        const EnergyFunctor &opt_energy,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &domain_coords,
        const VectorX &init_codomain_coords,
        const VectorX &gradient,
        const VectorX &descent_direction,
        VectorX &optimized_domain_coords,
        Scalar &beta,
        std::shared_ptr<ProjectionParameters> proj_params)
    {
        spdlog::get("optimize_metric")->info("Beginning line search");

        // Parameters for the line search
        Scalar shrink_factor = 0.5;

        // Get initial energy
        Scalar initial_energy = compute_domain_coordinate_energy(
            m,
            opt_energy,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            init_codomain_coords,
            proj_params);
        spdlog::get("optimize_metric")->info("Initial energy is {}", initial_energy);

        // Get the slope along the descent direction
        Scalar descent_slope = gradient.dot(descent_direction);
        spdlog::get("optimize_metric")->info("Descent direction slope is {}", descent_slope);

        // Make an initial line step and compute the energy and gradient
        spdlog::get("optimize_metric")->info("Making step of size {}", beta);
        optimized_domain_coords = domain_coords + beta * descent_direction;
        SPDLOG_TRACE(
            "Optimized domain coordinates in range [{}, {}]",
            optimized_domain_coords.minCoeff(),
            optimized_domain_coords.maxCoeff());
        VectorX domain_metric_coords = constraint_domain_matrix * domain_coords;
        SPDLOG_TRACE(
            "Optimized domain metric in range [{}, {}]",
            domain_metric_coords.minCoeff(),
            domain_metric_coords.maxCoeff());
        Scalar energy;
        VectorX step_gradient;
        bool success = compute_domain_coordinate_energy_with_gradient(
            m,
            opt_energy,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            optimized_domain_coords,
            init_codomain_coords,
            proj_params,
            energy,
            step_gradient);
        spdlog::get("optimize_metric")->info("Step energy is {}", energy);

        // TODO Backtrack until the Armijo condition is satisfied
        // Scalar control_parameter = 1e-4;
        // while (energy > initial_energy + beta * control_parameter * descent_slope)
        // Backtrack until the energy decreases sufficiently and the gradient sign condition
        // is satisfied
        while ((!success) || (energy > initial_energy * (1 + 1e-8)) ||
               (step_gradient.dot(descent_direction) > 0))
        {
            if (beta < 1e-16)
            {
                spdlog::get("optimize_metric")->warn("Terminating line step as beta too small");
                return compute_domain_coordinate_metric(
                    m,
                    constraint_domain_matrix,
                    constraint_codomain_matrix,
                    domain_coords,
                    init_codomain_coords,
                    proj_params);
            }

            // Reduce beta
            beta *= shrink_factor;

            // Make a line step
            optimized_domain_coords = domain_coords + beta * descent_direction;

            // Compute the new energy for the line step
            spdlog::get("optimize_metric")->info("Making step of size {}", beta);
            success = compute_domain_coordinate_energy_with_gradient(
                m,
                opt_energy,
                constraint_domain_matrix,
                constraint_codomain_matrix,
                optimized_domain_coords,
                init_codomain_coords,
                proj_params,
                energy,
                step_gradient);
            spdlog::get("optimize_metric")->info("Step energy is {}", energy);
            if (!success)
            {
                spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
            }
        }

        return compute_domain_coordinate_metric(
            m,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            optimized_domain_coords,
            init_codomain_coords,
            proj_params);
    }

    VectorX optimize_domain_coordinates(
        const DifferentiableConeMetric &m,
        const EnergyFunctor &opt_energy,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const VectorX &init_domain_coords,
        const VectorX &init_codomain_coords,
        std::shared_ptr<ProjectionParameters> proj_params,
        std::shared_ptr<OptimizationParameters> opt_params)
    {
        // Build default parameters if none given
        if (proj_params == nullptr)
            proj_params = std::make_shared<ProjectionParameters>();
        if (opt_params == nullptr)
            opt_params = std::make_shared<OptimizationParameters>();
        VectorX domain_coords = init_domain_coords;
        VectorX codomain_coords = init_codomain_coords;

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
        spdlog::get("optimize_metric")->set_level(spdlog::level::off);
        spdlog::get("optimize_metric")->info("Beginning explicit optimization");

        // Create per iteration data log if an output directory is specified
        std::filesystem::path data_log_path;
        if (!output_dir.empty())
        {
            data_log_path = join_path(output_dir, "iteration_data_log.csv");
            initialize_explicit_data_log(data_log_path);
        }
        spdlog::get("optimize_metric")
            ->info(
                "Optimizing {} coordinates with codomain of dimension {}",
                constraint_domain_matrix.cols(),
                constraint_codomain_matrix.cols());

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

        for (int iter = 0; iter < num_iter; ++iter)
        {
            spdlog::get("optimize_metric")->info("Beginning iteration {}", iter);

            // Compute the gradient for the shear metric coordinates
            Scalar energy;
            prev_gradient = gradient;
            compute_domain_coordinate_energy_with_gradient(
                m,
                opt_energy,
                constraint_domain_matrix,
                constraint_codomain_matrix,
                domain_coords,
                codomain_coords,
                proj_params,
                energy,
                gradient);
            spdlog::get("optimize_metric")->info("Energy at start of iteration is {}", energy);
            SPDLOG_TRACE(
                "Domain coordinates in range [{}, {}]",
                domain_coords.minCoeff(),
                domain_coords.maxCoeff());
            SPDLOG_TRACE(
                "Gradient coefficients in range [{}, {}]",
                gradient.minCoeff(),
                gradient.maxCoeff());

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
                        approximate_hessian_inverse);
                }
                else if (opt_params->direction_choice == "lbfgs")
                {
                    delta_variables.push_front(domain_coords - prev_domain_coords);
                    delta_gradients.push_front(gradient - prev_gradient);
                    if (delta_variables.size() > 10)
                        delta_variables.pop_back();
                    if (delta_gradients.size() > 10)
                        delta_gradients.pop_back();
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
                descent_direction);

            // Ensure the descent direction range is stable
            VectorX metric_descent_direction = constraint_domain_matrix * descent_direction;
            Scalar grad_range = beta * (metric_descent_direction.maxCoeff() - metric_descent_direction.minCoeff());
            if ((max_grad_range > 0) && (grad_range >= max_grad_range))
            {
                beta *= (max_grad_range / grad_range);
                spdlog::get("optimize_metric")->info("Reducing beta to {} for stability", beta);
            }

            // Perform backtracking gradient descent
            VectorX optimized_domain_coords;
            std::unique_ptr<DifferentiableConeMetric> optimized_cone_metric =
                backtracking_domain_line_search(
                    m,
                    opt_energy,
                    constraint_domain_matrix,
                    constraint_codomain_matrix,
                    domain_coords,
                    codomain_coords,
                    gradient,
                    descent_direction,
                    optimized_domain_coords,
                    beta,
                    proj_params);

            // Write iteration data if output directory specified
            if (!output_dir.empty())
            {
                write_explicit_data_log_entry(
                    data_log_path,
                    m,
                    opt_energy,
                    constraint_domain_matrix,
                    constraint_codomain_matrix,
                    optimized_domain_coords,
                    domain_coords,
                    codomain_coords,
                    gradient,
                    proj_params,
                    beta);
            }

            if (beta < 1e-16)
            {
                spdlog::get("optimize_metric")->warn("Terminating optimization as beta too small");
                break;
            }

            // Update for next iteration
            prev_domain_coords = domain_coords;
            domain_coords = optimized_domain_coords;
            codomain_coords = compute_codomain_coordinates(
                *optimized_cone_metric,
                constraint_domain_matrix,
                constraint_codomain_matrix,
                domain_coords);
            beta = std::min(2.0 * beta, max_beta);
        }

        // Compute final projection
        std::unique_ptr<DifferentiableConeMetric> optimized_cone_metric =
            compute_domain_coordinate_metric(
                m,
                constraint_domain_matrix,
                constraint_codomain_matrix,
                domain_coords,
                codomain_coords,
                proj_params);

        // Get final energy
        Scalar energy = opt_energy.energy(*optimized_cone_metric);
        spdlog::get("optimize_metric")->info("Final energy is {}", energy);

        // Close loggers
        spdlog::drop("mesh_data");
        spdlog::drop("optimize_metric");

        // Return final metric
        return optimized_cone_metric->get_reduced_metric_coordinates();
    }

    VectorX optimize_shear_basis_coordinates(
        const DifferentiableConeMetric &m,
        const EnergyFunctor &opt_energy,
        const MatrixX &shear_basis_matrix,
        std::shared_ptr<ProjectionParameters> proj_params,
        std::shared_ptr<OptimizationParameters> opt_params)
    {
        // Build independent and dependent basis vectors by adding a global scaling term
        // to the shear basis and removing and arbitrary basis vector from the scale factors
        MatrixX constraint_domain_matrix, constraint_codomain_matrix;
        VectorX domain_coords, codomain_coords;
        compute_optimization_domain(
            m,
            shear_basis_matrix,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            codomain_coords);

        // Perform optimization in domain coordinates
        return optimize_domain_coordinates(
            m,
            opt_energy,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            codomain_coords,
            proj_params,
            opt_params);
    }

} // namespace CurvatureMetric
