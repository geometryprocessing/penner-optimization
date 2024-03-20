#include "convergence.hh"

#include "projection.hh"
#include "constraint.hh"
#include "implicit_optimization.hh"
#include "explicit_optimization.hh"

namespace CurvatureMetric
{

  void
  compute_direction_energy_values(
      const DifferentiableConeMetric &m,
      const EnergyFunctor &opt_energy,
      std::shared_ptr<OptimizationParameters> opt_params,
      std::shared_ptr<ProjectionParameters> proj_params,
      const VectorX &direction,
      const VectorX &step_sizes,
      VectorX &unprojected_energies,
      VectorX &projected_energies)
  {
    VectorX reduced_metric_coords = m.get_reduced_metric_coordinates();

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
      std::unique_ptr<DifferentiableConeMetric> cone_metric = m.clone_cone_metric();
      unprojected_energies[i] = opt_energy.energy(*cone_metric);

      // Project to the constraint
      VectorX u = compute_constraint_scale_factors(*cone_metric, proj_params, opt_params->output_dir);
      std::unique_ptr<DifferentiableConeMetric> constrained_cone_metric = cone_metric->scale_conformally(u);

      // Compute the projected energy
      projected_energies[i] = opt_energy.energy(*constrained_cone_metric);
    }
  }

  /*
  TODO: Refactor this energy landscape code and expose to python
    void
    compute_domain_direction_energy_values(
        const DifferentiableConeMetric &m,
        const ReductionMaps &reduction_maps,
        const VectorX &reduced_metric_target,
        const VectorX &domain_coords,
        const MatrixX &constraint_domain_matrix,
        const MatrixX &constraint_codomain_matrix,
        const std::shared_ptr<OptimizationParameters> opt_params,
        const std::shared_ptr<ProjectionParameters> proj_params,
        const VectorX &direction,
        const VectorX &step_sizes,
        VectorX &energies)
    {
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

    void
    compute_projected_descent_direction_stability_values(
        const DifferentiableConeMetric &m,
        const VectorX &reduced_metric_coords,
        const VectorX &reduced_metric_target,
        const OptimizationParameters &opt_params,
        const std::shared_ptr<ProjectionParameters> proj_params,
        const VectorX &step_sizes,
        VectorX &max_errors,
        VectorX &norm_changes_in_metric_coords,
        VectorX &convergence_ratios,
        VectorX &gradient_signs)
    {
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
          projected_descent_direction);

      // Get edge maps
      std::vector<int> he2e;
      std::vector<int> e2he;
      build_edge_maps(m, he2e, e2he);

      // Build refl projection and embedding
      std::vector<int> proj;
      std::vector<int> embed;
      build_refl_proj(m, he2e, e2he, proj, embed);

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
            step_projected_descent_direction);

        // Compute numerics
        max_errors[i] = constraint.maxCoeff();
        norm_changes_in_metric_coords[i] = change_in_metric_coords.norm();
        convergence_ratios[i] = compute_convergence_ratio(step_descent_direction, step_projected_descent_direction);
        gradient_signs[i] = -step_projected_descent_direction.dot(projected_descent_direction);
      }
    }
    */

}
