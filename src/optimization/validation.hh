#pragma once
#include <random>
#include <vector>
#include "common.hh"
#include "energies.hh"
#include "embedding.hh"
#include "constraint.hh"

namespace CurvatureMetric {

inline std::vector<Scalar>
validate_hencky_strain_face(Scalar l1, Scalar l2, Scalar l3)
{
  std::vector<Scalar> cgret(9);

  Scalar Asq;
  Scalar csa1;
  Scalar csa2;
  Scalar t10;
  Scalar t12;
  Scalar t14;
  Scalar t17;
  Scalar t23;
  Scalar t25;
  Scalar t26;
  Scalar t27;
  Scalar t30;
  Scalar t34;
  Scalar t35;
  Scalar t42;
  Scalar t45;
  Scalar t47;
  Scalar t49;
  Scalar t55;
  Scalar t57;
  Scalar t72;
  Scalar t75;
  Scalar t8;
  Scalar t9;
  Asq =
    (l1 + l2 + l3) * (l1 + l2 - l3) * (l1 - l2 + l3) * (-l1 + l2 + l3) / 0.16e2;
  t8 = l1 * l1;
  t9 = l2 * l2;
  t10 = l3 * l3;
  t12 = 0.1e1 / l2;
  t14 = 0.1e1 / l3;
  csa1 = t14 * t12 * (-t8 + t9 + t10) / 0.2e1;
  t17 = 0.1e1 / l1;
  csa2 = t14 * t17 * (t8 - t9 + t10) / 0.2e1;
  t23 = t8 * t8;
  t25 = Asq * Asq;
  t26 = 0.1e1 / t25;
  t27 = t26 * t10;
  t30 = 0.1e1 / Asq;
  t34 = l1 * t8;
  t35 = l2 * t9;
  t42 = t8 * t30;
  t45 = csa1 * csa2 * t26 * t10 * t35 * t34 / 0.4e1 - t9 * t42 / 0.2e1;
  t47 = l3 * t10;
  t49 = t17 * t12 * (t8 + t9 - t10) * t26 / 0.2e1;
  t55 = csa1 * t49 * t47 * t9 * t34 / 0.4e1 - t10 * t42 / 0.2e1;
  t57 = t9 * t9;
  t72 = csa2 * t49 * t47 * t35 * t8 / 0.4e1 - t10 * t9 * t30 / 0.2e1;
  t75 = t10 * t10;
  cgret[0] = t27 * t9 * t23 / 0.4e1 - t23 * t30 / 0.2e1;
  cgret[1] = t45;
  cgret[2] = t55;
  cgret[3] = t45;
  cgret[4] = t27 * t57 * t8 / 0.4e1 - t57 * t30 / 0.2e1;
  cgret[5] = t72;
  cgret[6] = t55;
  cgret[7] = t72;
  cgret[8] = t26 * t75 * t9 * t8 / 0.4e1 - t75 * t30 / 0.2e1;

  return cgret;
}

inline void
generate_perturbation_vector(
  int perturbation_vector_length,
  Scalar perturbation_scale,
  VectorX& perturbation_vector
) {
  std::mt19937 e2(std::random_device{}());
  std::uniform_real_distribution<double> distribution(-perturbation_scale, perturbation_scale);
  perturbation_vector.setZero(perturbation_vector_length);
  for (size_t i = 0; i < perturbation_vector_length; ++i)
  {
    perturbation_vector[i] = distribution(e2);
  }
}

inline void
validate_energy_functor(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_target,
  const std::string& energy_choice,
  int p=2,
  int num_tests = 10
) {
  // Expand the target metric coordinates to the doubled surface
  ReductionMaps reduction_maps(m);
  VectorX metric_target;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);

  // Build perturbed metric coordinates
  VectorX deformation;
  generate_perturbation_vector(metric_target.size(), 1, deformation);
  VectorX metric_coords = metric_target + deformation;

  // Build energy functions for given energy
  OptimizationParameters opt_params;
  opt_params.energy_choice = energy_choice;
  opt_params.p = p;
  EnergyFunctor opt_energy(m, metric_target, opt_params);

  // Validate energy
  VectorX gradient = opt_energy.gradient(metric_coords);
  for (int i = 0; i < num_tests; ++i)
  {
    VectorX h;
    generate_perturbation_vector(metric_coords.size(), 1e-8, h);
    Scalar energy_plus = opt_energy.energy(metric_coords + h);
    Scalar energy_minus = opt_energy.energy(metric_coords - h);
    Scalar directional_derivative = gradient.dot(h) / h.norm();
    Scalar finite_difference_derivative = (energy_plus - energy_minus) / (2.0 * h.norm());
    Scalar error = abs(directional_derivative - finite_difference_derivative);
    spdlog::error("Energy error for norm {} vector is {}", h.norm(), error);
  }
}  
  
inline void
validate_constraint(
	const Mesh<Scalar>& m,
	const VectorX& reduced_metric_target,
  int num_tests = 10
) {
  // Expand the target metric coordinates to the doubled surface
  ReductionMaps reduction_maps(m);
  VectorX metric_target;
  expand_reduced_function(
    reduction_maps.proj, reduced_metric_target, metric_target);

  // Build perturbed metric coordinates
  VectorX deformation;
  generate_perturbation_vector(metric_target.size(), 1, deformation);
  VectorX metric_coords = metric_target + deformation;

  // Validate energy
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  bool use_edge_lengths = false;
  constraint_with_jacobian(m,
                            metric_coords,
                            constraint,
                            J_constraint,
                            flip_seq,
                            need_jacobian,
                            use_edge_lengths);
  for (int i = 0; i < num_tests; ++i)
  {
    VectorX h;
    generate_perturbation_vector(metric_coords.size(), 1e-8, h);

    need_jacobian = false;
    MatrixX J_constraint_temp;
    VectorX constraint_plus, constraint_minus;
    constraint_with_jacobian(m,
                              metric_coords + h,
                              constraint_plus,
                              J_constraint_temp,
                              flip_seq,
                              need_jacobian,
                              use_edge_lengths);
    constraint_with_jacobian(m,
                              metric_coords - h,
                              constraint_minus,
                              J_constraint_temp,
                              flip_seq,
                              need_jacobian,
                              use_edge_lengths);

    VectorX directional_derivative = (J_constraint * h) / h.norm();
    VectorX finite_difference_derivative = (constraint_plus - constraint_minus) / (2.0 * h.norm());
    Scalar error = sup_norm(directional_derivative - finite_difference_derivative);
    spdlog::error("Constraint error for norm {} vector is {}", h.norm(), error);
  }
}  
}
