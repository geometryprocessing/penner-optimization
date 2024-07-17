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

#pragma once
#include <random>
#include <vector>
#include "optimization/core/common.h"
#include "optimization/metric_optimization/energy_functor.h"
#include "optimization/core/constraint.h"
#include "optimization/core/cone_metric.h"

namespace Penner {
namespace Optimization {

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
	const DifferentiableConeMetric& cone_metric,
  const EnergyFunctor& opt_energy,
  int num_tests = 10
) {
  // Get initial metric coordinates
  VectorX metric_coords = cone_metric.get_reduced_metric_coordinates();

  // Validate energy
  VectorX gradient = opt_energy.gradient(cone_metric);
  for (int i = 0; i < num_tests; ++i)
  {
    VectorX h;
    generate_perturbation_vector(metric_coords.size(), 1e-8, h);

    std::unique_ptr<DifferentiableConeMetric> cone_metric_plus = cone_metric.set_metric_coordinates(metric_coords + h);
    std::unique_ptr<DifferentiableConeMetric> cone_metric_minus = cone_metric.set_metric_coordinates(metric_coords - h);
    Scalar energy_plus = opt_energy.energy(*cone_metric_plus);
    Scalar energy_minus = opt_energy.energy(*cone_metric_minus);

    Scalar directional_derivative = gradient.dot(h) / h.norm();
    Scalar finite_difference_derivative = (energy_plus - energy_minus) / (2.0 * h.norm());

    Scalar error = abs(directional_derivative - finite_difference_derivative);
    spdlog::error("Energy error for norm {} vector is {}", h.norm(), error);
  }
}  
  
inline void
validate_constraint(
	const DifferentiableConeMetric& cone_metric,
  int num_tests = 10
) {
  // Get initial metric coordinates
  VectorX metric_coords = cone_metric.get_reduced_metric_coordinates();

  // Validate energy
  VectorX constraint;
  MatrixX J_constraint;
  bool need_jacobian = true;
  bool only_free_vertices = true;
  cone_metric.constraint(constraint, J_constraint, need_jacobian, only_free_vertices);
  for (int i = 0; i < num_tests; ++i)
  {
    VectorX h;
    generate_perturbation_vector(metric_coords.size(), 1e-8, h);

    need_jacobian = false;
    MatrixX J_constraint_temp;
    VectorX constraint_plus, constraint_minus;
    std::unique_ptr<DifferentiableConeMetric> cone_metric_plus = cone_metric.set_metric_coordinates(metric_coords + h);
    std::unique_ptr<DifferentiableConeMetric> cone_metric_minus = cone_metric.set_metric_coordinates(metric_coords - h);
    cone_metric_plus->constraint(constraint_plus, J_constraint_temp, need_jacobian, only_free_vertices);
    cone_metric_minus->constraint(constraint_minus, J_constraint_temp, need_jacobian, only_free_vertices);

    VectorX directional_derivative = (J_constraint * h) / h.norm();
    VectorX finite_difference_derivative = (constraint_plus - constraint_minus) / (2.0 * h.norm());
    Scalar error = sup_norm(directional_derivative - finite_difference_derivative);
    spdlog::error("Constraint error for norm {} vector is {}", h.norm(), error);
  }
}  
} // namespace Optimization
} // namespace Penner
