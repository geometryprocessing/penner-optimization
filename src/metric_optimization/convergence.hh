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

#include "common.hh"
#include "cone_metric.hh"
#include "embedding.hh"
#include "energy_functor.hh"

/// @file Methods to analyze the convergence of a metric to a global minimum on the
/// constraint surface.

namespace CurvatureMetric {

/// Given a metric and direction, compute the energy values for optimization
/// at given (potentially negative) step sizes before and after the projection to the
/// constraint.
///
/// @param[in] m: input mesh
/// @param[in] opt_energy: optimization energy
/// @param[in] opt_params: parameters for the optimization method
/// @param[in] proj_params: parameters for the projection
/// @param[in] direction: direction for the optimization
/// @param[in] step_sizes: step sizes to compute the energy at
/// @param[out] unprojected_energies: energies at the step sizes before projection
/// @param[out] projected_energies: energies at the step sizes after projection
void compute_direction_energy_values(
    const DifferentiableConeMetric& m,
    const EnergyFunctor& opt_energy,
    std::shared_ptr<OptimizationParameters> opt_params,
    std::shared_ptr<ProjectionParameters> proj_params,
    const VectorX& direction,
    const VectorX& step_sizes,
    VectorX& unprojected_energies,
    VectorX& projected_energies);


} // namespace CurvatureMetric
