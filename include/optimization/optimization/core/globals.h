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

#include "util/common.h"

namespace Penner {
namespace Optimization {

/// Energies available for optimization
enum class EnergyChoice {
    log_length, // 2-norm of the metric coordinates
    log_scale, // scaling energy of best fit scale factors
    quadratic_sym_dirichlet, // quadratic approximation to symmetric Dirichlet
    sym_dirichlet, // symmetric Dirichlet
    p_norm // 4-norm of the metric coordinates
};

// Parameters to pass to the conformal method for projecting to the constraint.
// More detail on these parameters can be found in the documentation for that
// method.
struct ProjectionParameters
{
    int max_itr = 100; // maximum number of iterations
    Scalar bound_norm_thres = 1e-10; // line search threshold for dropping the gradient norm bound
#ifdef MULTIPRECISION
    Scalar error_eps = 1e-24; // minimum error termination condition
#else
    Scalar error_eps = 1e-8; // minimum error termination condition
#endif
    bool do_reduction =
        true; // reduce the initial line step if the range of coordinate values is large
    bool initial_ptolemy = true; // initial_ptolemy: use ptolemy flips for the initial make_delaunay
    bool use_edge_flips = true; // use intrinsic edge flips
    std::string output_dir = "";
};

// Parameters for the optimization method
struct OptimizationParameters
{
    // Logging
    std::string output_dir = ""; // output directory for file logs
    bool use_checkpoints = false; // if true, checkpoint state to output directory

    // Convergence parameters
    Scalar min_ratio =
        0.0; // minimum ratio of projected to ambient descent direction for convergence
    int num_iter = 200; // maximum number of iterations

    // Line step choice parameters
    bool require_energy_decr = true; // if true, require energy to decrease in each iteration
    bool require_gradient_proj_negative = true; // if true, require projection of the gradient onto
    // the descent direction to remain negative
    Scalar max_angle_incr = INF; // maximum allowed angle error increase in line step
    Scalar max_energy_incr = 1e-8; // maximum allowed energy increase in iteration

    // Optimization method choices
    std::string direction_choice = "projected_gradient"; // choice of direction

    // Numerical stability parameters
    Scalar beta_0 = 1.0; // initial line step size to try
    Scalar max_beta = 1e16; // maximum allowed line step size
    Scalar max_grad_range = 10; // maximum allowed gradient range (reduce if larger)
    Scalar max_angle = INF; // maximum allowed cone angle error (reduce if larger)
};

} // namespace Optimization
} // namespace Penner