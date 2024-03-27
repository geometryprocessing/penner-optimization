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
#include "nonlinear_optimization.hh"
#include "vector.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void compute_conjugate_gradient_direction(
    const VectorX& gradient,
    const VectorX& prev_gradient,
    const VectorX& prev_descent_direction,
    VectorX& descent_direction,
    std::string coefficient)
{
    Scalar beta = 0.0;
    Scalar numerator, denominator;
    if (coefficient == "fletcher_reeves") {
        numerator = gradient.dot(gradient);
        denominator = prev_gradient.dot(prev_gradient);
    } else if (coefficient == "polak_ribiere") {
        numerator = gradient.dot(gradient - prev_gradient);
        denominator = prev_gradient.dot(prev_gradient);
    } else if (coefficient == "hestenes_stiefel") {
        numerator = gradient.dot(gradient - prev_gradient);
        denominator = prev_descent_direction.dot(gradient - prev_gradient);
    } else if (coefficient == "dai_yuan") {
        numerator = gradient.dot(gradient);
        denominator = prev_descent_direction.dot(gradient - prev_gradient);
    } else {
        spdlog::error("Invalid coefficient option. Using default Fletcher-Reeves.");
        numerator = gradient.dot(gradient);
        denominator = prev_gradient.dot(prev_gradient);
    }

    // Compute conjugate direction
    beta = max(numerator / denominator, 0.0); // avoid negative beta
    descent_direction = -gradient + beta * prev_descent_direction;
}

void update_bfgs_hessian_inverse(
    const VectorX& gradient,
    const VectorX& prev_gradient,
    const VectorX& delta_variables,
    MatrixX& approximate_hessian_inverse)
{
    // Convert vectors to sparse matrices
    const MatrixX& H = approximate_hessian_inverse;
    MatrixX y, s;
    convert_dense_vector_to_sparse(gradient - prev_gradient, y);
    convert_dense_vector_to_sparse(delta_variables, s);

    // Build components of Hessian inverse update
    MatrixX sTy_mat = y.transpose() * s;
    Scalar sTy = sTy_mat.coeff(0, 0);
    MatrixX Hy = H * y;
    MatrixX yHy_mat = y.transpose() * Hy;
    Scalar yHy = yHy_mat.coeff(0, 0);
    MatrixX ssT = s * s.transpose();
    MatrixX HysT = Hy * s.transpose();
    MatrixX syTHT = HysT.transpose();

    // Update Hessian inverse
    MatrixX first_term = ((sTy + yHy) * ssT) / (sTy * sTy);
    MatrixX second_term = (HysT + syTHT) / (sTy);
    approximate_hessian_inverse += first_term;
    approximate_hessian_inverse += second_term;
}

void compute_lbfgs_direction(
    const std::deque<VectorX>& delta_variables,
    const std::deque<VectorX>& delta_gradients,
    const VectorX& gradient,
    VectorX& descent_direction)
{
    int m = delta_variables.size();
    int n = delta_variables.size();
    if (m != n) {
        spdlog::error("Inconsistent number of gradient and variable data");
        return;
    }

    // Initialize descent direction with a forward pass
    VectorX q = gradient;
    std::vector<Scalar> rho(m);
    std::vector<Scalar> alpha(m);
    for (int i = 0; i < m; ++i) {
        rho[i] = 1.0 / delta_variables[i].dot(delta_gradients[i]);
        alpha[i] = rho[i] * delta_variables[i].dot(q);
        q -= alpha[i] * delta_gradients[i];
    }

    // Scale descent direction
    Scalar gamma = delta_variables[0].dot(delta_gradients[0]) / delta_gradients[0].squaredNorm();
    VectorX z = gamma * q;

    // Finish computation of descent direction with a back pass
    std::vector<Scalar> beta(m);
    for (int i = m - 1; i >= 0; --i) {
        beta[i] = rho[i] * delta_gradients[i].dot(z);
        z += (alpha[i] - beta[i]) * delta_variables[i];
    }
    descent_direction = -z;
}

} // namespace CurvatureMetric
