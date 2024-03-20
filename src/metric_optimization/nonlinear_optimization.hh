#pragma once

#include <deque>
#include "common.hh"

/// @file Methods to perform advanced nonlinear optimization, including conjugate gradient
/// and L-BFGS-B, using the current gradient, previous gradient, and previous descent direction.
/// These methods are intended to be general enough to apply to both implicit projected gradient
/// descent and unconstrained gradient descent.

namespace CurvatureMetric {

/// Compute a nonlinear conjugate gradient descent direction from a given gradient and
/// previous iteration data. Supported choices for the coefficient of "beta" are:
///   fletcher_reeves
///   polak_ribiere
///   hestenes_stiefel
///   dai_yuan
///
/// @param[in] gradient: current iteration gradient
/// @param[in] prev_gradient: previous iteration gradient
/// @param[in] prev_descent_direction: previous iteration descent direction
/// @param[out] descent_direction: computed descent direction
/// @param[in] coefficient: choice of coefficient for the previous descent direction
void compute_conjugate_gradient_direction(
    const VectorX& gradient,
    const VectorX& prev_gradient,
    const VectorX& prev_descent_direction,
    VectorX& descent_direction,
    std::string coefficient = "fletcher_reeves");

/// @brief Update the approximate hessian inverse for BFGS using the change in gradient
/// and change in variables
///
/// @param[in] gradient: current iteration gradient
/// @param[in] prev_gradient: previous iteration gradient
/// @param[in] delta_variables: change in variables in the current iteration
/// @param[in, out] approximate_hessian_inverse: Hessian inverse approximation to update
void update_bfgs_hessian_inverse(
    const VectorX& gradient,
    const VectorX& prev_gradient,
    const VectorX& delta_variables,
    MatrixX& approximate_hessian_inverse);

/// @brief Compute the L-BFGS descent direction given a history of variable and gradients.
///
/// It is assumed that the history of variables and gradients is of the same length with
/// more recent values at lower indices and that the given gradient g_k corresponds to the
/// delta_gradients[0] = g_k - g_{k-1}.
///
/// @param[in] delta_variables: change in variables over several iterations
/// @param[in] delta_gradients: change in variables over several iterations
/// @param[in] gradient: current iteration gradient
/// @param[out] descent_direction: L-BFGS descent direction
void compute_lbfgs_direction(
    const std::deque<VectorX>& delta_variables,
    const std::deque<VectorX>& delta_gradients,
    const VectorX& gradient,
    VectorX& descent_direction);

} // namespace CurvatureMetric