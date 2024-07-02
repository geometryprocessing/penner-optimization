#include "holonomy/similarity/conformal.h"

#include <Eigen/SparseLU>
#include "holonomy/similarity/constraint.h"
#include "holonomy/holonomy/holonomy.h"

namespace PennerHolonomy {

// Compute the descent direction for a similarity metric with given angles and angle cotangents
VectorX compute_descent_direction(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& alpha,
    const VectorX& cot_alpha)
{
    // Build constraint system for holonomy constraints in terms of the scaling one form
    // Also includes closed one form constraints
    Eigen::SparseMatrix<Scalar> J = compute_similarity_constraint_jacobian(similarity_metric, cot_alpha);
    VectorX constraint = compute_similarity_constraint(similarity_metric, alpha);

    // Solve for descent direction (in one form edge coordinates)
    VectorX descent_direction = CurvatureMetric::solve_linear_system(J, constraint);

    // TODO Determine method to solve for newton decrement and add steepest descent
    // weighting if necessary

    return descent_direction;
}

// Make a line step with direction lambda along the metric's one form descent direction
void line_step_one_form(SimilarityPennerConeMetric& similarity_metric, Scalar lambda)
{
    // Update the metric
    VectorX xi0 = similarity_metric.get_one_form();
    VectorX d = similarity_metric.get_one_form_direction();
    VectorX xi = xi0 + lambda * d;
    similarity_metric.set_one_form(xi);

    // Make sure the metric remains Delaunay
    similarity_metric.make_discrete_metric();
}

// Compute the gradient of the closed one form constraint energy given the angles of the metric
void compute_gradient(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& alpha,
    VectorX& gradient)
{
    int n_v = similarity_metric.n_vertices();
    int n_s = similarity_metric.n_homology_basis_loops();

    // Extract the gradient from the constraint vector
    VectorX constraint = compute_similarity_constraint(similarity_metric, alpha);
    gradient = constraint.topRows(n_v - 1 + n_s);
}

// Compute the gradient of the closed one form constraint energy
void compute_gradient(const SimilarityPennerConeMetric& similarity_metric, VectorX& gradient)
{
    // Compute the angles and cotangents of the scaled metric
    VectorX alpha, cot_alpha;
    similarity_corner_angles(similarity_metric, alpha, cot_alpha);

    // Compute the gradient using the computed angles
    compute_gradient(similarity_metric, alpha, gradient);
}

// Compute the Newton decrement in terms of the reduced vertex and dual loop variables
Scalar compute_newton_decrement(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& gradient,
    const VectorX& descent_direction)
{
    // Convert descent direction to signed halfedge coordinates
    VectorX d(similarity_metric.n_halfedges());
    for (int h = 0; h < similarity_metric.n_halfedges(); h++) {
        d[h] = similarity_metric.sign(h) * descent_direction[similarity_metric.he2e[h]];
    }
    assert(is_closed_one_form(similarity_metric, d));

    // Get reduced descent direction coefficients
    VectorX y = similarity_metric.reduce_one_form(d);

    // Get newton decrement
    return gradient.dot(y);
}

// Perform backtracking line search along the given descent direction, starting from
// step size lambda, and update the metric and its gradient
void line_search(
    SimilarityPennerConeMetric& similarity_metric,
    VectorX& gradient,
    const VectorX& descent_direction,
    Scalar& lambda,
    bool& bound_norm,
    const AlgorithmParameters& alg_params,
    const LineSearchParameters& ls_params)
{
    // Convert descent direction to signed halfedge coordinates
    VectorX d(similarity_metric.n_halfedges());
    for (int h = 0; h < similarity_metric.n_halfedges(); h++) {
        d[h] = similarity_metric.sign(h) * descent_direction[similarity_metric.he2e[h]];
    }
    similarity_metric.set_one_form_direction(d);

    // Get reduced descent direction coefficients
    VectorX y = similarity_metric.reduce_one_form(d);

    // Line step reduction to avoid nans/infs
    if (ls_params.do_reduction) {
        while (lambda * (d.maxCoeff() - d.minCoeff()) > 2.5) {
            lambda /= 2;
            spdlog::info("Reducing lambda to {}", lambda);
        }
    }

    // Get initial gradient before the line step and its norm
    compute_gradient(similarity_metric, gradient);
    Scalar l2_g0_sq = gradient.squaredNorm();

    // Initial line search
    line_step_one_form(similarity_metric, lambda);
    compute_gradient(similarity_metric, gradient);
    Scalar l2_g_sq = gradient.squaredNorm(); // Squared norm of the gradient
    Scalar proj_grad = y.dot(gradient); // Projected gradient onto descent direction

    // Backtrack until the gradient norm decreases and the projected gradient is negative
    while ((proj_grad > 0) || (l2_g_sq > l2_g0_sq && bound_norm)) {
        // Backtrack one step
        lambda /= 2;
        line_step_one_form(similarity_metric, -lambda); // Backtrack by halved lambda
        compute_gradient(similarity_metric, gradient);

        // TODO Line search condition to ensure quadratic convergence

        // Update squared gradient norm and projected gradient
        l2_g_sq = gradient.squaredNorm();
        proj_grad = gradient.dot(y);

        // Check if gradient norm is below the threshold to drop the bound
        if ((bound_norm) && (lambda <= ls_params.bound_norm_thres)) {
            bound_norm = false;
            spdlog::debug("Dropping norm bound.");
        }

        // Check if lambda is below the termination threshold
        if (lambda < alg_params.min_lambda) break;
    }
    spdlog::debug("Used lambda {} ", lambda);
    return;
}

void compute_conformal_similarity_metric(
    SimilarityPennerConeMetric& similarity_metric,
    const AlgorithmParameters& alg_params,
    const LineSearchParameters& ls_params)
{
    Scalar lambda = ls_params.lambda0;
    bool bound_norm =
        (ls_params.lambda0 > ls_params.bound_norm_thres); // prevents the grad norm from increasing
    if (bound_norm) spdlog::debug("Using norm bound.");

    // Get initial angles
    similarity_metric.make_discrete_metric();
    VectorX alpha, cot_alpha, gradient;
    similarity_corner_angles(similarity_metric, alpha, cot_alpha);
    compute_gradient(similarity_metric, alpha, gradient);

    // Iterate until the gradient has sup norm below a threshold
    spdlog::info("itr(0) lm({}) max_error({}))", lambda, gradient.cwiseAbs().maxCoeff());
    int itr = 0;
    while (gradient.cwiseAbs().maxCoeff() >= alg_params.error_eps) {
        itr++;

        // Compute gradient and descent direction from Hessian (with efficient solver)
        // Warning: need to have updated angles
        VectorX descent_direction = compute_descent_direction(similarity_metric, alpha, cot_alpha);

        // Terminate if newton decrement sufficiently smalll
        Scalar newton_decr =
            compute_newton_decrement(similarity_metric, gradient, descent_direction);

        // Alternative termination conditons to error threshold
        if (lambda < alg_params.min_lambda) {
            spdlog::info("Stopping projection as step size {} too small", lambda);
            break;
        }
        if (itr >= alg_params.max_itr) {
            spdlog::info("Stopping projection as reached maximum iteration {}", alg_params.max_itr);
            break;
        }
        if (newton_decr > alg_params.newton_decr_thres) {
            spdlog::info("Stopping projection as newton decrement {} large enough", newton_decr);
            break;
        }

        // Determine initial lambda for line search based on method parameters
        if (ls_params.reset_lambda) {
            lambda = ls_params.lambda0;
        } else {
            lambda = std::min<Scalar>(1, 2 * lambda); // adaptive step length
        }

        // reset lambda when it goes above norm bound threshold
        if ((lambda > ls_params.bound_norm_thres) && (!bound_norm)) {
            bound_norm = true;
            lambda = ls_params.lambda0;
            spdlog::debug("Using norm bound.");
        }

        // Search for updated metric, gradient, and angles
        line_search(
            similarity_metric,
            gradient,
            descent_direction,
            lambda,
            bound_norm,
            alg_params,
            ls_params);

        // Update current angles
        similarity_corner_angles(similarity_metric, alpha, cot_alpha);

        // Display current iteration information
        spdlog::info(
            "itr({}) lm({}) newton_decr({}) max_error({}))",
            itr,
            lambda,
            newton_decr,
            gradient.cwiseAbs().maxCoeff());
    }
}

} // namespace PennerHolonomy