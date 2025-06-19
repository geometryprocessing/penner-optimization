#include "feature/dirichlet/constraint.h"
#include "feature/core/component_mesh.h"
#include "holonomy/holonomy/constraint.h"
#include "holonomy/holonomy/holonomy.h"
#include "optimization/core/constraint.h"
#include "optimization/core/viewer.h"
#include "util/io.h"
#include "util/vector.h"

namespace Penner {
namespace Feature {


// build the length system values
VectorX Length(DirichletPennerConeMetric& dirichlet_metric)
{
    int n_bd = dirichlet_metric.n_boundary_paths();
    const auto& boundary_paths = dirichlet_metric.get_boundary_paths();

    // compute boundary length constraints
    VectorX boundary_lengths(n_bd);
    for (int i = 0; i < n_bd; i++) {
        boundary_lengths[i] = boundary_paths[i].compute_length(dirichlet_metric);
    }
    const auto& A = dirichlet_metric.get_boundary_constraint_system();
    return A * boundary_lengths;
}

// build the log length system values used for boundary strengths
VectorX Ell(DirichletPennerConeMetric& dirichlet_metric)
{
    int n_bd = dirichlet_metric.n_boundary_paths();
    const auto& boundary_paths = dirichlet_metric.get_boundary_paths();
    if (n_bd == 0) return VectorX();

    // compute all boundary log length values
    VectorX boundary_lengths(n_bd);
    for (int i = 0; i < n_bd; i++) {
        boundary_lengths[i] = boundary_paths[i].compute_log_length(dirichlet_metric);
    }

    // compute constrained length values from boundary constraint system matrix
    const auto& A = dirichlet_metric.get_boundary_constraint_system();
    return A * boundary_lengths;
}

// helper function to add boundary constraints to a constraint vector with given offset
void add_boundary_constraints(
    DirichletPennerConeMetric& dirichlet_metric,
    VectorX& constraint,
    int offset = 0,
    Scalar weight = 1.)
{
    const auto& ell_hat = dirichlet_metric.ell_hat;
    int num_constraints = ell_hat.size();
    if (num_constraints == 0) return;

    // add segment as boundary system value minus the target
    VectorX ell = Ell(dirichlet_metric);
    constraint.segment(offset, num_constraints) = weight * (ell_hat - ell);
}

VectorX compute_boundary_constraint(DirichletPennerConeMetric& dirichlet_metric)
{
    // Add boundary length constraints
    int n_bd_constraints = dirichlet_metric.n_boundary_constraints();
    VectorX constraint = VectorX::Zero(n_bd_constraints);
    add_boundary_constraints(dirichlet_metric, constraint);

    return constraint;
}

MatrixX compute_boundary_constraint_jacobian(DirichletPennerConeMetric& dirichlet_metric)
{
    int n_bd = dirichlet_metric.n_boundary_paths();
    if (n_bd == 0) return MatrixX();

    // get boundary paths
    const auto& boundary_paths = dirichlet_metric.get_boundary_paths();

    // create IJV boundary log length Jacobian matrix
    int num_halfedges = dirichlet_metric.n_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int i = 0; i < n_bd; i++) {
        // build jacobian vector for log length
        auto log_length_jacobian = boundary_paths[i].compute_log_length_jacobian(dirichlet_metric);

        // add jacobian row to global system as row i
        for (const auto& jv : log_length_jacobian) {
            int j = jv.first;
            Scalar v = jv.second;
            tripletList.push_back(T(i, j, -v));
        }
    }

    // build reduced coordinate Jacobian from triplet list
    const auto& boundary_constraint_system = dirichlet_metric.get_boundary_constraint_system();
    MatrixX boundary_length_jacobian =
        dirichlet_metric.change_metric_to_reduced_coordinates(tripletList, n_bd);

    // compute jacobian of the system by the chain rule
    return (boundary_constraint_system * boundary_length_jacobian);
}

VectorX compute_dirichlet_constraint(
    DirichletPennerConeMetric& dirichlet_metric,
    const VectorX& angles)
{
    const MatrixX& angle_constraint_system = dirichlet_metric.get_angle_constraint_system();

    // get constraint counts
    int n_v = angle_constraint_system.rows();
    int n_s = dirichlet_metric.n_homology_basis_loops();
    int n_bd_constraints = dirichlet_metric.n_boundary_constraints();

    // build all constraints
    VectorX constraint = VectorX::Zero(n_v + n_s + n_bd_constraints);
    add_vertex_constraints(dirichlet_metric, angle_constraint_system, angles, constraint, 0);
    add_basis_loop_constraints(dirichlet_metric, angles, constraint, n_v);
    add_boundary_constraints(dirichlet_metric, constraint, n_v + n_s);

    return constraint;
}

MatrixX compute_dirichlet_constraint_jacobian(
    DirichletPennerConeMetric& dirichlet_metric,
    const VectorX& cotangents)
{
    // build holonomy angle constraints
    MatrixX J_corner_angle_metric =
        Holonomy::compute_metric_corner_angle_jacobian(dirichlet_metric, cotangents);
    const MatrixX& angle_constraint_system = dirichlet_metric.get_angle_constraint_system();
    const auto& homology_basis_loops = dirichlet_metric.get_homology_basis_loops();
    MatrixX holonomy_matrix = Holonomy::compute_holonomy_matrix(
        dirichlet_metric,
        angle_constraint_system,
        homology_basis_loops);
    MatrixX J_holonomy_constraint = holonomy_matrix * J_corner_angle_metric;

    // case: no boundary constraints
    int n_bd_constraints = dirichlet_metric.n_boundary_constraints();
    if (n_bd_constraints == 0) return J_holonomy_constraint;

    // compute boundary length constraints
    MatrixX J_boundary_constraint = compute_boundary_constraint_jacobian(dirichlet_metric);

    // assemble matrix from two components
    int r0 = J_holonomy_constraint.rows();
    int r1 = J_boundary_constraint.rows();
    int c = J_holonomy_constraint.cols();
    assert(c == J_boundary_constraint.cols());
    MatrixX J_transpose(c, r0 + r1);
    J_transpose.leftCols(r0) = J_holonomy_constraint.transpose();
    J_transpose.rightCols(r1) = J_boundary_constraint.transpose();

    // TODO Remove double transpose
    return J_transpose.transpose();
}

// helper function to compute dirichlet constraint, assuming a discrete metric
void dirichlet_constraint_with_jacobian_helper(
    DirichletPennerConeMetric& dirichlet_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian)
{
    // Get angles and cotangent of angles of faces opposite halfedges
    VectorX he2angle;
    VectorX cotangents;
    dirichlet_metric.get_corner_angles(he2angle, cotangents);

    // Compute constraint and (optionally) the Jacobian
    constraint = compute_dirichlet_constraint(dirichlet_metric, he2angle);
    if (need_jacobian) {
        J_constraint = compute_dirichlet_constraint_jacobian(dirichlet_metric, cotangents);
    }
}

void compute_dirichlet_constraint_with_jacobian(
    const DirichletPennerConeMetric& dirichlet_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian)
{
    // ensure current cone metric coordinates are log lengths
    DirichletPennerConeMetric dirichlet_metric_copy = dirichlet_metric;
    dirichlet_metric_copy.make_discrete_metric();

    // build constraint from discrete metric
    dirichlet_constraint_with_jacobian_helper(
        dirichlet_metric_copy,
        constraint,
        J_constraint,
        need_jacobian);
}

} // namespace Feature
} // namespace Penner