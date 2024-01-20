#include "projection.hh"

#include <igl/Timer.h>
#include <map>
#include <stack>
#include "constraint.hh"
#include "embedding.hh"
#include "globals.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {


MatrixX conformal_scaling_matrix(const Mesh<Scalar>& m)
{
    // Create projection matrix
    int num_halfedges = m.n_halfedges();
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        int v0 = m.v_rep[m.to[h]];
        int v1 = m.v_rep[m.to[m.opp[h]]];
        tripletList.push_back(T(h, v0, 1.0));
        tripletList.push_back(T(h, v1, 1.0));
    }
    MatrixX B;
    B.resize(num_halfedges, m.n_ind_vertices());
    B.reserve(tripletList.size());
    B.setFromTriplets(tripletList.begin(), tripletList.end());

    return B;
}

VectorX best_fit_conformal(
    const DifferentiableConeMetric& target_cone_metric,
    const VectorX& metric_coords)
{
    // Construct psuedoinverse for the conformal scaling matrix
    MatrixX B = conformal_scaling_matrix(target_cone_metric);
    MatrixX A = B.transpose() * B;

    // Solve for the best fit conformal scale factor
    VectorX metric_target = target_cone_metric.get_metric_coordinates();
    VectorX w = B.transpose() * (metric_coords - metric_target);
    return solve_psd_system(A, w);
}

std::tuple<std::vector<int>, SolveStats<Scalar>> compute_constraint_scale_factors(
    const DifferentiableConeMetric& cone_metric,
    VectorX& u,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::string output_dir)
{
    if (proj_params == nullptr) proj_params = std::make_shared<ProjectionParameters>();

    // Create parameters for conformal method using restricted set of projection
    // parameters
    output_dir = proj_params->output_dir;
    AlgorithmParameters alg_params;
    LineSearchParameters ls_params;
    StatsParameters stats_params;
    alg_params.initial_ptolemy = proj_params->initial_ptolemy;
    alg_params.max_itr = proj_params->max_itr;
    alg_params.error_eps = double(proj_params->error_eps);
    alg_params.use_edge_flips = proj_params->use_edge_flips;
    ls_params.bound_norm_thres = double(proj_params->bound_norm_thres);
    ls_params.do_reduction = proj_params->do_reduction;
    if (!output_dir.empty()) {
        stats_params.error_log = true;
        stats_params.flip_count = true;
        stats_params.output_dir = output_dir;
    }

    // Run conformal method
    OverlayMesh<Scalar> mo(cone_metric);
    std::vector<int> pt_fids;
    std::vector<Eigen::Matrix<Scalar, 3, 1>> pt_bcs;
    auto conformal_out = ConformalIdealDelaunay<Scalar>::FindConformalMetric(
        mo,
        u,
        pt_fids,
        pt_bcs,
        alg_params,
        ls_params,
        stats_params);

    // Update u with conformal output
    u = std::get<0>(conformal_out);
    std::vector<int> flip_seq = std::get<1>(conformal_out);
    SolveStats<Scalar> solve_stats = std::get<2>(conformal_out);

    return std::make_tuple(flip_seq, solve_stats);
}

VectorX compute_constraint_scale_factors(
    const DifferentiableConeMetric& cone_metric,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::string output_dir)
{
    VectorX u;
    u.setZero(cone_metric.n_ind_vertices());
    compute_constraint_scale_factors(cone_metric, u, proj_params, output_dir);

    return u;
}

std::unique_ptr<DifferentiableConeMetric> project_metric_to_constraint(
    const DifferentiableConeMetric& cone_metric,
    std::shared_ptr<ProjectionParameters> proj_params,
    std::string output_dir)
{
    VectorX metric_coords = cone_metric.get_metric_coordinates();
    MatrixX scaling_matrix = conformal_scaling_matrix(cone_metric);
    VectorX u = compute_constraint_scale_factors(cone_metric, proj_params, output_dir);
    VectorX constrained_metric_coords = metric_coords + scaling_matrix * u;
    return cone_metric.set_metric_coordinates(constrained_metric_coords);
}

MatrixX compute_descent_direction_projection_matrix(const MatrixX& J_constraint)
{
    // Solve for correction matrix
    MatrixX L = J_constraint * J_constraint.transpose();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(L);
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> rhs = -J_constraint;
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> temp = solver.solve(rhs);
    MatrixX I = id_matrix(J_constraint.cols());
    MatrixX M = J_constraint.transpose() * temp;

    // Compute lambdas line search direction projection
    return I + M;
}

VectorX project_descent_direction(
    const VectorX& descent_direction,
    const VectorX& constraint,
    const MatrixX& J_constraint)
{
    // Solve for correction vector mu
    MatrixX L = J_constraint * J_constraint.transpose();
    VectorX w = -(J_constraint * descent_direction + constraint);
    igl::Timer timer;
    timer.start();
    VectorX mu = solve_psd_system(L, w);
    double time = timer.getElapsedTime();
    spdlog::info("Direction projection solve took {} s", time);
    SPDLOG_INFO("Correction mu has norm {}", mu.norm());

    // Compute lambdas line search direction
    return descent_direction + (J_constraint.transpose() * mu);
}

VectorX project_descent_direction(
    const DifferentiableConeMetric& cone_metric,
    const VectorX& descent_direction)
{
    // Compute the constraint function and its Jacobian
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = true;
    bool only_free_vertices = true;
    bool success =
        cone_metric.constraint(constraint, J_constraint, need_jacobian, only_free_vertices);
    if (!success) {
        spdlog::get("optimize_metric")->warn("Conformal projection did not converge");
    }
    SPDLOG_INFO("Constraint has norm {}", constraint.norm());

    // Project the descent direction to the constraint tangent plane
    return project_descent_direction(descent_direction, constraint, J_constraint);
}


} // namespace CurvatureMetric
