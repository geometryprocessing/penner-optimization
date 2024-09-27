#include "holonomy/holonomy/constraint.h"

#include "holonomy/holonomy/holonomy.h"

#include "optimization/core/constraint.h"

namespace Penner {
namespace Holonomy {

VectorX Theta(const Mesh<Scalar>& m, const VectorX& alpha)
{
    // Sum up angles around vertices
    VectorX t(m.n_ind_vertices());
    t.setZero();
    for (int h = 0; h < m.n_halfedges(); h++) {
        t[m.v_rep[m.to[m.n[h]]]] += alpha[h];
    }
    SPDLOG_DEBUG("Cone angles with mean {} and norm {}", t.mean(), t.norm());
    return t;
}

VectorX Kappa(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const VectorX& alpha)
{
    int n_s = homology_basis_loops.size();
    VectorX k(n_s);
    for (int i = 0; i < n_s; ++i) {
        k[i] = compute_dual_loop_holonomy(m, alpha, *homology_basis_loops[i]);
    }

    return k;
}

// Compute dual loop holonomy for a similarity metric with given dual loops
VectorX Kappa(const MarkedPennerConeMetric& marked_metric, const VectorX& alpha)
{
    const auto& homology_basis_loops = marked_metric.get_homology_basis_loops();
    return Kappa(marked_metric, homology_basis_loops, alpha);
}

MatrixX build_free_vertex_system(const Mesh<Scalar>& m)
{
    // build map to free vertices
    std::vector<int> v_map;
    int num_free_vertices;
    Optimization::build_free_vertex_map(m, v_map, num_free_vertices);

    // make map into a matrix
    int num_vertices = v_map.size();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_vertices);
    for (int i = 0; i < num_vertices; i++) {
        int vi = v_map[i];
        if (vi < 0) continue;
        tripletList.push_back(T(vi, i, 1.));
    }

    // Create the matrix from the triplets
    MatrixX vertex_system;
    vertex_system.resize(num_free_vertices, num_vertices);
    vertex_system.reserve(tripletList.size());
    vertex_system.setFromTriplets(tripletList.begin(), tripletList.end());
    return vertex_system;
}

// Add vertex angle constraints
// TODO Allow for additional fixed degrees of freedom
void add_vertex_constraints(
    const MarkedPennerConeMetric& marked_metric,
    const MatrixX& angle_constraint_system,
    const VectorX& angles,
    VectorX& constraint,
    int offset)
{
    int n_constraints = angle_constraint_system.rows();
    VectorX t = Theta(marked_metric, angles);
    VectorX t_hat;
    convert_std_to_eigen_vector(marked_metric.Th_hat, t_hat);
    constraint.segment(offset, n_constraints) = angle_constraint_system * (t_hat - t);
}

void add_basis_loop_constraints(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles,
    VectorX& constraint,
    int offset)
{
    // Add holonomy constraints
    // TODO Allow for additional fixed degrees of freedom
    int n_s = marked_metric.n_homology_basis_loops();
    VectorX k = Kappa(marked_metric, angles);
    for (int i = 0; i < n_s; i++) {
        constraint[offset + i] = marked_metric.kappa_hat[i] - k(i);
    }
}

std::vector<int> compute_dependent_edges(const MarkedPennerConeMetric& marked_metric)
{
    // find boundary edges
    int num_halfedges = marked_metric.n_halfedges();
    int num_edges = marked_metric.n_edges();
    std::vector<bool> is_boundary_edge(num_edges, false);
    for (int h = 0; h < num_halfedges; ++h) {
        if (marked_metric.opp[marked_metric.R[h]] == h) {
            is_boundary_edge[marked_metric.he2e[h]] = true;
        }
    }

    // make list of dependent edges (using lowest index halfedge)
    std::vector<int> dependent_edges = {};
    for (int h = 0; h < num_halfedges; ++h) {
        if (is_boundary_edge[marked_metric.he2e[h]]) continue;
        if (marked_metric.opp[h] < h) continue;
        if (marked_metric.R[h] < h) continue;
        if (marked_metric.opp[marked_metric.R[h]] < h) continue;

        dependent_edges.push_back(marked_metric.he2e[h]);
    }
    spdlog::debug("{}/{} dependent edges", dependent_edges.size(), num_edges);

    return dependent_edges;
}

void add_symmetry_constraints(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& dependent_edges,
    VectorX& constraint,
    int offset)
{
    // add symmetry constraints
    int num_dep_edges = dependent_edges.size();
    for (int i = 0; i < num_dep_edges; i++) {
        int e = dependent_edges[i];
        int h = marked_metric.e2he[e];
        int Rh = marked_metric.R[h];
        constraint[offset + i] = marked_metric.original_coords[h] - marked_metric.original_coords[Rh];
    }
}

VectorX compute_vertex_constraint(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles)
{
    // Use all vertices
    std::vector<int> v_map = marked_metric.v_rep;
    int n_v = marked_metric.n_ind_vertices();
    MatrixX angle_constraint_system = id_matrix(n_v);

    // Build the constraint
    VectorX constraint = VectorX::Zero(n_v);
    add_vertex_constraints(marked_metric, angle_constraint_system, angles, constraint, 0);

    return constraint;
}

VectorX compute_metric_constraint(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& angles,
    bool only_free_vertices)
{
    MatrixX angle_constraint_system;
    if (only_free_vertices) {
        angle_constraint_system = build_free_vertex_system(marked_metric);
    } else {
        angle_constraint_system = id_matrix(marked_metric.n_ind_vertices());
    }
    int n_v = angle_constraint_system.rows();
    int n_s = marked_metric.n_homology_basis_loops();


    // Build the constraint
    VectorX constraint = VectorX::Zero(n_v + n_s);
    add_vertex_constraints(marked_metric, angle_constraint_system, angles, constraint, 0);
    add_basis_loop_constraints(marked_metric, angles, constraint, n_v);

    return constraint;
}

// Compute the derivatives of metric corner angles with respect to halfedge lengths
MatrixX compute_metric_corner_angle_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& cotangents)
{
    // Create list of triplets of Jacobian indices and values
    int num_halfedges = marked_metric.n_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(3 * num_halfedges);
    for (int hjk = 0; hjk < num_halfedges; ++hjk) {
        int hki = marked_metric.n[hjk];
        int hij = marked_metric.n[hki];

        // Get cotangents (opposite halfedge)
        Scalar cj = cotangents[hki];
        Scalar ck = cotangents[hij];

        // Add entries
        tripletList.push_back(T(hjk, hij, 0.5 * cj));
        tripletList.push_back(T(hjk, hjk, -0.5 * (cj + ck)));
        tripletList.push_back(T(hjk, hki, 0.5 * ck));
    }

    // Build reduced coordinate Jacobian from triplet list
    return marked_metric.change_metric_to_reduced_coordinates(tripletList, num_halfedges);
}

MatrixX compute_loop_holonomy_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops)
{
    int num_halfedges = m.n_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(2 * num_halfedges);

    // Add dual loop holonomy constraints
    int num_loops = dual_loops.size();
    for (int i = 0; i < num_loops; ++i) {
        for (const auto& dual_segment : *dual_loops[i]) {
            int h_start = dual_segment[0];
            int h_end = dual_segment[1];

            // Negative angle if the segment subtends an angle to the right
            if (h_end == m.n[h_start]) {
                int h_opp = m.n[h_end]; // halfedge opposite subtended angle
                tripletList.push_back(T(i, h_opp, -1.0));
            }
            // Positive angle if the segment subtends an angle to the left
            else if (h_start == m.n[h_end]) {
                int h_opp = m.n[h_start]; // halfedge opposite subtended angle
                tripletList.push_back(T(i, h_opp, 1.0));
            }
        }
    }

    // Create the matrix from the triplets
    MatrixX holonomy_matrix;
    holonomy_matrix.resize(num_loops, num_halfedges);
    holonomy_matrix.reserve(tripletList.size());
    holonomy_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return holonomy_matrix;
}

MatrixX compute_vertex_holonomy_matrix(const Mesh<Scalar>& m)
{
    int num_ind_vertiecs = m.n_ind_vertices();
    int num_halfedges = m.n_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_ind_vertiecs);

    // Add vertex holonomy
    for (int h = 0; h < num_halfedges; ++h) {
        int v = m.v_rep[m.to[m.n[h]]];
        tripletList.push_back(T(v, h, 1.0));
    }

    // Create the matrix from the triplets
    MatrixX holonomy_matrix;
    holonomy_matrix.resize(num_ind_vertiecs, num_halfedges);
    holonomy_matrix.reserve(tripletList.size());
    holonomy_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return holonomy_matrix;
}

// Compute the linear map from corner angles to vertex and dual loop holonomy constraints
MatrixX compute_holonomy_matrix(
    const Mesh<Scalar>& m,
    const MatrixX& angle_constraint_system,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops)
{
    MatrixX J_vertex_holonomy = angle_constraint_system * compute_vertex_holonomy_matrix(m);
    MatrixX J_loop_holonomy = compute_loop_holonomy_matrix(m, dual_loops);

    // Assemble matrix from two components
    int r0 = J_vertex_holonomy.rows();
    int r1 = J_loop_holonomy.rows();
    int c = J_vertex_holonomy.cols();
    assert(c == J_loop_holonomy.cols());
    MatrixX J_transpose(c, r0 + r1);
    J_transpose.leftCols(r0) = J_vertex_holonomy.transpose();
    J_transpose.rightCols(r1) = J_loop_holonomy.transpose();

    // TODO Remove double transpose
    return J_transpose.transpose();
}

MatrixX compute_metric_constraint_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& cotangents,
    bool only_free_vertices)
{
    // Get vertex representation
    MatrixX angle_constraint_system;
    if (only_free_vertices) {
        angle_constraint_system = build_free_vertex_system(marked_metric);
    } else {
        angle_constraint_system = id_matrix(marked_metric.n_ind_vertices());
    }

    // Get corner angle derivatives with respect to metric coordinates
    MatrixX J_corner_angle_metric = compute_metric_corner_angle_jacobian(marked_metric, cotangents);

    // Get matrix summing up corner angles to form holonomy matrix
    const auto& homology_basis_loops = marked_metric.get_homology_basis_loops();
    MatrixX holonomy_matrix =
        compute_holonomy_matrix(marked_metric, angle_constraint_system, homology_basis_loops);

    // Build holonomy constraint jacobian
    return holonomy_matrix * J_corner_angle_metric;
}

MatrixX compute_symmetry_constraint_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& dependent_edges)
{
    int num_dep_edges = dependent_edges.size();
    int num_edges = marked_metric.n_edges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(2 * num_dep_edges);

    for (int i = 0; i < num_dep_edges; i++) {
        int e = dependent_edges[i];
        int h = marked_metric.e2he[e];
        int Rh = marked_metric.R[h];
        int Re = marked_metric.he2e[Rh];
        tripletList.push_back(T(i, e, 1.));
        tripletList.push_back(T(i, Re, -1.));
    }

    // Create the matrix from the triplets
    MatrixX symmetry_matrix;
    symmetry_matrix.resize(num_dep_edges, num_edges);
    symmetry_matrix.reserve(tripletList.size());
    symmetry_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    //spdlog::info("Matrix {}", symmetry_matrix);
    return symmetry_matrix;
}

// Helper function to compute metric constraint assuming a discrete metric
void metric_constraint_with_jacobian_helper(
    const MarkedPennerConeMetric& marked_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices)
{
    // Get angles and cotangent of angles of faces opposite halfedges
    VectorX he2angle;
    VectorX cotangents;
    marked_metric.get_corner_angles(he2angle, cotangents);

    // Compute constraint and (optionally) the Jacobian
    constraint = compute_metric_constraint(marked_metric, he2angle, only_free_vertices);
    if (need_jacobian) {
        J_constraint = compute_metric_constraint_jacobian(marked_metric, cotangents, only_free_vertices);
    }
}

void compute_metric_constraint_with_jacobian(
    const MarkedPennerConeMetric& marked_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices)
{
    // Ensure current cone metric coordinates are log lengths
    if (marked_metric.is_discrete_metric()) {
        metric_constraint_with_jacobian_helper(
            marked_metric,
            constraint,
            J_constraint,
            need_jacobian,
            only_free_vertices);
    } else {
        MarkedPennerConeMetric marked_metric_copy = marked_metric;
        marked_metric_copy.make_discrete_metric();
        metric_constraint_with_jacobian_helper(
            marked_metric_copy,
            constraint,
            J_constraint,
            need_jacobian,
            only_free_vertices);
    }
}

} // namespace Holonomy
} // namespace Penner
