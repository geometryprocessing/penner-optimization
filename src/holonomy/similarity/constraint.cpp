#include "holonomy/similarity/constraint.h"

#include "optimization/core/constraint.h"
#include "holonomy/holonomy/holonomy.h"
#include "holonomy/holonomy/constraint.h"

namespace Penner {
namespace Holonomy {

VectorX compute_similarity_constraint(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& angles)
{
    VectorX constraint;
    int n_f = similarity_metric.n_faces();
    int n_s = similarity_metric.n_homology_basis_loops();

    std::vector<int> v_map;
    int num_vertex_forms;
    Optimization::build_free_vertex_map(similarity_metric, v_map, num_vertex_forms);

    // Initialize the constraint
    constraint.setZero(num_vertex_forms + n_s + n_f - 1);
    constraint.head(num_vertex_forms + n_s) = compute_metric_constraint(similarity_metric, angles);

    return constraint;
}

// The Jacobian of the constraints
MatrixX compute_one_form_constraint_jacobian(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& cotangents)
{
    int n_f = similarity_metric.n_faces();
    int n_e = similarity_metric.n_edges();
    int n_h = similarity_metric.n_halfedges();
    int n_s = similarity_metric.n_homology_basis_loops();

    typedef Eigen::Triplet<Scalar> Trip;
    std::vector<Trip> trips(0);
    trips.reserve(3 * n_h + 3 * n_f);

    // Get free vertex map
    std::vector<int> v_rep;
    int num_vertex_forms;
    Optimization::build_free_vertex_rep(similarity_metric, v_rep, num_vertex_forms);

    // Add entries for vertex angles constraints
    for (int h = 0; h < n_h; h++) {
        int v0 = similarity_metric.v0(h);
        if (v_rep[v0] >= 0) {
            trips.push_back(Trip(
                v_rep[v0],
                similarity_metric.he2e[h],
                similarity_metric.sign(h) * 0.5 * cotangents[h]));
        }

        int v1 = similarity_metric.v1(h);
        if (v_rep[v1] >= 0) {
            trips.push_back(Trip(
                v_rep[v1],
                similarity_metric.he2e[h],
                -similarity_metric.sign(h) * 0.5 * cotangents[h]));
        }
    }

    // Add entries for holonomy constraints
    const auto& homology_basis_loops = similarity_metric.get_homology_basis_loops();
    for (int s = 0; s < n_s; ++s) {
        for (const auto& dual_segment : *homology_basis_loops[s]) {
            int h = dual_segment[0]; // Get just one halfedge of the path
            int ho = similarity_metric.opp[h];
            Scalar val = similarity_metric.sign(h) * 0.5 * (cotangents[h] + cotangents[ho]);
            int i = num_vertex_forms + s;
            int j = similarity_metric.he2e[h];
            trips.push_back(Trip(i, j, val));
            spdlog::trace("Adding constraint ({}, {}, {})", i, j, val);
        }
    }

    // Add closed 1-form constraints
    for (int f = 0; f < n_f - 1; f++) {
        int hi = similarity_metric.h[f];
        int hj = similarity_metric.n[hi];
        int hk = similarity_metric.n[hj];
        int ei = similarity_metric.he2e[hi];
        int ej = similarity_metric.he2e[hj];
        int ek = similarity_metric.he2e[hk];
        trips.push_back(Trip(num_vertex_forms + n_s + f, ei, similarity_metric.sign(hi)));
        trips.push_back(Trip(num_vertex_forms + n_s + f, ej, similarity_metric.sign(hj)));
        trips.push_back(Trip(num_vertex_forms + n_s + f, ek, similarity_metric.sign(hk)));
    }


    assert((num_vertex_forms + n_s + n_f - 1) == n_e);
    Eigen::SparseMatrix<Scalar> J(n_e, n_e);
    J.reserve(trips.size());
    J.setFromTriplets(trips.begin(), trips.end());
    return J;
}

// Matrix to expand per-edge one form to per-halfedge one form
MatrixX compute_one_form_expansion_matrix(
    const DifferentiableConeMetric& cone_metric)
{
    int n_h = cone_metric.n_halfedges();
    int n_e = cone_metric.n_edges();
    typedef Eigen::Triplet<Scalar> Trip;
    std::vector<Trip> trips(0);
    trips.reserve(n_h);

    // Add entries
    for (int h = 0; h < n_h; h++) {
        int e = cone_metric.he2e[h];
        trips.push_back(Trip(h, e, cone_metric.sign(h)));
    }

    Eigen::SparseMatrix<Scalar> M(n_h, n_e);
    M.reserve(trips.size());
    M.setFromTriplets(trips.begin(), trips.end());
    return M;
}

// Matrix to reduce per-halfedge one form to per-edge one form
MatrixX compute_one_form_reduction_matrix(
    const DifferentiableConeMetric& cone_metric)
{
    int n_h = cone_metric.n_halfedges();
    int n_e = cone_metric.n_edges();
    typedef Eigen::Triplet<Scalar> Trip;
    std::vector<Trip> trips(0);
    trips.reserve(n_h);

    // Add entries
    for (int e = 0; e < n_e; e++) {
        int h = cone_metric.e2he[e];
        trips.push_back(Trip(e, h, cone_metric.sign(h)));
        // trips.push_back(Trip(e, h, 1.0));
    }

    Eigen::SparseMatrix<Scalar> M(n_e, n_h);
    M.reserve(trips.size());
    M.setFromTriplets(trips.begin(), trips.end());
    return M;
}

MatrixX compute_similarity_constraint_jacobian(
    const SimilarityPennerConeMetric& similarity_metric,
    const VectorX& cotangents)
{
    // Get vertex representation
    int num_vertices = similarity_metric.n_ind_vertices();

    // Get component matrices
    MatrixX J_constraint_0 = compute_metric_constraint_jacobian(similarity_metric, cotangents);

    MatrixX J_constraint_one_form =
        compute_one_form_constraint_jacobian(similarity_metric, cotangents);
    MatrixX M_one_form_reduction = compute_one_form_reduction_matrix(similarity_metric);
    MatrixX M_dual_loop_basis_one_form = build_dual_loop_basis_one_form_matrix(
        similarity_metric,
        similarity_metric.get_homology_basis_loops());
    MatrixX J_constraint_1 =
        -J_constraint_one_form * (M_one_form_reduction * M_dual_loop_basis_one_form);

    // Combine matrices and remove closed one form constraint
    typedef Eigen::Triplet<Scalar> Trip;
    std::vector<Trip> trips(0);
    int num_metric_coords = J_constraint_0.cols();
    int num_form_coords = similarity_metric.n_homology_basis_loops();
    int num_coords = num_metric_coords + num_form_coords;
    int num_constraints = num_vertices + num_form_coords - 1;
    trips.reserve(3 * similarity_metric.n_halfedges());
    for (int k = 0; k < J_constraint_0.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(J_constraint_0, k); it; ++it) {
            assert(it.row() < num_form_coords + num_vertices - 1);
            assert(it.col() < num_metric_coords);
            trips.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    for (int k = 0; k < J_constraint_1.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(J_constraint_1, k); it; ++it) {
            assert(it.col() < num_form_coords);
            if (it.row() >= num_constraints) continue; // Skip closed form constraints
            trips.push_back(Trip(it.row(), it.col() + num_metric_coords, it.value()));
        }
    }

    Eigen::SparseMatrix<Scalar> J_constraint(num_constraints, num_coords);
    J_constraint.reserve(trips.size());
    J_constraint.setFromTriplets(trips.begin(), trips.end());
    return J_constraint;
}

// Helper function to compute similarity constraint assuming a discrete metric
void similarity_constraint_with_jacobian_helper(
    const SimilarityPennerConeMetric& similarity_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian)
{
    // Get angles and cotangent of angles of faces opposite halfedges
    VectorX he2angle;
    VectorX cotangents;
    similarity_metric.get_corner_angles(he2angle, cotangents);

    // Compute constraint and (optionally) the Jacobian
    int num_vertices = similarity_metric.n_ind_vertices();
    int num_form_coords = similarity_metric.n_homology_basis_loops();
    constraint = compute_similarity_constraint(similarity_metric, he2angle);
    constraint = constraint.head(num_vertices + num_form_coords - 1);
    if (need_jacobian) {
        J_constraint = compute_similarity_constraint_jacobian(similarity_metric, cotangents);
    }
}

void compute_similarity_constraint_with_jacobian(
    const SimilarityPennerConeMetric& similarity_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian)
{
    // Ensure current cone metric coordinates are log lengths
    if (similarity_metric.is_discrete_metric()) {
        similarity_constraint_with_jacobian_helper(
            similarity_metric,
            constraint,
            J_constraint,
            need_jacobian);
    } else {
        SimilarityPennerConeMetric similarity_metric_copy = similarity_metric;
        similarity_metric_copy.make_discrete_metric();
        similarity_constraint_with_jacobian_helper(
            similarity_metric_copy,
            constraint,
            J_constraint,
            need_jacobian);
    }
}

} // namespace Holonomy
} // namespace Penner