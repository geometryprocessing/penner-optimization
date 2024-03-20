#include "constraint.hh"

#include "area.hh"
#include "embedding.hh"
#include "linear_algebra.hh"

namespace CurvatureMetric {

bool satisfies_triangle_inequality(const Mesh<Scalar>& cone_metric)
{
    int num_halfedges = cone_metric.n_halfedges();

    // Check triangle inequality for each halfedge
    for (int hi = 0; hi < num_halfedges; ++hi) {
        int hj = cone_metric.n[hi];
        int hk = cone_metric.n[hj];

        // Lengths of the halfedges
        Scalar li = cone_metric.l[hi];
        Scalar lj = cone_metric.l[hj];
        Scalar lk = cone_metric.l[hk];
        // TODO: Replace with function to get triangle scaled lengths
        // Can subtract Scalar llijk_avg = (lli + llj + llk) / 3.0;

        // Check triangle inequality
        if (li > lj + lk) return false;
    }

    return true;
}

void corner_angles(const Mesh<Scalar>& cone_metric, VectorX& he2angle, VectorX& he2cot)
{
    int num_halfedges = cone_metric.n_halfedges();
    he2angle.setZero(num_halfedges);
    he2cot.setZero(num_halfedges);
    const Scalar cot_infty = 1e10;

    // Compute maps from halfedges to opposite angles and cotangents of opposite
    // angles
    // #pragma omp parallel for
    int num_faces = cone_metric.h.size();
    for (int f = 0; f < num_faces; f++) {
        // Halfedges of face f
        int hi = cone_metric.h[f];
        int hj = cone_metric.n[hi];
        int hk = cone_metric.n[hj];

        // Lengths of the halfedges
        Scalar li = cone_metric.l[hi];
        Scalar lj = cone_metric.l[hj];
        Scalar lk = cone_metric.l[hk];
        // TODO: Replace with function to get triangle scaled lengths
        // Can subtract Scalar llijk_avg = (lli + llj + llk) / 3.0;

        // Compute the cotangent of the angles
        // (following "A Cotangent Laplacian for Images as Surfaces")
        Scalar Aijk4 = 4 * sqrt(std::max<Scalar>(squared_area(li, lj, lk), 0.0));
        Scalar Ijk = (-li * li + lj * lj + lk * lk);
        Scalar iJk = (li * li - lj * lj + lk * lk);
        Scalar ijK = (li * li + lj * lj - lk * lk);
        he2cot[hi] = Aijk4 == 0.0 ? copysign(cot_infty, Ijk) : (Ijk / Aijk4);
        he2cot[hj] = Aijk4 == 0.0 ? copysign(cot_infty, iJk) : (iJk / Aijk4);
        he2cot[hk] = Aijk4 == 0.0 ? copysign(cot_infty, ijK) : (ijK / Aijk4);

#define USE_ACOS
#ifdef USE_ACOS
        he2angle[hi] = acos(std::min<Scalar>(std::max<Scalar>(Ijk / (2.0 * lj * lk), -1.0), 1.0));
        he2angle[hj] = acos(std::min<Scalar>(std::max<Scalar>(iJk / (2.0 * lk * li), -1.0), 1.0));
        he2angle[hk] = acos(std::min<Scalar>(std::max<Scalar>(ijK / (2.0 * li * lj), -1.0), 1.0));
#else
        // atan2 is prefered for stability
        he2angle[hi] = 0.0, he2angle[hj] = 0.0, he2angle[hk] = 0.0;
        // li: l12, lj: l23, lk: l31
        Scalar l12 = li, l23 = lj, l31 = lk;
        const Scalar t31 = +l12 + l23 - l31, t23 = +l12 - l23 + l31, t12 = -l12 + l23 + l31;
        // valid triangle
        if (t31 > 0 && t23 > 0 && t12 > 0) {
            const Scalar l123 = l12 + l23 + l31;
            const Scalar denom = sqrt(t12 * t23 * t31 * l123);
            he2angle[hj] = 2 * atan2(t12 * t31, denom); // a1 l23
            he2angle[hk] = 2 * atan2(t23 * t12, denom); // a2 l31
            he2angle[hi] = 2 * atan2(t31 * t23, denom); // a3 l12
        } else if (t31 <= 0)
            he2angle[hk] = pi;
        else if (t23 <= 0)
            he2angle[hj] = pi;
        else if (t12 <= 0)
            he2angle[hi] = pi;
        else
            he2angle[hj] = pi;
#endif
    }
}


void build_free_vertex_map(const Mesh<Scalar>& m, std::vector<int>& v_rep, int& num_angles)
{
    // Build map from independent vertices to free vertices
    int num_ind_vertices = m.n_ind_vertices();
    num_angles = 0;
    std::vector<int> v_map(num_ind_vertices, -1);
    for (int v = 0; v < num_ind_vertices; ++v) {
        if (!m.fixed_dof[v]) {
            v_map[v] = num_angles;
            num_angles++;
        }
    }

    // Build map from vertices to free vertices
    int num_vertices = m.v_rep.size();
    v_rep.resize(num_vertices);
    for (int v = 0; v < num_vertices; ++v) {
        v_rep[v] = v_map[m.v_rep[v]];
    }
}

void vertex_angles_with_jacobian_helper(
    const DifferentiableConeMetric& cone_metric,
    VectorX& vertex_angles,
    MatrixX& J_vertex_angles,
    bool need_jacobian,
    bool only_free_vertices)
{
    // Get angles and cotangent of angles of faces opposite halfedges
    VectorX he2angle;
    VectorX he2cot;
    cone_metric.get_corner_angles(he2angle, he2cot);

    // Get matrix arrays
    const auto& next = cone_metric.n;
    const auto& to = cone_metric.to;

    // Build v_rep
    std::vector<int> v_rep;
    int num_angles;
    if (only_free_vertices) {
        build_free_vertex_map(cone_metric, v_rep, num_angles);
    } else {
        v_rep = cone_metric.v_rep;
        num_angles = cone_metric.n_ind_vertices();
    }

    // Sum up angles around vertices
    int num_halfedges = cone_metric.n_halfedges();
    vertex_angles.setZero(num_angles);
    for (int h = 0; h < num_halfedges; ++h) {
        int v = v_rep[to[h]];
        if (v < 0) continue;
        vertex_angles[v] += he2angle[next[next[h]]];
    }

    // Build Jacobian if needed
    if (need_jacobian) {
        // Create list of triplets of Jacobian indices and values
        auto [I, J, V] = allocate_triplet_matrix(3 * (num_halfedges / 2));
        for (int h = 0; h < num_halfedges; ++h) {
            int v = v_rep[to[next[h]]];
            if (v < 0) continue;

            I.push_back(v);
            J.push_back(next[next[h]]);
            V.push_back(-0.5 * he2cot[next[h]]);

            I.push_back(v);
            J.push_back(h);
            V.push_back(0.5 * he2cot[next[h]] + 0.5 * he2cot[next[next[h]]]);

            I.push_back(v);
            J.push_back(next[h]);
            V.push_back(-0.5 * he2cot[next[next[h]]]);
        }

        // Build reduced coordinate Jacobian from IJV halfedge Jacobian
        J_vertex_angles = cone_metric.change_metric_to_reduced_coordinates(I, J, V, num_angles);
    }
}

void vertex_angles_with_jacobian(
    const DifferentiableConeMetric& cone_metric,
    VectorX& vertex_angles,
    MatrixX& J_vertex_angles,
    bool need_jacobian,
    bool only_free_vertices)
{
    // Ensure current cone metric coordinates are log lengths
    if (cone_metric.is_discrete_metric()) {
        vertex_angles_with_jacobian_helper(
            cone_metric,
            vertex_angles,
            J_vertex_angles,
            need_jacobian,
            only_free_vertices);
    } else {
        std::unique_ptr<DifferentiableConeMetric> cone_metric_copy =
            cone_metric.clone_cone_metric();
        cone_metric_copy->make_discrete_metric();
        vertex_angles_with_jacobian_helper(
            *cone_metric_copy,
            vertex_angles,
            J_vertex_angles,
            need_jacobian,
            only_free_vertices);
    }
}


bool constraint_with_jacobian(
    const DifferentiableConeMetric& cone_metric,
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices)
{
    constraint.setZero(0);
    J_constraint.setZero();

    // Compute the current vertex angles (Jacobian is the same as the constraint)
    VectorX vertex_angles;
    vertex_angles_with_jacobian(
        cone_metric,
        vertex_angles,
        J_constraint,
        need_jacobian,
        only_free_vertices);

    // Subtract the target angles from the vertex angles to compute constraint
    constraint.resize(vertex_angles.size());
    int angle_count = 0;
    int num_vertices = cone_metric.n_ind_vertices();
    for (int v = 0; v < num_vertices; ++v) {
        // Skip fixed if only want free vertices
        if ((only_free_vertices) && (cone_metric.fixed_dof[v])) continue;

        constraint[angle_count] = vertex_angles[angle_count] - cone_metric.Th_hat[v];
        angle_count++;
    }

    return true;
}

Scalar compute_max_constraint(const DifferentiableConeMetric& cone_metric)
{
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = false;
    constraint_with_jacobian(
        cone_metric,
        constraint,
        J_constraint,
        need_jacobian,
        only_free_vertices);
    return sup_norm(constraint);
}

} // namespace CurvatureMetric
