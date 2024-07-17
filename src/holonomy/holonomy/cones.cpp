#include "holonomy/holonomy/cones.h"

#include "util/boundary.h"
#include "holonomy/core/forms.h"
#include "holonomy/holonomy/holonomy.h"

#include "optimization/core/constraint.h"

#include <random>

namespace Penner {
namespace Holonomy {

// Check cones computed from rotation form match more direct vertex iteration computation
bool validate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    const std::vector<Scalar>& Th_hat)
{
    // Compute the corner angles
    VectorX he2angle, he2cot;
    Optimization::corner_angles(m, he2angle, he2cot);

    // Get boundary vertices if symmetric mesh with boundary
    int num_vertices = m.n_vertices();
    bool is_symmetric = (m.type[0] != 0);
    std::vector<bool> is_boundary_vertex(num_vertices, false);
    if (is_symmetric) {
        std::vector<int> boundary_vertices = find_boundary_vertices(m);
        convert_index_vector_to_boolean_array(
            boundary_vertices,
            num_vertices,
            is_boundary_vertex);
    }

    // Compare cones with direct per-vertex computation
    for (int vi = 0; vi < num_vertices; ++vi) {
        DualLoopList dual_loop(build_counterclockwise_vertex_dual_segment_sequence(m, vi));
        Scalar rotation = compute_dual_loop_rotation(m, rotation_form, dual_loop);
        Scalar holonomy = compute_dual_loop_holonomy(m, he2angle, dual_loop);

        // Special treatment for vertices in interior of doubled mesh
        if ((is_symmetric) && (!is_boundary_vertex[vi])) {
            if (!float_equal(Th_hat[m.v_rep[vi]] / 2., holonomy - rotation, 1e-3)) {
                spdlog::warn(
                    "Inconsistent interior cones {} and {}",
                    Th_hat[m.v_rep[vi]] / 2.,
                    holonomy - rotation);
                return false;
            }
        }
        // General case
        else {
            if (!float_equal(Th_hat[m.v_rep[vi]], holonomy - rotation, 1e-3)) {
                spdlog::warn(
                    "Inconsistent cones {} and {}",
                    Th_hat[m.v_rep[vi]],
                    holonomy - rotation);
                return false;
            }
        }
    }

    return true;
}

std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form)
{
    assert(is_valid_one_form(m, rotation_form));
    // Compute the corner angles
    VectorX he2angle, he2cot;
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute cones from the rotation form as holonomy - rotation around each vertex
    // Per-halfedge iteration is used for faster computation
    int num_vertices = m.n_ind_vertices();
    std::vector<Scalar> Th_hat(num_vertices, 0.);
    for (int h = 0; h < m.n_halfedges(); h++) {
        // Add angle to vertex opposite the halfedge
        Th_hat[m.v_rep[m.to[m.n[h]]]] += he2angle[h];

        // Add rotation to the vertex at the tip of the halfedge
        // NOTE: By signing convention, this is the negative of the rotation ccw around
        // the vertex
        Th_hat[m.v_rep[m.to[h]]] += rotation_form[h];
    }
    assert(validate_cones_from_rotation_form(m, rotation_form, Th_hat));

    return Th_hat;
}

std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const VectorX& rotation_form,
    bool has_boundary)
{
    std::vector<Scalar> Th_hat_mesh = generate_cones_from_rotation_form(m, rotation_form);

    // Compute cones from the rotation form
    int num_vertices = m.n_ind_vertices();
    std::vector<Scalar> Th_hat(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (has_boundary) {
            Th_hat[vtx_reindex[vi]] = Th_hat_mesh[vi] / 2.;
        } else {
            Th_hat[vtx_reindex[vi]] = Th_hat_mesh[vi];
        }
    }

    return Th_hat;
}

bool contains_small_cones(const std::vector<Scalar>& Th_hat, int min_cone_index)
{
    int num_vertices = Th_hat.size();
    for (int vi = 0; vi < num_vertices; ++vi) {
        // Check for cones below threshold
        if (Th_hat[vi] < (min_cone_index * M_PI / 2.) - 1e-3) {
            spdlog::warn("{} cone found at {}", Th_hat[vi], vi);
            return true;
        }
    }

    return false;
}

bool contains_zero_cones(const std::vector<Scalar>& Th_hat)
{
    return contains_small_cones(Th_hat, 0.);
}

// Count negative and positive cones
std::pair<int, int> count_cones(const Mesh<Scalar>& m)
{
    const auto& Th_hat = m.Th_hat;

    // Get boundary vertices
    int num_vertices = m.n_vertices();
    bool is_symmetric = (m.type[0] != 0);
    std::vector<bool> is_boundary_vertex(num_vertices, false);
    if (is_symmetric) {
        std::vector<int> boundary_vertices = find_boundary_vertices(m);
        convert_index_vector_to_boolean_array(
            boundary_vertices,
            num_vertices,
            is_boundary_vertex);
    }

    // Check for cones
    int num_ind_vertices = Th_hat.size();
    int num_neg_cones = 0;
    int num_pos_cones = 0;
    std::vector<bool> is_seen(num_ind_vertices, false);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (is_seen[m.v_rep[vi]]) continue;

        // Get flat curvature
        Scalar flat_angle = 2. * M_PI;
        if ((is_symmetric) && (!is_boundary_vertex[vi])) {
            flat_angle = 4. * M_PI;
        }

        // Count negative and positive curvature cones
        if (Th_hat[m.v_rep[vi]] > flat_angle + 1e-3) {
            spdlog::trace("{} cone found", Th_hat[vi]);
            num_neg_cones++;
        }
        if (Th_hat[m.v_rep[vi]] < flat_angle - 1e-3) {
            spdlog::trace("{} cone found", Th_hat[vi]);
            num_pos_cones++;
        }

        // Mark vertex as seen
        is_seen[m.v_rep[vi]] = true;
    }

    return std::make_pair(num_neg_cones, num_pos_cones);
}

// Get the total curvature of the mesh from the cones
Scalar compute_total_curvature(const Mesh<Scalar>& m)
{
    const auto& Th_hat = m.Th_hat;

    // Get boundary vertices
    int num_vertices = m.n_vertices();
    bool is_symmetric = (m.type[0] != 0);
    std::vector<bool> is_boundary_vertex(num_vertices, false);
    if (is_symmetric) {
        std::vector<int> boundary_vertices = find_boundary_vertices(m);
        convert_index_vector_to_boolean_array(
            boundary_vertices,
            num_vertices,
            is_boundary_vertex);
    }

    // Incrementally compute total curvature
    int num_ind_vertices = Th_hat.size();
    Scalar total_curvature = 0.0;
    std::vector<bool> is_seen(num_ind_vertices, false);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (is_seen[m.v_rep[vi]]) continue;

        // Get flat curvature
        Scalar flat_angle = 2. * M_PI;
        if ((is_symmetric) && (!is_boundary_vertex[vi])) {
            flat_angle = 4. * M_PI;
        }

        // Total curvature is the deviation from 2 pi
        total_curvature += (Th_hat[m.v_rep[vi]] - flat_angle);

        // Mark vertex as seen
        is_seen[m.v_rep[vi]] = true;
    }

    return total_curvature;
}


bool is_trivial_torus(const Mesh<Scalar>& m)
{
    auto [num_neg_cones, num_pos_cones] = count_cones(m);
    return ((num_neg_cones == 0) && (num_pos_cones == 0));
}

bool is_torus_with_cone_pair(const Mesh<Scalar>& m)
{
    // Get the cone counts
    auto [num_neg_cones, num_pos_cones] = count_cones(m);

    // Compute genus of the surface from the total curvature
    Scalar total_curvature = compute_total_curvature(m);
    int genus = (int)(round(1 + total_curvature / (4. * M_PI)));
    spdlog::info("Total curvature is {}", total_curvature);
    spdlog::info("genus is {}", genus);

    // Check for tori with a pair of cones
    return ((genus == 1) && (num_neg_cones == 1) && (num_pos_cones == 1));
}

bool validate_cones(const Mesh<Scalar>& m)
{
    if (contains_zero_cones(m.Th_hat)) return false;
    if (is_torus_with_cone_pair(m)) return false;

    return true;
}

// Helper to fix small cones
void remove_minimum_cone(Mesh<Scalar>& m)
{
    bool is_symmetric = (m.type[0] != 0);
    Scalar angle_delta = (is_symmetric) ? M_PI : (M_PI / 2.);

    // Add pi/2 to the minimum cone
    auto min_cone = std::min_element(m.Th_hat.begin(), m.Th_hat.end());
    *min_cone += angle_delta;

    // Subtract pi/2 from the maximum cone
    auto max_cone = std::max_element(m.Th_hat.begin(), m.Th_hat.end());
    *max_cone -= angle_delta;
}

bool is_interior(const Mesh<Scalar>& m, int vi)
{
    int h_start = m.out[vi];
    int hij = h_start;
    do {
        int hji = m.opp[hij];
        if ((m.type[hij] == 1) && (m.type[hji] == 2)) return false;
        if ((m.type[hij] == 2) && (m.type[hji] == 1)) return false;

        hij = m.n[m.opp[hij]];
    } while (hij != h_start);

    return true;
}

int get_flat_vertex(const Mesh<Scalar>& m, bool only_interior)
{
    int num_halfedges = m.n_halfedges();
    std::mt19937 rng(0);
    std::uniform_int_distribution<> dist(0, num_halfedges - 1);

    // Find a flat cone in the interior of the mesh
    bool is_symmetric = (m.type[0] != 0);
    Scalar flat_angle = (is_symmetric) ? 4. * M_PI : 2. * M_PI;
    while (true) {
        int h = dist(rng);
        int vi = m.v_rep[m.to[h]];
        if ((is_interior(m, m.to[h])) && (float_equal(m.Th_hat[vi], flat_angle))) {
            return vi;
        }
        if (only_interior) continue;

        if ((m.type[h] == 1) && (m.R[m.opp[h]] == h) && float_equal(m.Th_hat[vi], 2. * M_PI)) {
            return vi;
        }
    }

    return -1;
}

void add_random_cone_pair(Mesh<Scalar>& m, bool only_interior)
{
    bool is_symmetric = (m.type[0] != 0);
    Scalar angle_delta = (is_symmetric) ? M_PI : (M_PI / 2.);

    // Add 5 cone
    int vi = get_flat_vertex(m, only_interior);
    spdlog::debug("Adding positive cone at {}", vi);
    m.Th_hat[vi] += angle_delta;

    // Add 3 cone
    int vj = get_flat_vertex(m, only_interior);
    spdlog::debug("Adding negative cone at {}", vj);
    m.Th_hat[vj] -= angle_delta;
}

std::tuple<int, int> get_constraint_outliers(
    MarkedPennerConeMetric& marked_metric,
    bool use_interior_vertices,
    bool use_flat_vertices)
{
    bool is_symmetric = (marked_metric.type[0] != 0);
    int num_vertices = marked_metric.n_vertices();
    int num_ind_vertices = marked_metric.n_ind_vertices();
    std::vector<int> bd_vertices = find_boundary_vertices(marked_metric);
    std::vector<bool> is_bd_vertex(num_ind_vertices, false);
    for (int vi : bd_vertices) {
        is_bd_vertex[marked_metric.v_rep[vi]] = true;
    }

    // get constraint errors
    VectorX constraint;
    MatrixX J_constraint;
    bool need_jacobian = false;
    bool only_free_vertices = false;
    marked_metric.constraint(constraint, J_constraint, need_jacobian, only_free_vertices);

    // get cone indices with minimum and maximum defect
    int i = 0;
    int j = 0;
    Scalar flat_angle = (is_symmetric) ? 4. * M_PI : 2. * M_PI;
    for (int k = 0; k < num_vertices; ++k) {
        int vi = marked_metric.v_rep[i];
        int vj = marked_metric.v_rep[j];
        int vk = marked_metric.v_rep[k];

        // only add (optionally) interior cone pairs at flat vertices
        if ((use_interior_vertices) && (is_symmetric) && (is_bd_vertex[vk])) continue;
        if ((use_flat_vertices) && (!float_equal(marked_metric.Th_hat[vk], flat_angle))) continue;

        if (constraint[vk] < constraint[vi]) i = k;
        if (constraint[vk] > constraint[vj]) j = k;
    }

    return std::make_tuple(i, j);
}

std::tuple<int, int> add_optimal_cone_pair(MarkedPennerConeMetric& marked_metric)
{
    auto [i, j] = get_constraint_outliers(marked_metric, true, true);
    spdlog::debug("Adding positive cone at {}", i);
    spdlog::debug("Adding negative cone at {}", j);
    bool is_symmetric = (marked_metric.type[0] != 0);
    Scalar angle_delta = (is_symmetric) ? M_PI : (M_PI / 2.);
    marked_metric.Th_hat[marked_metric.v_rep[i]] += angle_delta;
    marked_metric.Th_hat[marked_metric.v_rep[j]] -= angle_delta;

    return std::make_tuple(i, j);
}

void fix_cones(Mesh<Scalar>& m, int min_cone_index)
{
    // Remove any zero cones
    while (contains_small_cones(m.Th_hat, min_cone_index)) {
        remove_minimum_cone(m);
    }

    // Add another cone pair to torus with a cone pair
    if (is_torus_with_cone_pair(m)) {
        add_random_cone_pair(m);
    }
}


// TODO May be worth supporting
void remove_trivial_boundaries(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    std::vector<Scalar>& Th_hat)
{
    std::vector<int> boundary_components = find_boundary_components(m);
    for (int h_start : boundary_components) {
        spdlog::info("Checking for trivial loop at {}", h_start);
        bool is_trivial = true;
        int h = h_start;
        do {
            // Circulate to next boundary edge
            while (m.type[h] != 2) {
                h = m.opp[m.n[h]];
            }
            h = m.opp[h];

            int vi = vtx_reindex[m.v_rep[m.to[h]]];
            if (!float_equal(Th_hat[vi], M_PI)) {
                is_trivial = false;
                break;
            }
        } while (h != h_start);

        if (is_trivial) {
            spdlog::info("Adjusting trivial loop at {}", h);
            int vi = vtx_reindex[m.v_rep[m.to[h]]];
            int vj = vtx_reindex[m.v_rep[m.to[m.opp[h]]]];
            Th_hat[vi] += M_PI / 2.;
            Th_hat[vj] -= M_PI / 2.;
        }
    }
}

void make_interior_free(Mesh<Scalar>& m)
{
    m.fixed_dof = std::vector<bool>(m.n_ind_vertices(), true);
    auto bd_vertices = find_boundary_vertices(m);
    for (int vi : bd_vertices) {
        m.fixed_dof[m.v_rep[vi]] = false;
    }
}

} // namespace Holonomy
} // namespace Penner