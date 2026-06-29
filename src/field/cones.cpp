// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "field/cones.h"

#include "util/boundary.h"
#include "util/vector.h"

// corner angle computation
#include "metric/constraint.h"

// check valid one forms
#include "field/forms.h"

#include <random>

namespace Penner {
namespace Field {


std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form)
{
    assert(Field::is_valid_one_form(m, rotation_form));
    // Compute the corner angles
    VectorX he2angle, he2cot;
    corner_angles(m, he2angle, he2cot);

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

    for (int vi = 0; vi < num_vertices; ++vi) {
        Th_hat[vi] = round(Th_hat[vi] / (M_PI / 2.)) * (M_PI / 2.);
    }

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
    return contains_small_cones(Th_hat, 1);
}

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

        if ((m.type[h] == 1) && (m.R[m.opp[h]] == h) && float_equal<Scalar>(m.Th_hat[vi], 2. * M_PI)) {
            return vi;
        }
    }

    return -1;
}

void add_random_cone_pair(Mesh<Scalar>& m, bool only_interior, int offset)
{
    bool is_symmetric = (m.type[0] != 0);
    Scalar angle_delta = (is_symmetric) ? M_PI : (M_PI / 2.);

    int num_vertices = m.n_vertices();
    for (int i = 0; i < num_vertices; ++i)
    {
        int vi = (i + offset) % num_vertices;
        // check if vertex is valid
        int Vi = m.v_rep[vi];
        if (!float_equal(m.Th_hat[Vi], 4. * angle_delta)) continue;
        if ((only_interior) && (!is_interior(m, vi))) continue;

        // try to find valid adjacent vertex
        int hik = m.out[vi];
        int hij = hik;
        int vj = m.to[hij];
        do {
            hij = m.n[m.opp[hij]];
            vj = m.to[hij];
            if ((only_interior) && (!is_interior(m, vj))) continue;
            if (!float_equal(m.Th_hat[m.v_rep[vj]], 4. * angle_delta)) continue;
            break;
        }
        while (hij != hik);

        // check if valid adjacent vertex found
        int Vj = m.v_rep[vj];
        if ((only_interior) && (!is_interior(m, vj))) continue;
        if (!float_equal(m.Th_hat[Vj], 4. * angle_delta)) continue;

        // add cones
        spdlog::info("Adding negative cone at {} with angle {}", Vi, m.Th_hat[Vi]);
        spdlog::info("Adding positive cone at {} with angle {}", Vj, m.Th_hat[Vj]);
        m.Th_hat[m.v_rep[vi]] -= angle_delta;
        m.Th_hat[m.v_rep[vj]] += angle_delta;
        return;
    }

    // try again with interior if fails
    if (only_interior)
    {
        spdlog::warn("Cannot add cone pair in interior");
        add_random_cone_pair(m, false);
    } else {
        spdlog::error("Cannot add cone pair");
    }
}


void fix_cones(Mesh<Scalar>& m, int min_cone_index)
{
    // Remove any zero cones
    while (contains_small_cones(m.Th_hat, min_cone_index)) {
        remove_minimum_cone(m);
    }

    // Add another cone pair to torus with a cone pair
    if (is_torus_with_cone_pair(m)) {
        add_random_cone_pair(m, true, m.n_vertices() / 4);
        add_random_cone_pair(m, true, m.n_vertices() / 2);
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
            if (!float_equal<Scalar>(Th_hat[vi], M_PI)) {
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


} // namespace Field
} // namespace Penner