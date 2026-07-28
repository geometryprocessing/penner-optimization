// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "metric/interface.h"
#include "util/vector.h"

// used to get cone vertices
#include "metric/constraint.h"

// get boundary vertices for free interior
#include "util/boundary.h"

// Delaunay flip algorithm for Penner coordinates
#include "conformal_ideal_delaunay/ConformalInterface.hh"

// TODO: Clean code

namespace Penner {

VectorX generate_log_edge_lengths(const Mesh<Scalar>& m)
{
    // Make copy of mesh delaunay
    // Get metric coordinates from copy
    int num_halfedges = m.n_halfedges();
    VectorX metric_coords(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        metric_coords[h] = 2.0 * log(m.l[h]);
        if (isnan(metric_coords[h])) spdlog::warn("generating NaN Penner coordinate");
    }

    return metric_coords;
}


VectorX generate_penner_coordinates(const Mesh<Scalar>& m)
{
    // Make copy of mesh delaunay
    Mesh<Scalar> m_copy = m;
    VectorX scale_factors;
    scale_factors.setZero(m.n_ind_vertices());
    bool use_ptolemy_flip = false;
    DelaunayStats del_stats;
    SolveStats<Scalar> solve_stats;
    ConformalIdealDelaunay<Scalar>::MakeDelaunay(
        m_copy,
        scale_factors,
        del_stats,
        solve_stats,
        use_ptolemy_flip);

    // Get flip sequence
    const auto& flip_sequence = del_stats.flip_seq;
    for (auto iter = flip_sequence.rbegin(); iter != flip_sequence.rend(); ++iter) {
        int flip_index = *iter;
        if (flip_index < 0) {
            flip_index = -flip_index - 1;
        }
        m_copy.flip_ccw(flip_index);
        m_copy.flip_ccw(flip_index);
        m_copy.flip_ccw(flip_index);
    }

    // Get metric coordinates from copy
    int num_halfedges = m.n_halfedges();
    VectorX metric_coords(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        metric_coords[h] = 2.0 * log(m_copy.l[h]);
        if (isnan(metric_coords[h])) spdlog::warn("generating NaN Penner coordinate");
    }

    return metric_coords;
}

void make_free_interior(Mesh<Scalar>& m) {
    m.fixed_dof = std::vector<bool>(m.n_ind_vertices(), true);
    auto bd_vertices = find_boundary_vertices(m);
    for (int vi : bd_vertices) {
        m.fixed_dof[m.v_rep[vi]] = false;
    }

    // handle trivial interior case
    int num_bd_vertices = bd_vertices.size();
    if (num_bd_vertices == m.n_ind_vertices()) {
        m.fixed_dof[0] = true;
    }
}

void remove_symmetry(Mesh<Scalar>& m) {
    // make copy
    Mesh<Scalar> _m = m;

    m.Th_hat = std::vector<Scalar>(m.n_vertices(), 0.);
    m.fixed_dof = std::vector<bool>(m.n_vertices(), false);
    arange(m.n_vertices(), m.v_rep);
    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        m.type[hij] = 0;
        // m.R[hij] = 0;

        // split interior cones
        m.Th_hat[m.v_rep[m.to[hij]]] = _m.Th_hat[_m.v_rep[_m.to[hij]]] / 2.;
        if (_m.type[hij] == 2) {
            m.fixed_dof[m.v_rep[m.to[hij]]] = true;
        } else {
            m.fixed_dof[m.v_rep[m.to[hij]]] = _m.fixed_dof[_m.v_rep[_m.to[hij]]];
        }
    }

    std::vector<int> bd_vertices = find_boundary_vertices(_m);
    for (int vi : bd_vertices) {
        m.Th_hat[m.v_rep[vi]] = _m.Th_hat[_m.v_rep[vi]];
        m.fixed_dof[m.v_rep[vi]] = _m.fixed_dof[_m.v_rep[vi]];
    }
}


DiscreteMetric generate_discrete_metric(const Mesh<Scalar>& m) {
    VectorX log_length_coords = generate_log_edge_lengths(m);
    return DiscreteMetric(m, log_length_coords);
}

std::tuple<PennerConeMetric, std::vector<int>> generate_cone_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Scalar>& Th_hat,
    std::vector<int> free_cones,
    ConeMetricParameters cone_metric_params)
{
    // Convert VF mesh to halfedge
    bool fix_boundary = false;
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        V,
        F,
        Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);
    
    return std::make_tuple(generate_cone_metric_from_mesh(m, cone_metric_params), vtx_reindex);
}

PennerConeMetric generate_cone_metric_from_mesh(
    const Mesh<Scalar>& m,
    ConeMetricParameters cone_metric_params)
{
    // Build initial metric coordinates, using zero, edge, or Penner coordinates
    VectorX metric_coords;
    if (cone_metric_params.use_initial_zero) {
        int num_halfedges = m.n_halfedges();
        metric_coords = VectorX::Zero(num_halfedges);
    } else if (cone_metric_params.use_log_length) {
        metric_coords = generate_log_edge_lengths(m);
    } else {
        metric_coords = generate_penner_coordinates(m);
    }

    // create metric
    PennerConeMetric cone_metric(m, metric_coords);

    // optional modifications
    if (cone_metric_params.free_interior) make_free_interior(cone_metric);
    if (cone_metric_params.remove_symmetry) remove_symmetry(cone_metric);

    // if set to free cones, instead mark all free cones
    if (cone_metric_params.use_free_cones)
    {
        std::vector<int> cones = enumerate_cone_vertices(cone_metric);
        if (!cones.empty())
        {
            convert_index_vector_to_boolean_array(cones, cone_metric.n_ind_vertices(), cone_metric.fixed_dof);
        }
    }

    return cone_metric;
}


} // namespace Penner