// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "optimization/interface.h"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "util/embedding.h"
#include "util/vector.h"
#include "parametrization/interpolation.h"
#include "parametrization/layout.h"
#include "metric/projection.h"
#include "parametrization/translation.h"

/// FIXME Do cleaning pass


namespace Penner {
namespace Optimization {

std::unique_ptr<DifferentiableConeMetric> generate_initial_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    std::vector<int>& vtx_reindex,
    std::vector<int> free_cones,
    bool fix_boundary,
    bool use_discrete_metric)
{
    // Convert VF mesh to halfedge
    std::vector<int> indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        uv,
        F_uv,
        Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);

    // Build initial metric and target metric from edge lengths
    VectorX scale_factors;
    scale_factors.setZero(m.n_ind_vertices());
    bool is_hyperbolic = false;
    InterpolationMesh<Scalar> interpolation_mesh(m, scale_factors, is_hyperbolic);
    VectorX reduced_metric_coords;
    if (use_discrete_metric) {
        reduced_metric_coords = interpolation_mesh.get_reduced_metric_coordinates();
        return std::make_unique<DiscreteMetric>(m, reduced_metric_coords);
    } else {
        std::vector<int> flip_sequence, hyperbolic_flip_sequence;
        interpolation_mesh.convert_to_hyperbolic_surface(flip_sequence, hyperbolic_flip_sequence);
        reduced_metric_coords = interpolation_mesh.get_reduced_metric_coordinates();
        return std::make_unique<PennerConeMetric>(m, reduced_metric_coords);
    }
}

std::unique_ptr<EnergyFunctor> generate_energy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Scalar>& Th_hat,
    const DifferentiableConeMetric& target_cone_metric,
    const EnergyChoice& energy_choice)
{
    // Generate chosen energy
    if (energy_choice == EnergyChoice::log_length) {
        return std::make_unique<LogLengthEnergy>(LogLengthEnergy(target_cone_metric));
    } else if (energy_choice == EnergyChoice::log_scale) {
        return std::make_unique<LogScaleEnergy>(LogScaleEnergy(target_cone_metric));
    } else if (energy_choice == EnergyChoice::quadratic_sym_dirichlet) {
        // Build discrete mesh
        std::vector<int> vtx_reindex;
        std::vector<int> free_cones = {};
        bool fix_boundary = false;
        std::unique_ptr<DifferentiableConeMetric> eucl_cone_metric =
            generate_initial_mesh(V, F, V, F, Th_hat, vtx_reindex, free_cones, fix_boundary, true);
        DiscreteMetric discrete_metric(
            *eucl_cone_metric,
            eucl_cone_metric->get_metric_coordinates());

        // Build energy
        return std::make_unique<QuadraticSymmetricDirichletEnergy>(
            QuadraticSymmetricDirichletEnergy(target_cone_metric, discrete_metric));
    } else if (energy_choice == EnergyChoice::sym_dirichlet) {
        // Build discrete mesh
        std::vector<int> vtx_reindex;
        std::vector<int> free_cones = {};
        bool fix_boundary = false;
        std::unique_ptr<DifferentiableConeMetric> eucl_cone_metric =
            generate_initial_mesh(V, F, V, F, Th_hat, vtx_reindex, free_cones, fix_boundary, true);
        DiscreteMetric discrete_metric(
            *eucl_cone_metric,
            eucl_cone_metric->get_metric_coordinates());

        // Build energy
        return std::make_unique<SymmetricDirichletEnergy>(
            SymmetricDirichletEnergy(target_cone_metric, discrete_metric));
    } else if (energy_choice == EnergyChoice::p_norm) {
        return std::make_unique<LogLengthEnergy>(LogLengthEnergy(target_cone_metric, 4));
    } else {
        return std::make_unique<LogLengthEnergy>(LogLengthEnergy(target_cone_metric));
    }
}

std::vector<Scalar> correct_cone_angles(const std::vector<Scalar>& initial_cone_angles)
{
    // Get precise value of pi
    Scalar pi;
#ifdef MULTIPRECISION
    pi = Scalar(mpfr::const_pi());
#else
    pi = Scalar(M_PI);
#endif

    // Correct angles
    int num_vertices = initial_cone_angles.size();
    std::vector<Scalar> corrected_cone_angles(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        Scalar angle = initial_cone_angles[i];
        int rounded_angle = lround(Scalar(60.0) * angle / pi);
        corrected_cone_angles[i] = (rounded_angle * pi) / Scalar(60.0);
    }

    return corrected_cone_angles;
}


} // namespace Optimization
} // namespace Penner 