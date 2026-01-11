/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#include "optimization/parameterization/translation.h"

#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "util/embedding.h"
#include "util/linear_algebra.h"
#include "util/vector.h"
#include "optimization/core/reparametrization.h"
#include "optimization/core/shear.h"

/// FIXME Do cleaning pass

namespace Penner {
namespace Optimization {

/// Generate the Lagrangian system Lx = b for the least squares solution to the halfedge
/// translations in the hyperbolic metric needed to satisfy the per halfedge shear
/// change and face zero sum condition.
///
/// The zero sum face condition is necessary to extend the edge translations
/// to a projective transformation on the entire face.
///
/// @param[in] m: mesh
/// @param[in] halfedge_shear_change: constraint for the change in shear per halfedge
/// @param[out] lagrangian_matrix: matrix L defining the lagrangian system
/// @param[out] right_hand_side: vector b defining the right hand side of the lagrangian system
void generate_translation_lagrangian_system(
    const Mesh<Scalar>& m,
    const VectorX& halfedge_shear_change,
    MatrixX& lagrangian_matrix,
    VectorX& right_hand_side)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Get number of halfedges, edges, and faces
    int n_h = m.n.size();
    int n_e = e2he.size();
    int n_f = m.h.size();

    // Initialize matrix entry list
    std::vector<T> tripletList;
    std::vector<Scalar> rhs_vec;
    tripletList.reserve(n_h + 4 * n_e + 6 * n_f);
    rhs_vec.reserve(n_h + 4 * n_e + 6 * n_f);

    // Add identity block
    for (int h = 0; h < n_h; ++h) {
        tripletList.push_back(T(h, h, 1.0));
        rhs_vec.push_back(0.0);
    }

    // Add halfedge sum constraints
    int mu_count = 0;
    for (int e = 0; e < n_e; ++e) {
        int h = e2he[e];
        int ho = m.opp[h];

        // Add edge sum block
        tripletList.push_back(T(n_h + mu_count, h, 1.0));
        tripletList.push_back(T(n_h + mu_count, ho, 1.0));

        // Add edge sum transpose block
        tripletList.push_back(T(h, n_h + mu_count, 1.0));
        tripletList.push_back(T(ho, n_h + mu_count, 1.0));

        // Add constrained sum values to RHS
        rhs_vec.push_back(halfedge_shear_change[h]);

        // Increment number of constraints by 1
        mu_count += 1;
    }

    // Add face sum constraints (leaving one out due to redundancy)
    int nu_count = 0;
    for (int f = 1; f < n_f; ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Add face sum block
        tripletList.push_back(T(n_h + mu_count + nu_count, hij, 1.0));
        tripletList.push_back(T(n_h + mu_count + nu_count, hjk, 1.0));
        tripletList.push_back(T(n_h + mu_count + nu_count, hki, 1.0));

        // Add face sum transpose block
        tripletList.push_back(T(hij, n_h + mu_count + nu_count, 1.0));
        tripletList.push_back(T(hjk, n_h + mu_count + nu_count, 1.0));
        tripletList.push_back(T(hki, n_h + mu_count + nu_count, 1.0));

        // Add 0 to RHS
        rhs_vec.push_back(0.0);

        // Increment constraint counter by 1
        nu_count += 1;
    }

    // Build matrix
    int n_var = n_h + mu_count + nu_count;
    lagrangian_matrix.resize(n_var, n_var);
    lagrangian_matrix.reserve(tripletList.size());
    lagrangian_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    // Build RHS
    right_hand_side.setZero(n_var);
    for (int i = 0; i < n_var; ++i) {
        right_hand_side[i] = rhs_vec[i];
    }
}

void _compute_as_symmetric_as_possible_translations(
    const Mesh<Scalar>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_translations)
{
    // Compute the change in shear from the target to the new metric
    spdlog::trace("Computing shear change");
    VectorX he_shear_change;
    compute_shear_change(m, he_metric_coords, he_metric_target, he_shear_change);

    // Build the lagrangian for the problem
    spdlog::trace("Computing lagrangian system");
    MatrixX lagrangian_matrix;
    VectorX right_hand_side;
    generate_translation_lagrangian_system(m, he_shear_change, lagrangian_matrix, right_hand_side);

    // Compute the solution of the lagrangian
    spdlog::trace(
        "Computing solution for {}x{} system with length {} rhs",
        lagrangian_matrix.rows(),
        lagrangian_matrix.cols(),
        right_hand_side.size());
    lagrangian_matrix.makeCompressed();
    VectorX lagrangian_solution = solve_linear_system(lagrangian_matrix, -right_hand_side);

    // The desired translations are at the head of the solution vector
    spdlog::trace("Extracting halfedges");
    int num_halfedges = he_shear_change.size();
    he_translations = lagrangian_solution.head(num_halfedges);
}

template <typename OverlayScalar>
void generate_translation_constraint_system(
    const Mesh<OverlayScalar>& m,
    const VectorX& halfedge_shear_change,
    Eigen::SparseMatrix<OverlayScalar>& constraint_matrix,
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1>& right_hand_side)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(change_mesh_type<OverlayScalar, Scalar>(m), he2e, e2he);

    // Get number of halfedges, edges, and faces
    int n_h = m.n.size();
    int n_e = e2he.size();
    int n_f = m.h.size();

    // Initialize matrix entry list
    std::vector<Eigen::Triplet<OverlayScalar>> tripletList;
    std::vector<Scalar> rhs_vec;
    tripletList.reserve(2 * n_e + 3 * n_f);
    rhs_vec.reserve(n_e + n_f);

    // Add halfedge sum constraints
    int mu_count = 0;
    for (int e = 0; e < n_e; ++e) {
        int h = e2he[e];
        int ho = m.opp[h];

        // Add edge sum block
        tripletList.push_back(Eigen::Triplet<OverlayScalar>(mu_count, h, OverlayScalar(1.0)));
        tripletList.push_back(Eigen::Triplet<OverlayScalar>(mu_count, ho, OverlayScalar(1.0)));

        // Add constrained sum values to RHS
        rhs_vec.push_back(halfedge_shear_change[h]);

        // Increment number of constraints by 1
        mu_count += 1;
    }

    // Add face sum constraints (leaving one out due to redundancy)
    int nu_count = 0;
    for (int f = 1; f < n_f; ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Add face sum block
        tripletList.push_back(Eigen::Triplet<OverlayScalar>(mu_count + nu_count, hij, OverlayScalar(1.0)));
        tripletList.push_back(Eigen::Triplet<OverlayScalar>(mu_count + nu_count, hjk, OverlayScalar(1.0)));
        tripletList.push_back(Eigen::Triplet<OverlayScalar>(mu_count + nu_count, hki, OverlayScalar(1.0)));

        // Add 0 to RHS
        rhs_vec.push_back(0.0);

        // Increment constraint counter by 1
        nu_count += 1;
    }

    // Build matrix
    int n_constraint = mu_count + nu_count;
    constraint_matrix.resize(n_constraint, n_h);
    constraint_matrix.reserve(tripletList.size());
    constraint_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    // Build RHS
    right_hand_side.setZero(n_constraint);
    for (int i = 0; i < n_constraint; ++i) {
        right_hand_side[i] = OverlayScalar(rhs_vec[i]);
    }
}

template <typename OverlayScalar>
void generate_edge_translation_constraint_system(
    const Mesh<OverlayScalar>& m,
    const VectorX& halfedge_shear_change,
    Eigen::SparseMatrix<OverlayScalar>& constraint_matrix,
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1>& right_hand_side)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(change_mesh_type<OverlayScalar, Scalar>(m), he2e, e2he);

    // Get number of halfedges, edges, and faces
    int n_h = m.n.size();
    int n_e = e2he.size();
    int n_f = m.h.size();

    // Initialize matrix entry list
    std::vector<Eigen::Triplet<OverlayScalar>> tripletList;
    right_hand_side.setZero(n_f - 1);
    tripletList.reserve(3 * n_f);

    // Add face sum constraints (leaving one out due to redundancy)
    for (int f = 0; f < n_f - 1; ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Add face sum block
        for (int h : {hij, hjk, hki})
        {
            int e = he2e[h];
            if (h == e2he[e])
            {
                tripletList.push_back(Eigen::Triplet<OverlayScalar>(f, e, OverlayScalar(1.0)));
                right_hand_side[f] -= halfedge_shear_change[h] / 2.;
            }
            else
            {
                tripletList.push_back(Eigen::Triplet<OverlayScalar>(f, e, OverlayScalar(-1.0)));
                right_hand_side[f] -= halfedge_shear_change[h] / 2.;
            }
        }
    }

    // Build matrix
    constraint_matrix.resize(n_f - 1, n_e);
    constraint_matrix.reserve(tripletList.size());
    constraint_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename OverlayScalar>
void compute_as_symmetric_as_possible_translations(
    const Mesh<OverlayScalar>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_translations)
{
    spdlog::debug("Computing least squares translations with psuedo-inverse");

    // Compute the change in shear from the target to the new metric
    spdlog::trace("Computing shear change");
    VectorX he_shear_change;
    compute_shear_change(
        change_mesh_type<OverlayScalar, Scalar>(m),
        he_metric_coords,
        he_metric_target,
        he_shear_change);
    SPDLOG_TRACE(
        "shear change in range [{}, {}]",
        he_shear_change.minCoeff(),
        he_shear_change.maxCoeff());

    // Build the constraint for the problem
    bool use_edge_system = false;
    // WARNING: edge sytem is faster, but produces low quality values and may be buggy
    if (use_edge_system)
    {
        spdlog::trace("Computing constraint system");
        Eigen::SparseMatrix<OverlayScalar> constraint_matrix;
        Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> right_hand_side;
        generate_edge_translation_constraint_system(m, he_shear_change, constraint_matrix, right_hand_side);

        // Compute the solution of the constraint
        Eigen::SparseMatrix<OverlayScalar> gram_matrix = constraint_matrix * constraint_matrix.transpose();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<OverlayScalar>> solver;
        solver.compute(gram_matrix);
        Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> constraint_solution = solver.solve(-right_hand_side);

        // The desired translations are at the head of the solution vector
        VectorX e_translations = convert_vector_type<OverlayScalar, Scalar>(constraint_matrix.transpose() * constraint_solution);

        // Get edge maps
        std::vector<int> he2e;
        std::vector<int> e2he;
        build_edge_maps(change_mesh_type<OverlayScalar, Scalar>(m), he2e, e2he);
        he_translations.resize(m.n_halfedges());
        for (int e = 0; e < e2he.size(); ++e)
        {
            int hij = e2he[e];
            int hji = m.opp[hij];
            he_translations[hij] = he_shear_change[hij] + e_translations[e];
            he_translations[hji] = he_shear_change[hij] - e_translations[e];
        }
    }
    else
    {
        // Build the constraint for the problem
        spdlog::trace("Computing constraint system");
        Eigen::SparseMatrix<OverlayScalar> constraint_matrix;
        Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> right_hand_side;
        generate_translation_constraint_system(m, he_shear_change, constraint_matrix, right_hand_side);

        // Compute the solution of the constraint
        spdlog::debug(
            "Computing solution for {}x{} system with length {} rhs",
            constraint_matrix.rows(),
            constraint_matrix.cols(),
            right_hand_side.size());
        Eigen::SparseMatrix<OverlayScalar> gram_matrix = constraint_matrix * constraint_matrix.transpose();
        //gram_matrix.makeCompressed();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<OverlayScalar>> solver;
        solver.compute(gram_matrix);
        Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> constraint_solution = solver.solve(-right_hand_side);

        // The desired translations are at the head of the solution vector
        he_translations = convert_vector_type<OverlayScalar, Scalar>(constraint_matrix.transpose() * constraint_solution);
    }

    // use zero coordinates as fallback for nans
    if (isnan(he_translations.maxCoeff()))
    {
        spdlog::warn("Setting halfedge translations to zero due to numerical instability");
        he_translations.setZero();
    }
}

template void compute_as_symmetric_as_possible_translations<Scalar>(
    const Mesh<Scalar>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_translations);

#ifdef WITH_MPFR
#ifndef MULTIPRECISION
template void compute_as_symmetric_as_possible_translations<mpfr::mpreal>(
    const Mesh<mpfr::mpreal>& m,
    const VectorX& he_metric_coords,
    const VectorX& he_metric_target,
    VectorX& he_translations);

#endif
#endif

} // namespace Optimization
} // namespace Penner
