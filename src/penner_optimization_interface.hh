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
#pragma once
#include "common.hh"
#include "cone_metric.hh"
#include "energy_functor.hh"

namespace CurvatureMetric {

/// Generate a mesh with initial target metric coordinates for optimization
///
/// @param[in] V: initial vertices
/// @param[in] F: initial faces
/// @param[in] uv: initial metric vertices
/// @param[in] F_uv: initial metric faces
/// @param[in] Th_hat: target angles
/// @param[out] vtx_reindex: vertices reindexing for new halfedge mesh
/// @param[in] free_cones: (optional) vertex cones to leave unconstrained
/// @param[in] fix_boundary: (optional) if true, fix boundary edge lengths
/// @param[in] use_discrete_metric: (optional) if true, use log edge lengths instead of penner coordinates
/// @return differentiable mesh with metric
std::unique_ptr<DifferentiableConeMetric> generate_initial_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    std::vector<int>& vtx_reindex,
    std::vector<int> free_cones = {},
    bool fix_boundary = false,
    bool use_discrete_metric = false);

/// Generate a distortion energy for a given target mesh.
///
/// @param[in] V: initial vertices
/// @param[in] F: initial faces
/// @param[in] Th_hat: target angles
/// @param[in] target_cone_metric: target mesh
/// @param[in] energy_choice: energy type to construct
/// @return energy functor for the chosen energy and mesh
std::unique_ptr<EnergyFunctor> generate_energy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Scalar>& Th_hat,
    const DifferentiableConeMetric& target_cone_metric,
    const EnergyChoice& energy_choice);

/// Correct cone angles that are multiples of pi/60 to machine precision.
///
/// @param[in] initial_cone_angles: target angles to correct
/// @return: corrected angles
std::vector<Scalar> correct_cone_angles(const std::vector<Scalar>& initial_cone_angles);

/// Write an obj file with uv coordinates.
///
/// @param[in] filename: obj output file location
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv corner coordinates
/// @param[in] F_uv: mesh uv faces
void write_obj_with_uv(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/// Given a mesh with initial target and final optimized metric coordinates, generate a corresponding
/// overlay VF mesh with parametrization.
///
/// @param[in] V: initial mesh vertices
/// @param[in] F: initial mesh faces
/// @param[in] Th_hat: initial target angles
/// @param[in] m: mesh
/// @param[in] vtx_reindex: map from old to new vertices
/// @param[in] reduced_metric_target: initial mesh Penner coordinates
/// @param[in] reduced_metric_coords: optimized mesh metric
/// @param[in] do_best_fit_scaling: if true, extract best fit scale factors from the metric
/// @return parametrized VF mesh with cut and topology mapping information
std::
    tuple<
        OverlayMesh<Scalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_metric(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<Scalar>& Th_hat,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords,
        std::vector<bool> is_cut = {},
        bool do_best_fit_scaling = false);
std::
    tuple<
        OverlayMesh<Scalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_halfedge_metric(
        const Eigen::MatrixXd& V,
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const DifferentiableConeMetric& initial_cone_metric,
        const VectorX& reduced_metric_coords,
        std::vector<bool> cut_h,
        bool do_best_fit_scaling);

std::
    tuple<
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool> // is_cut_h
        >
    generate_VF_mesh_from_discrete_metric(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<Scalar>& Th_hat,
        const VectorX& reduced_log_edge_lengths,
        std::vector<bool> cut_h = {});

} // namespace CurvatureMetric
