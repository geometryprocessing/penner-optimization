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

#include "optimization/core/common.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace Penner {
namespace Optimization {

/// Generate an overlay mesh for the mesh m with given metric coordinates integrated
/// as the mesh metric.
///
/// Note that here the original mesh metric is overwritten, whereas during the optimization
/// the two are kept separate and the mesh maintains the original input length metric.
///
/// @param[in] m: mesh to add overlay to
/// @param[in] reduced_metric_coords: reduced metric coordinates for overlay mesh
/// @return: overlay mesh with new metric coordinates
template <typename OverlayScalar>
OverlayMesh<OverlayScalar> add_overlay(const Mesh<Scalar>& m, const VectorX& reduced_metric_coords);

/// @brief: Make an overlay mesh into a tufted double cover
///
/// @param[in] mo: mesh to make tufted
template <typename OverlayScalar>
void make_tufted_overlay(OverlayMesh<OverlayScalar>& mo);

/// Given a VF mesh, check that the signed face areas are nonnegative
///
/// @param[in] V: mesh vertices in 2D
/// @param[in] F: mesh faces
/// @return true iff the face areas are all nonnegative
bool check_areas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

/// Given a VF mesh with uv coordinates, get the maximum error of the uv lengths
/// across cuts
///
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv coordinates in 2D
/// @param[in] F_uv: mesh uv faces
/// @return maximum uv length error across cuts
double compute_uv_length_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/// Given a VF mesh with uv coordinates, check that it satisfies fundamental uv
/// consistency constraints:
///     - uv lengths match up across cuts
///     - uv face areas are nonnegative
///
/// @param[in] V: mesh vertices in 3D
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv coordinates in 2D
/// @param[in] F_uv: mesh uv faces
/// @return true iff the mesh passes all of the tests
bool check_uv(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

/// Given a halfedge mesh, do a bfs on dual graph of mesh to produce a cut
///
/// Note that this only lays out the connected component containing the start halfedge.
///
/// @param m: mesh data structure
/// @param is_cut_h: (optional) pre-defined cuts to be included
/// @param start_h: the first halfedge to be laid out, can be used to control the axis-alignment for the whole patch
/// @return #h vector, mark whether the current halfedge is part of cut graph
std::vector<bool>
compute_layout_topology(const Mesh<Scalar>& m, const std::vector<bool>& is_cut_h, int start_h = -1);

/// Given a cut defined on the original or current mesh, pull it back to a cut defined on
/// an overlay mesh for the given mesh (possibly after flips)
///
/// @param[in] m_o: overlay mesh data structure
/// @param[in] is_cut_h: cuts on the original or current mesh
/// @param[in] is_original_cut: if true, use cut mask on the original mesh
/// @return cuts on the overlay mesh
template <typename Scalar>
std::vector<bool> pullback_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h,
    bool is_original_cut = true);


// TODO: Document this technical function
// Exposed for usage in other libraries
template <typename OverlayScalar>
std::
    tuple<
        OverlayMesh<OverlayScalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    consistent_overlay_mesh_to_VL(
        const Mesh<Scalar>& _m,
        OverlayMesh<OverlayScalar>& mo,
        const std::vector<int>& vtx_reindex,
        const std::vector<bool>& is_bd,
        std::vector<Scalar>& u,
        std::vector<std::vector<OverlayScalar>>& V_overlay,
        std::vector<std::pair<int, int>>& endpoints,
        const std::vector<bool>& is_cut_orig,
        const std::vector<bool>& is_cut,
        bool use_uniform_bc=false);

    double signed_area(
        const Eigen::Vector2d& A,
        const Eigen::Vector2d& B,
        const Eigen::Vector2d& C);

template <typename Scalar>
std::tuple<
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
compute_layout_VF(OverlayMesh<Scalar>& m_o);

#ifdef PYBIND
#endif


} // namespace Optimization
} // namespace Penner