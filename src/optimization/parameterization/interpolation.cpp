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
#include "optimization/parameterization/interpolation.h"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "util/embedding.h"
#include "util/vector.h"
#include "optimization/core/projection.h"
#include "optimization/core/reparametrization.h"
#include "optimization/parameterization/translation.h"

/// FIXME Do cleaning pass

namespace Penner {
namespace Optimization {

bool is_vector_equal(const std::vector<int>& v, const std::vector<int>& w)
{
    if (v.size() != w.size()) return false;
    int size = v.size();
    for (int i = 0; i < size; ++i)
    {
        if (v[i] != w[i]) return false;
    }

    return true;
}

template <typename OverlayScalar>
void interpolate_penner_coordinates(
    const Mesh<Scalar>& mesh,
    const VectorX& halfedge_metric_coords,
    const VectorX& scale_factors,
    InterpolationMesh<OverlayScalar>& interpolation_mesh,
    InterpolationMesh<OverlayScalar>& reverse_interpolation_mesh)
{
    // Check scale factor size
    if (scale_factors.size() != mesh.n_ind_vertices()) {
        spdlog::error("Invalid scale factor array for mesh");
        return;
    }

    // Get forward Euclidean interpolation mesh
    spdlog::trace("Building interpolation mesh");
    VectorX trivial_scale_factors;
    trivial_scale_factors.setZero(mesh.n_ind_vertices());
    bool is_hyperbolic = false;
    interpolation_mesh = InterpolationMesh<OverlayScalar>(mesh, trivial_scale_factors, is_hyperbolic);

    // Convert the surface to a hyperbolic surface with Penner coordiantes
    bool undo_flips = true;
    if (undo_flips) {
        spdlog::trace("Converting surface to hyperbolic surface");
        std::vector<int> euclidean_flip_sequence, hyperbolic_flip_sequence;
        interpolation_mesh.convert_to_hyperbolic_surface(
            euclidean_flip_sequence,
            hyperbolic_flip_sequence);
        Mesh<OverlayScalar>& mc = interpolation_mesh.get_mesh();
        std::vector<char> initial_type = mc.type;
        std::vector<int> initial_R = mc.R;

        // Get the metric coordinates of the original metric
        VectorX initial_halfedge_metric_coords =
            interpolation_mesh.get_halfedge_metric_coordinates();
        SPDLOG_TRACE(
            "Initial metric coordiantes in range [{}, {}]",
            initial_halfedge_metric_coords.minCoeff(),
            initial_halfedge_metric_coords.maxCoeff());

        // Compute translations for reparametrization
        VectorX halfedge_translations;
        compute_as_symmetric_as_possible_translations(
            mc,
            halfedge_metric_coords,
            initial_halfedge_metric_coords,
            halfedge_translations);
        SPDLOG_TRACE(
            "Translations in range [{}, {}]",
            halfedge_translations.minCoeff(),
            halfedge_translations.maxCoeff());

        // Change the metric and reparameterize
        spdlog::debug("Changing underlying metric");
        interpolation_mesh.change_hyperbolic_surface_metric(
            halfedge_metric_coords,
            scale_factors,
            halfedge_translations);
        SPDLOG_TRACE(
            "Min change in coordinates is {}",
            (halfedge_metric_coords - initial_halfedge_metric_coords).minCoeff());
        SPDLOG_TRACE(
            "Max change in coordinates is {}",
            (halfedge_metric_coords - initial_halfedge_metric_coords).maxCoeff());

        // Make delaunay with new metric
        spdlog::trace("Making new surface Delaunay");
        std::vector<int> flip_sequence;
        interpolation_mesh.make_delaunay(flip_sequence);
        // interpolation_mesh.check_bc_values();

        // Build a clean overlay mesh with the final metric
        spdlog::trace("Building reverse interpolation mesh");
        Mesh<OverlayScalar> m_layout = interpolation_mesh.get_mesh();
        is_hyperbolic = true;
        reverse_interpolation_mesh = InterpolationMesh<OverlayScalar>(
            change_mesh_type<OverlayScalar, Scalar>(m_layout),
            trivial_scale_factors,
            is_hyperbolic);

        // Undo the flips to make the hyperbolic surface with new metric Delaunay
        reverse_interpolation_mesh.reverse_flip_sequence(flip_sequence);
        reverse_interpolation_mesh.get_mesh().type = initial_type;
        reverse_interpolation_mesh.get_mesh().R = initial_R;
        if (!is_vector_equal(mesh.n, reverse_interpolation_mesh.get_mesh().n))
        {
            spdlog::error("inconsistent reverse mesh before making euclidean");
        }

        // Undo the reparametrization
        spdlog::debug("Inverting reparameterization");
        VectorX inverse_halfedge_translations = -halfedge_translations;
        reverse_interpolation_mesh.change_hyperbolic_surface_metric(
            initial_halfedge_metric_coords,
            trivial_scale_factors,
            inverse_halfedge_translations);

        // Generate reverse map for Euclidean flips
        reverse_interpolation_mesh.reverse_flip_sequence(hyperbolic_flip_sequence);
        reverse_interpolation_mesh.force_convert_to_euclidean_surface();
        reverse_interpolation_mesh.reverse_flip_sequence(euclidean_flip_sequence);
        // reverse_interpolation_mesh.check_bc_values();
        // reverse_interpolation_mesh.force_convert_to_hyperbolic_surface();

        if (!is_vector_equal(mesh.n, reverse_interpolation_mesh.get_mesh().n))
        {
            spdlog::error("inconsistent reverse mesh");
        }
    } else {
        // Get initial reflection structure
        Mesh<OverlayScalar>& mc = interpolation_mesh.get_mesh();
        std::vector<char> initial_type = mc.type;
        std::vector<int> initial_R = mc.R;

        spdlog::trace("Making surface Delaunay");
        std::vector<int> euclidean_flip_sequence;
        interpolation_mesh.convert_to_delaunay_hyperbolic_surface(euclidean_flip_sequence);
        VectorX initial_halfedge_metric_coords =
            interpolation_mesh.get_halfedge_metric_coordinates();

        // Get Euclidean Delaunay reflection structure
        std::vector<char> eucl_del_type = mc.type;
        std::vector<int> eucl_del_R = mc.R;

        // Copy flip sequence with ptolemy flips and new metric to get metric coordinates
        spdlog::debug("Getting flipped metric coordinates");
        VectorX trivial_halfedge_translations;
        trivial_halfedge_translations.setZero(halfedge_metric_coords.size());
        InterpolationMesh<OverlayScalar> metric_interpolation_mesh =
            InterpolationMesh<OverlayScalar>(mesh, trivial_scale_factors, true);
        metric_interpolation_mesh.change_hyperbolic_surface_metric(
            halfedge_metric_coords,
            trivial_scale_factors,
            trivial_halfedge_translations);
        metric_interpolation_mesh.follow_flip_sequence(euclidean_flip_sequence);
        VectorX flipped_halfedge_metric_coords =
            metric_interpolation_mesh.get_halfedge_metric_coordinates();

        // Compute translations for reparametrization
        VectorX halfedge_translations;
        compute_as_symmetric_as_possible_translations(
            mc,
            flipped_halfedge_metric_coords,
            initial_halfedge_metric_coords,
            halfedge_translations);
        SPDLOG_DEBUG(
            "Translations in range [{}, {}]",
            halfedge_translations.minCoeff(),
            halfedge_translations.maxCoeff());

        // Change the metric and reparameterize
        spdlog::debug("Changing underlying metric");
        interpolation_mesh.change_hyperbolic_surface_metric(
            flipped_halfedge_metric_coords,
            scale_factors,
            halfedge_translations);

        // Make delaunay with new metric
        spdlog::trace("Making new surface Delaunay");
        std::vector<int> flip_sequence;
        interpolation_mesh.make_delaunay(flip_sequence);
        // interpolation_mesh.check_bc_values();

        // Build a clean overlay mesh with the final metric
        spdlog::trace("Building reverse interpolation mesh");
        Mesh<OverlayScalar> m_layout = interpolation_mesh.get_mesh();
        is_hyperbolic = true;
        reverse_interpolation_mesh = InterpolationMesh<OverlayScalar>(
            change_mesh_type<OverlayScalar, Scalar>(m_layout),
            trivial_scale_factors,
            is_hyperbolic);

        // Undo the flips to make the hyperbolic surface with new metric Delaunay
        reverse_interpolation_mesh.reverse_flip_sequence(flip_sequence);
        reverse_interpolation_mesh.get_mesh().type = eucl_del_type;
        reverse_interpolation_mesh.get_mesh().R = eucl_del_R;

        // Undo the reparametrization
        spdlog::debug("Inverting reparameterization");
        VectorX inverse_halfedge_translations = -halfedge_translations;
        reverse_interpolation_mesh.change_hyperbolic_surface_metric(
            initial_halfedge_metric_coords,
            trivial_scale_factors,
            inverse_halfedge_translations);

        // Generate reverse map for Euclidean flips
        reverse_interpolation_mesh.force_convert_to_euclidean_surface();
        reverse_interpolation_mesh.reverse_flip_sequence(euclidean_flip_sequence);
        reverse_interpolation_mesh.get_mesh().type = initial_type;
        reverse_interpolation_mesh.get_mesh().R = initial_R;
        // reverse_interpolation_mesh.check_bc_values();
    }
}

template <typename OverlayScalar>
void interpolate_vertex_positions(
    const Eigen::MatrixXd& V,
    const std::vector<int> vtx_reindex,
    const InterpolationMesh<OverlayScalar>& interpolation_mesh,
    const InterpolationMesh<OverlayScalar>& reverse_interpolation_mesh,
    Eigen::MatrixXd& V_overlay)
{
    // Get the vertex map between the forward and reverse maps
    OverlayMesh<OverlayScalar> overlay_mesh = interpolation_mesh.get_overlay_mesh();
    OverlayMesh<OverlayScalar> reverse_overlay_mesh = reverse_interpolation_mesh.get_overlay_mesh();
    if ((overlay_mesh.bypass_overlay) || (reverse_overlay_mesh.bypass_overlay))
    {
        spdlog::warn("overlay bypassed due to numerical issue or as instructed.");
    }
    if (overlay_mesh.n_halfedges() != reverse_overlay_mesh.n_halfedges())
    {
        spdlog::error("overlay has {} halfedges", overlay_mesh.n_halfedges());
        spdlog::error("reverse overlay has {} halfedges", reverse_overlay_mesh.n_halfedges());
    }
    std::vector<int> v_map =
        ConformalIdealDelaunay<OverlayScalar>::GetVertexMap(overlay_mesh, reverse_overlay_mesh);

    // Reindex the original vertex positions
    Mesh<OverlayScalar>& mesh = overlay_mesh.cmesh();
    std::vector<std::vector<OverlayScalar>> V_reindex(3);
    for (int i = 0; i < 3; i++) {
        V_reindex[i].resize(mesh.out.size(), 0);
        for (int j = 0; j < V.rows(); j++) {
            V_reindex[i][j] = OverlayScalar(V(vtx_reindex[j], i));
        }
    }

    // Interpolate vertex positions along the original edges
    Mesh<OverlayScalar>& reverse_mesh = reverse_overlay_mesh.cmesh();
    std::vector<std::vector<OverlayScalar>> V_overlay_rev(3);
    for (int j = 0; j < 3; j++) {
        V_overlay_rev[j] = reverse_overlay_mesh.interpolate_along_o_bc(
            reverse_mesh.opp,
            reverse_mesh.to,
            V_reindex[j]);
    }

    // Reindex vertices
    V_overlay.resize(V_overlay_rev[0].size(), 3);
    for (size_t j = 0; j < 3; j++) {
        for (size_t i = 0; i < V_overlay_rev[j].size(); i++) {
            V_overlay(i, j) = double(V_overlay_rev[j][v_map[i]]);
        }
    }
}

// Utility function to build an overlay mesh from a VF mesh
template <typename OverlayScalar>
OverlayMesh<OverlayScalar> build_overlay_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Theta_hat)
{
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<OverlayScalar> m =
        FV_to_double<OverlayScalar>(
            V,
            F,
            uv,
            F_uv,
            convert_vector_type<Scalar, OverlayScalar>(Theta_hat),
            vtx_reindex,
            indep_vtx,
            dep_vtx,
            v_rep,
            bnd_loops);
    return OverlayMesh<OverlayScalar>(m);
}

template <typename OverlayScalar>
InterpolationMesh<OverlayScalar>::InterpolationMesh()
    : m_overlay_mesh(OverlayMesh<OverlayScalar>(Mesh<OverlayScalar>()))
{
    // Mark the mesh as a invalid Euclidean mesh
    m_is_hyperbolic = false;
    m_is_valid = false;
}

template <typename OverlayScalar>
InterpolationMesh<OverlayScalar>::InterpolationMesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Theta_hat)
    : m_overlay_mesh(build_overlay_mesh<OverlayScalar>(V, F, uv, F_uv, Theta_hat))
{
    // Build the (trivial) conformal scale factors and scaling matrix
    Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();
    m_scale_factors.setZero(mc.n_ind_vertices());

    // Mark the mesh as a valid Euclidean mesh
    m_is_hyperbolic = false;
    m_is_valid = true;
}

template <typename OverlayScalar>
InterpolationMesh<OverlayScalar>::InterpolationMesh(
    const Mesh<Scalar>& mesh,
    const VectorX& scale_factors,
    bool is_hyperbolic)
    : m_overlay_mesh(change_mesh_type<Scalar, OverlayScalar>(mesh))
    , m_scale_factors(convert_vector_type<Scalar, OverlayScalar>(scale_factors))
{
    // Mark the mesh as a valid Hyperbolic mesh
    m_is_hyperbolic = is_hyperbolic;
    m_is_valid = true;
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::add_points(
    const std::vector<int>& pt_fids,
    const std::vector<Eigen::Matrix<OverlayScalar, 3, 1>>& pt_bcs)
{
    m_overlay_mesh.init_pts(pt_fids, pt_bcs);
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::get_points(
    std::vector<int>& pt_fids,
    std::vector<Eigen::Matrix<OverlayScalar, 3, 1>>& pt_bcs)
{
    Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();

    // Map to scaled triangles if the surface is hyperbolic
    if (is_hyperbolic()) {
        equilateral_to_scaled(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, m_scale_factors);
    }

    // Extract points
    int cnt = 0;
    for (auto pt : mc.pts) {
        pt_fids[cnt] = pt.f_id;
        pt_bcs[cnt] = pt.bc;
        cnt++;
    }

    // Undo scaling
    if (is_hyperbolic()) {
        scaled_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, m_scale_factors);
    }
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::convert_to_delaunay_hyperbolic_surface(std::vector<int>& flip_sequence)
{
    spdlog::trace("Converting to Delaunay Hyperbolic surface");
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Make the mesh delaunay
    flip_sequence.clear();
    make_delaunay(flip_sequence);

    // Convert original metric barycentric coordinates to reference triangle if the
    // mesh is currently Euclidean
    if (is_euclidean()) {
        Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();
        m_overlay_mesh.bc_original_to_eq(mc.n, mc.to, mc.l);
        original_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.l);
    }

    // Mark the surface as hyperbolic
    m_is_hyperbolic = true;
    m_is_valid = is_valid_interpolation_mesh();
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::convert_to_delaunay_euclidean_surface(std::vector<int>& flip_sequence)
{
    spdlog::trace("Converting to Delaunay Euclidean surface");
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Make the mesh delaunay
    flip_sequence.clear();
    make_delaunay(flip_sequence);

    // Convert reference triangle barycentric coordinates to the metric if the mesh
    // is currently Hyperbolic
    if (is_hyperbolic()) {
        Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();

        // Remap vertices to representatives and reparametrize the surface
        std::vector<int> to_rep(mc.n_halfedges());
        for (int h = 0; h < mc.n_halfedges(); h++) {
            to_rep[h] = mc.v_rep[mc.to[h]];
        }
        m_overlay_mesh.bc_eq_to_scaled(mc.n, to_rep, mc.l, m_scale_factors);
        equilateral_to_scaled(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, m_scale_factors);
    }

    // Mark the surface as Euclidean
    m_is_hyperbolic = false;
    // m_is_valid = is_valid_interpolation_mesh(); TODO Euclidean case
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::convert_to_hyperbolic_surface(
    std::vector<int>& euclidean_flip_sequence,
    std::vector<int>& hyperbolic_flip_sequence)
{
    spdlog::trace("Converting to Hyperbolic surface");
    euclidean_flip_sequence.clear();
    hyperbolic_flip_sequence.clear();
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Check mesh is trivial
    if (!is_trivial_symmetric_mesh()) {
        spdlog::error("Input mesh must have a trivial symmetry");
        return;
    }

    // Do nothing if the mesh is already hyperbolic
    if (is_hyperbolic()) {
        spdlog::warn("Mesh is already hyperbolic");
        return;
    }

    // Record initial connectivity information for validation and copying
    std::vector<int> initial_n = m_overlay_mesh.cmesh().n;
    std::vector<int> initial_opp = m_overlay_mesh.cmesh().opp;
    std::vector<char> initial_type = m_overlay_mesh.cmesh().type;
    std::vector<int> initial_R = m_overlay_mesh.cmesh().R;

    // Make the mesh delaunay (with Euclidean flips)
    spdlog::trace("Making mesh Delaunay");
    make_delaunay(euclidean_flip_sequence);
    // check_bc_values();

    // Convert original metric barycentric coordinates to reference triangle
    spdlog::trace("Remapping interpolated points");
    Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();
    m_overlay_mesh.bc_original_to_eq(mc.n, mc.to, mc.l);
    original_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.l);
    // check_bc_values();

    // Mark the surface as hyperbolic
    // WARNING: Must be done before undoing flips
    spdlog::trace("Marking the surface as hyperbolic");
    m_is_hyperbolic = true;
    m_is_valid = is_valid_interpolation_mesh();

    // Undo the flips with hyperbolic flips
    spdlog::trace("Undoing Euclidean flips with Ptolemy flips");
    size_t num_flips = euclidean_flip_sequence.size();
    hyperbolic_flip_sequence.reserve(num_flips);
    for (auto iter = euclidean_flip_sequence.rbegin(); iter != euclidean_flip_sequence.rend();
         ++iter) {
        int halfedge_index = *iter;
        flip_clockwise(halfedge_index);
        hyperbolic_flip_sequence.push_back(halfedge_index);
        hyperbolic_flip_sequence.push_back(halfedge_index);
        hyperbolic_flip_sequence.push_back(halfedge_index);
    }
    m_overlay_mesh.garbage_collection();
    SPDLOG_TRACE("Euclidean flip sequence is {}", formatted_vector(euclidean_flip_sequence));
    SPDLOG_TRACE("Hyperbolic flip sequence is {}", formatted_vector(hyperbolic_flip_sequence));
    // check_bc_values();

    // Check that the initial and final next arrays agree
    for (int h = 0; h < num_mesh_halfedges(); ++h) {
        if (mc.n[h] != initial_n[h]) {
            spdlog::error(
                "Halfedge {} has next {} in the original connectivity and {} in the flipped "
                "connectivity",
                h,
                initial_n[h],
                mc.n[h]);
        }

        if (mc.opp[h] != initial_opp[h]) {
            spdlog::error(
                "Halfedge {} has opp {} in the original connectivity and {} in the flipped "
                "connectivity",
                h,
                initial_opp[h],
                mc.opp[h]);
        }
    }

    // Replicate the original type and reflection information
    // WARNING: This is not ideal, but the flipping bookkeeping is nontrivial
    mc.type = initial_type;
    mc.R = initial_R;

    // Check final validity
    m_is_valid = is_valid_interpolation_mesh();

    // Check output mesh is trivial
    if (!is_trivial_symmetric_mesh()) {
        spdlog::error("Output mesh must have a trivial symmetry");
        return;
    }

    // check_bc_values();
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::force_convert_to_euclidean_surface()
{
    spdlog::trace("Forcing conversion to Euclidean surface");
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Convert reference triangle barycentric coordinates to the metric if the mesh
    // is currently Hyperbolic
    if (is_hyperbolic()) {
        m_overlay_mesh.garbage_collection();
        Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();

        // Check scale factor size
        if (m_scale_factors.size() != mc.n_ind_vertices()) {
            spdlog::error("Invalid scale factor array for mesh");
            return;
        }

        // Remap vertices to representatives and reparametrize the surface
        std::vector<int> to_rep(mc.n_halfedges());
        for (int h = 0; h < mc.n_halfedges(); h++) {
            to_rep[h] = mc.v_rep[mc.to[h]];
        }
        m_overlay_mesh.bc_eq_to_scaled(mc.n, to_rep, mc.l, m_scale_factors);
        equilateral_to_scaled(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.to, mc.l, m_scale_factors);
    }

    // Mark the surface as Euclidean
    m_is_hyperbolic = false;
    // m_is_valid = is_valid_interpolation_mesh(); TODO Euclidean case
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::force_convert_to_hyperbolic_surface()
{
    spdlog::trace("Forcing conversion to Hyperbolic surface");
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Convert reference triangle barycentric coordinates to the metric if the mesh
    // is currently Hyperbolic
    if (is_euclidean()) {
        Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();
        m_overlay_mesh.bc_original_to_eq(mc.n, mc.to, mc.l);
        original_to_equilateral(mc.pts, mc.pt_in_f, mc.n, mc.h, mc.l);
    }

    // Mark the surface as Euclidean
    m_is_hyperbolic = true;
    m_is_valid = is_valid_interpolation_mesh();
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::change_hyperbolic_surface_metric(
    const VectorX& halfedge_metric_coords,
    const VectorX& scale_factors,
    const VectorX& halfedge_translations)
{
    spdlog::debug("Changing the surface metric");
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Check input validity
    if (!are_valid_halfedge_lengths(halfedge_metric_coords)) {
        spdlog::error("Invalid halfedge lengths used");
        return;
    }
    if (!are_valid_halfedge_translations(halfedge_translations)) {
        spdlog::error("Invalid halfedge translations used");
        return;
    }

    // Change lengths to target values
    Mesh<OverlayScalar>& mc = m_overlay_mesh.cmesh();
    spdlog::trace("Changing mesh edge lengths");
    for (int h = 0; h < halfedge_metric_coords.size(); ++h) {
        mc.l[h] = exp(halfedge_metric_coords[h] / 2.0);
    }

    // Update scale factors
    m_scale_factors = convert_vector_type<Scalar, OverlayScalar>(scale_factors);

    // Translate barycentric coordinates on edges to rectify the shear
    spdlog::trace("Reparametrizing interpolation points");
    bc_reparametrize_eq(m_overlay_mesh, halfedge_translations);
    reparametrize_equilateral(mc.pts, mc.n, mc.h, halfedge_translations);

    // Check validity
    m_is_valid = is_valid_interpolation_mesh();
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::make_delaunay(std::vector<int>& flip_sequence)
{
    spdlog::trace("Making mesh Delaunay");
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Make overlay mesh Delaunay
    bool use_ptolemy_flip = is_hyperbolic();
    if (use_ptolemy_flip) {
        spdlog::trace("Making mesh Delaunay with Ptolemy flips");
    } else {
        spdlog::trace("Making mesh Delaunay with Euclidean flips");
    }
    DelaunayStats del_stats;
    SolveStats<OverlayScalar> solve_stats;
    ConformalIdealDelaunay<OverlayScalar>::MakeDelaunay(
        m_overlay_mesh,
        m_scale_factors,
        del_stats,
        solve_stats,
        use_ptolemy_flip);
    m_overlay_mesh.garbage_collection();
    // FIXME
    if (!overlay_has_all_original_halfedges(m_overlay_mesh)) {
        spdlog::error("Overlay mesh is missing an original edge");
    }

    // Get flip sequence
    flip_sequence = del_stats.flip_seq;

    // Check validity
    if (is_hyperbolic()) {
        m_is_valid = is_valid_interpolation_mesh();
    }
}


template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::flip_ccw(int flip_index)
{
    // Get halfedge index from flip index
    int halfedge_index = get_flip_halfedge_index(flip_index);
    if (!is_valid_halfedge_index(halfedge_index)) {
        spdlog::error("Invalid halfedge index {} for flip", halfedge_index);
        return;
    }
    bool use_ptolemy_flip = is_hyperbolic();
    m_overlay_mesh.flip_ccw(halfedge_index, use_ptolemy_flip);
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::flip_clockwise(int flip_index)
{
    // Equivalent to flipping clockwise thrice
    for (size_t i = 0; i < 3; ++i) {
        flip_ccw(flip_index);
    }
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::follow_flip_sequence(const std::vector<int>& flip_sequence)
{
    spdlog::trace("Following flip sequence of {} flips", flip_sequence.size());
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Flip edges counterclockwise in order
    for (auto iter = flip_sequence.begin(); iter != flip_sequence.end(); ++iter) {
        int halfedge_index = *iter;
        flip_ccw(halfedge_index);
    }
    m_overlay_mesh.garbage_collection();

    // Check validity
    if (is_hyperbolic()) {
        m_is_valid = is_valid_interpolation_mesh();
    }
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::reverse_flip_sequence(const std::vector<int>& flip_sequence)
{
    spdlog::trace("Reversing flip sequence of {} flips", flip_sequence.size());
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return;
    }

    // Flip edges clockwise in the reverse order
    for (auto iter = flip_sequence.rbegin(); iter != flip_sequence.rend(); ++iter) {
        int halfedge_index = *iter;
        flip_clockwise(halfedge_index);
    }
    m_overlay_mesh.garbage_collection();

    // Check validity
    if (is_hyperbolic()) {
        m_is_valid = is_valid_interpolation_mesh();
    }
}

template <typename OverlayScalar>
OverlayMesh<OverlayScalar> InterpolationMesh<OverlayScalar>::get_tufted_overlay_mesh(
    int num_vertices,
    const std::vector<int>& indep_vtx,
    const std::vector<int>& dep_vtx,
    const std::vector<int>& bnd_loops) const
{
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
    }

    OverlayMesh<OverlayScalar> tufted_overlay_mesh = m_overlay_mesh;
    if (bnd_loops.size() != 0) {
        auto mc = tufted_overlay_mesh.cmesh();
        create_tufted_cover(mc.type, mc.R, indep_vtx, dep_vtx, mc.v_rep, mc.out, mc.to);
        mc.v_rep = range(0, num_vertices);
    }
    return tufted_overlay_mesh;
}

template <typename OverlayScalar>
VectorX InterpolationMesh<OverlayScalar>::get_halfedge_metric_coordinates()
{
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return VectorX();
    }

    auto cmesh = m_overlay_mesh.cmesh();
    int num_halfedges = cmesh.n_halfedges();
    VectorX halfedge_metric_coords(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        halfedge_metric_coords[h] = 2.0 * (Scalar)(log(cmesh.l[h]));
    }

    return halfedge_metric_coords;
}

template <typename OverlayScalar>
VectorX InterpolationMesh<OverlayScalar>::get_metric_coordinates()
{
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return VectorX();
    }

    auto cmesh = m_overlay_mesh.cmesh();

    // Build edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(cmesh.opp, he2e, e2he);

    // Restrict coordinates
    int num_edges = e2he.size();
    VectorX metric_coords(num_edges);
    for (int e = 0; e < num_edges; ++e) {
        int h = e2he[e];
        metric_coords[e] = 2.0 * (Scalar)(log(cmesh.l[h]));
    }

    return metric_coords;
}

template <typename OverlayScalar>
VectorX InterpolationMesh<OverlayScalar>::get_reduced_metric_coordinates()
{
    if (!m_is_valid) {
        spdlog::error("Invalid interpolation mesh");
        return VectorX();
    }

    auto cmesh = m_overlay_mesh.cmesh();

    // Build edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(cmesh.opp, he2e, e2he);

    // Get reflection projection and embedding
    std::vector<int> proj;
    std::vector<int> embed;
    build_refl_proj(cmesh, he2e, e2he, proj, embed);

    // Restrict coordinates
    int num_reduced_edges = embed.size();
    VectorX reduced_metric_coords(num_reduced_edges);
    for (int E = 0; E < num_reduced_edges; ++E) {
        int e = embed[E];
        int h = e2he[e];
        reduced_metric_coords[e] = 2.0 * (Scalar)(log(cmesh.l[h]));
    }

    return reduced_metric_coords;
}

template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::is_euclidean() const
{
    return !(is_hyperbolic());
}

template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::is_hyperbolic() const
{
    return m_is_hyperbolic;
}

template <typename OverlayScalar>
int InterpolationMesh<OverlayScalar>::num_mesh_halfedges() const
{
    return m_overlay_mesh._m.n_halfedges();
}

// Remove sign convention used to distinguish Euclidean and Ptolemy flips
template <typename OverlayScalar>
int InterpolationMesh<OverlayScalar>::get_flip_halfedge_index(int flip_index) const
{
    if (flip_index >= 0) {
        return flip_index;
    } else {
        return -flip_index - 1;
    }
}

template <typename OverlayScalar>
void InterpolationMesh<OverlayScalar>::check_bc_values()
{
    for (size_t h = 0; h < m_overlay_mesh.seg_bcs.size(); h++) {
        if (m_overlay_mesh.edge_type[m_overlay_mesh.e(h)] == ORIGINAL_EDGE) continue;
        if (float_equal(m_overlay_mesh.seg_bcs[h][0], OverlayScalar(0.))) continue;
        if (float_equal(m_overlay_mesh.seg_bcs[h][1], OverlayScalar(0.))) continue;
        spdlog::trace(
            "Boundary conditions are {} and {}",
            m_overlay_mesh.seg_bcs[h][0],
            m_overlay_mesh.seg_bcs[h][1]);
    }
}


template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::is_valid_interpolation_mesh()
{
    spdlog::trace("Checking validity");
    // TODO Add separate variable for halfedge and edge coordinates
    return true;

    Mesh<OverlayScalar> mesh = m_overlay_mesh.cmesh();

    // TODO The below only make sense for edge coordinates (not halfedge coordinates)

    // Check opposite halfedges are consistent
    if (!check_length_consistency(mesh)) {
        return false;
    }

    // Check bc alignment of hyperbolic meshes
    if (is_hyperbolic()) {
        for (int h = 0; h < mesh.n_halfedges(); ++h) {
            m_overlay_mesh.check_bc_alignment(&mesh, h);
        }
    }

    return true;
}

template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::is_trivial_symmetric_mesh()
{
    spdlog::trace("Checking for triviality of the symmetry");
    Mesh<OverlayScalar> mc = m_overlay_mesh.cmesh();
    for (int h = 0; h < num_mesh_halfedges(); ++h) {
        if (mc.type[h] > 2) {
            spdlog::error(
                "Halfedge {} is type {} and crosses the symmetry line",
                h,
                static_cast<int>(mc.type[h]));
            return false;
        }
    }

    return true;
}

template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::is_valid_halfedge_index(int halfedge_index)
{
    return ((halfedge_index >= 0) && (halfedge_index < m_overlay_mesh.cmesh().n_halfedges()));
}

// Check if a vector describes valid halfedge lengths
// TODO
template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::are_valid_halfedge_lengths(const VectorX& halfedge_lengths)
{
    auto cmesh = m_overlay_mesh.cmesh();

    // Check correct number of halfedges
    if (halfedge_lengths.size() != num_mesh_halfedges()) {
        spdlog::error(
            "Incorrect number {} of halfedge lengths for mesh with {} halfedges",
            halfedge_lengths.size(),
            num_mesh_halfedges());
        return false;
    }

    // Check that edge pairing and symmetry structures are respected
    for (int h = 0; h < num_mesh_halfedges(); ++h) {
        Scalar l = halfedge_lengths[h];

        // Check symmetry structure
        if ((cmesh.type[h] == 1) || (cmesh.type[h] == 2)) {
            int Rh = cmesh.R[h];
            Scalar Rl = halfedge_lengths[Rh];
            if (!float_equal(l, Rl)) {
                spdlog::error(
                    "Inconsistent reflection pair halfedges {} and {} with lengths {} and {}",
                    h,
                    Rh,
                    l,
                    Rl);
                return false;
            }
        }
    }

    return true;
}

// Check if a vector describes valid halfedge translations
// TODO
template <typename OverlayScalar>
bool InterpolationMesh<OverlayScalar>::are_valid_halfedge_translations(const VectorX& halfedge_translations) const
{
    if (halfedge_translations.size() != num_mesh_halfedges()) {
        spdlog::error(
            "Incorrect number {} of halfedge translations for mesh with {} halfedges",
            halfedge_translations.size(),
            num_mesh_halfedges());
        return false;
    }

    return true;
}

template <typename OverlayScalar>
bool overlay_has_all_original_halfedges(OverlayMesh<OverlayScalar>& mo)
{
    std::vector<bool> has_original_halfedge(mo.cmesh().n_halfedges(), false);
    for (int hi = 0; hi < mo.n_halfedges(); ++hi) {
        if (mo.n[hi] == -1) continue; // Deleted halfedge
        if (mo.edge_type[hi] == ORIGINAL_EDGE) {
            has_original_halfedge[mo.origin_of_origin[hi]] = true;
        } else if (mo.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE) {
            has_original_halfedge[mo.origin_of_origin[hi]] = true;
        }
    }
    int num_missing_original_halfedges =
        std::count(has_original_halfedge.begin(), has_original_halfedge.end(), false);

    return (num_missing_original_halfedges == 0);
}

template class InterpolationMesh<Scalar>;
template bool overlay_has_all_original_halfedges<Scalar>(OverlayMesh<Scalar>& mo);
template void interpolate_penner_coordinates<Scalar>(
    const Mesh<Scalar>& mesh,
    const VectorX& halfedge_metric_coords,
    const VectorX& scale_factors,
    InterpolationMesh<Scalar>& interpolation_mesh,
    InterpolationMesh<Scalar>& reverse_interpolation_mesh);
template void interpolate_vertex_positions<Scalar>(
    const Eigen::MatrixXd& V,
    const std::vector<int> vtx_reindex,
    const InterpolationMesh<Scalar>& interpolation_mesh,
    const InterpolationMesh<Scalar>& reverse_interpolation_mesh,
    Eigen::MatrixXd& V_overlay);

#ifdef WITH_MPFR
#ifndef MULTIPRECISION
template class InterpolationMesh<mpfr::mpreal>;
template bool overlay_has_all_original_halfedges<mpfr::mpreal>(OverlayMesh<mpfr::mpreal>& mo);
template void interpolate_penner_coordinates<mpfr::mpreal>(
    const Mesh<Scalar>& mesh,
    const VectorX& halfedge_metric_coords,
    const VectorX& scale_factors,
    InterpolationMesh<mpfr::mpreal>& interpolation_mesh,
    InterpolationMesh<mpfr::mpreal>& reverse_interpolation_mesh);
template void interpolate_vertex_positions<mpfr::mpreal>(
    const Eigen::MatrixXd& V,
    const std::vector<int> vtx_reindex,
    const InterpolationMesh<mpfr::mpreal>& interpolation_mesh,
    const InterpolationMesh<mpfr::mpreal>& reverse_interpolation_mesh,
    Eigen::MatrixXd& V_overlay);
#endif
#endif

} // namespace Optimization
} // namespace Penner
