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
#include "penner_optimization_interface.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "embedding.hh"
#include "interpolation.hh"
#include "layout.hh"
#include "projection.hh"
#include "translation.hh"
#include "vector.hh"
#include "viewers.hh"

/// FIXME Do cleaning pass


namespace CurvatureMetric {

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
    InterpolationMesh interpolation_mesh(m, scale_factors, is_hyperbolic);
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

void write_obj_with_uv(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    Eigen::MatrixXd N;
    Eigen::MatrixXi FN;
    igl::writeOBJ(filename, V, F, N, FN, uv, F_uv);
}


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
    consistent_overlay_mesh_to_VL(
        const Eigen::MatrixXi& F,
        const std::vector<Scalar>& Theta_hat,
        OverlayMesh<Scalar>& mo,
        std::vector<Scalar>& u,
        std::vector<std::vector<Scalar>>& V_overlay,
        std::vector<int>& vtx_reindex,
        std::vector<std::pair<int, int>>& endpoints,
        const std::vector<bool>& is_cut_orig,
        const std::vector<bool>& is_cut)
{
    // get cones and bd
    std::vector<int> cones, bd;
    std::vector<bool> is_bd = igl::is_border_vertex(F);
    for (size_t i = 0; i < is_bd.size(); i++) {
        if (is_bd[i]) {
            bd.push_back(i);
        }
    }
    for (size_t i = 0; i < Theta_hat.size(); i++) {
        if ((!is_bd[i]) && abs(Theta_hat[i] - 2 * M_PI) > 1e-15) {
            cones.push_back(i);
        }
    }

    std::vector<int> f_labels = get_overlay_face_labels(mo);

    // reindex cones and bd
    std::vector<int> vtx_reindex_rev(vtx_reindex.size());
    for (size_t i = 0; i < vtx_reindex.size(); i++) {
        vtx_reindex_rev[vtx_reindex[i]] = i;
    }
    for (size_t i = 0; i < cones.size(); i++) {
        cones[i] = vtx_reindex_rev[cones[i]];
    }
    for (size_t i = 0; i < bd.size(); i++) {
        bd[i] = vtx_reindex_rev[bd[i]];
    }

    spdlog::trace("#bd_vt: {}", bd.size());
    spdlog::trace("#cones: {}", cones.size());
    spdlog::trace("vtx reindex size: {}", vtx_reindex.size());
    spdlog::trace("mc.out size: {}", mo.cmesh().out.size());

    // get layout
    auto layout_res = get_consistent_layout(mo, u, cones, is_cut_orig, is_cut);
    auto u_o = std::get<0>(layout_res);
    auto v_o = std::get<1>(layout_res);
    auto is_cut_h = std::get<2>(layout_res);
    auto is_cut_o = std::get<3>(layout_res);

    // get output VF and metric
    auto FVFT_res = get_FV_FTVT(mo, endpoints, is_cut_o, V_overlay, u_o, v_o);
    auto v3d = std::get<0>(FVFT_res);
    auto u_o_out = std::get<1>(FVFT_res);
    auto v_o_out = std::get<2>(FVFT_res);
    auto F_out = std::get<3>(FVFT_res);
    auto FT_out = std::get<4>(FVFT_res);
    auto Fn_to_F = std::get<5>(FVFT_res);
    auto remapped_endpoints = std::get<6>(FVFT_res);

    // v3d_out = v3d^T
    std::vector<std::vector<Scalar>> v3d_out(v3d[0].size());
    for (size_t i = 0; i < v3d[0].size(); i++) {
        v3d_out[i].resize(3);
        for (int j = 0; j < 3; j++) {
            v3d_out[i][j] = v3d[j][i];
        }
    }

    // reindex back
    auto u_o_out_copy = u_o_out;
    auto v_o_out_copy = v_o_out;
    auto v3d_out_copy = v3d_out;
    auto endpoints_out = remapped_endpoints;
    int num_vertices = vtx_reindex.size();
    for (size_t i = 0; i < F_out.size(); i++) {
        for (int j = 0; j < 3; j++) {
            if (F_out[i][j] < num_vertices) {
                F_out[i][j] = vtx_reindex[F_out[i][j]];
            }
            if (FT_out[i][j] < num_vertices) {
                FT_out[i][j] = vtx_reindex[FT_out[i][j]];
            }
        }
    }
    for (size_t i = 0; i < vtx_reindex.size(); i++) {
        u_o_out[vtx_reindex[i]] = u_o_out_copy[i];
        v_o_out[vtx_reindex[i]] = v_o_out_copy[i];
        v3d_out[vtx_reindex[i]] = v3d_out_copy[i];
    }
    for (size_t i = vtx_reindex.size(); i < endpoints_out.size(); i++) {
        int a = vtx_reindex[endpoints_out[i].first];
        int b = vtx_reindex[endpoints_out[i].second];
        endpoints_out[i] = std::make_pair(a, b);
    }

    // Convert vector formats to matrices
    Eigen::MatrixXd V_o, uv_o;
    Eigen::VectorXd u_o_col, v_o_col;
    Eigen::MatrixXi F_o, FT_o;
    convert_std_to_eigen_matrix(v3d_out, V_o);
    convert_std_to_eigen_matrix(F_out, F_o);
    convert_std_to_eigen_matrix(FT_out, FT_o);
    convert_std_to_eigen_vector(u_o_out, u_o_col);
    convert_std_to_eigen_vector(v_o_out, v_o_col);
    uv_o.resize(u_o_col.size(), 2);
    uv_o.col(0) = u_o_col;
    uv_o.col(1) = v_o_col;

    // Check for validity
    if (!check_uv(V_o, F_o, uv_o, FT_o)) {
        spdlog::error("Inconsistent uvs");
    }

    return std::make_tuple(mo, V_o, F_o, uv_o, FT_o, is_cut_h, is_cut_o, Fn_to_F, endpoints_out);
}

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
        std::vector<bool> cut_h,
        bool do_best_fit_scaling)
{
    // Get mesh with vertex reindexing
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m =
        FV_to_double(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    // Get metric target coordinates
    auto cone_metric = initial_cone_metric.set_metric_coordinates(reduced_metric_coords);
    VectorX metric_target = initial_cone_metric.get_metric_coordinates();
    VectorX metric_coords = cone_metric->get_metric_coordinates();

    // Fit conformal scale factors
    VectorX metric_coords_scaled = metric_coords;
    VectorX scale_factors;
    scale_factors.setZero(initial_cone_metric.n_ind_vertices());
    if (do_best_fit_scaling) {
        scale_factors = best_fit_conformal(initial_cone_metric, metric_coords);
        MatrixX B = conformal_scaling_matrix(initial_cone_metric);
        metric_coords_scaled = metric_coords - B * scale_factors;
    }
    VectorX metric_diff = metric_coords_scaled - metric_target;
    SPDLOG_DEBUG(
        "Scale factors in range [{}, {}]",
        scale_factors.minCoeff(),
        scale_factors.maxCoeff());
    SPDLOG_DEBUG(
        "Scaled metric coordinates in range [{}, {}]",
        metric_coords_scaled.minCoeff(),
        metric_coords_scaled.maxCoeff());
    SPDLOG_DEBUG(
        "Differences from target to optimized metric in range [{}, {}]",
        metric_diff.minCoeff(),
        metric_diff.maxCoeff());

    // Compute interpolation overlay mesh
    Eigen::MatrixXd V_overlay;
    InterpolationMesh interpolation_mesh, reverse_interpolation_mesh;
    spdlog::trace("Interpolating penner coordinates");
    interpolate_penner_coordinates(
        m,
        metric_coords_scaled,
        scale_factors,
        interpolation_mesh,
        reverse_interpolation_mesh);
    spdlog::trace("Interpolating vertex positions");
    interpolate_vertex_positions(
        V,
        vtx_reindex,
        interpolation_mesh,
        reverse_interpolation_mesh,
        V_overlay);
    OverlayMesh<Scalar> m_o = interpolation_mesh.get_overlay_mesh();
    make_tufted_overlay(m_o, V, F, Th_hat);

    // Get endpoints
    std::vector<std::pair<int, int>> endpoints;
    find_origin_endpoints(m_o, endpoints);

    // Convert overlay mesh to transposed vector format
    std::vector<std::vector<Scalar>> V_overlay_vec(3);
    for (int i = 0; i < 3; ++i) {
        V_overlay_vec[i].resize(V_overlay.rows());
        for (int j = 0; j < V_overlay.rows(); ++j) {
            V_overlay_vec[i][j] = V_overlay(j, i);
        }
    }

    // Get layout topology from original mesh
    std::vector<bool> is_cut = compute_layout_topology(m, cut_h);

    // Convert overlay mesh to VL format
    spdlog::trace("Getting layout");
    std::vector<int> vtx_reindex_mutable = vtx_reindex;
    std::vector<Scalar> u; // (m_o._m.Th_hat.size(), 0.0);
    convert_eigen_to_std_vector(scale_factors, u);
    // auto parametrize_res = overlay_mesh_to_VL<Scalar>(V, F, Th_hat, m_o, u, V_overlay_vec,
    // vtx_reindex_mutable, endpoints, -1); FIXME
    return consistent_overlay_mesh_to_VL(
        F,
        Th_hat,
        m_o,
        u,
        V_overlay_vec,
        vtx_reindex_mutable,
        endpoints,
        is_cut,
        {});
}

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
        std::vector<bool> cut_h)
{
    // Get mesh with vertex reindexing
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m =
        FV_to_double(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    // Get layout topology from mesh
    std::vector<bool> is_cut = compute_layout_topology(m, cut_h);

    // Set metric for layout
		DiscreteMetric discrete_metric(m, reduced_log_edge_lengths);

    // Create trivial overlay mesh
    OverlayMesh<Scalar> m_o(discrete_metric);
    make_tufted_overlay(m_o, V, F, Th_hat);

    // Convert vertices to transposed vector format
    std::vector<std::vector<Scalar>> V_overlay_vec(3);
    for (int i = 0; i < 3; ++i) {
        V_overlay_vec[i].resize(V.rows());
        for (int j = 0; j < V.rows(); ++j) {
            V_overlay_vec[i][j] = V(vtx_reindex[j], i);
        }
    }

    // Get endpoints
    std::vector<std::pair<int, int>> endpoints;
    find_origin_endpoints(m_o, endpoints);

		// Compute layout
    std::vector<Scalar> u_vec(m.n_ind_vertices(), 0.0);
    std::vector<int> vtx_reindex_mutable = vtx_reindex;
    auto layout_res = consistent_overlay_mesh_to_VL(
        F,
        Th_hat,
        m_o,
        u_vec,
        V_overlay_vec,
        vtx_reindex_mutable,
        endpoints,
        is_cut,
        {});

		Eigen::MatrixXd V_l = std::get<1>(layout_res);
		Eigen::MatrixXi F_l = std::get<2>(layout_res);
		Eigen::MatrixXd uv = std::get<3>(layout_res);
		Eigen::MatrixXi FT = std::get<4>(layout_res);

		return std::make_tuple(V_l, F_l, uv, FT, cut_h);
}

} // namespace CurvatureMetric
