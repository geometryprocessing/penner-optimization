#include "feature/surgery/cut_mesh_layout.h"

#include "util/vf_mesh.h"

#include "optimization/parameterization/interpolation.h"
#include "optimization/parameterization/layout.h"

#include "feature/feature/error.h"
#include "feature/core/component_mesh.h"

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Layout.hh"

#include <igl/facet_components.h>

namespace Penner {
namespace Feature {

template <typename OverlayScalar>
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
parameterize_cut_mesh(
    const MarkedPennerConeMetric& embedding_metric,
    const MarkedPennerConeMetric& original_metric,
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::MatrixXd& V_cut,
    const std::vector<int>& vtx_reindex,
    const std::vector<int>& face_reindex,
    bool use_uniform_bc,
    std::string output_dir)
{
#ifdef WITH_MPFR
    mpfr::mpreal::set_default_prec(100);
    mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
    mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

    // Separate mesh into components
    ComponentMesh component_embedding(embedding_metric);
    ComponentMesh component_original(original_metric);
    ComponentMesh component_mesh(marked_metric);
    const auto& embedding_components = component_embedding.get_mesh_components();
    //const auto& init_components = component_original.get_mesh_components();
    const auto& mesh_components = component_mesh.get_mesh_components();
    const auto& he_maps = component_mesh.get_halfedge_maps();
    spdlog::info("Extracted {} components", mesh_components.size());

    // Parameterize each component
    int num_components = embedding_components.size();
    std::vector<Eigen::MatrixXd> V_components, uv_components;
    std::vector<Eigen::MatrixXi> F_components, FT_components;
    std::vector<std::vector<int>> fn_to_f_components;
    std::vector<std::vector<std::pair<int, int>>> endpoint_components;
    for (int i = 0; i < num_components; ++i) {
        const auto& embedding_component = embedding_components[i];
        //const auto& init_component = init_components[i];
        const auto& mesh_component = mesh_components[i];
        const auto& he_index = he_maps[i];

        // get initial and final metric coordinates
        VectorX metric_init = Holonomy::generate_penner_coordinates(embedding_component);
        VectorX metric_coords(mesh_component.n_halfedges());
        for (int i = 0; i < mesh_component.n_halfedges(); ++i) {
        //    metric_init[i] = 2. * log(init_component.l[i]);
            metric_coords[i] = 2. * log(mesh_component.l[i]);
        }
        Optimization::PennerConeMetric component_init(mesh_component, metric_init);
        Optimization::PennerConeMetric component_metric(mesh_component, metric_coords);

        // Get component vertices
        int n_v_component = component_metric.n_vertices();
        int n_ind_v_component = component_metric.n_ind_vertices();
        Eigen::MatrixXd V_component(n_ind_v_component, 3);
        for (int i = 0; i < n_v_component; ++i) {
            // get vertex in component
            int hij = component_metric.out[i];
            int vi = component_metric.v_rep[i];

            // get vertex in total mesh
            int Hij = he_index[hij];
            int Vi = marked_metric.v_rep[marked_metric.to[marked_metric.opp[Hij]]];

            // build v_rep and vertex positions
            V_component.row(vi) = V_cut.row(vtx_reindex[Vi]);
        }

        // undouble Th_hat
        std::vector<int> vtx_reindex_component;
        arange(n_ind_v_component, vtx_reindex_component);

        // Generate overlay
        std::vector<bool> is_cut = {};
        std::string component_layout_path = join_path(output_dir, "component_" + std::to_string(i) + ".obj");
        auto vf_res = Optimization::generate_VF_mesh_from_halfedge_metric<OverlayScalar>(
            V_component,
            embedding_component,
            vtx_reindex_component,
            component_init,
            component_metric.get_metric_coordinates(),
            is_cut,
            false,
            use_uniform_bc,
            "");
        OverlayMesh<OverlayScalar> m_o = std::get<0>(vf_res);
        Eigen::MatrixXd V_o = std::get<1>(vf_res);
        Eigen::MatrixXi F_o = std::get<2>(vf_res);
        Eigen::MatrixXd uv_o = std::get<3>(vf_res);
        Eigen::MatrixXi FT_o = std::get<4>(vf_res);
        std::vector<bool> is_cut_h_final = std::get<5>(vf_res);
        std::vector<bool> is_cut_o = std::get<6>(vf_res);
        std::vector<int> fn_to_f_o = std::get<7>(vf_res);
        std::vector<std::pair<int, int>> endpoints_o = std::get<8>(vf_res);

        // Add components to lists
        V_components.push_back(V_o);
        F_components.push_back(F_o);
        uv_components.push_back(uv_o);
        FT_components.push_back(FT_o);
        fn_to_f_components.push_back(fn_to_f_o);
        endpoint_components.push_back(endpoints_o);
    }

    // Combine components into single mesh
    auto [V_r, F_r, fn_to_f, endpoints] = component_mesh.combine_refined_components(
        face_reindex,
        V_components,
        F_components,
        fn_to_f_components,
        endpoint_components);
    auto [uv_r, FT_r] = combine_mesh_components(uv_components, FT_components);

    auto [V_rr, F_rr] = reindex_mesh(V_r, F_r, vtx_reindex);
    endpoints = reindex_endpoints(endpoints, vtx_reindex);

    return std::make_tuple(V_rr, F_rr, uv_r, FT_r, fn_to_f, endpoints);
}

Eigen::MatrixXd align_to_hard_features(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXi& F_is_hard_feature)
{
    // Get components and boundary
    Eigen::VectorXi components;
    igl::facet_components(FT, components);
    int num_components = components.maxCoeff() + 1;

    // for each component, find feature edge to align to
    int num_faces = FT.rows();
    std::vector<std::pair<int, int>> component_corners(num_components);
    std::vector<double> edge_lengths(num_components, -1.);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        // check if component already seen
        int ci = components(fijk);
        for (int i = 0; i < 3; ++i) {
            if (!F_is_hard_feature(fijk, i)) continue;
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;
            Eigen::Vector2d edge = uv.row(FT(fijk, j)) - uv.row(FT(fijk, k));
            double edge_length = edge.norm();
            if (edge_length > edge_lengths[ci]) {
                component_corners[ci] = {fijk, i};
                edge_lengths[ci] = edge_length;
            }
        }
    }

    // make sure each hard feature edge is aligned
    std::vector<Eigen::Matrix2d> component_rotations(num_components);
    for (int ci = 0; ci < num_components; ++ci) {
        auto [fijk, k] = component_corners[ci];
        int i = (k + 1) % 3;
        int j = (i + 1) % 3;
        int vi = FT(fijk, i);
        int vj = FT(fijk, j);
        Eigen::Vector2d dij = uv.row(vj) - uv.row(vi);
        dij.normalize();
        double alignment_error = compute_edge_alignment(dij);
        if (float_equal<Scalar>(alignment_error, 0.)) {
            component_rotations[ci] = Eigen::Matrix2d::Identity(2, 2);
        } else {
            component_rotations[ci](0, 0) = dij[0];
            component_rotations[ci](0, 1) = dij[1];
            component_rotations[ci](1, 0) = -dij[1];
            component_rotations[ci](1, 1) = dij[0];
        }
    }

    // get the vertex to face map
    int num_vertices = uv.rows();
    Eigen::VectorXi vertex_component(num_vertices);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            int vi = FT(fijk, i);
            vertex_component[vi] = components[fijk];
        }
    }

    // align uv
    Eigen::MatrixXd aligned_uv = uv;
    for (int vi = 0; vi < num_vertices; ++vi) {
        int ci = vertex_component[vi];
        Eigen::Vector2d uvi = uv.row(vi);
        aligned_uv.row(vi) = component_rotations[ci] * uvi;
    }

    // check alignment
    for (int ci = 0; ci < num_components; ++ci) {
        auto [fijk, k] = component_corners[ci];
        int i = (k + 1) % 3;
        int j = (i + 1) % 3;
        int vi = FT(fijk, i);
        int vj = FT(fijk, j);
        Eigen::Vector2d dij = aligned_uv.row(vj) - aligned_uv.row(vi);
        dij.normalize();
        double alignment_error = compute_edge_alignment(dij);
        if (!float_equal<Scalar>(alignment_error, 0.)) {
            spdlog::error("component {} not aligned with error {}", ci, alignment_error);
            spdlog::error("alignment edge has length {}", edge_lengths[ci]);
        }
    }

    return aligned_uv;
}

// TODO: Redundnant; would be better to unify with conformal code
template <typename OverlayScalar>
std::tuple<
        std::vector<OverlayScalar>,                // u_out
        std::vector<OverlayScalar>,                // v_out
        std::vector<std::vector<int>>>      // FT_out
get_FV_FTVT(OverlayMesh<OverlayScalar> &mo,
            std::vector<bool> &is_cut_o,
            std::vector<int> &vtx_reindex,
            const Eigen::MatrixXi& F,
            const std::vector<OverlayScalar> u_o,
            const std::vector<OverlayScalar> v_o)
{
    // get h_group and to_map
    std::vector<int> f_labels = get_overlay_face_labels(mo);
    
    int origin_size = mo.cmesh().out.size();
    std::vector<int> h_group(mo.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    std::vector<int> to_group(origin_size, -1);
    std::vector<int> which_to_group(mo.out.size(), -1);
    for (size_t i = 0; i < to_group.size(); i++)
    {
        to_group[i] = i;
        which_to_group[i] = i;
    }
    int num_halfedges = mo.n.size();
    for (int i = 0; i < num_halfedges; i++)
    {
        if (h_group[i] != -1 || f_labels[mo.f[i]] == 2) continue;
        if (which_to_group[mo.to[i]] == -1)
        {
            which_to_group[mo.to[i]] = to_group.size();
            to_group.push_back(mo.to[i]);
        }
        if (mo.to[i] < origin_size && to_map[mo.to[i]] == -1)
        {
            h_group[i] = mo.to[i];
            to_map[mo.to[i]] = i;
        }    
        else
        {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = mo.n[i];
        while (is_cut_o[cur] == false && mo.opp[cur] != i)
        {
            cur = mo.opp[cur];
            h_group[cur] = h_group[i];
            cur = mo.n[cur];
        }
        cur = mo.opp[i];
        while (is_cut_o[cur] == false && mo.prev[cur] != i)
        {
            cur = mo.prev[cur];
            h_group[cur] = h_group[i];
            cur = mo.opp[cur];
        }
    }

    std::vector<OverlayScalar> u_o_out(to_map.size());
    std::vector<OverlayScalar> v_o_out(to_map.size());
    for (size_t i = 0; i < to_map.size(); i++)
    {
        u_o_out[i] = u_o[to_map[i]];
        v_o_out[i] = v_o[to_map[i]];
    }

    std::vector<std::vector<int>> FT_out;
    for (size_t i = 0; i < mo.h.size(); i++)
    {
        if (f_labels[i] == 2) continue;
        int h0 = mo.h[i];
        while (vtx_reindex[mo.to[h0]] != F(i, 0))
        {
            h0 = mo.n[h0];
        }
        int hc = mo.n[h0];
        std::vector<int> hf;
        hf.push_back(h0);
        while (hc != h0)
        {
            hf.push_back(hc);
            hc = mo.n[hc];
        }
        for (size_t j = 0; j < hf.size() - 2; j++)
        {
            FT_out.push_back(std::vector<int>{h_group[h0], h_group[hf[j + 1]], h_group[hf[j + 2]]});
        }
    }

    return std::make_tuple(u_o_out, v_o_out, FT_out);

}

// get error of lengths across halfedges
template <typename OverlayScalar>
OverlayScalar compute_length_error(const Mesh<OverlayScalar>& m)
{
    int num_halfedges = m.n_halfedges();
    OverlayScalar max_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = m.opp[hij];
        if (hij < hji) continue;
        max_error = max(max_error, abs(m.l[hij] - m.l[hji]));
    }

    return max_error;
}

// get average of lengths across halfedges
template <typename OverlayScalar>
void average_length(Mesh<OverlayScalar>& m)
{
    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = m.opp[hij];
        if (hij < hji) continue;
        m.l[hij] = m.l[hji] = (m.l[hij] + m.l[hji]) / 2.;
    }
}

template <typename OverlayScalar>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_connected_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv
) {
#ifdef WITH_MPFR
    mpfr::mpreal::set_default_prec(100);
    mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
    mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

    VectorX cone_angles = compute_cone_angles(V, F, uv, F_uv);
    std::vector<Scalar> Th_hat;
    convert_eigen_to_std_vector(cone_angles, Th_hat);
    spdlog::info("found {} cone angles", Th_hat.size());

    std::vector<int> vtx_reindex;
    std::vector<int> indep_vtx;
    std::vector<int> dep_vtx;
    std::vector<int> v_rep;
    std::vector<int> bnd_loops;
    Mesh<Scalar> m_scalar = FV_to_double<Scalar>(V, F, uv, F_uv, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);
    Mesh<OverlayScalar> m = FV_to_double<OverlayScalar>(V, F, uv, F_uv, convert_vector_type<Scalar, OverlayScalar>(Th_hat), vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);
    //average_length(m);
    OverlayMesh<OverlayScalar> m_o(m);
    Optimization::make_tufted_overlay(m_o);
    spdlog::info("generated overlay mesh with {} vertices", m_o.n_vertices());
    spdlog::info("length error is {}", compute_length_error(m));

    // Get original overlay face labels
    auto f_labels = get_overlay_face_labels(m_o);

    // get cones and bd
    std::vector<int> cones;
    Scalar flat_angle = 2. * M_PI;
    for (size_t i = 0; i < m.Th_hat.size(); i++) {
        if (abs(m.Th_hat[i] - flat_angle) > 1e-12) {
            cones.push_back(i);
        }
    }
    spdlog::info("found {} cones", cones.size());

    // find most aligned edge
    int num_faces = F_uv.rows();
    Eigen::SparseMatrix<int> vv2h = generate_VV_to_halfedge_map(m_scalar, vtx_reindex);
    int h_start = 0;
    Scalar min_alignment = 1.;
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int k = 0; k < 3; ++k)
        {
            Scalar corner_alignment = compute_corner_alignment(uv, F_uv, {fijk, k});
            if (corner_alignment < min_alignment)
            {
                min_alignment = corner_alignment;
                int i = (k + 1) % 3;
                int j = (k + 2) % 3;
                int vi = F(fijk, i);
                int vj = F(fijk, j);
                h_start = vv2h.coeff(vi, vj) - 1;
            }
        }
    }
    spdlog::info("starting from h {} with alignment {}", h_start, min_alignment);


    // Now directly do layout on triangulated overlay mesh
    std::vector<OverlayScalar> phi(m.n_vertices(), 0.0);
    std::vector<bool> is_cut;
    auto overlay_layout_res = Optimization::compute_layout_components(m_o.cmesh(), phi, is_cut, h_start);
    std::vector<OverlayScalar> u_o = std::get<0>(overlay_layout_res);
    std::vector<OverlayScalar> v_o = std::get<1>(overlay_layout_res);
    std::vector<bool> is_cut_o = std::get<2>(overlay_layout_res);
    spdlog::info("found initial layout");

    // Trim unnecessary branches of the cut graph
    bool do_trim = true;
    if (do_trim) {
        trim_open_branch(m_o, f_labels, cones, is_cut_o);
    }
    spdlog::info("trimmed cut graph");

    auto [u_o_out, v_o_out, FT_out] = get_FV_FTVT<OverlayScalar>(m_o, is_cut_o, vtx_reindex, F, u_o, v_o);


    // Convert vector formats to matrices
    Eigen::MatrixXd  uv_o;
    Eigen::VectorXd u_o_col, v_o_col;
    Eigen::MatrixXi FT_o;
    convert_std_to_eigen_matrix(FT_out, FT_o);
    convert_std_to_eigen_vector(convert_vector_type<OverlayScalar, Scalar>(u_o_out), u_o_col);
    convert_std_to_eigen_vector(convert_vector_type<OverlayScalar, Scalar>(v_o_out), v_o_col);
    uv_o.resize(u_o_col.size(), 2);
    uv_o.col(0) = u_o_col;
    uv_o.col(1) = v_o_col;

    return std::make_tuple(uv_o, FT_o);
}

template std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
parameterize_cut_mesh<Scalar>(
    const MarkedPennerConeMetric& embedding_metric,
    const MarkedPennerConeMetric& original_metric,
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::MatrixXd& V_cut,
    const std::vector<int>& vtx_reindex,
    const std::vector<int>& face_reindex,
    bool use_uniform_bc,
    std::string output_dir);

template
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_connected_parameterization<Scalar>(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

#ifdef WITH_MPFR
#ifndef MULTIPRECISION
template std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
parameterize_cut_mesh<mpfr::mpreal>(
    const MarkedPennerConeMetric& embedding_metric,
    const MarkedPennerConeMetric& original_metric,
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::MatrixXd& V_cut,
    const std::vector<int>& vtx_reindex,
    const std::vector<int>& face_reindex,
    bool use_uniform_bc,
    std::string output_dir);

template
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_connected_parameterization<mpfr::mpreal>(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);
#endif
#endif

} // namespace Feature
} // namespace Penner
