#include "resolution/hierarchy/overlay.h"
#include "metric/viewer.h"
#include "holonomy/core/viewer.h"
#include "util/union_find.h"
#include "util/map.h"
#include "util/vf_mesh.h"
#include "holonomy/interface.h"
#include "field/frame_field.h"
#include "parametrization/interpolation.h"
#include "parametrization/refinement.h"
#include "parametrization/parametrize.h"
#include "metric/constraint.h"
#include "resolution/core/triangulation.h"
#include "optimization/metric_optimization/implicit_optimization.h"
//#include <omp.h>

#include <igl/per_vertex_normals.h>
#include <igl/predicates/predicates.h>

#ifdef USE_SUITESPARSE
//#include <Eigen/src/CholmodSupport/CholmodSupport.h>
#include <Eigen/CholmodSupport>
#endif

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

#include <cmath>

namespace Penner {
namespace Resolution {


// --------------
// Core Interface
// --------------

SimplificationOverlayMesh::SimplificationOverlayMesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& mesh)
: HalfedgeMesh(mesh.opp, mesh.n, mesh.to, mesh.f, build_edge_map(mesh.opp))
, V_orig(V)
, vtx_reindex_orig(vtx_reindex)
, orig(mesh)
, simp(out.size(), e2h.size(), f2h.size())
{
}

Mesh<Scalar> SimplificationOverlayMesh::generate_mesh() const
{
    auto [opp_r, n_r, prev_r] = reduce_halfedge_maps();
    auto [face_r, f2h_r] = reduce_face_maps();
    auto [to_r, out_r] = reduce_vertex_maps();
    auto [h_new2old, h_old2new] = get_halfedge_reductions();

    Mesh<Scalar> m;
    m.n = n_r;
    m.opp = opp_r;
    m.f = face_r;
    m.h = f2h_r;
    m.to = to_r;
    m.out = out_r;
    m.l = std::vector<Scalar>(m.n.size(), 1.);
    for (int hij = 0; hij < m.n.size(); ++hij)
    {
        int hji = m.opp[hij];
        if (hij < hji) continue;

        int h = h_new2old[hij];
        m.l[hji] = m.l[hij] = halfedge_length(h);
    }
    m.type = std::vector<char>(m.n.size(), 0);
    m.type_input = std::vector<char>(m.n.size(), 0);
    m.R = std::vector<int>(m.n.size(), 0);
    m.v_rep = arange(m.out.size());
    m.Th_hat = std::vector<Scalar>(m.out.size(), 2 * PI);
    m.fixed_dof = std::vector<bool>(m.out.size(), false);
    m.fixed_dof[0] = true;

    return m;
}

// -----------
// Frame Field
// -----------

void SimplificationOverlayMesh::set_original_field(
    const Eigen::MatrixXd& reference_vector,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXi& period_jump)
{
    reference_field_orig = reference_vector;
    thetas_orig = theta;
    period_jumps_orig = Field::transfer_period_jumps_to_halfedge(
        orig.m,
        vtx_reindex_orig,
        F_orig,
        period_jump
    );
}


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi> 
SimplificationOverlayMesh::refine_frame_field(
    const Eigen::MatrixXi& F_orig,
    const Eigen::MatrixXd& reference_vector,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump)
{
    auto [f_new2old, f_old2new] = get_face_reductions();
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [F, F_halfedge] = generate_overlay_faces();

    // build refined field
    int num_faces = count_faces();
    Eigen::MatrixXd reference_vector_r(num_faces, 3);
    Eigen::VectorXd theta_r(num_faces);
    Eigen::MatrixXd kappa_r(num_faces, 3);
    Eigen::MatrixXi period_jump_r(num_faces, 3);
    for (int fi = 0; fi < num_faces; ++fi)
    {
        int f_orig = orig.face_parent[f_new2old[fi]];
        reference_vector_r.row(fi) = reference_vector.row(f_orig);
        theta_r[fi] = theta[f_orig];

        for (int j = 0; j < 3; ++j)
        {
            int hij = prev[he_new2old[F_halfedge(fi, j)]];
            if (orig.edge_type[h2e[hij]] == EdgeType::EF)
            {
                kappa_r(fi, j) = 0.;
                period_jump_r(fi, j) = 0;
            }
            else
            {
                int Hij = orig.halfedge_parent[hij];
                int Vk = vtx_reindex_orig[orig.m.to[orig.m.n[Hij]]];
                int i = 0;
                while (F_orig(f_orig, i) != Vk)
                {
                    i++;
                    if (i>= 3)
                    {
                        spdlog::error("could not match offset for {}", f_orig);
                        break;
                    }
                }
                kappa_r(fi, j) = kappa(f_orig, i);
                period_jump_r(fi, j) = period_jump(f_orig, i);
            }
        }
    }

    return std::make_tuple(reference_vector_r, theta_r, kappa_r, period_jump_r);
}

// ---------------
// Parametrization
// ---------------

std::vector<int> SimplificationOverlayMesh::project_dual_loop(
    const std::vector<int>& dual_loop,
    const OverlayStructure& overlay) const
{
    // get parent face for each face
    int num_faces = dual_loop.size();
    std::vector<int> projected_dual_loop(num_faces);
    for (int i = 0; i < num_faces; ++i)
    {
        projected_dual_loop[i] = overlay.face_parent[dual_loop[i]];
    }

    // trim redundant elements from cycle
    return trim_cycle(projected_dual_loop);
}


std::tuple<Eigen::MatrixXd, OverlayMesh<Scalar>>
SimplificationOverlayMesh::generate_simplified_parameterization(
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa
)
{
    spdlog::info("mesh lengths in range {}, {}", vector_min(simp.m.l), vector_max(simp.m.l));
    std::vector<int> start_halfedge = find_start_halfedges(simp);

    // compute metric coordiantes from path lengths
    VectorX metric_coords;
    if (use_zero_lengths)
    {
        simp.m.l = std::vector<Scalar>(simp.m.n_halfedges(), 1.);
        metric_coords = VectorX::Zero(simp.m.n_halfedges());
    }
    else if (use_intrinsic_lengths)
    {
        set_simplified_halfedge_extrinsic_lengths();
        correct_simplified_halfedge_lengths();
        metric_coords = compute_simplified_penner_coordinates();
    }
    else
    {
        set_simplified_halfedge_extrinsic_lengths();
        correct_simplified_halfedge_lengths();
        metric_coords = generate_log_edge_lengths(simp.m);
    }

    spdlog::info("Initial coordinates in range {}, {}", metric_coords.minCoeff(), metric_coords.maxCoeff());

    MarkedPennerConeMetric marked_metric(simp.m, metric_coords, homology_basis_loops, kappa);
    Optimization::LogLengthEnergy opt_energy(marked_metric);
    double max_triangle_quality = 1e3;
    Holonomy::regularize_metric(marked_metric, max_triangle_quality);

    NewtonParameters alg_params;
    alg_params.solver = "ldlt";
    alg_params.do_reduction = true;
    alg_params.max_itr = 100;
    alg_params.log_level = -1;

    MatrixX identity = id_matrix(marked_metric.n_reduced_coordinates());
    Holonomy::OptimizeNewton solver;
    MarkedPennerConeMetric opt_marked_metric = solver.run(marked_metric, identity, alg_params);


    if (solver.get_log().max_error > 1e-10)
    {
        spdlog::warn("falling back to zero coordinates");
        alg_params.max_itr = 10000;
        simp.m.l = std::vector<Scalar>(simp.m.n_halfedges(), 1.);
        metric_coords = VectorX::Zero(simp.m.n_halfedges()); 
        marked_metric.change_metric(marked_metric, metric_coords);
        opt_marked_metric = solver.run(marked_metric, identity, alg_params);
    }

    if ((optimize_metric) && (opt_iters > 0))
    {
        auto proj_params = std::make_shared<ProjectionParameters>();
        auto opt_params = std::make_shared<Optimization::OptimizationParameters>();
        opt_params->num_iter = opt_iters;
        proj_params->max_itr = 100;
        proj_params->error_eps = 0;
        auto opt_cone_metric = Optimization::optimize_metric(opt_marked_metric, opt_energy, proj_params, opt_params);
        opt_marked_metric.change_metric(marked_metric, opt_cone_metric->get_metric_coordinates());
    }


    VectorX opt_metric_coords = opt_marked_metric.get_reduced_metric_coordinates(); 
    VectorX he_metric_coords = opt_marked_metric.get_metric_coordinates(); 
    spdlog::info("optimized coordinates in range {}, {}", he_metric_coords.minCoeff(), he_metric_coords.maxCoeff());

    auto [V_simp, F_simp] = generate_simplified_vf_mesh();
    std::vector<int> vtx_reindex_simp = arange(simp.m.n_vertices());
    VectorX scale_factors = VectorX::Zero(simp.m.n_ind_vertices());
    InterpolationMesh<Scalar> interpolation_mesh, reverse_interpolation_mesh;
    interpolate_penner_coordinates(
        simp.m,
        he_metric_coords,
        scale_factors,
        interpolation_mesh,
        reverse_interpolation_mesh);

    Eigen::MatrixXd V_overlay;
    interpolate_vertex_positions(
        V_simp,
        vtx_reindex_simp,
        interpolation_mesh,
        reverse_interpolation_mesh,
        V_overlay);

    bool save_coarse_mesh = false;
    if (save_coarse_mesh)
    {
        std::vector<bool> cut_h = {};

        auto [m_o, V_o, F_o, uv_o, FT_o, is_cut_h, is_cut_o, fn_to_f_o, endpoints_o] = generate_VF_mesh_from_halfedge_metric<Scalar>(
            V_simp,
            simp.m,
            vtx_reindex_simp,
            marked_metric,
            opt_metric_coords,
            cut_h,
            false);

        Holonomy::view_seamless_parameterization(V_o, F_o, uv_o, FT_o, "overlay", true);
        //Holonomy::view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "refined", true);

        // Write the output mesh
        std::string output_filename = "coarse_mesh.obj";
        write_obj_with_uv(output_filename, V_o, F_o, uv_o, FT_o);
    }

    OverlayMesh<Scalar> m_o = interpolation_mesh.get_overlay_mesh();

    return std::make_tuple(V_overlay, m_o);
}

// -----------------------
// Barycentric Coordinates
// -----------------------

SimplificationOverlayMesh::BaryCoords
SimplificationOverlayMesh::tip_face_coordinates(
    int halfedge_index,
    int face_index,
    const OverlayStructure& overlay) const
{
    int vertex_index = to[halfedge_index];
    int hij = overlay.m.h[face_index];
    int hjk = overlay.m.n[hij];
    int hki = overlay.m.n[hjk];


    if (overlay.vertex_type[vertex_index] == VertexType::VV)
    {
        int h_bd = halfedge_index;
        while (overlay.edge_type[h2e[h_bd]] != EdgeType::EE)
        {
            h_bd = prev[opp[h_bd]];
        }

        if (overlay.halfedge_parent[h_bd] == hki) return {1., 0.};
        if (overlay.halfedge_parent[h_bd] == hij) return {0., 1.};
        if (overlay.halfedge_parent[h_bd] == hjk) return {0., 0.};
    }
    else if (overlay.vertex_type[vertex_index] == VertexType::VE)
    {
        int h_bd = halfedge_index;
        while (overlay.edge_type[h2e[h_bd]] != EdgeType::EE)
        {
            h_bd = prev[opp[h_bd]];
        }

        BaryCoords edge_coords = overlay.bary_coords[vertex_index];
        return overlay.convert_halfedge_to_face_coordinates(overlay.halfedge_parent[h_bd], face_index, edge_coords);
    }
    else if (overlay.vertex_type[vertex_index] == VertexType::VF)
    {
        return overlay.bary_coords[vertex_index];
    }

    spdlog::error("tip not adjacent to face");
    assert(false);
    return {0., 0.};
}

SimplificationOverlayMesh::BaryCoords
SimplificationOverlayMesh::tip_edge_coordinates(
    int halfedge_index,
    int edge_index,
    const OverlayStructure& overlay) const
{
    int vertex_index = to[halfedge_index];
    int hij = overlay.e2h[edge_index];
    int hji = overlay.m.opp[edge_index];
    if (overlay.vertex_type[vertex_index] == VertexType::VV)
    {
        if (overlay.halfedge_parent[halfedge_index] == hji) return {1., 0.};
        if (overlay.halfedge_parent[halfedge_index] == hij) return {0., 0.};
    }

    return overlay.vertex_edge_coordinates(vertex_index, edge_index);
}

Vector2 SimplificationOverlayMesh::project_halfedge_tip(int halfedge_index) const
{
    int hij = halfedge_index;
    int vj = to[hij];
    int fijk = face[hij];
    int Fijk = orig.face_parent[fijk];

    return orig.vertex_face_coordinates(vj, Fijk);
}

SimplificationOverlayMesh::BaryCoords SimplificationOverlayMesh::convert_overlay_to_original_edge_coordinates(
    int halfedge_index,
    Scalar overlay_edge_coord) const
{
    assert(orig.edge_type[h2e[halfedge_index]] == EdgeType::EE);

    // get edge with correct orientation
    int vi = to[opp[halfedge_index]];
    int vj = to[halfedge_index];
    int h_parent = orig.halfedge_parent[halfedge_index];
    int Eij = orig.h2e[h_parent];
    BaryCoords Vi = orig.vertex_edge_coordinates(vi, Eij);
    BaryCoords Vj = orig.vertex_edge_coordinates(vj, Eij);

    Scalar t = overlay_edge_coord;
    return ((t * Vj) + ((1. - t) * Vi));
}

SimplificationOverlayMesh::BaryCoords SimplificationOverlayMesh::convert_overlay_to_original_face_coordinates(
    int halfedge_index,
    Scalar overlay_edge_coord) const
{
    assert(orig.edge_type[h2e[halfedge_index]] == EdgeType::EF);

    // get edge with correct orientation
    int vi = to[opp[halfedge_index]];
    int vj = to[halfedge_index];
    int Fijk = orig.halfedge_parent[halfedge_index];
    BaryCoords Vi = orig.vertex_face_coordinates(vi, Fijk);
    BaryCoords Vj = orig.vertex_face_coordinates(vj, Fijk);

    Scalar t = overlay_edge_coord;
    return ((t * Vj) + ((1. - t) * Vi));
}

SimplificationOverlayMesh::BaryCoords SimplificationOverlayMesh::convert_overlay_to_original_coordinates(
    int halfedge_index,
    Scalar overlay_edge_coord) const
{
    int eij = h2e[halfedge_index];
    if (orig.edge_type[eij] == EdgeType::EE)
    {
        return convert_overlay_to_original_edge_coordinates(halfedge_index, overlay_edge_coord);
    }
    else
    {
        return convert_overlay_to_original_face_coordinates(halfedge_index, overlay_edge_coord);
    }
}


// ------------
// Edge Lengths
// ------------

Scalar SimplificationOverlayMesh::halfedge_length(int halfedge_index) const
{
    int hij = halfedge_index;
    int vj = to[hij];
    int vi = to[opp[hij]];

    // use edge coordinates if contained in edge
    if (orig.edge_type[h2e[hij]] == EdgeType::EE) 
    {
        int Hij = orig.halfedge_parent[hij];
        int Eij = orig.h2e[Hij];
        Scalar s = orig.vertex_edge_coordinates(vi, Eij)[0];
        Scalar t = orig.vertex_edge_coordinates(vj, Eij)[0];
        Scalar l = orig.m.l[Hij];
        return abs(s - t) * l;
    }
    // build geometry from 3D positions otherwise
    else
    {
        Eigen::Vector3d Vi = generate_overlay_vertex(vi, V_orig, vtx_reindex_orig, orig);
        Eigen::Vector3d Vj = generate_overlay_vertex(vj, V_orig, vtx_reindex_orig, orig);
        return (Vi - Vj).norm();
    }
}

Scalar SimplificationOverlayMesh::compute_simplified_halfedge_length(int halfedge_index) const
{
    std::vector<int> hij_path = enumerate_halfedges(halfedge_index, simp);
    Scalar length = 0.;
    for (int h : hij_path)
    {
        length += halfedge_length(h);
    }
    return length;
}

Scalar SimplificationOverlayMesh::compute_extrinsic_halfedge_length(int halfedge_index) const
{
    int hij = halfedge_index;
    int hji = simp.m.opp[hij];
    int Vi = simp.m.to[hji];
    int Vj = simp.m.to[hij];
    int vi = simp.vertex_index[Vi];
    int vj = simp.vertex_index[Vj];
    if (vi == vj) spdlog::warn("self adjacent edge");
    Eigen::Vector3d pi = generate_overlay_vertex(vi, V_orig, vtx_reindex_orig, orig);
    Eigen::Vector3d pj = generate_overlay_vertex(vj, V_orig, vtx_reindex_orig, orig);
    return (pi - pj).norm();
}

void SimplificationOverlayMesh::set_simplified_halfedge_lengths()
{
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    for (int hij = 0; hij < simp.m.n_halfedges(); ++hij)
    {
        int hji = simp.m.opp[hij];
        if (hij < hji) continue;
        Scalar length = compute_simplified_halfedge_length(start_halfedge[hij]);
        simp.m.l[hij] = length;
        simp.m.l[hji] = length;
    }
}

VectorX SimplificationOverlayMesh::compute_simplified_penner_coordinates()
{
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    int num_halfedges = simp.m.n_halfedges();
    VectorX metric_coords(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = simp.m.opp[hij];
        if (hij < hji) continue;
        Scalar length = compute_simplified_halfedge_length(start_halfedge[hij]);
        metric_coords[hij] = (float_equal<Scalar>(length, 0.)) ? -10 : 2. * log(length);
        metric_coords[hji] = metric_coords[hij];
    }

    return metric_coords;
}

VectorX SimplificationOverlayMesh::compute_extrinsic_penner_coordinates()
{
    int num_halfedges = simp.m.n_halfedges();
    VectorX metric_coords(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = simp.m.opp[hij];
        if (hij < hji) continue;
        Scalar length = compute_extrinsic_halfedge_length(hij);
        if (isnan(length)) spdlog::error("edge length is NaN");
        metric_coords[hij] = (float_equal<Scalar>(length, 0.)) ? -10 : 2. * log(length);
        metric_coords[hji] = metric_coords[hij];
    }

    return metric_coords;
}

void SimplificationOverlayMesh::set_simplified_halfedge_extrinsic_lengths()
{
    for (int hij = 0; hij < simp.m.n_halfedges(); ++hij)
    {
        int hji = simp.m.opp[hij];
        if (hij < hji) continue;
        Scalar length = compute_extrinsic_halfedge_length(hij);
        if (isnan(length))
        {
            spdlog::warn("edge length is NaN");
            length = 0.;
        }
        simp.m.l[hij] = length;
        simp.m.l[hji] = length;
    }
}

void SimplificationOverlayMesh::set_simplified_halfedge_curvature_lengths(Scalar normal_scale)
{
    // compute extrinsic mesh and approximated normals
    auto [V_simp, F_simp] = generate_simplified_vf_mesh();
    std::vector<int> vtx_reindex_simp = arange(simp.m.n_vertices());
    simp.m.l = compute_curvature_adaptive_lengths(V_simp, F_simp, simp.m, vtx_reindex_simp, normal_scale);
}

void SimplificationOverlayMesh::correct_simplified_halfedge_lengths()
{
    // Check triangle inequality for each halfedge
    int num_halfedges = simp.m.n_halfedges();
    bool is_degenerate = false;
    do
    {
        is_degenerate = false;

        // check for degenerate edge
        for (int hi = 0; hi < num_halfedges; ++hi) {
            int hj = simp.m.n[hi];
            int hk = simp.m.n[hj];

            // Lengths of the halfedges
            Scalar li = simp.m.l[hi];
            Scalar lj = simp.m.l[hj];
            Scalar lk = simp.m.l[hk];

            // Check for zero length
            if (float_equal<Scalar>(li, 0.))
            {
                spdlog::trace("zero edge length {}", li);
                is_degenerate = true;
            }

            // Check triangle inequality
            if ((li > lj + lk) || float_equal(li, lj + lk))
            {
                spdlog::trace("edge length {} > {} + {}", li, lj, lk);
                is_degenerate = true;
            }
        }

        if (!is_degenerate) return;

        // make more numerically stable
        for (int hij = 0; hij < num_halfedges; ++hij) {
            int hji = simp.m.opp[hij];
            if (hij < hji) continue;
            if (simp.m.l[hij] <= 0.) simp.m.l[hij] = 1e-4;
            simp.m.l[hij] = exp(0.9 * log(simp.m.l[hij]));
            simp.m.l[hji] = simp.m.l[hij];
            if (isnan(simp.m.l[hij])) spdlog::error("edge length nan");
        }
    } while (is_degenerate);
}

// ------------------
// Embedding Geometry
// ------------------

void SimplificationOverlayMesh::set_uniform_edge_coordinates(int start_halfedge, OverlayStructure& overlay)
{
    int hij = start_halfedge;
    if (overlay.edge_type[h2e[hij]] != EdgeType::EE) return;
    int Hij = overlay.halfedge_parent[hij];
    int Eij = overlay.h2e[Hij];
    std::vector<int> halfedges = enumerate_halfedges(start_halfedge, overlay);
    int hi = opp[halfedges.front()];
    int hj = halfedges.back();
    Scalar delta = 1. / halfedges.size();
    BaryCoords Vi = tip_edge_coordinates(hi, Eij, overlay);
    BaryCoords Vj = tip_edge_coordinates(hj, Eij, overlay);
    assert(Vi != Vj);
    for (int k = 1; k < halfedges.size(); ++k)
    {
        int vk = to[opp[halfedges[k]]];
        Scalar t = delta * k;
        overlay.bary_coords[vk] = (1. - t) * Vi + t * Vj;
    }
}

void SimplificationOverlayMesh::set_uniform_edge_coordinates(OverlayStructure& overlay)
{
    std::vector<int> start_halfedge = find_start_halfedges(overlay);

    // find boundary coordiantes
    int num_overlay_edges = overlay.e2h.size();
    for (int Eij = 0; Eij < num_overlay_edges; ++Eij)
    {
        int Hij = overlay.e2h[Eij];
        set_uniform_edge_coordinates(start_halfedge[Hij], overlay);
    }
}

void SimplificationOverlayMesh::find_tutte_coordinates(OverlayStructure& overlay, std::vector<bool>& use_face, bool use_adaptive_weight)
{
    assert(is_valid());
    //remove_parallel_edges(overlay); // need to ensure no parallel edges
    //make_three_connected(overlay);

    // nothing to do if no faces flagged
    if (!vector_contains(use_face, true)) return;
    std::vector<int> faces, old_faces, face_map;
    enumerate_boolean_array(use_face, faces, old_faces, face_map);

    std::vector<int> start_halfedge = find_start_halfedges(overlay);

    // get unique index for each vertex in each face
    int num_halfedges = n.size();
    UnionFind face_vertices(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // union across face edges
        int hkj = opp[n[hij]];
        if (overlay.edge_type[h2e[hkj]] == EdgeType::EF) face_vertices.union_sets(hij, hkj);
    }

    // get representative halfedges for face vertices
    int num_face_vertices = face_vertices.count_sets();
    std::vector<int> vertex_representatives = face_vertices.get_representative_elements();
    std::vector<int> h2fv = face_vertices.index_sets();
    std::vector<Scalar> mean_value_weights;
    if (use_adaptive_weight) mean_value_weights = compute_mean_value_weights(orig);

    // determine boundary vertices
    std::vector<bool> is_boundary_vertex(num_face_vertices, false);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;
        int hji = opp[hij];
        if (overlay.face_parent[face[hij]] == overlay.face_parent[face[hji]]) continue;

        is_boundary_vertex[h2fv[hij]] = true;
    }

    // enumerate interior and boundary vertices per face
    int num_faces = faces.size();
    std::vector<std::vector<int>> interior_face_vertices(num_faces, std::vector<int>());
    std::vector<std::vector<int>> boundary_face_vertices(num_faces, std::vector<int>());
    std::vector<int> vertex_index(num_face_vertices);
    for (int vj = 0; vj < num_face_vertices; ++vj)
    {
        // get a representative halfedge
        int hij = vertex_representatives[vj];
        if (is_halfedge_deleted(hij)) continue;
        int Fijk = overlay.face_parent[face[hij]];
        if (!use_face[Fijk]) continue;
        int f_parent = face_map[Fijk];

        if (is_boundary_vertex[vj])
        {
            vertex_index[vj] = boundary_face_vertices[f_parent].size();
            boundary_face_vertices[f_parent].push_back(vj);
        }
        else
        {
            vertex_index[vj] = interior_face_vertices[f_parent].size();
            interior_face_vertices[f_parent].push_back(vj);
        }
    }

    // build tutte matrix triplets for each face
    std::vector<std::vector<T>> harmonic_matrix_trips(num_faces, std::vector<T>());
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 2> BaryMat;
    //typedef Eigen::MatrixXd BaryMat;
    std::vector<std::vector<BaryCoords>> boundary_coords(num_faces);
    for (int i = 0; i < num_faces; ++i)
    {
        boundary_coords[i] = std::vector<BaryCoords>(boundary_face_vertices[i].size());
    }

    // find boundary coordiantes
    int num_overlay_halfedges = overlay.m.n.size();
    for (int Hij = 0; Hij < num_overlay_halfedges; ++Hij)
    {
        int Eij = overlay.h2e[Hij];
        int Fijk = overlay.m.f[Hij];
        if (!use_face[Fijk]) continue;
        std::vector<int> halfedges = enumerate_halfedges(start_halfedge[Hij], overlay);
        for (int k = 1; k < halfedges.size(); ++k)
        {
            int hij = halfedges[k];
            int vi = to[prev[hij]];
            int i = vertex_index[h2fv[prev[hij]]];
            BaryCoords edge_coords = overlay.bary_coords[vi];
            boundary_coords[face_map[Fijk]][i] = overlay.convert_halfedge_to_face_coordinates(Hij, Fijk, edge_coords);
        }

        // set first and last vertex
        int hkj = halfedges.back();
        int vj = to[hkj];
        int j = vertex_index[h2fv[hkj]];
        boundary_coords[face_map[Fijk]][j] = tip_face_coordinates(hkj, Fijk, overlay);

        //int hik = halfedges.front();
        //int vi = to[prev[hik]];
        //int i = vertex_index[h2fv[prev[hik]]];
        //boundary_coords[face_map[Fijk]][i] = overlay.tip_face_coordinates(hik, vi, Fijk);
    }

    // initialize rhs to zero
    std::vector<std::vector<BaryCoords>> harmonic_rhs(num_faces);
    std::vector<std::vector<std::vector<int>>> harmonic_connections(num_faces);
    for (int i = 0; i < num_faces; ++i)
    {
        harmonic_rhs[i] = std::vector<BaryCoords>(interior_face_vertices[i].size(), {0., 0.});
        harmonic_connections[i].resize(interior_face_vertices[i].size());
    }

    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        int fijk = face[hij];
        int Fijk = overlay.face_parent[fijk];
        if (!use_face[Fijk]) continue;
        int f_parent = face_map[Fijk];

        int vi = h2fv[prev[hij]];
        int vj = h2fv[hij];
        if (is_boundary_vertex[vi]) continue;

        Scalar weight = (use_adaptive_weight) ? mean_value_weights[hij] : 1.;
        if (weight <= 0.)
        {
            spdlog::warn("negative weight {}", weight);
            continue;
        }

        int i = vertex_index[vi];
        int j = vertex_index[vj];
        harmonic_matrix_trips[f_parent].push_back({i, i, weight});
        if (is_boundary_vertex[vj])
        {
            harmonic_rhs[f_parent][i] += weight * boundary_coords[f_parent][j];
            harmonic_connections[f_parent][i].push_back(j);
        }
        else
        {
            harmonic_matrix_trips[f_parent].push_back({i, j, -weight});
        }
        //harmonic_matrix_trips[f_parent].push_back({vi, vj, 1.});
        //harmonic_matrix_trips[f_parent].push_back({vi, vi, -1.});
    }

    //#pragma omp parallel for
    int num_remeshed_faces = 0;
    int total_variables = 0;
    int max_size = 0;
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        int num_variables = interior_face_vertices[fijk].size();
        if (num_variables == 0) continue;
        ++num_remeshed_faces;
        total_variables += num_variables;
        max_size = max(max_size, num_variables);

        BaryMat rhs(num_variables, 2);
        for (int i = 0; i < num_variables; ++i)
        {
            rhs.row(i) = harmonic_rhs[fijk][i];
        }

        MatrixX tutte_system(num_variables, num_variables);
        tutte_system.setFromTriplets(
            harmonic_matrix_trips[fijk].begin(),
            harmonic_matrix_trips[fijk].end());

        // use LU as general weights may not be symmetric
        Eigen::SparseLU<MatrixX> solver;
        //Eigen::CholmodSupernodalLLT<MatrixX> solver;
        //Eigen::SimplicialLDLT<MatrixX> solver;
        solver.compute(tutte_system);
        if (solver.info() != Eigen::Success) spdlog::error("matrix factorization failed");
        BaryMat bary_coords = solver.solve(rhs);

        if (solver.info() != Eigen::Success) spdlog::error("matrix solve failed");
        Scalar solver_error = (tutte_system * bary_coords - rhs).norm();
        spdlog::debug("tutte solver error for {} is {}", fijk, solver_error);

        //overlay.he_bary_coords.resize(n.size());
        Scalar min_dist = 1.;
        for (int i = 0; i < num_variables; ++i)
        {
            int vi = interior_face_vertices[fijk][i];
            int hji = vertex_representatives[vi];
            overlay.bary_coords[to[hji]] = bary_coords.row(i);
            //overlay.he_bary_coords[hji] = bary_coords.row(i); // TODO remove
            if (show_failure)
            {
                Scalar u = bary_coords(i, 0);
                Scalar v = bary_coords(i, 1);
                min_dist = min(min_dist, u);
                min_dist = min(min_dist, 1 - u);
                min_dist = min(min_dist, v);
                min_dist = min(min_dist, 1 - v);
                min_dist = min(min_dist, 1 - (u + v));
            }
        }


        //if (false)
        if ((show_failure) && ((!float_equal<Scalar>(solver_error, 0.)) || (min_dist <= 0)))
        {
            spdlog::error("tutte solver failed");
            spdlog::error("minimum distance is {}", min_dist);
            BaryMat bary_coords = solve_high_precision<mpfr::mpreal>(tutte_system, rhs).cast<Scalar>();

            int num_constants = boundary_face_vertices[fijk].size();
            int num_face_vertices = num_variables + num_constants;
            BaryMat face_coords(num_face_vertices, 2);
            face_coords.topRows(num_variables) = bary_coords;
            for (int k = 0; k < num_constants; ++k)
            {
                face_coords.row(num_variables + k) = boundary_coords[fijk][k];
            }
            std::vector<std::array<int, 2>> edges = {};
            for (int k = 0; k < tutte_system.outerSize(); ++k) {
                for (Eigen::SparseMatrix<Scalar>::InnerIterator it(tutte_system, k); it; ++it) {
                    int row = it.row();     // row index
                    int col = it.col();     // column index
                    //double val = it.value();
                    if (row == col) continue;
                    edges.push_back({row, col});
                }
            }
            for (int i = 0; i < num_variables; ++i)
            {
                for (int j : harmonic_connections[fijk][i])
                {
                    edges.push_back({i, num_variables + j});
                }
            }
#ifdef ENABLE_VISUALIZATION
            polyscope::init();
            polyscope::registerCurveNetwork2D("face layout", face_coords, edges);
            polyscope::registerPointCloud2D("face boundary", boundary_coords[fijk]);
            polyscope::show();
#endif
            write_sparse_matrix(tutte_system, "tutte_matrix.txt", "matlab");
            write_matrix(bary_coords, "bary_coords.txt");
            write_matrix(rhs, "rhs.txt");
        }
    }
    spdlog::info("{} faces remeshed with {} total variables", num_remeshed_faces, total_variables);
    spdlog::info("largest system is {}", max_size);

    int num_flipped_faces = 0;
    for (int fijk = 0; fijk < f2h.size(); ++fijk)
    {
        if (is_face_deleted(fijk)) continue;
        int Fijk = overlay.face_parent[fijk];
        int hij = f2h[fijk];
        int hjk = n[hij];
        int hki = n[hjk];
        BaryCoords uvi = tip_face_coordinates(hki, Fijk, overlay);
        BaryCoords uvj = tip_face_coordinates(hij, Fijk, overlay);
        BaryCoords uvk = tip_face_coordinates(hjk, Fijk, overlay);
        if (igl::predicates::orient2d(uvi, uvj, uvk) != igl::predicates::Orientation::POSITIVE)
        {
            spdlog::error("flipped face {}", fijk);
            ++num_flipped_faces;
        }
    }
    spdlog::info("{} flipped faces", num_flipped_faces);
}  

void SimplificationOverlayMesh::find_tutte_coordinates(OverlayStructure& overlay, bool use_adaptive_weight)
{
    remove_parallel_edges(overlay); // need to ensure no parallel edges
    std::vector<bool> use_face(overlay.m.n_faces(), true);
    find_tutte_coordinates(overlay, use_face, use_adaptive_weight);
}

std::vector<Scalar> SimplificationOverlayMesh::compute_mean_value_weights(
    const OverlayStructure& overlay) const
{
    const Scalar eps = 1e-12;
    const Scalar pi = acos(Scalar(-1.));
    std::vector<Scalar> weights(n.size(), -1.);

    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;
        int hji = opp[hij];

        // compute edge length and check for degenerate edge
        Vector3 vi = generate_overlay_vertex(to[hji], V_orig, vtx_reindex_orig, overlay);
        Vector3 vj = generate_overlay_vertex(to[hij], V_orig, vtx_reindex_orig, overlay);
        Scalar edge_length = (vj - vi).norm();
        if (edge_length <= eps)
        {
            spdlog::warn("edge numerically degenerate");
            weights[hij] = 0.;
            continue;
        }

        // compute triangle angles at halfedge tip
        std::array<Scalar, 2> angles;
        Vector3 vk = generate_overlay_vertex(to[n[hij]], V_orig, vtx_reindex_orig, overlay);
        Vector3 vl = generate_overlay_vertex(to[n[hji]], V_orig, vtx_reindex_orig, overlay);
        angles[0] = abs(compute_triangle_angle(vi, vj, vk));
        angles[1] = abs(compute_triangle_angle(vi, vl, vj));

        // compute tangent sum
        Scalar tan_half_angle_sum = 0.;
        for (Scalar& angle : angles)
        {
            angle = min(max(angle, eps), pi - eps);
            tan_half_angle_sum += tan(angle / 2.);
        }

        // add weight to halfedge tip 
        Scalar weight = tan_half_angle_sum / edge_length;
        assert(weight >= 0.);
        weights[hij] = weight;
    }

    return weights;
}

// ------------------
// Embedding Connectivity
// ------------------

// TODO make local and fast
void SimplificationOverlayMesh::flood_face(int h, int f_parent)
{
    // get unique index for each vertex in each face
    UnionFind faces(n.size());
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // always union with next halfedge in face
        int hjk = n[hij];
        faces.union_sets(hij, hjk);

        // union across face edges, but not edge edges
        int hji = opp[hij];
        if (simp.edge_type[h2e[hij]] == EdgeType::EF) faces.union_sets(hij, hji);
    }

    // propagate face indices to all faces, edges, and vertices
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // set parent of face
        //int f = faces.find_set(hij);
        if (faces.find_set(hij) != faces.find_set(h)) continue;
        simp.face_parent[face[hij]] = f_parent;

        // check edge type
        int eij = h2e[hij];
        if (simp.edge_type[eij] == EdgeType::EF)
        {
            simp.halfedge_parent[hij] = f_parent;
        }

        // check tip vertex type
        int vj = to[hij];
        if (simp.vertex_type[vj] == VertexType::VF)
        {
            simp.vertex_parent[vj] = f_parent;
        }
    }
}

void SimplificationOverlayMesh::flood_faces_union_find()
{
    assert(is_valid_edge_network());
    // get first halfedge of each simplified halfedge path
    std::vector<int> start_halfedge = find_start_halfedges(simp);

    // get unique index for each vertex in each face
    UnionFind faces(n.size());
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // always union with next halfedge in face
        int hjk = n[hij];
        faces.union_sets(hij, hjk);

        // union across face edges, but not edge edges
        int hji = opp[hij];
        if (simp.edge_type[h2e[hij]] == EdgeType::EF) faces.union_sets(hij, hji);
    }

    // set face indices from start halfedges
    auto& simp_faces = simp.m.f;
    // TODO make option for base mesh, or unify representation
    std::vector<int> face_parent(faces.count_sets(), -1);
    std::vector<int> face_indices = faces.index_sets();
    for (int hij = 0; hij < simp_faces.size(); ++hij)
    {
        int h = start_halfedge[hij];
        if (!is_valid_halfedge_index(h)) spdlog::error("start halfedge not valid");
        if (is_halfedge_deleted(h)) spdlog::error("halfedge deleted");
        assert(simp.halfedge_parent[h] == hij);

        //int f = faces.find_set(h);
        int face_index = face_indices[h];
        face_parent[face_index] = simp_faces[hij];
    }

    // sanity check
    for (int hij = 0; hij < simp_faces.size(); ++hij)
    {
        int h = start_halfedge[hij];
        int face_index = face_indices[h];
        assert(face_parent[face_index] == simp_faces[hij]);
    }

    // propagate face indices to all faces, edges, and vertices
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // set parent of face
        //int f = faces.find_set(hij);
        int face_index = face_indices[hij];
        int f_parent = face_parent[face_index];
        assert(is_valid_simplified_face_index(f_parent));
        simp.face_parent[face[hij]] = f_parent;

        // check edge type
        int eij = h2e[hij];
        if (simp.edge_type[eij] == EdgeType::EF)
        {
            simp.halfedge_parent[hij] = f_parent;
        }

        // check tip vertex type
        int vj = to[hij];
        if (simp.vertex_type[vj] == VertexType::VF)
        {
            simp.vertex_parent[vj] = f_parent;
        }
    }
}

void SimplificationOverlayMesh::flood_faces()
{
    assert(is_valid_edge_network());
    // get first halfedge of each simplified halfedge path
    std::vector<int> start_halfedge = find_start_halfedges(simp);

    // set face indices from start halfedges
    // TODO make option for base mesh, or unify representation
    int num_simp_faces = simp.m.f.size();
    std::vector<int> faces_to_process = {};
    faces_to_process.reserve(num_simp_faces);
    for (int hij = 0; hij < num_simp_faces; ++hij)
    {
        int h = start_halfedge[hij];
        if (!is_valid_halfedge_index(h)) spdlog::error("start halfedge not valid");
        if (is_halfedge_deleted(h)) spdlog::error("halfedge deleted");
        assert(simp.halfedge_parent[h] == hij);

        //int f = faces.find_set(h);
        int face_index = face[h];
        simp.face_parent[face_index] = simp.m.f[hij];
        faces_to_process.push_back(face_index);
    }

    int num_faces = f2h.size();
    std::vector<bool> is_processed_face(num_faces, false);
    while (!faces_to_process.empty()) {
        // Get the next face to process
        int fi = faces_to_process.back();
        faces_to_process.pop_back();

        // Skip already processed face 
        if (is_processed_face[fi]) continue;
        is_processed_face[fi] = true;

        // Iterate over adjacent faces without crossing overlay edges
        int f_parent = simp.face_parent[fi];
        int h_start = f2h[fi];
        int hij = h_start;
        do {
            // propagate labels across edge if not overlay edge
            if (simp.edge_type[h2e[hij]] == EdgeType::EF)
            {
                // Get the face opposite the edge
                int hji = opp[hij];
                int fj = face[hji];
                simp.halfedge_parent[hij] = f_parent;
                simp.halfedge_parent[hji] = f_parent;
                simp.face_parent[fj] = f_parent;
                faces_to_process.push_back(fj);
            }

            // label tip vertex if a face vertex
            int vj = to[hij];
            if (simp.vertex_type[vj] == VertexType::VF)
            {
                simp.vertex_parent[vj] = f_parent;
            }

            // Progress to the next halfedge in the vertex circulator
            hij = n[hij];
        } while (hij != h_start);
    }

    for (int f = 0; f < num_simp_faces; ++f)
    {
        if (is_face_deleted(f)) continue;
        if (!is_processed_face[f]) spdlog::error("face {} not processed", f);
    }
}

std::vector<int> SimplificationOverlayMesh::find_start_halfedges(const OverlayStructure& overlay) const
{
    int num_overlay_halfedges = overlay.m.n_halfedges();
    std::vector<int> start_halfedge(num_overlay_halfedges, -1);
    int num_halfedges_seen = 0;
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // check if halfedge is in a overlay edge and adjacent to an overlay vertex
        if (overlay.edge_type[h2e[hij]] != EdgeType::EE) continue;
        if (overlay.vertex_type[to[opp[hij]]] != VertexType::VV) continue;

        // set as start halfedge for edge if so
        int Hij = overlay.halfedge_parent[hij];
        assert(Hij >= 0);
        assert(Hij < num_overlay_halfedges);
        start_halfedge[Hij] = hij;
        ++num_halfedges_seen;
    }
    if (num_halfedges_seen > start_halfedge.size())
    {
        spdlog::error("halfedge seen multiple times");
        assert(false);
    }
    if (num_halfedges_seen < start_halfedge.size())
    {
        spdlog::error("not all halfedges seen");
        for (int i = 0; i < num_overlay_halfedges; ++i)
        {
            if (start_halfedge[i] < 0) spdlog::error("{} not seen", i);
        }
        assert(false);
    }

    return start_halfedge;
}

std::vector<int> SimplificationOverlayMesh::find_overlay_vertices(const OverlayStructure& overlay) const
{
    int num_overlay_vertices = overlay.m.n_vertices();
    std::vector<int> overlay_vertices(num_overlay_vertices, -1);
    int num_vertices_seen = 0;
    for (int vi = 0; vi < out.size(); ++vi)
    {
        if (is_vertex_deleted(vi)) continue;
        if (overlay.vertex_type[vi] != VertexType::VV) continue;
        overlay_vertices[overlay.vertex_parent[vi]] = vi;
        num_vertices_seen++;
    }
    if (num_vertices_seen != num_overlay_vertices)
    {
        spdlog::error("inconsistent vertex counts");
        assert(false);
    }

    return overlay_vertices;
}

std::vector<int> SimplificationOverlayMesh::find_end_halfedges(const OverlayStructure& overlay) const
{
    int num_overlay_halfedges = overlay.m.n_halfedges();
    std::vector<int> end_halfedge(num_overlay_halfedges, -1);
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // check if halfedge is in a overlay edge and adjacent to an overlay vertex
        if (overlay.edge_type[h2e[hij]] != EdgeType::EE) continue;
        if (overlay.vertex_type[to[hij]] != VertexType::VV) continue;

        // set as start halfedge for edge if so
        int Hij = overlay.halfedge_parent[hij];
        assert(Hij >= 0);
        assert(Hij < num_overlay_halfedges);
        end_halfedge[Hij] = hij;
    }

    return end_halfedge;
}

std::vector<int> SimplificationOverlayMesh::enumerate_halfedges(int halfedge_index, const OverlayStructure& overlay) const
{
    int hij = halfedge_index;
    std::vector<int> halfedges = { hij };
    while (overlay.vertex_type[to[hij]] != VertexType::VV)
    {
        assert(overlay.vertex_type[to[hij]] == VertexType::VE);
        hij = get_next_overlay_edge(hij, overlay);
        halfedges.push_back(hij);
    }

    return halfedges;
}

std::vector<std::vector<int>> SimplificationOverlayMesh::enumerate_simplified_edges() const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    int num_edges = simp.e2h.size();
    std::vector<std::vector<int>> paths;
    for (int e = 0; e < num_edges; ++e)
    {
        int h = simp.e2h[e];
        std::vector<int> halfedges = enumerate_halfedges(start_halfedge[h], simp);
        paths.push_back(std::vector<int>());
        auto& curr_path = paths.back();
        curr_path.reserve(halfedges.size() + 1);
        curr_path.push_back(v_old2new[to[opp[halfedges.front()]]]);
        for (int hij : halfedges)
        {
            curr_path.push_back(v_old2new[to[hij]]);
        }
    }

    return paths;
}

bool SimplificationOverlayMesh::is_start_cyclic_orientation(
    int h_start,
    int halfedge_index) const
{
    if (!use_consistency_check) return true;

    // get first counterclockwise edge around halfedge base
    int hij = h_start;
    int e_ccw = -1;
    do {
        hij = opp[prev[hij]];
        int eij = h2e[hij];
        if (simp.edge_type[eij] == EdgeType::EE)
        {
            e_ccw = simp.h2e[simp.halfedge_parent[hij]];
            break;
        }
    } while (hij != h_start);
    if (e_ccw < 0) return true; // first edge adjacent to vertex

    // get first clockwise edge around halfedge base
    hij = h_start;
    int e_cw = -1;
    do {
        hij = n[opp[hij]];
        int eij = h2e[hij];
        if (simp.edge_type[eij] == EdgeType::EE)
        {
            e_cw = simp.h2e[simp.halfedge_parent[hij]];
            break;
        }
    } while (hij != h_start);
    if (e_ccw == e_cw) return true; // second edge adjacent to vertex

    // determine if three edges are consistent in simplified edge
    // just need to check if first edge seen clockwise is e_cw
    const auto& m = simp.m;
    int Hij = halfedge_index;
    do {
        Hij = m.n[m.opp[Hij]];
        if (simp.h2e[Hij] == e_ccw) return false;
        if (simp.h2e[Hij] == e_cw) return true;
    } while (Hij != halfedge_index);

    assert(false); // should never get here
    return false;
}

bool SimplificationOverlayMesh::is_end_cyclic_orientation(
    int h_start,
    int halfedge_index) const
{
    if (!use_consistency_check) return true;

    // get first counterclockwise edge around halfedge tip
    int hij = h_start;
    int e_ccw = -1;
    do {
        hij = prev[opp[hij]];
        int eij = h2e[hij];
        if (simp.edge_type[eij] == EdgeType::EE)
        {
            e_ccw = simp.h2e[simp.halfedge_parent[hij]];
            break;
        }
    } while (hij != h_start);
    if (e_ccw < 0) return true; // first edge adjacent to vertex

    // get first clockwise edge around halfedge tip
    hij = h_start;
    int e_cw = -1;
    do {
        hij = opp[n[hij]];
        int eij = h2e[hij];
        if (simp.edge_type[eij] == EdgeType::EE)
        {
            e_cw = simp.h2e[simp.halfedge_parent[hij]];
            break;
        }
    } while (hij != h_start);
    if (e_ccw == e_cw) return true; // second edge adjacent to vertex

    // determine if three edges are consistent in simplified edge
    // just need to check if first edge seen clockwise is e_cw
    const auto& m = simp.m;
    int Hij = halfedge_index;
    do {
        Hij = m.opp[m.n[Hij]];
        if (simp.h2e[Hij] == e_ccw) return false;
        if (simp.h2e[Hij] == e_cw) return true;
    } while (Hij != halfedge_index);

    assert(false); // should never get here
    return false;
}

std::vector<std::vector<int>> SimplificationOverlayMesh::enumerate_simplified_halfedges() const
{
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    int num_edges = simp.e2h.size();
    std::vector<std::vector<int>> paths;
    for (int e = 0; e < num_edges; ++e)
    {
        int h = simp.e2h[e];
        std::vector<int> halfedges = enumerate_halfedges(start_halfedge[h], simp);
        paths.push_back(std::vector<int>());
        auto& curr_path = paths.back();
        curr_path.reserve(halfedges.size());
        for (int hij : halfedges)
        {
            curr_path.push_back(he_old2new[hij]);
        }
    }

    return paths;
}


// ------------------
// Overlay Navigation
// ------------------

int SimplificationOverlayMesh::get_next_overlay_edge(int halfedge_index, const OverlayStructure& overlay) const
{
    int hij = halfedge_index;

    // get halfedge with base at tip of input halfedge
    hij = opp[hij];

    // rotate ccw around base until hit another overlay edge
    do {
        hij = n[opp[hij]];
    } while (overlay.edge_type[h2e[hij]] != EdgeType::EE);
    assert(overlay.vertex_type[to[hij]] != VertexType::VF);

    return hij;
}

int SimplificationOverlayMesh::get_ccw_overlay_edge(int halfedge_index, const OverlayStructure& overlay) const
{
    int hij = halfedge_index;
    do {
        hij = opp[prev[hij]];
    } while (overlay.edge_type[h2e[hij]] != EdgeType::EE);
    assert(overlay.vertex_type[to[hij]] != VertexType::VF);

    return hij;
}

int SimplificationOverlayMesh::get_clockwise_overlay_edge(int halfedge_index, const OverlayStructure& overlay) const
{
    int hij = halfedge_index;
    do {
        hij = n[opp[hij]];
    } while (overlay.edge_type[h2e[hij]] != EdgeType::EE);
    assert(overlay.vertex_type[to[hij]] != VertexType::VF);

    return hij;
}

// --------------
// Robust Cleanup
// --------------

bool SimplificationOverlayMesh::is_degenerate_edge_coordinates(int start_halfedge, OverlayStructure& overlay)
{
    int hij = start_halfedge;
    if (overlay.edge_type[h2e[hij]] != EdgeType::EE) return false;

    int Hij = overlay.halfedge_parent[hij];
    int Eij = overlay.h2e[Hij];
    std::vector<int> halfedges = enumerate_halfedges(start_halfedge, overlay);
    if (halfedges.size() == 1) return false;

    int vi = to[opp[halfedges.front()]];
    int vj = to[halfedges.back()];
    int vk = to[halfedges.front()];
    int vl = to[opp[halfedges.back()]];
    BaryCoords Vi = overlay.vertex_edge_coordinates(vi, Eij);
    BaryCoords Vj = overlay.vertex_edge_coordinates(vj, Eij);
    if (float_equal<Scalar>(Vi[0], overlay.bary_coords[vk][0], 1e-8)) return true;
    if (float_equal<Scalar>(Vj[0], overlay.bary_coords[vl][0], 1e-8)) return true;
    for (int k = 1; k < halfedges.size() - 1; ++k)
    {
        int vk = to[opp[halfedges[k]]];
        int vl = to[halfedges[k]];
        Scalar s = overlay.bary_coords[vk][0];
        Scalar t = overlay.bary_coords[vl][0];
        if (float_equal<Scalar>(s, t, 1e-8)) return true;
    }

    return false;
}

void SimplificationOverlayMesh::fix_degenerate_edge_coordinates(int start_halfedge, OverlayStructure& overlay)
{
    int hij = start_halfedge;
    if (overlay.edge_type[h2e[hij]] != EdgeType::EE) return;
    int Hij = overlay.halfedge_parent[hij];
    int Eij = overlay.h2e[Hij];
    std::vector<int> halfedges = enumerate_halfedges(start_halfedge, overlay);
    for (int k = 0; k < halfedges.size()-1; ++k)
    {
        int vi = to[opp[halfedges[k]]];
        int vj = to[halfedges[k]];
        int vk = to[halfedges[k + 1]];
        BaryCoords Vi = overlay.vertex_edge_coordinates(vi, Eij);
        BaryCoords Vj = overlay.vertex_edge_coordinates(vj, Eij);
        BaryCoords Vk = overlay.vertex_edge_coordinates(vk, Eij);
        if (float_equal<Scalar>(Vi[0], Vj[0], 1e-8))
        {
            Scalar t = 0.01;
            overlay.bary_coords[vj] = (1. - t) * Vk + t * Vj;
        }
        else if (float_equal<Scalar>(Vk[0], Vj[0], 1e-8))
        {
            Scalar t = 0.01;
            overlay.bary_coords[vj] = (1. - t) * Vi + t * Vj;
        }
    }
}

void SimplificationOverlayMesh::fix_degenerate_edge_coordinates(OverlayStructure& overlay)
{
    std::vector<int> start_halfedge = find_start_halfedges(overlay);

    // find boundary coordiantes
    int num_overlay_edges = overlay.e2h.size();
    int count = 0;
    for (int Eij = 0; Eij < num_overlay_edges; ++Eij)
    {
        int Hij = overlay.e2h[Eij];
        while (is_degenerate_edge_coordinates(start_halfedge[Hij], overlay))
        {
            fix_degenerate_edge_coordinates(start_halfedge[Hij], overlay);
            ++count;
        }
    }
    spdlog::info("{} edges made uniform", count);
}

void SimplificationOverlayMesh::triangulate_with_tutte()
{
    // first ensure there are no degenerate edges
    fix_degenerate_edge_coordinates(orig);

    std::vector<bool> remesh_face(orig.m.n_faces(), false);
    int num_faces = f2h.size();
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f)) continue;
        bool success = triangulate_face(f);
        if (!success) remesh_face[orig.face_parent[f]] = true;
    }

    remove_parallel_edges(simp); // need to ensure no parallel edges
    remove_parallel_edges(orig); // need to ensure no parallel edges

    // check for degenerate triangles
    num_faces = f2h.size();
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f)) continue;
        if (compute_polygon_face_area(f) < 0.)
        {
            remesh_face[orig.face_parent[f]] = true;
        }
    }

    find_tutte_coordinates(orig, remesh_face); 
}


// --------------
// Simplification
// --------------

void SimplificationOverlayMesh::remove_excess_refinement()
{
    assert(is_valid());
    if (keep_all_refinement)
    {
        collect_garbage();
        return;
    }

    // ensure triangle mesh first for edge collapse
    triangulate();
    assert(is_valid());

    // collapse any possible halfedges to straighten the paths
    int edges_collapsed = 0;
    do
    {
        edges_collapsed = 0;
        break; // TODO skipping
        for (int hij = 0; hij < n.size(); ++hij)
        {
            if (is_halfedge_deleted(hij)) continue;
            if (!is_collapse_edge_valid(hij)) continue;

            // TODO: need to prevent geometric collapse. The link condition should help
            // but we should add an explicit check. This might limit how much refinement we
            // can do.
            //if (!is_collapse_link_valid(hij)) continue;

            if (!is_straightening_collapse(hij)) continue;
            collapse_edge(hij);
            ++edges_collapsed;
        }
        spdlog::info("{} edges collapsed", edges_collapsed);
    } while (edges_collapsed > 0);

    // remove any possible halfedges

    int edges_removed = 0;
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;
        if (!is_edge_removable(hij)) continue;
        if (!is_join_edge_faces_valid(hij)) continue;
        join_edge_faces(hij);
        ++edges_removed;
    }
    spdlog::info("{} edges removed", edges_removed);

    // remove any degree one vertices (always removable)
    int d1_vertices_removed = 0;
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;
        int vj = to[hij];
        while (is_vertex_degree_one(vj))
        {
            ++d1_vertices_removed;
            vj = remove_degree_one_vertex(vj);
        }
    }
    spdlog::info("{} degree one vertices removed", d1_vertices_removed);

    // remove any possible degree two vertices
    int d2_vertices_removed = 0;
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (skip_degree_two) break;
        if (is_halfedge_deleted(hij)) continue;
        int vj = to[hij];
        int vi = to[opp[hij]];

        // check if interior degree two vertex
        if (!is_vertex_degree_two(vj)) continue;
        if (orig.vertex_type[vj] == VertexType::VV) continue;
        if (simp.vertex_type[vj] == VertexType::VV) continue;
        if ((orig.vertex_type[vj] == VertexType::VE) && (orig.edge_type[h2e[hij]] != EdgeType::EE)) continue;

        if (simp.vertex_type[vj] == VertexType::VE)
        {
            // only collapse simplified vertices along simplified edges
            if (simp.edge_type[h2e[hij]] != EdgeType::EE) continue;

            // don't move simplified vertices along or into original edge
            if (orig.edge_type[h2e[hij]] == EdgeType::EE) continue;
            if (orig.vertex_type[vi] != VertexType::VF) continue;
        }

        if (!is_join_halfedge_endpoints_valid(hij)) continue;
        join_halfedge_endpoints(hij);
        ++d2_vertices_removed;
    }
    spdlog::info("{} degree two vertices removed", d2_vertices_removed);
    assert(is_valid());

    // restore triangulation
    triangulate_with_tutte();

    // reindex to compress
    collect_garbage();

    // check if degenerate
    std::vector<Scalar> face_areas = compute_face_areas();
    spdlog::info("minimum face area after triangulation is {}", vector_min(face_areas));
}

/*
void SimplificationOverlayMesh::simplify_original_face(int halfedge_index)
{
    if (keep_all_refinement) return;
}
*/

bool SimplificationOverlayMesh::is_edge_removable(int halfedge_index)
{
    int hij = halfedge_index;
    if (is_halfedge_deleted(hij)) return false;

    if (orig.edge_type[h2e[hij]] != EdgeType::EF) return false;
    if (simp.edge_type[h2e[hij]] != EdgeType::EF) return false;

    return true;
}

bool SimplificationOverlayMesh::is_vertex_removable(int vertex_index)
{
    if (is_vertex_deleted(vertex_index)) return false;

    if (orig.vertex_type[vertex_index] != VertexType::VF) return false;
    if (simp.vertex_type[vertex_index] != VertexType::VF) return false;

    return true;
}

bool SimplificationOverlayMesh::is_halfedge_collapsable(int halfedge_index)
{
    int hij = halfedge_index;
    int vj = to[hij];
    if (is_halfedge_deleted(hij)) return false;

    // can never collapse original or simplified vertex
    if (orig.vertex_type[vj] == VertexType::VV) return false;
    if (simp.vertex_type[vj] == VertexType::VV) return false;

    // cannot collapse edge vertex if the collapsed halfedge is not an original edge
    int hjk = n[hij];
    int hlj = prev[opp[hij]];
    if (orig.vertex_type[vj] == VertexType::VE)
    {
        if (simp.vertex_type[vj] != VertexType::VF) return false;
        if (orig.edge_type[h2e[hij]] != EdgeType::EE) return false;
        if (!is_edge_removable(hlj)) return false;
        if (!is_edge_removable(hjk)) return false;
    }

    // likewise for simplified mesh
    if (simp.vertex_type[vj] == VertexType::VE)
    {
        if (orig.vertex_type[vj] != VertexType::VF) return false;
        if (simp.edge_type[h2e[hij]] != EdgeType::EE) return false;
        if (!is_edge_removable(hlj)) return false;
        if (!is_edge_removable(hjk)) return false;
    }

    return true;
}

bool SimplificationOverlayMesh::is_straightening_collapse(int halfedge_index)
{
    if (is_halfedge_deleted(halfedge_index)) return false;
    int hij = halfedge_index;
    int eij = h2e[hij];
    int vi = to[opp[hij]];
    int vj = to[hij];
    if (simp.edge_type[eij] != EdgeType::EE) return false;

    // can never collapse across a vertex
    if (orig.vertex_type[vj] == VertexType::VV) return false;
    if (simp.vertex_type[vj] == VertexType::VV) return false;

    // can only collapse joint intersecting original edge if contained entirely
    int hjk = get_next_overlay_edge(hij, simp);
    int vk = to[hjk];
    if (orig.vertex_type[vj] == VertexType::VE)
    {
        if (orig.edge_type[h2e[hij]] != EdgeType::EE) return false;
        if (orig.edge_type[h2e[hjk]] != EdgeType::EE) return false;
    }
    // otherwise, must be entirely in face
    else
    {
        if (orig.edge_type[h2e[hij]] != EdgeType::EF) return false;
        if (orig.edge_type[h2e[hjk]] != EdgeType::EF) return false;

        // cannot collapse into edge
        if ((orig.vertex_type[vi] == VertexType::VV) && (orig.vertex_type[vk] == VertexType::VV)) return false;
        else if ((orig.vertex_type[vi] == VertexType::VE) && (orig.vertex_type[vk] == VertexType::VE))
        {
            if (orig.vertex_parent[vi] == orig.vertex_parent[vk]) return false;
        }
        else if ((orig.vertex_type[vi] == VertexType::VV) && (orig.vertex_type[vk] == VertexType::VE))
        {
            int Vi = orig.vertex_parent[vi];
            int Eij = orig.vertex_parent[vk];
            int Hij = orig.e2h[Eij];
            if (orig.m.to[Hij] == Vi) return false;
            if (orig.m.to[orig.m.opp[Hij]] == Vi) return false;
        }
        else if ((orig.vertex_type[vi] == VertexType::VE) && (orig.vertex_type[vk] == VertexType::VV))
        {
            int Vk = orig.vertex_parent[vk];
            int Eij = orig.vertex_parent[vi];
            int Hij = orig.e2h[Eij];
            if (orig.m.to[Hij] == Vk) return false;
            if (orig.m.to[orig.m.opp[Hij]] == Vk) return false;
        }
    }

    return true;
}

// ----------------------------
// Embedded Geometry Generation
// ----------------------------

Eigen::Vector3d SimplificationOverlayMesh::generate_overlay_vertex(
    int vi,
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const OverlayStructure& overlay) const
{
    const auto& m = overlay.m;
    if (is_vertex_deleted(vi)) return Eigen::Vector3d();

    if (overlay.vertex_type[vi] == VertexType::VV) 
    {
        int wi = overlay.vertex_parent[vi];
        return V.row(vtx_reindex[m.v_rep[wi]]);
    }
    else if (overlay.vertex_type[vi] == VertexType::VE) 
    {
        int eij = overlay.vertex_parent[vi];
        int hij = overlay.e2h[eij];
        int wi = m.to[m.opp[hij]];
        int wj = m.to[hij];
        auto Wi = V.row(vtx_reindex[m.v_rep[wi]]);
        auto Wj = V.row(vtx_reindex[m.v_rep[wj]]);
        double s = (double)(overlay.bary_coords[vi][0]);
        return s * Wi + (1. - s) * Wj;
    }
    else if (overlay.vertex_type[vi] == VertexType::VF) 
    {
        int fijk = overlay.vertex_parent[vi];
        int hij = m.h[fijk];
        int wj = m.to[hij];
        int wk = m.to[m.n[hij]];
        int wi = m.to[m.n[m.n[hij]]];
        auto Wi = V.row(vtx_reindex[m.v_rep[wi]]);
        auto Wj = V.row(vtx_reindex[m.v_rep[wj]]);
        auto Wk = V.row(vtx_reindex[m.v_rep[wk]]);
        double s = (double)(overlay.bary_coords[vi][0]);
        double t = (double)(overlay.bary_coords[vi][1]);
        return s * Wi + t * Wj + (1. - s - t) * Wk;
    }
    else
    {
        spdlog::warn("unexpected vertex type");
        return Eigen::Vector3d();
    }
}

std::vector<int> SimplificationOverlayMesh::get_face_parent(const OverlayStructure& overlay) const
{
    auto [f_new2old, f_old2new] = get_face_reductions();
    return vector_compose(overlay.face_parent, f_new2old);
}

Eigen::MatrixXd SimplificationOverlayMesh::generate_overlay_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const OverlayStructure& overlay) const
{

    int num_vertices = count_vertices();
    Eigen::MatrixXd V_o(num_vertices, 3);
    int i = 0;
    for (int vi = 0; vi < out.size(); ++vi)
    {
        if (is_vertex_deleted(vi)) continue;
        V_o.row(i) = generate_overlay_vertex(vi, V, vtx_reindex, overlay);
        i++;
    }

    return V_o;
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> SimplificationOverlayMesh::generate_overlay_faces() const
{
    return generate_halfedge_faces();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> SimplificationOverlayMesh::generate_simplified_vf_mesh() const
{
    Eigen::MatrixXi F_simp, F_halfedge;
    std::tie(F_simp, F_halfedge) = generate_connectivity_faces(simp.m.n, simp.m.h, simp.m.to);
    int num_vertices = simp.m.n_vertices();
    Eigen::MatrixXd V_simp(num_vertices, 3);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        int Vi = simp.vertex_index[vi];
        V_simp.row(vi) = generate_overlay_vertex(Vi, V_orig, vtx_reindex_orig, orig);
    }

    return std::make_tuple(V_simp, F_simp);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
SimplificationOverlayMesh::generate_simplified_edges(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    int num_edges = e2h.size();
    std::vector<std::array<int, 2>> edges;
    std::vector<int> parent;
    edges.reserve(count_edges());
    for (int eij = 0; eij < num_edges; ++eij)
    {
        // only add edges contained in simpified mesh edges
        if (is_edge_deleted(eij)) continue;
        if (simp.edge_type[eij] == EdgeType::EF) continue;

        int hij = e2h[eij];
        int vi = v_old2new[to[opp[hij]]];
        int vj = v_old2new[to[hij]];
        edges.push_back({vi, vj});
        parent.push_back(simp.halfedge_parent[e2h[eij]]); // TODO: get edge in original mesh
    }
    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    auto [V_s, edges_s] = remove_unreferenced_vertices(V_o, edges);
    Eigen::MatrixXi E_s(edges_s.size(), 2);
    for (int i = 0; i < edges_s.size(); ++i)
    {
        E_s(i, 0) = edges_s[i][0];
        E_s(i, 1) = edges_s[i][1];
    }
    
    return std::make_tuple(V_s, E_s);
}

// ---------------------
// Regularity Conditions
// ---------------------

bool SimplificationOverlayMesh::is_regular_face(int face_index) const
{
    if (is_face_deleted(face_index)) return false;
    int hij = f2h[face_index];
    int hkl = hij;
    int num_embedded_edges = 0;

    // count number of embedded edges
    do
    {
        if (simp.edge_type[h2e[hkl]] == EdgeType::EE)
        {
            ++num_embedded_edges;
        }
        hkl = n[hkl];
    } while (hij != hkl);

    return (num_embedded_edges < 2);
}

bool SimplificationOverlayMesh::is_regular_edge(int edge_index) const
{
    if (is_edge_deleted(edge_index)) return false;

    // if the edge is embedded, its always regular
    if (simp.edge_type[edge_index] == EdgeType::EE) return true;

    // if not, it must have at least one non-embedded vertex
    int hij = e2h[edge_index];
    int hji = opp[hij];
    if (simp.vertex_type[to[hij]] == VertexType::VF) return true;
    if (simp.vertex_type[to[hji]] == VertexType::VF) return true;

    return false;
}

bool SimplificationOverlayMesh::is_parallel_edge(int edge_index, const OverlayStructure& overlay) const
{
    if (is_edge_deleted(edge_index)) return false;

    // an embedded edge cannot be parallel
    if (overlay.edge_type[edge_index] == EdgeType::EE) return false;

    // not parallel if one of the edge endpoints is embedded in a face interior
    int hij = e2h[edge_index];
    int hji = opp[hij];
    int vi = to[hji];
    int vj = to[hij];
    if (overlay.vertex_type[vi] == VertexType::VF) return false;
    if (overlay.vertex_type[vj] == VertexType::VF) return false;

    // parallel if both endpoints are embedded vertices (assuming edge is not embedded)
    if ((overlay.vertex_type[vi] == VertexType::VV) && (overlay.vertex_type[vj] == VertexType::VV)) return true;

    // complex case: first endpoint is embedded in an edge
    if (overlay.vertex_type[vi] == VertexType::VE)
    {
        // get index of edge containing the first endpoint 
        int Ekl = overlay.vertex_parent[vi];
        int Hkl = overlay.e2h[Ekl];

        // get endpoints of the embedded mesh edge
        int Vk = overlay.m.to[overlay.m.opp[Hkl]];
        int Vl = overlay.m.to[Hkl];

        // parallel if the other endpoint is embedded in the same edge
        if ((overlay.vertex_type[vj] == VertexType::VE) && (overlay.vertex_parent[vj] == Ekl)) return true;

        // parallel if the other endpoint is one of the edge endpoints
        if ((overlay.vertex_type[vj] == VertexType::VV) && (overlay.vertex_parent[vj] == Vk)) return true;
        if ((overlay.vertex_type[vj] == VertexType::VV) && (overlay.vertex_parent[vj] == Vl)) return true;
    }

    // equivalent check for other endpoint
    if (overlay.vertex_type[vj] == VertexType::VE)
    {
        int Ekl = overlay.vertex_parent[vj];
        int Hkl = overlay.e2h[Ekl];
        int Vk = overlay.m.to[overlay.m.opp[Hkl]];
        int Vl = overlay.m.to[Hkl];
        if ((overlay.vertex_type[vi] == VertexType::VE) && (overlay.vertex_parent[vi] == Ekl)) return true;
        if ((overlay.vertex_type[vi] == VertexType::VV) && (overlay.vertex_parent[vi] == Vk)) return true;
        if ((overlay.vertex_type[vi] == VertexType::VV) && (overlay.vertex_parent[vi] == Vl)) return true;
    }

    return false;
}

bool SimplificationOverlayMesh::is_parallel_free(const OverlayStructure& overlay)
{
    // ensure all edges have at most one embedded vertex
    int num_edges = e2h.size();
    for (int e = 0; e < num_edges; ++e)
    {
        if (is_edge_deleted(e)) continue;
        if (is_parallel_edge(e, overlay))
        {
            spdlog::warn("{} is parallel", e);
            return false;
        }
    }

    return true;
}

bool SimplificationOverlayMesh::is_three_connected(const OverlayStructure& overlay)
{
    // ensure all edges have at most one embedded vertex
    int num_edges = e2h.size();
    for (int e = 0; e < num_edges; ++e)
    {
        if (is_edge_deleted(e)) continue;
        int h = e2h[e];
        int vi = to[h];
        int vj = to[opp[h]];
        if (overlay.edge_type[e] == EdgeType::EE) continue;
        if (overlay.vertex_type[vi] == VertexType::VF) continue;
        if (overlay.vertex_type[vj] == VertexType::VF) continue;
        spdlog::warn("{} connects two edges", e);
        return false;
    }

    return true;
}

void SimplificationOverlayMesh::make_edge_regular()
{
    // ensure all edges have at most one embedded vertex
    int num_edges = e2h.size();
    int num_refined_edges = 0;
    for (int e = 0; e < num_edges; ++e)
    {
        if (is_edge_deleted(e)) continue;

        // nothing to do if already regular
        if (is_regular_edge(e)) continue;

        // split edge (while maintaining triangulation) if not
        int h = e2h[e];
        int hik = split_edge(e2h[e]);
        int hjk = opp[n[hik]];
        split_face(hik, n[n[hik]]);
        split_face(hjk, n[n[hjk]]);

        // increment count if split
        num_refined_edges++;
    }
    spdlog::info("{} refined edges", num_refined_edges);
}

void SimplificationOverlayMesh::make_face_regular()
{
    // ensure all faces have at most one embedded edge
    int num_faces = f2h.size();
    int num_refined_faces = 0;
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f)) continue;

        // nothing to do if already regular
        if (is_regular_face(f)) continue;

        // simply add center to face to make regular
        add_center_to_face(f);

        // increment count if split
        num_refined_faces++;
    }
    spdlog::info("{} refined faces", num_refined_faces);
}

void SimplificationOverlayMesh::make_regular()
{
    // WARNING: order matters
    make_edge_regular();
    make_face_regular();
}

void SimplificationOverlayMesh::make_adjacent_regular(const std::vector<int>& halfedges)
{
    // get edges adjacent to the interior vertices of the path
    std::vector<int> adjacent_edges = {};
    int num_halfedges = halfedges.size();
    adjacent_edges.reserve(6 * num_halfedges);
    for (int i = 0; i < num_halfedges; ++i)
    {
        int hij = halfedges[i];
        if (simp.vertex_type[to[opp[hij]]] == VertexType::VV) continue;

        // circulate around vertex
        int h = hij;
        do
        {
            adjacent_edges.push_back(h2e[hij]);
            hij = n[opp[hij]];
        } while (hij != h);
    }

    // ensure all edges have at most one embedded vertex
    int num_refined_edges = 0;
    for (int e : adjacent_edges)
    {
        if (is_edge_deleted(e)) continue;

        // split irregular edges
        if (is_regular_edge(e)) continue;
        int hik = split_edge(e2h[e]);
        int hjk = opp[n[hik]];
        split_face(hik, n[n[hik]]);
        split_face(hjk, n[n[hjk]]);
        num_refined_edges++;
    }
    spdlog::trace("{} refined edges", num_refined_edges);
}

void SimplificationOverlayMesh::make_oriented_adjacent_regular(const std::vector<int>& halfedges)
{
    // get edges adjacent to the interior vertices of the path on the same side as the halfedge
    std::vector<int> adjacent_edges = {};
    int num_halfedges = halfedges.size();
    adjacent_edges.reserve(6 * num_halfedges);
    for (int i = 0; i < num_halfedges - 1; ++i)
    {
        // circulate clockwise until the next edge in the path is found
        int hij = halfedges[i];
        int hjk = halfedges[i + 1];
        do
        {
            adjacent_edges.push_back(h2e[hij]);
            hij = opp[n[hij]];
        } while (hij != opp[hjk]);
    }

    // ensure all edges have at most one embedded vertex
    int num_refined_edges = 0;
    for (int e : adjacent_edges)
    {
        if (is_regular_edge(e)) continue;
        int hik = split_edge(e2h[e]);
        int hjk = opp[n[hik]];
        split_face(hik, n[n[hik]]);
        split_face(hjk, n[n[hjk]]);
        num_refined_edges++;
    }
    spdlog::trace("{} refined edges", num_refined_edges);
}


void SimplificationOverlayMesh::remove_parallel_edges(const OverlayStructure& overlay)
{
    // ensure all edges have at most one embedded vertex
    int num_refined_edges = 0;
    do
    {
        num_refined_edges = 0;
        int num_edges = e2h.size();
        for (int e = 0; e < num_edges; ++e)
        {
            if (is_edge_deleted(e)) continue;

            // check if edge is parallel
            if (!is_parallel_edge(e, overlay)) continue;

            // split parallel edges
            int h = e2h[e];
            int hik = split_edge(e2h[e]);
            int hjk = opp[n[hik]];
            split_face(hik, n[n[hik]]);
            split_face(hjk, n[n[hjk]]);
            num_refined_edges++;
        }
        spdlog::info("{} refined edges to remove parallel edges", num_refined_edges);
    }
    while (num_refined_edges > 0);
    assert(is_parallel_free(overlay));
}


void SimplificationOverlayMesh::make_three_connected(const OverlayStructure& overlay)
{
    // ensure all edges have at most one embedded vertex
    int num_edges = e2h.size();
    int num_refined_edges = 0;
    for (int e = 0; e < num_edges; ++e)
    {
        if (is_edge_deleted(e)) continue;
        int h = e2h[e];
        int vi = to[h];
        int vj = to[opp[h]];

        // check if interior edge spans the boundary
        if (overlay.edge_type[e] == EdgeType::EE) continue;
        if (overlay.vertex_type[vi] == VertexType::VF) continue;
        if (overlay.vertex_type[vj] == VertexType::VF) continue;

        // split edge if necessary 
        int hik = split_edge(e2h[e]);
        int hjk = opp[n[hik]];
        split_face(hik, n[n[hik]]);
        split_face(hjk, n[n[hjk]]);
        num_refined_edges++;
    }
    spdlog::info("{} refined edges to make three connected", num_refined_edges);
    if (!is_three_connected(overlay)) spdlog::warn("not three connected");
    assert(is_three_connected(overlay));
}

std::vector<Scalar> SimplificationOverlayMesh::compute_corner_angles() const
{
    int num_halfedges = n.size();
    std::vector<Scalar> corner_angles(num_halfedges, 0.);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;
        Vector2 vi = project_halfedge_tip(prev[hij]);
        Vector2 vj = project_halfedge_tip(hij);
        Vector2 vk = project_halfedge_tip(n[hij]);
        corner_angles[hij] = compute_triangle_angle(vj, vk, vi);
    }

    return corner_angles;
}

std::vector<Scalar> SimplificationOverlayMesh::compute_face_areas() const
{
    int num_faces = f2h.size();
    std::vector<Scalar> face_areas;
    face_areas.reserve(num_faces);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        if (is_face_deleted(fijk)) continue;
        int hij = f2h[fijk];
        Vector2 vi = project_halfedge_tip(prev[hij]);
        Vector2 vj = project_halfedge_tip(hij);
        Vector2 vk = project_halfedge_tip(n[hij]);
        face_areas.push_back(compute_triangle_area(vj, vk, vi));
    }

    return face_areas;
}

Scalar SimplificationOverlayMesh::compute_polygon_face_area(int face_index) const
{
    int hij = f2h[face_index];
    int degree = face_degree(face_index);

    std::vector<int> face_halfedges(degree);
    std::vector<Vector2> face_vertices(degree);
    int h = hij;
    for (int i = 0; i < degree; ++i)
    {
        face_halfedges[i] = h;
        face_vertices[i] = project_halfedge_tip(h);
        h = n[h];
    }

    Scalar area = 0.;
    for (int i = 0; i < degree - 2; ++i)
    {
        int vi = i;
        int vj = (i + 1) % degree;
        int vk = (i + 2) % degree;
        area += compute_triangle_area(face_vertices[vi], face_vertices[vj], face_vertices[vk]);
    }

    return area;
}

// -----------------
// Atomic Operations
// -----------------

int SimplificationOverlayMesh::join_edge_faces(int halfedge_index)
{
    assert(is_edge_removable(halfedge_index));
    int f = HalfedgeMesh::join_edge_faces(halfedge_index);
    join_edge_faces_overlay(f, halfedge_index, orig);
    join_edge_faces_overlay(f, halfedge_index, simp);

    return f;
}

/*
int SimplificationOverlayMesh::collapse_edge(int halfedge_index)
{
    int hij = HalfedgeMesh::collapse_edge(halfedge_index);
    //collapse_edge_overlay(halfedge_index, orig);
    //collapse_edge_overlay(halfedge_index, simp);

    return hij;
}
*/

int SimplificationOverlayMesh::collapse_edge(int halfedge_index)
{
    int hij = halfedge_index;
    int hji = opp[hij];
    assert(face_degree(face[hij]) == 3);
    assert(face_degree(face[hji]) == 3);
    int hjk = n[hij];
    int hlj = prev[hji];
    int hki = n[hjk];
    int hil = prev[hlj];
    int vi = to[hji];
    int vj = to[hij];
    join_halfedge_endpoints(hij);
    join_edge_faces(hjk);
    if (opp[hlj] != hjk)
    {
        join_edge_faces(hlj);
        assert(face_degree(face[hil]) == 3);
        assert(face_degree(face[opp[hil]]) == 3);
        return hki;
    }
    else if (is_edge_removable(hil))
    {
        join_edge_faces(hil);
        return hki;
    }
    else if (is_edge_removable(hki))
    {
        join_edge_faces(hki);
        return opp[hil];
    }
    else
    {
        spdlog::error("could not collapse edge");
        assert(false);
        return -1;
    }
}

bool SimplificationOverlayMesh::triangulate_face(int face_index)
{
    int hij = f2h[face_index];
    int degree = face_degree(face_index);

    if (degree == 3)
    {
        return (compute_polygon_face_area(face_index) > 0.);
        //return true;
    }
    else if (degree < 3)
    {
        spdlog::error("cannot triangulate degree {} face", degree);
        assert(false);
        return false;
    }

    // TODO special case for degree 4 to avoid long computation
    //spdlog::debug("{} split of degree {} face", i - 2, degree);
    //face_index = clip_ear(face_index);

    std::vector<int> face_halfedges(degree);
    std::vector<Vector2> face_vertices(degree);
    int h = hij;
    for (int i = 0; i < degree; ++i)
    {
        face_halfedges[i] = h;
        face_vertices[i] = project_halfedge_tip(h);
        h = n[h];
    }

    // special case for degree 4; just try both options
    if (degree == 4)
    {
        std::array<Scalar, 4> areas;
        for (int i = 0; i < 4; ++i)
        {
            areas[(i + 1) % 4] = compute_triangle_area(
                face_vertices[i],
                face_vertices[(i + 1) % 4],
                face_vertices[(i + 2) % 4]);
        }

        Scalar max_area;
        if (min(areas[0], areas[2]) < min(areas[1], areas[3]))
        {
            split_face(face_halfedges[0], face_halfedges[2]);
            max_area = min(areas[1], areas[3]);
        }
        else
        {
            split_face(face_halfedges[1], face_halfedges[3]);
            max_area = min(areas[0], areas[2]);
        }
        
        if (max_area <= 0.)
        {
            spdlog::debug("degenerate triangle in {}-gon with area {}", degree, max_area);
            spdlog::debug("areas are {}", formatted_vector(areas));

            //polyscope::init();
            //polyscope::registerPointCloud2D("boundary", face_vertices);
            //polyscope::show();
        }
        return (max_area > 0.);
    }

    EarClipper ear_clipper(face_vertices);
    ear_clipper.triangulate();
    //if (degree > 4) ear_clipper.view();

    Scalar min_area = 1e10;
    for (int i = 0; i < degree - 2; ++i)
    {
        auto [vi, vj, vk] = ear_clipper.triangles[i];
        Scalar area = compute_triangle_area(face_vertices[vi], face_vertices[vj], face_vertices[vk]);
        min_area = min(min_area, area);
    }
    if (min_area <= 0.)
    {
        spdlog::debug("degenerate triangle in {}-gon with area {}", degree, min_area);
        bool view_degenerate_polygons = false;
        if (view_degenerate_polygons) ear_clipper.view();
    }

    int num_edges = ear_clipper.edges.size();
    if (num_edges != (degree - 3)) ear_clipper.view();
    assert(num_edges == (degree - 3));
    for (int i = 0; i < num_edges; ++i)
    {
        int vi = ear_clipper.edges[i][0];
        int vj = ear_clipper.edges[i][1];
        int h = face_halfedges[vi];
        int g = face_halfedges[vj];
        split_face(h, g);

        // update halfedge to be in same face
        // WARNING: depends on ordering of edge vertices
        face_halfedges[vj] = opp[n[g]];
    }

    // TODO: replace area with angles
    return (min_area > 0.);
}


int SimplificationOverlayMesh::add_center_to_face(int face_index)
{
    int vi = HalfedgeMesh::add_center_to_face(face_index);
    add_center_to_face_overlay(vi, face_index, orig);
    add_center_to_face_overlay(vi, face_index, simp);

    return vi;
}

void SimplificationOverlayMesh::remove_path(int start_halfedge, OverlayStructure& overlay)
{
    // set edge and tip vertices to face interior until reach end vertex
    int hij = start_halfedge;
    while (overlay.vertex_type[to[hij]] != VertexType::VV)
    {
        assert(overlay.vertex_type[to[hij]] == VertexType::VE);
        int hjk = get_next_overlay_edge(hij, overlay);
        overlay.edge_type[h2e[hij]] = EdgeType::EF;
        overlay.vertex_type[to[hij]] = VertexType::VF;
        hij = hjk;
    }

    // set last edge type
    overlay.edge_type[h2e[hij]] = EdgeType::EF;
}

void SimplificationOverlayMesh::view() const
{
    add_mesh_to_viewer(V_orig, vtx_reindex_orig);
    add_original_to_viewer(V_orig, vtx_reindex_orig);
    add_simplified_vertices_to_viewer(V_orig, vtx_reindex_orig);
    add_simplified_edges_to_viewer(V_orig, vtx_reindex_orig);

    //add_simplified_to_viewer(V, vtx_reindex);
#ifdef ENABLE_VISUALIZATION
    polyscope::show();
#endif
}


int SimplificationOverlayMesh::create_face()
{
    int f = HalfedgeMesh::create_face();
    orig.face_parent.push_back(-1);
    simp.face_parent.push_back(-1);

    return f;
}

int SimplificationOverlayMesh::create_edge()
{
    int e = HalfedgeMesh::create_edge();
    orig.halfedge_parent.push_back(-1);
    orig.halfedge_parent.push_back(-1);
    orig.edge_type.push_back(EdgeType::EF);

    simp.halfedge_parent.push_back(-1);
    simp.halfedge_parent.push_back(-1);
    simp.edge_type.push_back(EdgeType::EF);

    return e;
}

int SimplificationOverlayMesh::create_vertex()
{
    int v = HalfedgeMesh::create_vertex();
    orig.vertex_parent.push_back(-1);
    orig.vertex_type.push_back(VertexType::VF);
    orig.bary_coords.push_back({0., 0.});

    simp.vertex_parent.push_back(-1);
    simp.vertex_type.push_back(VertexType::VF);
    simp.bary_coords.push_back({0., 0.});

    return v;
}

int SimplificationOverlayMesh::split_edge(int halfedge_index)
{
    int edge_index = h2e[halfedge_index];

    spdlog::debug("splitting edge {} = ({},{})", edge_index, to[opp[halfedge_index]], to[halfedge_index]);
    int hij = HalfedgeMesh::split_edge(halfedge_index);

    // TODO: need to be careful with face parent; probably overwritten
    split_edge_overlay(hij, halfedge_index, edge_index, orig);
    split_edge_overlay(hij, halfedge_index, edge_index, simp);

    return hij;
}

int SimplificationOverlayMesh::split_face(int h, int g)
{
    int face_index = face[h];

    spdlog::debug("splitting face {} between {} and {}", face[h], to[h], to[g]);
    int h_split = HalfedgeMesh::split_face(h, g);
    int e = h2e[h_split];

    split_face_overlay(e, face_index, orig);
    split_face_overlay(e, face_index, simp);

    return h_split;
}

bool SimplificationOverlayMesh::is_valid_overlay(const OverlayStructure& overlay) const
{
    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // check if face halfedge label is consistent
        int hji = opp[hij];
        int eij = h2e[hij];
        int f = face[hij];
        int vj = to[hij];
        if (overlay.edge_type[eij] == EdgeType::EF)
        {
            int F = overlay.face_parent[f];
            if (overlay.halfedge_parent[hij] != F) return errmess("interior halfedge parent inconsistent");
            if (overlay.halfedge_parent[hji] != F) return errmess("interior halfedge parent inconsistent");
        }
        else if (overlay.edge_type[eij] == EdgeType::EE)
        {
            int F = overlay.face_parent[f];
            int Hij = overlay.halfedge_parent[hij];
            int Hji = overlay.halfedge_parent[hji];
            if (overlay.m.opp[Hij] != Hji) return errmess("halfedge opp inconsistent with overlay mesh");
            if (overlay.m.f[Hij] != F) return errmess("halfedge face inconsistent with overlay mesh");

            // check if edge halfedge tip label is consistent
            if (overlay.vertex_type[vj] == VertexType::VE)
            {
                int Hij = overlay.halfedge_parent[hij];
                if (overlay.vertex_parent[vj] != overlay.h2e[Hij]) return errmess("edge vertex parent inconsistent");
            }
            else if (overlay.vertex_type[vj] == VertexType::VV)
            {
                int Hij = overlay.halfedge_parent[hij];
                int Vj = overlay.vertex_parent[vj];
                if (overlay.m.to[Hij] != Vj) return errmess("halfedge tip inconsistent with overlay mesh");
            }
            else
            {
                assert(false);
                return errmess("edge edge tip should not be in a face");
            }
        }
    }

    for (int vi = 0; vi < out.size(); ++vi)
    {
        if (is_vertex_deleted(vi)) continue;

        // check if face vertex label is consistent
        if (overlay.vertex_type[vi] == VertexType::VF)
        {
            int f = face[out[vi]];
            if (overlay.vertex_parent[vi] != overlay.face_parent[f]) return errmess("face vertex parent inconsistent");
        }
    }

    return true;
}

bool SimplificationOverlayMesh::is_valid_edge_network() const
{
    if (!HalfedgeMesh::is_valid()) return false;
    if (!is_valid_overlay(orig)) return errmess("original overlay not valid");

    int simp_vertex_count = 0;
    for (int vi = 0; vi < out.size(); ++vi)
    {
        if (is_vertex_deleted(vi)) continue;
        if (simp.vertex_type[vi] == VertexType::VV) ++simp_vertex_count;
    }

    std::vector<int> start_halfedge = find_start_halfedges(simp);
    int num_vertices = simp.m.out.size();
    if (simp_vertex_count != num_vertices) return errmess("isolated vertex found");
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        int Hij = simp.m.out[vi];
        int Hik = Hij;
        do
        {
            int Hil = simp.m.n[simp.m.opp[Hik]];

            // check consistency on mesh surface
            int hik = start_halfedge[Hik];
            int hil = start_halfedge[Hil];
            if (get_clockwise_overlay_edge(hik, simp) != hil) return errmess("edge orientation not consistent");

            // iterate
            Hik = Hil;
        } while (Hik != Hij);
    }

    for (int hij = 0; hij < n.size(); ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;

        // check if face halfedge label is consistent
        int hji = opp[hij];
        int eij = h2e[hij];
        int vj = to[hij];
        if (simp.edge_type[eij] == EdgeType::EE)
        {
            int Hij = simp.halfedge_parent[hij];
            int Hji = simp.halfedge_parent[hji];
            if (simp.m.opp[Hij] != Hji) return errmess("halfedge opp inconsistent with overlay mesh");


            // check if edge halfedge tip label is consistent
            if (simp.vertex_type[vj] == VertexType::VE)
            {
                int Hij = simp.halfedge_parent[hij];
                if (simp.vertex_parent[vj] != simp.h2e[Hij]) return errmess("edge vertex parent inconsistent");

                int hjk = get_next_overlay_edge(hij, simp);
                if (simp.halfedge_parent[hjk] != Hij) return errmess("next overlay edge is inconsistent");
            }
            else if (simp.vertex_type[vj] == VertexType::VV)
            {
                int Hij = simp.halfedge_parent[hij];
                int Vj = simp.vertex_parent[vj];
                if (simp.m.to[Hij] != Vj) return errmess("halfedge tip inconsistent with overlay mesh");
            }
            else
            {
                assert(false);
                return errmess("edge edge tip should not be in a face");
            }
        }
    }

    return true;
}

bool SimplificationOverlayMesh::is_valid() const
{
    if (!HalfedgeMesh::is_valid()) return false;
    if (!is_valid_overlay(orig)) return errmess("original overlay not valid");
    if (!is_valid_overlay(simp)) return errmess("simplified overlay not valid");

    // TODO: make member function of OverlayStructure
    //if (face_parent.size() != f2h.size()) return errmess("face parent size is inconsistent");
    //if (edge_parent.size() != e2h.size()) return errmess("edge parent size is inconsistent");
    //if (edge_type.size() != e2h.size()) return errmess("edge type size is inconsistent");
    //if (vertex_parent.size() != out.size()) return errmess("vertex parent size is inconsistent");
    //if (vertex_type.size() != out.size()) return errmess("vertex type size is inconsistent");
    //if (bary_coords.size() != out.size()) return errmess("vertex type size is inconsistent");

    return true;
}

bool SimplificationOverlayMesh::is_valid_simplified_vertex_index(int vertex_index) const
{
    return (vertex_index >= 0) && (vertex_index < simp.m.n_vertices());
}
bool SimplificationOverlayMesh::is_valid_simplified_edge_index(int edge_index) const
{
    return (edge_index >= 0) && (edge_index < simp.m.n_edges());
}
bool SimplificationOverlayMesh::is_valid_simplified_face_index(int face_index) const
{
    return (face_index >= 0) && (face_index < simp.m.n_faces());
}

bool SimplificationOverlayMesh::is_valid_original_vertex_index(int vertex_index) const
{
    return (vertex_index >= 0) && (vertex_index < orig.m.n_vertices());
}
bool SimplificationOverlayMesh::is_valid_original_edge_index(int edge_index) const
{
    return (edge_index >= 0) && (edge_index < orig.m.n_edges());
}
bool SimplificationOverlayMesh::is_valid_original_face_index(int face_index) const
{
    return (face_index >= 0) && (face_index < orig.m.n_faces());
}
bool SimplificationOverlayMesh::is_valid_vertex_index(int vertex_index) const
{
    return (vertex_index >= 0) && (vertex_index < out.size());
}
bool SimplificationOverlayMesh::is_valid_halfedge_index(int halfedge_index) const
{
    return (halfedge_index >= 0) && (halfedge_index < n.size());
}

int SimplificationOverlayMesh::enumerate_vertex_type(const VertexType& vertex_type) const
{
    switch (vertex_type) {
        case VertexType::VV:
            return 0;
        case VertexType::VE:
            return 1;
        case VertexType::VF:
            return 2;
    }

    return -1;
}

std::vector<int> SimplificationOverlayMesh::enumerate_vertex_type(const std::vector<VertexType>& vertex_type) const
{
    int n = vertex_type.size();
    std::vector<int> vertex_enum(n);
    for (int i = 0; i < n; ++i)
    {
        vertex_enum[i] = enumerate_vertex_type(vertex_type[i]);
    }

    return vertex_enum;
}

SimplificationOverlayMesh::BaryCoords
SimplificationOverlayMesh::OverlayStructure::vertex_edge_coordinates(
    int vertex_index,
    int edge_index) const
{
    int hij = e2h[edge_index];
    int vi = m.to[m.opp[hij]];
    int vj = m.to[hij];
    if (vertex_type[vertex_index] == VertexType::VV)
    {
        if (vertex_parent[vertex_index] == vi) return {1., 0.};
        if (vertex_parent[vertex_index] == vj) return {0., 0.};
    }
    else if (vertex_type[vertex_index] == VertexType::VE)
    {
        assert (vertex_parent[vertex_index] == edge_index);
        return bary_coords[vertex_index];
    }

    spdlog::error("vertex not adjacent to edge");
    assert(false);
    return {0., 0.};
}


SimplificationOverlayMesh::BaryCoords
SimplificationOverlayMesh::OverlayStructure::vertex_face_coordinates(
    int vertex_index,
    int face_index) const
{
    int hij = m.h[face_index];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];
    if (vertex_type[vertex_index] == VertexType::VV)
    {
        if (vertex_parent[vertex_index] == vi) return {1., 0.};
        if (vertex_parent[vertex_index] == vj) return {0., 1.};
        if (vertex_parent[vertex_index] == vk) return {0., 0.};
    }
    else if (vertex_type[vertex_index] == VertexType::VE)
    {
        int eij = vertex_parent[vertex_index];
        BaryCoords edge_coords = bary_coords[vertex_index];
        return convert_edge_to_face_coordinates(eij, face_index, edge_coords);
    }
    else if (vertex_type[vertex_index] == VertexType::VF)
    {
        assert (vertex_parent[vertex_index] == face_index);
        return bary_coords[vertex_index];
    }

    spdlog::error("vertex not adjacent to face");
    assert(false);
    return {0., 0.};
}

SimplificationOverlayMesh::BaryCoords
SimplificationOverlayMesh::OverlayStructure::convert_halfedge_to_face_coordinates(
    int halfedge_index,
    int face_index,
    BaryCoords edge_coords) const
{
    int hij = m.h[face_index];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    Scalar s = edge_coords[0]; // second coord superfluous
    Scalar t = 1. - s;
    int h = halfedge_index;
    if (halfedge_index == hij)
    {
        if (h == e2h[h2e[h]]) return {s, t};
        else return {t, s};
    }
    else if (halfedge_index == hjk)
    {
        if (h == e2h[h2e[h]]) return {0., s};
        else return {0., t};
    }
    else if (halfedge_index == hki)
    {
        if (h == e2h[h2e[h]]) return {t, 0.};
        else return {s, 0.};
    }

    spdlog::error("halfedge not adjacent to face");
    assert(false);
    return {0., 0.};
}

SimplificationOverlayMesh::BaryCoords
SimplificationOverlayMesh::OverlayStructure::convert_edge_to_face_coordinates(
    int edge_index,
    int face_index,
    BaryCoords edge_coords) const
{
    int hij = m.h[face_index];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];

    Scalar s = edge_coords[0]; // second coord superfluous
    Scalar t = 1. - s;
    int h = e2h[edge_index];
    if (edge_index == h2e[hij])
    {
        if (m.to[h] == vj) return {s, t};
        else return {t, s};
    }
    else if (edge_index == h2e[hjk])
    {
        if (m.to[h] == vk) return {0., s};
        else return {0., t};
    }
    else if (edge_index == h2e[hki])
    {
        if (m.to[h] == vi) return {t, 0.};
        else return {s, 0.};
    }

    spdlog::error("edge not adjacent to face");
    assert(false);
    return {0., 0.};
}

void SimplificationOverlayMesh::split_edge_overlay(
    int hij,
    int halfedge_index,
    int edge_index,
    OverlayStructure& overlay)
{
    EdgeType e_type = overlay.edge_type[edge_index];
    int e_parent = overlay.halfedge_parent[halfedge_index];
    overlay.edge_type[h2e[hij]] = e_type;
    overlay.edge_type[h2e[n[hij]]] = e_type;
    overlay.halfedge_parent[hij] = e_parent;
    overlay.halfedge_parent[n[hij]] = e_parent;
    // if part of an edge, set opposite halfedge parent to parent opposite halfedge
    if (e_type == EdgeType::EE)
    {
        overlay.halfedge_parent[opp[hij]] = overlay.m.opp[e_parent];
        overlay.halfedge_parent[opp[n[hij]]] = overlay.m.opp[e_parent];
    }
    // if part of a face, just propogate the face label
    else
    {
        overlay.halfedge_parent[opp[hij]] = e_parent;
        overlay.halfedge_parent[opp[n[hij]]] = e_parent;
    }

    // set vertex parent
    int vj = to[hij];
    int vi = to[opp[hij]];
    int vk = to[n[hij]];
    if (e_type == EdgeType::EE)
    {
        overlay.vertex_type[vj] = VertexType::VE;
        overlay.vertex_parent[vj] = overlay.h2e[e_parent]; // actually use edge, not halfedge
    }
    else
    {
        overlay.vertex_type[vj] = VertexType::VF;
        overlay.vertex_parent[vj] = e_parent;
    }

    if (overlay.is_bary_init)
    {
        std::array<int, 2> adjacent_verts = {vi, vk};
        std::array<BaryCoords, 2> adjacent_coords;
        for (int i : {0, 1})
        {
            int v = adjacent_verts[i];
            if ((e_type == EdgeType::EE) && (overlay.vertex_type[v] == VertexType::VV))
            {
                adjacent_coords[i] = overlay.vertex_edge_coordinates(v, overlay.h2e[e_parent]);
            }
            else if ((e_type == EdgeType::EF) && (overlay.vertex_type[v] == VertexType::VV))
            {
                adjacent_coords[i] = overlay.vertex_face_coordinates(v, e_parent);
            }
            else if ((e_type == EdgeType::EF) && (overlay.vertex_type[v] == VertexType::VE))
            {
                adjacent_coords[i] = overlay.convert_edge_to_face_coordinates(
                    overlay.vertex_parent[v],
                    e_parent,
                    overlay.bary_coords[v]
                );
            }
            else
            {
                adjacent_coords[i] = overlay.bary_coords[v];
            }
        }

        overlay.bary_coords[vj] = (adjacent_coords[0] + adjacent_coords[1]) / 2.;
    }
}

int SimplificationOverlayMesh::split_face_overlay(
    int e,
    int face_index,
    OverlayStructure& overlay)
{
    int f_parent = overlay.face_parent[face_index];
    overlay.edge_type[e] = EdgeType::EF;
    int hij = e2h[e];
    int hji = opp[hij];
    overlay.halfedge_parent[hij] = f_parent;
    overlay.halfedge_parent[hji] = f_parent;
    overlay.face_parent[face[hij]] = f_parent;
    overlay.face_parent[face[hji]] = f_parent;

    return e;
}

void SimplificationOverlayMesh::join_edge_faces_overlay(
    int f,
    int halfedge_index,
    OverlayStructure& overlay)
{
    int f_parent = overlay.halfedge_parent[halfedge_index];
    overlay.face_parent[f] = f_parent;
}

void SimplificationOverlayMesh::collapse_edge_overlay(
    int hij,
    int halfedge_index,
    OverlayStructure& overlay)
{

    overlay.edge_type[h2e[hij]] = overlay.edge_type[h2e[halfedge_index]];
    overlay.halfedge_parent[hij] = overlay.halfedge_parent[opp[halfedge_index]];
    overlay.halfedge_parent[opp[hij]] = overlay.halfedge_parent[halfedge_index];
}

void SimplificationOverlayMesh::add_center_to_face_overlay(int vi, int face_index, OverlayStructure& overlay)
{
    int f_parent = overlay.face_parent[face_index];
    overlay.vertex_type[vi] = VertexType::VF;
    overlay.vertex_parent[vi] = f_parent;
    int hij = out[vi];
    int hik = hij;
    std::vector<int> adjacent_verts;
    do
    {
        int eik = h2e[hik];
        overlay.edge_type[eik] = EdgeType::EF;
        overlay.halfedge_parent[hik] = f_parent;
        overlay.halfedge_parent[opp[hik]] = f_parent;
        overlay.face_parent[face[hik]] = f_parent;
        adjacent_verts.push_back(to[hik]);
        hik = opp[prev[hik]];
    }
    while (hik != hij);

    if (overlay.is_bary_init)
    {
        int degree = adjacent_verts.size();
        overlay.bary_coords[vi] = {0., 0.};
        for (int i = 0; i < degree; ++i)
        {
            int v = adjacent_verts[i];
            if (overlay.vertex_type[v] == VertexType::VV)
            {
                overlay.bary_coords[vi] += overlay.vertex_face_coordinates(v, f_parent);
            }
            else if (overlay.vertex_type[v] == VertexType::VE)
            {
                overlay.bary_coords[vi] += overlay.convert_edge_to_face_coordinates(
                    overlay.vertex_parent[v],
                    f_parent,
                    overlay.bary_coords[v]
                );
            }
            else
            {
                overlay.bary_coords[vi] += overlay.bary_coords[v];
            }
        }

        overlay.bary_coords[vi] /= degree;
    }
}


void SimplificationOverlayMesh::add_simplified_edges_to_viewer(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    int num_edges = e2h.size();
    std::vector<std::array<int, 2>> edges;
    std::vector<int> parent;
    edges.reserve(count_edges());
    for (int eij = 0; eij < num_edges; ++eij)
    {
        // only add edges contained in simpified mesh edges
        if (is_edge_deleted(eij)) continue;
        if (simp.edge_type[eij] == EdgeType::EF) continue;

        int hij = e2h[eij];
        int vi = v_old2new[to[opp[hij]]];
        int vj = v_old2new[to[hij]];
        edges.push_back({vi, vj});
        parent.push_back(simp.halfedge_parent[e2h[eij]]); // TODO: get edge in original mesh
    }
    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    auto [V_s, edges_s] = remove_unreferenced_vertices(V_o, edges);

    std::vector<int> node_parent(V_s.rows(), -1);
    for (int i = 0; i < edges.size(); ++i)
    {
        auto [vi, vj] = edges[i];
        auto [wi, wj] = edges_s[i];
        node_parent[wi] = simp.vertex_parent[v_new2old[vi]];
        node_parent[wj] = simp.vertex_parent[v_new2old[vj]];
    }
#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle;
    mesh_handle = "simplified edges";
    polyscope::registerCurveNetwork(mesh_handle, V_s, edges_s)
        ->setColor(BLACK)
        ->setRadius(0.0005);
    polyscope::getCurveNetwork(mesh_handle)->addEdgeScalarQuantity("edge parent", parent);
    polyscope::getCurveNetwork(mesh_handle)->addNodeScalarQuantity("node parent", node_parent);
#endif
}

void SimplificationOverlayMesh::add_simplified_vertices_to_viewer(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    std::vector<int> vertices = {};
    std::vector<int> parent;
    vertices.reserve(count_vertices());
    int num_vertices = out.size();
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        // only add simplified vertices
        if (is_vertex_deleted(vi)) continue;
        if (simp.vertex_type[vi] != VertexType::VV) continue;

        vertices.push_back(v_old2new[vi]);
        parent.push_back(simp.vertex_parent[vi]);
    }

    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    Eigen::MatrixXd V_s = subset_vertices(V_o, vertices);
    


#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle;
    mesh_handle = "simplified vertices";
    polyscope::registerPointCloud(mesh_handle, V_s)
        ->setPointColor(BLACK)
        ->setPointRadius(0.001);
    polyscope::getPointCloud(mesh_handle)->addScalarQuantity("parent", parent);
#endif
}

void SimplificationOverlayMesh::add_overlay_to_viewer(
    const std::string& mesh_handle,
    const OverlayStructure& overlay,
    const std::string& prefix 
) const {
    auto [f_new2old, f_old2new] = get_face_reductions();
    auto [v_new2old, v_old2new] = get_vertex_reductions();
    std::vector<int> vertex_type = enumerate_vertex_type(overlay.vertex_type);
    std::vector<int> vertex_type_r = vector_compose(vertex_type, v_new2old);
    std::vector<int> face_parent_r = vector_compose(overlay.face_parent, f_new2old);
    std::vector<int> shuffled_face_parent = shuffle_map_image(face_parent_r);
    std::vector<Scalar> u_coord, v_coord;
    for (int vi = 0; vi < out.size(); ++vi)
    {
        if (is_vertex_deleted(vi)) continue;
        u_coord.push_back(overlay.bary_coords[vi][0]);
        v_coord.push_back(overlay.bary_coords[vi][1]);
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity(prefix + " face parent", face_parent_r);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity(prefix + " shuffled face parent", shuffled_face_parent);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity(prefix + " vertex type", vertex_type_r);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity(prefix + " u coord", u_coord);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity(prefix + " v coord", v_coord);
#endif
}

void SimplificationOverlayMesh::add_original_to_viewer(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [f_new2old, f_old2new] = get_face_reductions();
    std::vector<int> to_reindex = vector_compose(vtx_reindex, orig.m.to);
    auto [F, F_halfedge] = generate_connectivity_faces(orig.m.n, orig.m.h, to_reindex);

    std::vector<Scalar> face_areas = compute_face_areas();
    std::vector<Scalar> min_area(F.rows(), 1e10);
    for (int f = 0; f < f2h.size(); ++f)
    {
        if (is_face_deleted(f)) continue;
        int Fi = orig.face_parent[f];
        min_area[Fi] = min(face_areas[f_old2new[f]], min_area[Fi]);
    }


#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle;
    mesh_handle = "original mesh";
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    //polyscope::getSurfaceMesh(mesh_handle)->addFaceVectorQuantity("basis 1", B1);
    //polyscope::getSurfaceMesh(mesh_handle)->addFaceVectorQuantity("basis 2", B2);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("min overlay area", min_area);
#endif
}

void SimplificationOverlayMesh::add_mesh_to_viewer(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [F, F_halfedge] = generate_overlay_faces();
    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    std::vector<Scalar> face_areas = compute_face_areas();
    std::vector<Scalar> min_area(V_o.rows(), 1e10);
    std::vector<int> degenerate = {};
    for (int f = 0; f < F.rows(); ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int vi = F(f, i);
            min_area[vi] = min(face_areas[f], min_area[vi]);
            if (face_areas[f] <= 0.) degenerate.push_back(vi);
        }
    }
    Eigen::MatrixXd V_degenerate = subset_vertices(V_o, degenerate);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle;
    mesh_handle = "overlay mesh";
    polyscope::registerSurfaceMesh(mesh_handle, V_o, F);
    if (!degenerate.empty()) polyscope::registerPointCloud("degenerate", V_degenerate);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("vertex indices", v_new2old);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("min adjacent area", min_area);
    add_overlay_to_viewer(mesh_handle, orig, "original");
    add_overlay_to_viewer(mesh_handle, simp, "simplified");
#endif
}

void SimplificationOverlayMesh::collect_garbage()
{
    if (skip_garbage_collection) return;

    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [v_new2old, v_old2new] = get_vertex_reductions();
    auto [e_new2old, e_old2new] = get_edge_reductions();
    auto [f_new2old, f_old2new] = get_face_reductions();

    orig.collect_garbage(he_new2old, v_new2old, e_new2old, f_new2old);
    simp.collect_garbage(he_new2old, v_new2old, e_new2old, f_new2old);
    HalfedgeMesh::collect_garbage(
        he_new2old,
        e_new2old,
        v_new2old,
        f_new2old,
        he_old2new,
        e_old2new,
        v_old2new,
        f_old2new);
}

std::tuple<Eigen::MatrixXd, std::vector<std::array<int, 2>>> remove_unreferenced_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<std::array<int, 2>>& edges)
{
    // get remaining vertices
    std::vector<bool> is_deleted(V.rows(), true);
    for (const auto& edge : edges)
    {
        for (int v : edge) is_deleted[v] = false;
    }
    std::vector<int> v_removed, v_new2old, v_old2new;
    enumerate_boolean_array(is_deleted, v_removed, v_new2old, v_old2new);

    int num_edges = edges.size();
    std::vector<std::array<int, 2>> new_edges(num_edges);
    for (int e = 0; e < num_edges; ++e)
    {
        for (int i : {0, 1})
        {
            new_edges[e][i] = v_old2new[edges[e][i]];
        }
    }

    // get vertex subset
    Eigen::MatrixXd new_V = subset_vertices(V, v_new2old);

    return std::make_tuple(new_V, new_edges);
}

// remove trivial cycle self intersections
std::vector<int> remove_cycle_tags(const std::vector<int>& cycle)
{
    // nothing to do for a cycle with two elements
    if (cycle.size() <= 2) return cycle;

    // check if the path loops back on itself
    int num_elem = cycle.size();
    std::vector<bool> is_trimmed(num_elem, false);

    // lambda to get previous untrimmed element
    auto get_prev = [&](int n)
    {
        do
        {
            n = (n - 1 + num_elem) % num_elem;
        } while (is_trimmed[n]);
        return n;
    };

    // lambda to get next untrimmed element
    auto get_next = [&](int n)
    {
        do
        {
            n = (n + 1) % num_elem;
        } while (is_trimmed[n]);
        return n;
    };

    // search for tags until none remain
    int i = 0;
    int j = 1;
    int k = 2;
    int count = 0;
    while (count < num_elem)
    {
        // if tag found, remove and reset count
        if (cycle[i] == cycle[k])
        {
            is_trimmed[j] = true;
            is_trimmed[k] = true;

            // move the window back one (to catch long tails quickly)
            j = i;
            i = get_prev(i);
            k = get_next(k);
            count = 0;
        }
        // otherwise, advance by one and increment counter
        else
        {
            i = j;
            j = k;
            k = get_next(k);
            ++count;
        }
    }

    // build trimmed vector
    std::vector<int> cycle_trimmed = {};
    cycle_trimmed.reserve(num_elem);
    for (int i = 0; i < num_elem; ++i)
    {
        if (!is_trimmed[i])
        {
            cycle_trimmed.push_back(cycle[i]);
        }
    }

    return cycle_trimmed;
}

// remove trivial cycle self intersections
std::vector<int> remove_one_cycle_tag(const std::vector<int>& cycle)
{
    // nothing to do for a cycle with two elements
    if (cycle.size() <= 2) return cycle;

    // look for a tag
    int num_elem = cycle.size();
    int tag = -1;
    for (int i = 0; i < num_elem; ++i)
    {
        int k = (i + num_elem - 1) % num_elem;
        int j = (i + 1) % num_elem;
        int prev = cycle[k];
        int next = cycle[j];
        if (prev == next)
        {
            tag = i;
            break;
        }
    }

    if (tag < 0) return cycle;

    // remove tag from cycle
    std::vector<int> cycle_trimmed = {};
    cycle_trimmed.reserve(num_elem);
    for (int i = 0; i < num_elem - 2; ++i)
    {
        int j = (i + 1 + tag) % num_elem;
        cycle_trimmed.push_back(cycle[j]);
    }

    return cycle_trimmed;
}

// given a cycle, remove redundant elements and trivial self intersections so that every three
// consecutive elements are unique
std::vector<int> trim_cycle(const std::vector<int>& cycle)
{
    // prepare new vector for trimmed cycle
    std::vector<int> cycle_trimmed = {};
    int num_elem = cycle.size();
    cycle_trimmed.reserve(num_elem);

    // first, remove repeated elements
    int prev = cycle.back();
    for (int i = 0; i < num_elem; ++i)
    {
        int curr = cycle[i];
        if (curr != prev)
        {
            cycle_trimmed.push_back(curr);
            prev = curr;
        }
    }
    
    // now remove any tags of the form A B A
    return remove_cycle_tags(cycle_trimmed);
}

std::vector<Scalar> compute_curvature_adaptive_lengths(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    Scalar normal_scale
)
{
    std::vector<Scalar> l = m.l;

    // compute normals for normal weights
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);

    // compute lengths and total length without normals term
    Scalar total_length = 0.;
    for (int hij = 0; hij < m.n_halfedges(); ++hij)
    {
        int hji = m.opp[hij];
        if (hij < hji) continue;
        total_length += m.l[hij];
    }

    // add normal distances, weighted by average length
    Scalar avg_length = total_length / m.n_edges();
    Scalar a = normal_scale * avg_length;
    for (int hij = 0; hij < m.n_halfedges(); ++hij)
    {
        int hji = m.opp[hij];
        if (hij < hji) continue;

        // get edge vertex positions
        int Vi = vtx_reindex[m.to[hji]];
        int Vj = vtx_reindex[m.to[hij]];

        // get edge normals
        Vector3 ni = N.row(Vi);
        Vector3 nj = N.row(Vj);

        // update lengths with normals term
        Scalar length = a * (ni - nj).norm();
        if (isnan(length)) spdlog::error("edge normal length is NaN");
        l[hij] += length;
        l[hji] += length;
    }

    return l;
}

} // namespace Resolution 
} // namespace Penner
