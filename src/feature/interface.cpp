#include "feature/interface.h"

#include "util/vf_mesh.h"
#include "util/boundary.h"

#include "optimization/parameterization/refinement.h"
#include "optimization/core/common.h"

#include "holonomy/interface.h"
#include "holonomy/field/frame_field.h"
#include "holonomy/core/viewer.h"

#include "feature/feature/gluing.h"
#include "feature/core/component_mesh.h"
#include "holonomy/core/viewer.h"
#include "feature/dirichlet/angle_constraint_relaxer.h"
#include "feature/core/vf_corners.h"
#include "feature/feature/error.h"
#include "feature/dirichlet/optimization.h"
#include "feature/surgery/cut_metric_generator.h"
#include "feature/surgery/cut_mesh_layout.h"
#include "feature/surgery/refinement.h"
#include "feature/surgery/stitching.h"

#include <igl/facet_components.h>

namespace Penner {
namespace Feature {

std::tuple<DirichletPennerConeMetric, std::vector<int>> generate_dirichlet_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    const VectorX& rotation_form,
    std::vector<int> free_cones,
    MarkedMetricParameters marked_metric_params)
{
    // Convert VF mesh to halfedge
    bool fix_boundary = false;
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
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

    // Use halfedge mesh method
    DirichletPennerConeMetric dirichlet_metric =
        generate_dirichlet_metric_from_mesh(m, rotation_form, marked_metric_params);
    if (marked_metric_params.remove_symmetry) {
        vtx_reindex = Holonomy::extend_vtx_reindex(m, vtx_reindex);
    }

    return std::make_tuple(dirichlet_metric, vtx_reindex);
}

MarkedPennerConeMetric generate_marked_metric_components(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_metric_params)
{
    // build marked metric components
    std::vector<MarkedPennerConeMetric> marked_metrics = {};
    ComponentMesh component_mesh(m);
    const auto& components = component_mesh.get_mesh_components();
    const auto& he_maps = component_mesh.get_halfedge_maps();

    // build marked metrics from meshes and rotation forms
    int num_components = components.size();
    for (int i = 0; i < num_components; ++i)
    {
        const auto& component_mesh = components[i];
        const auto& he_index = he_maps[i];
        VectorX component_form = vector_compose(rotation_form, he_index);

        marked_metrics.push_back(
            generate_marked_metric_from_mesh(component_mesh, component_form, marked_metric_params));
    }

    // union components
    return union_marked_metrics(marked_metrics);
}

DirichletPennerConeMetric generate_dirichlet_metric_from_mesh(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_metric_params)
{
    // Generate the base underlying marked metric
    MarkedPennerConeMetric marked_metric =
        generate_marked_metric_components(m, rotation_form, marked_metric_params);

    // Find all boundary vertices
    auto boundary_vertices = find_boundary_vertices(m);
    int num_bd = boundary_vertices.size();

    // Build boundary paths and compute initial lengths
    std::vector<BoundaryPath> boundary_paths = {};
    boundary_paths.reserve(num_bd);
    std::vector<Scalar> ell_vec;
    ell_vec.reserve(num_bd);
    for (int i = 0; i < num_bd; ++i) {
        if (i == 0) continue;
        int vi = boundary_vertices[i];
        if (!float_equal<Scalar>(m.Th_hat[m.v_rep[vi]], 2. * M_PI)) continue;
        boundary_paths.push_back(BoundaryPath(m, vi));
        ell_vec.push_back(boundary_paths.back().compute_log_length(m));
    }
    VectorX ell;
    convert_std_to_eigen_vector(ell_vec, ell);

    // Use identity for matrix system
    MatrixX identity = id_matrix(ell.size());

    return DirichletPennerConeMetric(
        m,
        marked_metric.get_reduced_metric_coordinates(),
        marked_metric.get_homology_basis_loops(),
        marked_metric.kappa_hat,
        boundary_paths,
        identity,
        ell);
}


std::tuple<std::vector<Mesh<Scalar>>, std::vector<Eigen::VectorXi>> generate_component_meshes(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut)
{
    // Get components and boundary
    Eigen::VectorXi components;
    igl::facet_components(F_cut, components);
    int num_components = components.maxCoeff() + 1;

    // Parameterize each component
    std::vector<Mesh<Scalar>> meshes;
    std::vector<Eigen::VectorXi> vertex_maps;
    for (int i = 0; i < num_components; ++i) {
        // Get component
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        std::vector<int> component_f;
        Eigen::VectorXi J;
        std::tie(V, F, component_f, J) = build_component(V_cut, F_cut, components, i);

        // Get initial dirichlet mesh for optimization
        int num_vertices = V.rows();
        std::vector<int> free_cones(0);
        std::vector<Scalar> Th_hat_flat(num_vertices, 2 * M_PI);
        bool fix_boundary = false;
        std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
        Mesh<Scalar> m = FV_to_double<Scalar>(
            V,
            F,
            V,
            F,
            Th_hat_flat,
            vtx_reindex,
            indep_vtx,
            dep_vtx,
            v_rep,
            bnd_loops,
            free_cones,
            fix_boundary);
        meshes.push_back(m);

        // Build vertex map for this component
        int domain_size = J.size();
        Eigen::VectorXi vertex_map(domain_size);
        for (int i = 0; i < domain_size; ++i) {
            vertex_map[i] = J[vtx_reindex[i]];
        }
        vertex_maps.push_back(vertex_map);
    }

    return std::make_tuple(meshes, vertex_maps);
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<VertexEdge>, std::vector<VertexEdge>> generate_refined_feature_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool use_minimal_forest)
{
    // generate default features
    FeatureFinder feature_finder(V, F);
    feature_finder.mark_dihedral_angle_features(60.);
    feature_finder.prune_small_components(4);
    feature_finder.prune_small_features(5);

    // refine faces to avoid faces with two boundary edges
    auto [V_ref_f, F_ref_f, feature_edges_f] = refine_corner_feature_faces(feature_finder);

    // refine edges to avoid feature components without a forest edge
    feature_finder = FeatureFinder(V_ref_f, F_ref_f);
    feature_finder.mark_features(feature_edges_f);
    return refine_feature_components(feature_finder, use_minimal_forest);
}

AlignedMetricGenerator::AlignedMetricGenerator(
    const Eigen::MatrixXd& V_,
    const Eigen::MatrixXi& F_,
    const std::vector<VertexEdge>& feature_edges_,
    const std::vector<VertexEdge>& hard_feature_edges_,
    const Eigen::MatrixXd& reference_field_,
    const Eigen::VectorXd& theta_,
    const Eigen::MatrixXd& kappa_,
    const Eigen::MatrixXi& period_jump_,
    Scalar regularization_factor,
    bool use_minimal_forest)
    : parameterized(false)
    , V(V_)
    , F(F_)
    , feature_edges(feature_edges_)
    , hard_feature_edges(hard_feature_edges_)
    , reference_field(reference_field_)
    , theta(theta_)
    , kappa(kappa_)
    , period_jump(period_jump_)
{
    // generate feature finders for all features
    FeatureFinder feature_finder(V, F);
    feature_finder.mark_features(feature_edges);

    // generate feature finders for hard features
    FeatureFinder hard_feature_finder(V, F);
    hard_feature_finder.mark_features(hard_feature_edges);

    if (use_minimal_forest)
    {
        UnionFind cut_feature_components = feature_finder.compute_cut_feature_components();
        hard_feature_finder.prune_refined_greedy(cut_feature_components);
    }

    // build cut mesh and feature masks
    std::tie(V_cut, F_cut, V_map, F_is_hard_feature) =
            hard_feature_finder.generate_feature_cut_mesh();
    std::tie(V_cut, F_cut, V_map, F_is_feature) = feature_finder.generate_feature_cut_mesh();

    // build soft feature mask and corresponding corners opposite relaxed edges
    F_is_soft_feature = mask_difference(F_is_feature, F_is_hard_feature);
    std::vector<std::pair<int, int>> relaxed_corners = compute_mask_corners(F_is_soft_feature);

    // build a cut metric generator for the field cut mesh and input field
    MarkedMetricParameters marked_metric_params;
    marked_metric_params.remove_trivial_torus = false;
    marked_metric_params.use_log_length = true;
    CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, relaxed_corners);
    cut_metric_generator.set_fields(F_cut, reference_field, theta, kappa, period_jump);

    // get the metric from the generator
    std::tie(embedding_metric, vtx_reindex, face_reindex, rotation_form, Th_hat) = cut_metric_generator.get_fixed_aligned_metric(V_map, marked_metric_params);

    // build an initial metric with Penner coordinates
    dirichlet_metric = embedding_metric;
    VectorX metric_coords = generate_penner_coordinates(embedding_metric);
    dirichlet_metric.change_metric(embedding_metric, metric_coords, true, false);

    // build the relaxed constraints
    std::vector<std::pair<int, int>> relaxed_edges = compute_relaxed_edges(relaxed_corners, dirichlet_metric, vtx_reindex, V_map, F_cut);
    AngleConstraintMatrixRelaxer relaxer;
    MatrixX relaxed_angle_constraint_system = relaxer.run(dirichlet_metric, relaxed_edges);
    dirichlet_metric.set_angle_constraint_system(relaxed_angle_constraint_system);

    // initialize metric for optimization
    opt_dirichlet_metric = dirichlet_metric;
    opt_dirichlet_metric.change_metric(embedding_metric, regularization_factor * metric_coords, true, false);
}

void make_minimal_forest();

void AlignedMetricGenerator::optimize_full(const NewtonParameters& alg_params)
{
    // make sure using full constraints
    opt_dirichlet_metric.use_relaxed_system = false;

    // optimize the metric to satisfy the constraints
    auto opt_marked_metric = optimize_metric_angles(opt_dirichlet_metric, alg_params);
    opt_metric_coords = opt_marked_metric.get_metric_coordinates();
    opt_dirichlet_metric.change_metric(opt_dirichlet_metric, opt_metric_coords);
}

void AlignedMetricGenerator::optimize_relaxed(const NewtonParameters& alg_params)
{
    opt_dirichlet_metric.use_relaxed_system = true;

    // optimize the metric to satisfy the constraints
    auto opt_marked_metric = optimize_relaxed_angles(opt_dirichlet_metric, alg_params);
    opt_metric_coords = opt_marked_metric.get_metric_coordinates();
    opt_dirichlet_metric.change_metric(opt_dirichlet_metric, opt_metric_coords);
}

Scalar AlignedMetricGenerator::compute_error() const
{
    return opt_dirichlet_metric.max_constraint_error();
}

Eigen::MatrixXd AlignedMetricGenerator::get_metric() const
{
    return transfer_halfedge_function_to_corner(
        opt_dirichlet_metric,
        vtx_reindex,
        V_map,
        F,
        opt_dirichlet_metric.get_metric_coordinates());
}

void AlignedMetricGenerator::parameterize(bool use_high_precision)
{
    Eigen::MatrixXd V_o, uv_o;
    Eigen::MatrixXi F_o, FT_o;
    std::vector<int> fn_to_f;
    std::vector<std::pair<int, int>> endpoints;

    // parameterize the cut mesh in low precision
    bool use_uniform_bc = false;
    if (use_high_precision) {
        mpfr::mpreal::set_default_prec(100);
        std::tie(V_o, F_o, uv_o, FT_o, fn_to_f, endpoints) = parameterize_cut_mesh<mpfr::mpreal>(
            embedding_metric,
            dirichlet_metric,
            opt_dirichlet_metric,
            V_cut,
            vtx_reindex,
            face_reindex,
            use_uniform_bc,
            "./");
    }
    else
    {
        std::tie(V_o, F_o, uv_o, FT_o, fn_to_f, endpoints) = parameterize_cut_mesh<Scalar>(
            embedding_metric,
            dirichlet_metric,
            opt_dirichlet_metric,
            V_cut,
            vtx_reindex,
            face_reindex,
            use_uniform_bc,
            "./");
    }

    // if using low precision, check if sufficiently accurate
    if (!use_high_precision)
    {
        auto [uv_length_error, uv_angle_error, uv_length, uv_angle] = Holonomy::compute_seamless_error(F_o, uv_o, FT_o);
        if ((uv_length_error.maxCoeff() > 1e-10) || (uv_angle_error.maxCoeff() > 1e-10) || Optimization::matrix_contains_nan(uv_o) || Optimization::matrix_contains_nan(V_o))
        {
            spdlog::warn("Falling back to high precision");
            return parameterize(true);
       } 
    }

    // get fetaure edges in the cut mesh
    Eigen::MatrixXi F_o_is_feature = generate_overlay_cut_mask(F_o, endpoints, F_cut, F_is_feature);
    Eigen::MatrixXi F_o_is_hard_feature =
        generate_overlay_cut_mask(F_o, endpoints, F_cut, F_is_hard_feature);
    Eigen::MatrixXi F_o_is_soft_feature =
        generate_overlay_cut_mask(F_o, endpoints, F_cut, F_is_soft_feature);

    // align the mesh to the hard faetures
    uv_o = align_to_hard_features(uv_o, FT_o, F_o_is_hard_feature);

    // stitch mesh together
    auto [V_s, F_s, uv_s, FT_s, fn_to_f_s, endpoints_s, F_s_is_feature] =
        stitch_cut_overlay(V_o, F_o, uv_o, FT_o, fn_to_f, endpoints, F_o_is_feature, V_map, use_uniform_bc);

    // refine mesh
    Optimization::RefinementMesh refinement_mesh(V_s, F_s, uv_s, FT_s, fn_to_f_s, endpoints_s);
    refinement_mesh.get_VF_mesh(V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r);

    // check error and fallback to high precision if high
    // if using low precision, check if sufficiently accurate
    if (!use_high_precision)
    {
        auto [uv_length_error, uv_angle_error, uv_length, uv_angle] = Holonomy::compute_seamless_error(F_r, uv_r, FT_r);
        if ((uv_length_error.maxCoeff() > 1e-10) || (uv_angle_error.maxCoeff() > 1e-10) || Optimization::matrix_contains_nan(uv_r) || Optimization::matrix_contains_nan(V_r))
        {
            spdlog::warn("Falling back to high precision");
            return parameterize(true);
        }
    }

    // refine feature and misaligned edges
    Eigen::MatrixXi F_r_is_feature = generate_overlay_cut_mask(F_r, endpoints_r, F, F_is_feature);
    auto feature_corners = compute_mask_corners(F_r_is_feature);
    auto full_feature_face_edges = compute_face_edges_from_corners(F_r, feature_corners);
    std::tie(feature_edges_r, misaligned_edges_r) = prune_misaligned_face_edges(uv_r, FT_r, full_feature_face_edges, 1e-10);

    // refine frame field
    std::tie(reference_field_r, theta_r, kappa_r, period_jump_r) = Holonomy::refine_frame_field(
        F_r,
        FT_r,
        fn_to_f_r,
        endpoints_r,
        F,
        reference_field,
        theta,
        kappa,
        period_jump);
    
    // mark mesh as parameterized
    parameterized = true;
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
AlignedMetricGenerator::get_parameterization()
{
    // do paramaterization if not done yet
    if (!parameterized) parameterize();

    return std::make_tuple(V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
    AlignedMetricGenerator::get_refined_field()
{
    // do paramaterization if not done yet
    if (!parameterized) parameterize();

    return std::make_tuple(reference_field_r, theta_r, kappa_r, period_jump_r);
}

std::tuple<std::vector<FaceEdge>, std::vector<FaceEdge>>
AlignedMetricGenerator::get_refined_features()
{
    // do paramaterization if not done yet
    if (!parameterized) parameterize();

    return std::make_tuple(feature_edges_r, misaligned_edges_r);
}


Eigen::MatrixXd generate_feature_aligned_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<VertexEdge>& feature_edges,
    const std::vector<VertexEdge>& hard_feature_edges,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    const NewtonParameters& alg_params)
{
    // use utility class
    AlignedMetricGenerator aligned_metric_generator(
        V,
        F,
        feature_edges,
        hard_feature_edges,
        reference_field,
        theta,
        kappa,
        period_jump);
    aligned_metric_generator.optimize_relaxed(alg_params);
    
    return aligned_metric_generator.get_metric();
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
generate_feature_aligned_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<VertexEdge>& feature_edges,
    const std::vector<VertexEdge>& hard_feature_edges,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    const NewtonParameters& alg_params)
{
    // use utility class
    AlignedMetricGenerator aligned_metric_generator(
        V,
        F,
        feature_edges,
        hard_feature_edges,
        reference_field,
        theta,
        kappa,
        period_jump);
    aligned_metric_generator.optimize_relaxed(alg_params);
    
    return aligned_metric_generator.get_parameterization();
}


} // namespace Feature
} // namespace Penner
