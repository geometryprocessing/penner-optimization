#include "feature/surgery/cut_metric_generator.h"

#include "holonomy/field/cross_field.h"
#include "holonomy/field/frame_field.h"
#include "holonomy/field/intrinsic_field.h"
#include "holonomy/holonomy/cones.h"

#include "feature/core/component_mesh.h"
#include "feature/core/union_meshes.h"
#include "feature/core/vf_corners.h"
#include "feature/dirichlet/cone_perturber.h"
#include "feature/feature/gluing.h"

#include <igl/per_face_normals.h>
#include <igl/facet_components.h>

namespace Penner {
namespace Feature {

CutMetricGenerator::CutMetricGenerator(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    MarkedMetricParameters marked_metric_params,
    std::vector<std::pair<int, int>> marked_corners)
{
    // build mask from the marked corners
    int num_faces = F_cut.rows();
    F_mask = compute_mask_from_corners(num_faces, marked_corners);

    // Get components
    igl::facet_components(F_cut, components);
    int num_components = components.maxCoeff() + 1;

    // Build each component
    for (int i = 0; i < num_components; ++i) {
        // Get component
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        std::vector<int> component_f;
        Eigen::VectorXi J;
        std::tie(V, F, component_f, J) = build_component(V_cut, F_cut, components, i);

        // build component halfedge mesh
        int num_vertices = V.rows();
        std::vector<int> free_cones(0);
        std::vector<Scalar> Th_hat_flat(num_vertices, 2 * M_PI);
        bool fix_boundary = false;
        std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
        meshes.push_back(FV_to_double<Scalar>(
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
            fix_boundary));
        auto& m = meshes.back();
        vtx_reindexes.push_back(vtx_reindex);

        // make fixed dof interior
        m.fixed_dof = std::vector<bool>(m.n_ind_vertices(), false);
        bool fixed_dof = false;
        for (int vi = 0; vi < m.n_vertices(); ++vi) {
            if (Holonomy::is_interior(m, vi)) {
                m.fixed_dof[m.v_rep[vi]] = true;
                fixed_dof = true;
                break;
            }
        }
        if (!fixed_dof) {
            spdlog::warn("Fixing boundary vertex");
            m.fixed_dof[0] = true;
        }

        // get component marked corners
        std::vector<std::pair<int, int>> component_corners;
        marked_halfedges.push_back({});
        for (int fijk = 0; fijk < F.rows(); ++fijk) {
            for (int i = 0; i < 3; ++i) {
                if (F_mask(component_f[fijk], i)) {
                    component_corners.push_back({fijk, i});
                }
            }
        }
        auto vv2he = generate_VV_to_halfedge_map(m, vtx_reindex);
        for (const auto& ci : component_corners) {
            int fijk = ci.first;
            int k = ci.second;
            int i = (k + 1) % 3;
            int j = (k + 2) % 3;
            int vi = F(fijk, i);
            int vj = F(fijk, j);
            int hij = vv2he.coeff(vi, vj) - 1;

            spdlog::debug("Adding edge ({}, {})", vi, vj);
            marked_halfedges.back().push_back(hij);
        }

        // (optionally) remove symmetry structure
        if (marked_metric_params.remove_symmetry) {
            vtx_reindex = Holonomy::extend_vtx_reindex(m, vtx_reindex);
        }

        // build face map for component
        std::vector<int> face_map(m.n_faces());
        for (int i = 0; i < m.n_faces(); ++i)
        {
            int f = (m.type[m.h[i]] > 1) ? m.f[m.R[m.h[i]]] : i;
            face_map[i] = component_f[f];
        }
        face_maps.push_back(face_map);

        // build vertex map for component
        int domain_size = J.size();
        Eigen::VectorXi vertex_map(domain_size);
        for (int i = 0; i < domain_size; ++i) {
            vertex_map[i] = J[vtx_reindex[i]];
        }
        vertex_maps.push_back(vertex_map);
    }
}

// generate vector of minimal cone index for each vertex
std::vector<int> generate_glued_min_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    int num_vertices = m.n_ind_vertices();

    // for closed mesh, minimum cone is 1
    if (m.type[0] == 0)
    {
        return std::vector<int>(num_vertices, 1);
    }
    
    // build map from glued mesh edge vertices to cut halfedges
    auto vv2he = generate_VV_to_halfedge_map(m, vtx_reindex, V_map);

    // initialize doubled mesh vertices with 4 (twice 2)
    std::vector<int> min_cones(num_vertices, 4);
    for (int hij = 0; hij < m.n_halfedges(); hij++) {
        int hji = m.opp[hij];

        // search for boundary edges
        if ((m.type[hij] == 1) && (m.type[hji] == 2))
        {
            // set boundary cone target to 2
            int vi = m.v_rep[m.to[hji]];
            int vj = m.v_rep[m.to[hij]];
            min_cones[vi] = 2;
            min_cones[vj] = 2;

            // get paired primal halfedge across cut to check for valence 1 vertices
            int Vi = V_map[vtx_reindex[vi]];
            int Vj = V_map[vtx_reindex[vj]];
            int h_opp = vv2he.coeff(Vj, Vi) - 1;
            if (h_opp < 0) continue; // skip if boundary

            // if base vertex is valence 1, set target back to 4
            if (m.v_rep[m.to[h_opp]] == vi)
            {
                min_cones[vi] = 4;
            }

            // if tip vertex is valence 1, set target back to 4
            if (m.v_rep[m.to[m.opp[h_opp]]] == vj)
            {
                min_cones[vj] = 4;
            }
        }
    }

    return min_cones;
}

void CutMetricGenerator::generate_fields(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXd& direction,
    const std::vector<bool>& is_fixed_direction)
{
    // initialize field data for the full mesh
    int num_faces = F_cut.rows();
    reference_corner.resize(num_faces);
    theta.resize(num_faces);
    kappa.resize(num_faces, 3);
    period_jump.resize(num_faces, 3);
    is_fixed_face.resize(num_faces);
    int num_components = meshes.size();

    // get per face normals 
    Eigen::MatrixXd N;
    igl::per_face_normals(V_cut, F_cut, N);

    // build field for each individual component separately
    rotation_forms.clear();
    for (int i = 0; i < num_components; ++i) {
        // get component data
        spdlog::debug("Generating field for component {}/{}", i + 1, num_components);
        auto& m = meshes[i];
        std::vector<int> vtx_reindex(vertex_maps[i].data(), vertex_maps[i].data() + vertex_maps[i].size());
        auto& face_map = face_maps[i];

        // TODO: make parameter
        Holonomy::FieldParameters field_params;
        field_params.min_cone = 1;

        // initialize field data for component (needed to generate reference corners)
        Holonomy::IntrinsicNRosyField field_generator;
        field_generator.min_cone = field_params.min_cone;
        if (V_map.size() > 0)
        {
            field_generator.set_min_cones(generate_glued_min_cones(m, vtx_reindex, V_map));
        }
        field_generator.initialize(m);
        field_generator.get_field(m, vtx_reindex, F_cut, face_map, reference_corner, theta, kappa, period_jump);

        // get fixed angles from input direction field
        int num_faces = face_map.size();
        std::vector<Scalar> target_theta(num_faces);
        std::vector<bool> is_fixed(num_faces);
        for (int i = 0; i < num_faces; ++i)
        {
            int fijk = face_map[i];
            is_fixed[i] = is_fixed_direction[fijk];

            // convert extrinsic direction to intrinsic angle
            Eigen::Vector3d reference_direction = Holonomy::generate_reference_direction(V_cut, F_cut, fijk, reference_corner[fijk]);
            target_theta[i] = -Holonomy::signed_angle<Eigen::Vector3d>(direction.row(fijk), reference_direction, N.row(fijk));
        }
        field_generator.set_fixed_directions(m, target_theta, is_fixed);

        // solve for field with given angle constraints
        field_generator.solve(m);

        // extract field data and rotation form with target cones
        rotation_forms.push_back(field_generator.compute_rotation_form(m));
        field_generator.get_field(m, vtx_reindex, F_cut, face_map, reference_corner, theta, kappa, period_jump);
        field_generator.get_fixed_faces(m, face_map, is_fixed_face);
        m.Th_hat = Holonomy::generate_cones_from_rotation_form(m, rotation_forms.back());

        // check Guass Bonnet
        GaussBonnetCheck(m);
    }

    // generate reference field from reference corners
    reference_field = Holonomy::generate_reference_field(V_cut, F_cut, reference_corner);
}


void CutMetricGenerator::set_fields(
    const Eigen::MatrixXi& F_cut,
    const Eigen::MatrixXd& face_reference_field,
    const Eigen::VectorXd& face_theta,
    const Eigen::MatrixXd& corner_kappa,
    const Eigen::MatrixXi& corner_period_jump)
{
    // set global fields directly
    reference_field = face_reference_field;
    theta = face_theta;
    kappa = corner_kappa;
    period_jump = corner_period_jump;

    // extract rotation forms and target cones for components
    int num_components = meshes.size();
    rotation_forms.clear();
    for (int i = 0; i < num_components; ++i) {
        auto& m = meshes[i];
        std::vector<int> vtx_reindex(vertex_maps[i].data(), vertex_maps[i].data() + vertex_maps[i].size());
        auto& face_map = face_maps[i];

        // initialize feild generator with the given field
        Holonomy::IntrinsicNRosyField field_generator;
        field_generator.min_cone = 1;
        field_generator.use_trivial_boundary = true;
        field_generator.initialize(m);
        field_generator.set_field(m, vtx_reindex, F_cut, face_map, theta, kappa, period_jump);

        // extract the rotation form and cone angles
        rotation_forms.push_back(field_generator.compute_rotation_form(m));
        m.Th_hat = Holonomy::generate_cones_from_rotation_form(m, rotation_forms.back());

        // check Guass Bonnet
        GaussBonnetCheck(m);
    }
}


std::tuple<Mesh<Scalar>, std::vector<int>, std::vector<int>>
CutMetricGenerator::get_union_mesh() const
{
    // generate total mesh
    auto union_mesh = union_meshes(meshes);

    // Reindex vertices
    Eigen::VectorXi vertex_map = union_vectors(vertex_maps);
    std::vector<int> vtx_reindex(vertex_map.data(), vertex_map.data() + vertex_map.size());

    // reindex faces
    std::vector<int> face_reindex = union_vectors(face_maps);

    return std::make_tuple(union_mesh, vtx_reindex, face_reindex);
}

std::tuple<MarkedPennerConeMetric, std::vector<int>, std::vector<int>, VectorX, std::vector<Scalar>>
CutMetricGenerator::get_union_metric(MarkedMetricParameters marked_metric_params)
{
    generate_marked_metrics(marked_metric_params);

    // generate total mesh
    auto union_metric = union_marked_metrics(marked_metrics);

    // Reindex vertices
    Eigen::VectorXi vertex_map = union_vectors(vertex_maps);
    std::vector<int> vtx_reindex(vertex_map.data(), vertex_map.data() + vertex_map.size());

    // reindex faces
    std::vector<int> face_reindex = union_vectors(face_maps);

    // generate union rotation data
    VectorX rotation_form = union_vectors(rotation_forms);
    std::vector<Scalar> Th_hat = vector_inverse_reindex(union_metric.Th_hat, vtx_reindex);

    return std::make_tuple(union_metric, vtx_reindex, face_reindex, rotation_form, Th_hat);
}

std::tuple<
    DirichletPennerConeMetric,
    std::vector<int>,
    std::vector<int>,
    VectorX,
    std::vector<Scalar>>
CutMetricGenerator::get_aligned_metric(const Eigen::VectorXi& V_map, MarkedMetricParameters marked_metric_params)
{
    // generate marked metric
    auto [marked_metric, vtx_reindex, face_reindex, rotation_form, Th_hat] =
        get_union_metric(marked_metric_params);

    // generate dirichlet metric
    DirichletPennerConeMetric dirichlet_metric =
        generate_dirichlet_metric_from_mesh(marked_metric, vtx_reindex, V_map);

    return std::make_tuple(dirichlet_metric, vtx_reindex, face_reindex, rotation_form, Th_hat);
}

// helper function to fix invalid cones with heuristics
void fix_cones(
    DirichletPennerConeMetric& dirichlet_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    VectorX& rotation_form,
    std::vector<Scalar>& Th_hat)
{
    // get genus from the angle defecits
    auto glued_angle_defects = compute_glued_angle_defects(dirichlet_metric, vtx_reindex, V_map);
    Scalar total_curvature = compute_total_sum(glued_angle_defects);
    int genus = (int)(round(1 + total_curvature / (4. * M_PI)));
    Scalar min_angle = (dirichlet_metric.type[0] > 0) ? M_PI : (M_PI / 2.);

    // If a small number of cone pairs can be removed, do so (possibly introducing zero cones)
    int min_cones = 2;
    int num_neg_cones, num_pos_cones;
    std::tie(num_neg_cones, num_pos_cones) = count_glued_cones(dirichlet_metric, vtx_reindex, V_map);
    while ((genus == 1) && ((num_neg_cones + num_pos_cones) <= min_cones))
    {
        if ((num_neg_cones + num_pos_cones) == 0) break;
        spdlog::info("Attempting to resolve cone pair with collapse");
        bool removal_success = ConePerturber().remove_cone_pair(dirichlet_metric, rotation_form, vtx_reindex, V_map, 0.);
        std::tie(num_neg_cones, num_pos_cones) = count_glued_cones(dirichlet_metric, vtx_reindex, V_map);
        if (!removal_success) break;
        spdlog::info("{} pos and {} neg cones remain", num_pos_cones, num_neg_cones);
    }

    // remove any zero cones on the boundary, without introducing glued cones if possible
    spdlog::info("Remove any zero cones");
    ConePerturber().perturb_boundary_cones(dirichlet_metric, rotation_form, vtx_reindex, V_map, min_angle);
    std::tie(num_neg_cones, num_pos_cones) = count_glued_cones(dirichlet_metric, vtx_reindex, V_map);

    // remove cone pairs, without introducing zero cones
    while ((genus == 1) && ((num_neg_cones + num_pos_cones) <= min_cones))
    {
        if ((num_neg_cones + num_pos_cones) == 0) break;
        spdlog::info("Attempting to resolve cone pair with collapse");
        bool removal_success = ConePerturber().remove_cone_pair(dirichlet_metric, rotation_form, vtx_reindex, V_map, min_angle);
        std::tie(num_neg_cones, num_pos_cones) = count_glued_cones(dirichlet_metric, vtx_reindex, V_map);

        // stop if unable to make progress
        if (!removal_success) break;
        spdlog::info("{} pos and {} neg cones remain", num_pos_cones, num_neg_cones);
    }

    // fix 3-5 torus case by adding two random cone pairs
    while ((genus == 1) && (num_neg_cones + num_pos_cones <= min_cones)) {
        if ((num_neg_cones + num_pos_cones) == 0) break;
        spdlog::info("Resolving cone pair with split");

        int num_vertices = dirichlet_metric.n_ind_vertices();
        add_random_cone_pair(dirichlet_metric, true, num_vertices / 3);
        add_random_cone_pair(dirichlet_metric, true, 2 * num_vertices / 3);
        std::tie(num_neg_cones, num_pos_cones) = count_glued_cones(dirichlet_metric, vtx_reindex, V_map);
    }

    // update constraints
    Th_hat = vector_inverse_reindex(dirichlet_metric.Th_hat, vtx_reindex);
    dirichlet_metric.kappa_hat = Holonomy::compute_kappa(dirichlet_metric, rotation_form, dirichlet_metric.get_homology_basis_loops());
}

std::tuple<
    DirichletPennerConeMetric,
    std::vector<int>,
    std::vector<int>,
    VectorX,
    std::vector<Scalar>>
CutMetricGenerator::get_fixed_aligned_metric(const Eigen::VectorXi& V_map, MarkedMetricParameters marked_metric_params)
{
    auto [dirichlet_metric, vtx_reindex, face_reindex, rotation_form, Th_hat] = get_aligned_metric(V_map, marked_metric_params);
    fix_cones(dirichlet_metric, vtx_reindex, V_map, rotation_form, Th_hat);

    // if trivial torus, regenereate with no holonomy loop constraints
    int num_neg_cones, num_pos_cones;
    std::tie(num_neg_cones, num_pos_cones) = count_glued_cones(dirichlet_metric, vtx_reindex, V_map);
    if (((num_neg_cones + num_pos_cones) == 0) && (marked_metric_params.max_loop_constraints > 0)) {
        spdlog::info("Trivial torus found. Removing loop constraints");
        marked_metric_params.max_loop_constraints = 0;
        return get_fixed_aligned_metric(V_map, marked_metric_params);
    }

    return std::make_tuple(dirichlet_metric, vtx_reindex, face_reindex, rotation_form, Th_hat);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi>
CutMetricGenerator::get_field() const
{
    return std::make_tuple(reference_field, theta, kappa, period_jump);
}

std::vector<bool> CutMetricGenerator::get_fixed_faces() const
{
    return is_fixed_face;
}

// generate all marked metrics
void CutMetricGenerator::generate_marked_metrics(MarkedMetricParameters marked_metric_params)
{
    int num_components = meshes.size();
    marked_metrics.clear();
    for (int i = 0; i < num_components; ++i) {
        marked_metrics.push_back(generate_marked_metric_from_mesh(
            meshes[i],
            rotation_forms[i],
            marked_metric_params,
            marked_halfedges[i]));
    }
}

// (untested) field optimization
void CutMetricGenerator::optimize_fields(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& F_cut)
{
    // generate cross field from current field
    std::array<Eigen::MatrixXd, 4> cross_field = Holonomy::generate_cross_field(V, F, reference_field, theta);

    // optimize cross field
    std::vector<int> fixed_faces;
    convert_boolean_array_to_index_vector(is_fixed_face, fixed_faces);
    cross_field = Holonomy::reduce_curl(V, F, cross_field, fixed_faces);

    // build glued mesh
    spdlog::info("Generating uncut mesh");
    std::vector<Scalar> Th_hat_flat(V.rows(), 2. * M_PI);
    std::vector<int> face_reindex_uncut;
    arange(F.rows(), face_reindex_uncut);
    auto [m, vtx_reindex_uncut] = Holonomy::generate_mesh(V, F, V, F, Th_hat_flat);

    // set field with optimized cross field
    spdlog::info("Setting field");
    Eigen::VectorXi reference_corner(F.rows());
    Holonomy::IntrinsicNRosyField field_generator;
    field_generator.initialize(m);
    field_generator.get_field(m, vtx_reindex_uncut, F, face_reindex_uncut, reference_corner, theta, kappa, period_jump);
    theta = Holonomy::infer_theta(V, F, reference_corner, cross_field[0]);
    reference_field = Holonomy::generate_reference_field(V, F, reference_corner);
    field_generator.set_field(m, vtx_reindex_uncut, F, face_reindex_uncut, theta, kappa, period_jump);

    // set principal matchings
    spdlog::info("Generating principal matchings");
    field_generator.compute_principal_matchings(m);
    field_generator.get_field(m, vtx_reindex_uncut, F, face_reindex_uncut, reference_corner, theta, kappa, period_jump);
    set_fields(F_cut, reference_field, theta, kappa, period_jump);
}


} // namespace Feature
} // namespace Penner
