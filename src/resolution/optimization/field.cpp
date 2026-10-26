#include "resolution/optimization/field.h"
#include "feature/feature/features.h"
#include "field/intrinsic_field.h"
#include "parametrization/layout.h"
#include "field/cones.h"
#include "holonomy/holonomy/holonomy.h"
#include "holonomy/core/homology_basis.h"
#include "metric/constraint.h"

#include <igl/doublearea.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/per_face_normals.h>

namespace Penner
{
namespace Resolution
{

std::vector<int> trace_spanning_tree_to_root(
    const std::vector<int>& parent,
    int start_index)
{
    std::vector<int> path = {start_index};

    // Trace up the dual tree until a root is reached
    int curr_index = start_index;
    while (parent[curr_index] >= 0) {
        curr_index = parent[curr_index];
        path.push_back(curr_index);
    }

    return path;
}

std::vector<int> get_spanning_tree_with_cut(
    const VoronoiOverlayMesh& voronoi_mesh,
    const std::vector<bool>& is_not_cut
)
{
    int num_faces = voronoi_mesh.f2h.size();
    int num_halfedges = voronoi_mesh.n.size();
    std::vector<bool> is_processed_face(num_faces, false);
    std::vector<int> parent_face(num_faces, -1);

    // Initialize queue with boundary faces to process
    std::deque<int> faces_to_process;
    int fi = 0;
    while (voronoi_mesh.is_face_deleted(fi)) ++fi;
    is_processed_face[fi] = true;
    faces_to_process.push_back(fi);

    // bfs
    while (!faces_to_process.empty()) {
        // Get the next face to process
        int fi = faces_to_process.front();
        faces_to_process.pop_front();

        // Iterate over the face circulator via halfedges
        int h_start = voronoi_mesh.f2h[fi];
        int hij = h_start;
        do {
            // Get the face in the one ring at the tip of the halfedge
            int fj = voronoi_mesh.face[voronoi_mesh.opp[hij]];

            bool is_invalid = false;
            is_invalid = (is_invalid) || (is_processed_face[fj]);
            if (voronoi_mesh.orig.edge_type[voronoi_mesh.h2e[hij]] == VoronoiOverlayMesh::EdgeType::EE)
            {
                int Hij = voronoi_mesh.orig.halfedge_parent[hij];
                is_invalid = (is_invalid) || (!is_not_cut[Hij]);
            }

            // Check if the edge to the tip face is the best seen so far
            if (!is_invalid) {
                is_processed_face[fj] = true;
                parent_face[fj] = fi;
                faces_to_process.push_back(fj);
            }

            // Progress to the next halfedge in the face circulator
            hij = voronoi_mesh.n[hij];
        } while (hij != h_start);
    }

    return parent_face;
}


VoronoiOverlayMesh generate_uniform_triangulation(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Scalar ratio)
{
    // compute target length from mesh area and vertices
    Eigen::VectorXd double_area;
    igl::doublearea(V, F, double_area);
    Scalar mesh_area = double_area.sum() / 2.;
    int num_vertices = V.rows();
    Scalar target_length = sqrt(mesh_area / (ratio * num_vertices));

    std::vector<Scalar> Th_hat(V.rows(), 2 * PI);
    auto [m, vtx_reindex] = Holonomy::generate_mesh(V, F, V, F, Th_hat);
    VoronoiOverlayMesh voronoi_mesh(V, vtx_reindex, m);
    voronoi_mesh.use_zero_lengths = true; // TODO

    // get feature vertices and junctions to use as seeds
    Feature::FeatureFinder feature_finder(V, F);
    Scalar feature_angle = 30.;
    feature_finder.mark_dihedral_angle_features(feature_angle);
    std::vector<int> feature_degrees = feature_finder.compute_feature_degrees();
    std::vector<int> seeds = {};
    std::vector<bool> is_feature_vertex(feature_degrees.size(), false);
    for (int i = 0; i < feature_degrees.size(); ++i)
    {
        if (feature_degrees[i] > 0) is_feature_vertex[i] = true;
        if (feature_degrees[i] > 2) seeds.push_back(i);
    }
    if (seeds.empty()) seeds = {0};

    // build point sampling, starting with feature vertices, then feature edges, then interiors
    voronoi_mesh.grow_voronoi_regions(seeds);
    voronoi_mesh.set_constained_max_voronoi_distance(target_length, is_feature_vertex);
    voronoi_mesh.set_max_voronoi_distance(target_length);

    // build triangulation
    voronoi_mesh.split_voronoi_edges();
    voronoi_mesh.split_voronoi_faces();
    voronoi_mesh.find_voronoi_dual_edges();
    voronoi_mesh.propagate_voronoi_vertices_to_faces();
    voronoi_mesh.trace_delaunay_edges();
    voronoi_mesh.flood_faces();

    // straighten and compute the length of the triangulation edges
    //voronoi_mesh.voronoi.reset(); // reset to allow overlay path tracing
    //voronoi_mesh.remove_excess_refinement();
    //voronoi_mesh.straighten_edges();
    //voronoi_mesh.set_simplified_halfedge_lengths();

    return voronoi_mesh;
}


// TODO: replace this with generate_frame_field (using the existing mesh)
Field::IntrinsicNRosyField compute_simplified_field(
    const VoronoiOverlayMesh& voronoi_mesh,
    const Eigen::MatrixXd& V_simp,
    const Eigen::MatrixXi& F_simp,
    const Field::FieldParameters& field_params)
{
    const Mesh<Scalar>& m = voronoi_mesh.simp.m;
    Field::IntrinsicNRosyField field_generator(field_params);
    field_generator.initialize(voronoi_mesh.simp.m);
    if (field_params.use_principal_directions)
    {
        int num_faces = F_simp.rows();
        std::vector<int> vtx_reindex = arange(V_simp.rows());
        std::vector<int> face_map = generate_face_map(voronoi_mesh.simp.m);
        Eigen::VectorXi reference_corner(num_faces);
        Eigen::VectorXd theta(num_faces);
        Eigen::MatrixXd kappa(num_faces, 3);
        Eigen::MatrixXi period_jump(num_faces, 3);
        field_generator.get_field(voronoi_mesh.simp.m, vtx_reindex, F_simp, face_map, reference_corner, theta, kappa, period_jump);

        // fit field directions
        int radius = 5;
        Scalar abs_anisotropy=0.5;
        Scalar rel_anisotropy=0.9;
        Scalar bb_diag = igl::bounding_box_diagonal(V_simp);
        auto [direction, is_fixed_direction] = Field::compute_field_direction(
            V_simp,
            F_simp,
            radius,
            abs_anisotropy / bb_diag,
            rel_anisotropy,
            0.05);
        // compute normals for angle computations

        Eigen::MatrixXd N;
        igl::per_face_normals(V_simp, F_simp, N);

        // get fixed angles from input direction field
        std::vector<Scalar> target_theta(num_faces);
        std::vector<bool> is_fixed(num_faces, false);
        for (int i = 0; i < num_faces; ++i)
        {
            int fijk = face_map[i];
            if (!is_fixed_direction[fijk]) continue;

            // check if adjacent faces also set
            bool is_adjacent_fixed = false;
            int hij = m.h[i];
            do
            {
                if (is_fixed[m.f[m.opp[hij]]])
                {
                    is_adjacent_fixed = true;
                }

                hij = m.n[hij];
            } while (hij != m.h[i]);
            if (is_adjacent_fixed) continue; 

            is_fixed[i] = is_fixed_direction[fijk];

            // convert extrinsic direction to intrinsic angle
            Eigen::Vector3d reference_direction = Field::generate_reference_direction(V_simp, F_simp, fijk, reference_corner[fijk]);
            target_theta[i] = -signed_angle<Eigen::Vector3d>(direction.row(fijk), reference_direction, N.row(fijk));
        }

        // set fixed directions 
        field_generator.set_fixed_directions(voronoi_mesh.simp.m, target_theta, is_fixed);
    }

    field_generator.solve(voronoi_mesh.simp.m);
    if (field_params.fix_cone_pair) field_generator.fix_cone_pair(voronoi_mesh.simp.m);
    if (field_params.min_cone_pair_distance > 0.) field_generator.remove_close_cone_pairs(voronoi_mesh.simp.m, field_params.min_cone_pair_distance);
    return field_generator;
}

void lift_simplified_field_to_original(
    const VoronoiOverlayMesh& voronoi_mesh,
    const VectorX& rotation_form_simp,
    Field::IntrinsicNRosyField& field_generator
)
{
    const auto& m = voronoi_mesh.orig.m;
    int num_halfedges = m.n_halfedges();
    int num_vertices = m.n_vertices();

    std::vector<int> start_halfedge = voronoi_mesh.find_start_halfedges(voronoi_mesh.orig);

    // get target cones from simplified mesh
    std::vector<Scalar> Th_hat_simp = Field::generate_cones_from_rotation_form(
        voronoi_mesh.simp.m,
        rotation_form_simp);

    // Compute the corner angles for the original and simplified mesh
    VectorX he2angle, he2cot;
    corner_angles(m, he2angle, he2cot);
    VectorX he2angle_simp, he2cot_simp;
    corner_angles(voronoi_mesh.simp.m, he2angle_simp, he2cot_simp);

    // get base kappas and field
    Eigen::VectorXd theta;
    Eigen::VectorXd kappa;
    Eigen::VectorXi period_jump;
    field_generator.get_halfedge_field(m, theta, kappa, period_jump);
    std::vector<Scalar> Th_hat_base(num_vertices, 0);
    for (int hij = 0; hij < m.n_halfedges(); hij++) {
        Th_hat_base[m.v_rep[m.to[m.n[hij]]]] += he2angle[hij];
        Th_hat_base[m.v_rep[m.to[hij]]] += kappa[hij];
    }

    // get cones determined by kappa
    std::vector<int> base_cones(num_vertices, 0);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        base_cones[vi] = lround(Th_hat_base[vi] / (M_PI / 2.));
    }

    // set dual tree halfedges as fixed
    std::vector<int> fixed_faces = {0};
    std::vector<int> halfedges_from_face = Field::build_double_dual_bfs_forest(m, fixed_faces);
    std::vector<bool> is_period_jump_fixed(num_halfedges, false);
    for (int h : halfedges_from_face) {
        if (h < 0) continue;
        for (int h_rel : { h, m.opp[h] }) {
            is_period_jump_fixed[h_rel] = true;
        }
    }

    // get primal cotree
    std::vector<Scalar> weights(num_halfedges, 1.);
    std::vector<bool> is_spanning_tree(num_halfedges, false);
    std::vector<int> halfedges_from_vertex = build_primal_forest(m, weights, is_period_jump_fixed);
    for (int h : halfedges_from_vertex) {
        if (h < 0) continue;
        for (int h_rel : { h, m.opp[h] }) {
            is_spanning_tree[h_rel] = true;
        }
    }

    // count the number of unset edges adjacent to each vertex
    std::vector<int> adjacent_unset_edges(num_vertices, 0);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (!is_period_jump_fixed[hij])
        {
            ++adjacent_unset_edges[m.to[hij]];
        }
    }

    // get dual path in overlay to base face that doesn't intersect the unset period jump edges
    // TODO: more efficient to build this after first branch trim in zippering
    std::vector<int> voronoi_face_parent = get_spanning_tree_with_cut(voronoi_mesh, is_period_jump_fixed);

    // set period jumps for leaves of the unset period jump graph
    std::vector<int> vertices_to_process = arange(num_vertices);

    // store loops on surface
    std::vector<Holonomy::DualLoopList> dual_loops = {};

    //DiscreteMetric discrete_metric_orig = generate_discrete_metric(voronoi_mesh.orig);
    //DiscreteMetric discrete_metric_simp = generate_discrete_metric(voronoi_mesh.simp);
    // iterate until all halfedge jumps set
    int h_test = 0;
    while (h_test < num_halfedges)
    {
        while (!vertices_to_process.empty())
        {
            int vi = vertices_to_process.back();
            vertices_to_process.pop_back();

            if (adjacent_unset_edges[vi] == 1)
            {
                // find adjacent unfixed edge
                int hij = m.out[vi];
                while (is_period_jump_fixed[hij])
                {
                    hij = m.n[m.opp[hij]];
                }

                // get target cone angle from simplified mesh
                int vi_overlay = voronoi_mesh.orig.vertex_index[vi];
                int target_cone = 4;
                if (voronoi_mesh.simp.vertex_type[vi_overlay] == SimplificationOverlayMesh::VertexType::VV)
                {
                    int vi_simp = voronoi_mesh.simp.vertex_parent[vi];
                    target_cone = lround(Th_hat_simp[vi_simp] / (M_PI / 2.));
                }

                // get cone index determined by adjacent halfedges
                int hik = m.n[m.opp[hij]];
                int cone = base_cones[vi];
                while (hik != hij)
                {
                    assert(is_period_jump_fixed[hik]);
                    cone -= period_jump[hik]; // subtract since vertex at base of halfedge
                    hik = m.n[m.opp[hik]];
                }

                // update data structures
                int vj = m.to[hij];
                if (vi == vj) spdlog::warn("self adjacent edge");
                int hji = m.opp[hij];
                period_jump[hij] = (cone - target_cone);
                period_jump[hji] = -(cone - target_cone);
                is_period_jump_fixed[hij] = true;
                is_period_jump_fixed[hji] = true;
                --adjacent_unset_edges[vi];
                --adjacent_unset_edges[vj];
                vertices_to_process.push_back(vj);
            }
        }

        while (h_test < num_halfedges)
        {
            if ((!is_period_jump_fixed[h_test]) && (!is_spanning_tree[h_test])) break;
            h_test++;
        }

        if (h_test < num_halfedges)
        {
            int hij = h_test;
            int hji = m.opp[hij];
            int hij_overlay = start_halfedge[hij];
            int hji_overlay = voronoi_mesh.opp[hij_overlay];
            int vj = m.to[hij];
            int vi = m.to[hji];

            // build a path of fixed period jumps connecting the opposite halfedges
            std::vector<int> path_i = trace_spanning_tree_to_root(voronoi_face_parent, voronoi_mesh.face[hij_overlay]);
            std::vector<int> path_j = trace_spanning_tree_to_root(voronoi_face_parent, voronoi_mesh.face[hji_overlay]);
            std::vector<int> basis_loop = concatenate_vector(path_i, reverse_vector(path_j));

            // project the loops to the origianl and simplified mesh
            std::vector<int> orig_basis_loop = voronoi_mesh.project_dual_loop(basis_loop, voronoi_mesh.orig);
            dual_loops.emplace_back(voronoi_mesh.orig.m, orig_basis_loop);
            Holonomy::DualLoopList & orig_dual_path = dual_loops.back(); 

            // compute target holonomy of the loop projected on the simplified mesh
            std::vector<int> simp_basis_loop = voronoi_mesh.project_dual_loop(basis_loop, voronoi_mesh.simp);
            Holonomy::DualLoopList simp_dual_path(voronoi_mesh.simp.m, simp_basis_loop);
            Scalar rotation_simp = compute_dual_loop_rotation(voronoi_mesh.simp.m, rotation_form_simp, simp_dual_path);
            Scalar holonomy_simp = compute_dual_loop_holonomy(voronoi_mesh.simp.m, he2angle_simp, simp_dual_path);
            Scalar target = holonomy_simp - rotation_simp;
            spdlog::debug("target angle {}", target);

            // get holonomy on the original mesh
            Scalar holonomy = compute_dual_loop_holonomy(m, he2angle, orig_dual_path);

            // compute rotation of base field along the dual loop
            Scalar rotation = 0.;
            period_jump[hij] = 0;
            period_jump[hji] = 0;
            for (const auto& dual_segment : orig_dual_path) {
                int h = dual_segment[0];
                rotation += kappa[h] + (period_jump[h] * (PI / 2.));
            }
            Scalar current = holonomy - rotation;
            int defect = round((current - target) / (PI / 2.));
            spdlog::debug("current holonomy {}", current);
            spdlog::debug("defect is {}", defect);

            period_jump[hij] = defect;
            period_jump[hji] = -defect;
            is_period_jump_fixed[hij] = true;
            is_period_jump_fixed[hji] = true;
            --adjacent_unset_edges[vi];
            --adjacent_unset_edges[vj];
            vertices_to_process.push_back(vi);
            vertices_to_process.push_back(vj);
        }
    }

    assert(vector_max(adjacent_unset_edges) == 0);
    // set field with new values
    field_generator.set_halfedge_field(m, theta, kappa, period_jump);

    VectorX rotation_form = field_generator.compute_rotation_form(voronoi_mesh.orig.m);
    std::vector<Scalar> Th_hat = Field::generate_cones_from_rotation_form(
        voronoi_mesh.orig.m,
        rotation_form);
    
    bool check_consistency = false;
    if (check_consistency)
    {
        for (int vi = 0; vi < voronoi_mesh.out.size(); ++vi)
        {
            if (voronoi_mesh.is_vertex_deleted(vi)) continue;
            if (voronoi_mesh.orig.vertex_type[vi] != SimplificationOverlayMesh::VertexType::VV) continue;
            int vi_orig = voronoi_mesh.orig.vertex_parent[vi];
            int actual_cone = lround(Th_hat[vi_orig] / (M_PI / 2.));
            int target_cone = 4;
            if (voronoi_mesh.simp.vertex_type[vi] == SimplificationOverlayMesh::VertexType::VV)
            {
                int vi_simp = voronoi_mesh.simp.vertex_parent[vi];
                target_cone = lround(Th_hat_simp[vi_simp] / (M_PI / 2.));
            }
            if (target_cone != actual_cone)
            {
                spdlog::error("inconsistent cone {} at {} with target {}", actual_cone, vi, target_cone);
            }
        }

        for (const auto& dual_loop : dual_loops)
        {
            Scalar rotation = compute_dual_loop_rotation(m, rotation_form, dual_loop);
            Scalar holonomy = compute_dual_loop_holonomy(m, he2angle, dual_loop);
            Scalar current = holonomy - rotation;
            spdlog::info("current holonomy {}", current);
        }
    }
}


Eigen::MatrixXd lift_simplified_directions_to_overlay(
    const VoronoiOverlayMesh& voronoi_mesh,
    const Field::IntrinsicNRosyField& field_generator,
    const Eigen::MatrixXd& V_simp,
    const Eigen::MatrixXi& F_simp)
{
    std::vector<int> vtx_reindex_simp = arange(V_simp.rows());
    Eigen::MatrixXd frame_field_orig(voronoi_mesh.orig.m.n_faces(), 3);
    int num_faces = F_simp.rows();
    std::vector<int> face_map = generate_face_map(voronoi_mesh.simp.m);
    Eigen::VectorXi reference_corner(num_faces);
    Eigen::VectorXd theta(num_faces);
    Eigen::MatrixXd kappa(num_faces, 3);
    Eigen::MatrixXi period_jump(num_faces, 3);
    field_generator.get_field(voronoi_mesh.simp.m, vtx_reindex_simp, F_simp, face_map, reference_corner, theta, kappa, period_jump);
    Eigen::MatrixXd reference_field = Field::generate_reference_field(V_simp, F_simp, reference_corner);
    //view_cross_field(V_simp, F_simp, reference_field, theta, kappa, period_jump);

    Eigen::MatrixXd frame_field = Field::generate_vector_field(V_simp, F_simp, reference_field, theta);
    auto [f_new2old, f_old2new] = voronoi_mesh.get_face_reductions();
    Eigen::MatrixXd frame_field_overlay(f_new2old.size(), 3);
    for (int f = 0; f < f_new2old.size(); ++f)
    {
        int Fijk = voronoi_mesh.simp.face_parent[f_new2old[f]];
        frame_field_overlay.row(f) = frame_field.row(Fijk);
    }
    return frame_field_overlay;
}

// experimental
Field::IntrinsicNRosyField compute_simplified_field(
    const VoronoiOverlayMesh& voronoi_mesh,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& V_simp,
    const Eigen::MatrixXi& F_simp,
    const Field::FieldParameters& field_params)
{
    Field::IntrinsicNRosyField field_generator(field_params);
    field_generator.initialize(voronoi_mesh.simp.m);
    if (field_params.use_principal_directions)
    {
        int num_faces = F_simp.rows();
        std::vector<int> vtx_reindex = arange(V_simp.rows());
        std::vector<int> face_map = generate_face_map(voronoi_mesh.simp.m);
        Eigen::VectorXi reference_corner(num_faces);
        Eigen::VectorXd theta(num_faces);
        Eigen::MatrixXd kappa(num_faces, 3);
        Eigen::MatrixXi period_jump(num_faces, 3);
        field_generator.get_field(voronoi_mesh.simp.m, vtx_reindex, F_simp, face_map, reference_corner, theta, kappa, period_jump);

        // fit field directions in original mesh
        int radius = 5;
        Scalar abs_anisotropy=0.5;
        Scalar rel_anisotropy=0.9;
        Scalar bb_diag = igl::bounding_box_diagonal(V);
        auto [direction, is_fixed_direction] = Field::compute_field_direction(
            V,
            F,
            radius,
            abs_anisotropy / bb_diag,
            rel_anisotropy,
            0.01);

        // compute normals for angle computations
        Eigen::MatrixXd N_simp;
        igl::per_face_normals(V_simp, F_simp, N_simp);

        // get fixed angles from input direction field
        std::vector<Scalar> target_theta(num_faces);
        std::vector<bool> is_fixed(num_faces);
        for (int f = 0; f < voronoi_mesh.f2h.size(); ++f)
        {
            if (voronoi_mesh.is_face_deleted(f)) continue;
            int f_simp = voronoi_mesh.simp.face_parent[f];
            int f_orig = voronoi_mesh.orig.face_parent[f];
            if (!is_fixed_direction[f_orig]) continue;

            // convert extrinsic direction to intrinsic angle
            is_fixed[f_simp] = true;
            Eigen::Vector3d reference_direction = Field::generate_reference_direction(V_simp, F_simp, f_simp, reference_corner[f_simp]);
            Eigen::Vector3d d = project_orthogonal_to(direction.row(f_orig), N_simp.row(f_simp)).stableNormalized();
            target_theta[f_simp] = -signed_angle<Eigen::Vector3d>(d, reference_direction, N_simp.row(f_simp));
        }

        // set fixed directions 
        field_generator.set_fixed_directions(voronoi_mesh.simp.m, target_theta, is_fixed);
    }

    field_generator.solve(voronoi_mesh.simp.m);
    if (field_params.fix_cone_pair) field_generator.fix_cone_pair(voronoi_mesh.simp.m);
    if (field_params.min_cone_pair_distance > 0.) field_generator.remove_close_cone_pairs(voronoi_mesh.simp.m, field_params.min_cone_pair_distance);
    return field_generator;
}

void comb_field_to_cut(
    const Mesh<Scalar>& m,
    const std::vector<bool> is_cut,
    Eigen::VectorXd& thetas,
    Eigen::VectorXi& period_jumps)
{
    // initialize queue and markings with first halfedge
    int h_start = 0;
    std::queue<int> Q;
    std::vector<bool> done(m.h.size());
    Q.push(h_start);
    done[m.f[h_start]] = true;

    while (!Q.empty()) {
        int h = Q.front();
        Q.pop();
        int hn = m.n[h];
        int hp = m.n[hn];

        for (int h_curr : {h, hn, hp})
        {
            int ho = m.opp[h_curr];
            int fo = m.f[ho];

            // stop at visited edges and explicitly cut edges
            if (done[m.f[ho]]) continue;
            if (is_cut[h_curr]) continue;

            // bake period jump across edge into the face angle
            int pj = period_jumps[ho];
            thetas[fo] += pj * (PI / 2.);

            // push period jumps through face to opposite wedge
            period_jumps[ho] = period_jumps[m.opp[ho]] = 0;
            period_jumps[m.n[ho]] -= pj;
            period_jumps[m.n[m.n[ho]]] -= pj;
            period_jumps[m.opp[m.n[ho]]] += pj;
            period_jumps[m.opp[m.n[m.n[ho]]]] += pj;

            // mark opposite face as seen and queue
            done[fo] = true;
            Q.push(ho);
        }
    }
}


std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi
>
lift_simplified_field_to_original(
    const VoronoiOverlayMesh& voronoi_mesh,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& rotation_form,
    const Eigen::MatrixXd& reference_field_simp,
    const Eigen::VectorXd& theta_simp,
    const Eigen::MatrixXd& kappa_simp,
    const Eigen::MatrixXi& period_jump_simp,
    int smooth_iters)
{
    const auto& m = voronoi_mesh.orig.m;
    int num_faces = F.rows();
    Field::IntrinsicNRosyField field_generator_int;
    field_generator_int.initialize(voronoi_mesh.orig.m);
    std::vector<int> face_map = generate_face_map(m);
    Eigen::VectorXi reference_corner(num_faces);
    Eigen::VectorXd theta(num_faces);
    Eigen::MatrixXd kappa(num_faces, 3);
    Eigen::MatrixXi period_jump(num_faces, 3);
    int num_overlay_faces = voronoi_mesh.f2h.size();
    field_generator_int.get_field(m, vtx_reindex, F, face_map, reference_corner, theta, kappa, period_jump);
    Eigen::MatrixXd reference_field = Field::generate_reference_field(V, F, reference_corner);

    // get overlay mesh and frame field
    Eigen::MatrixXd V_o = voronoi_mesh.generate_overlay_vertices(V, vtx_reindex, voronoi_mesh.orig);
    auto [F_o, F_halfedge] = voronoi_mesh.generate_overlay_faces();

    // set period jumps in original mesh from simplified mesh
    lift_simplified_field_to_original(voronoi_mesh, rotation_form, field_generator_int);

    // lift field directions to overlay
    auto [reference_field_overlay, theta_overlay, period_jump_overlay] = voronoi_mesh.sample_field_from_overlay(
        reference_field_simp,
        theta_simp,
        period_jump_simp,
        voronoi_mesh.simp);

    // comb field to original mesh seam
    spdlog::info("combing field to original mesh");
    Eigen::VectorXd kappa_halfedge;
    Eigen::VectorXi period_jump_halfedge;
    field_generator_int.get_halfedge_field(m, theta, kappa_halfedge, period_jump_halfedge);
    std::vector<bool> is_cut_place_holder = {};
    std::vector<bool> is_cut_orig = compute_layout_topology(voronoi_mesh.orig.m, is_cut_place_holder);
    voronoi_mesh.comb_field_to_overlay_seam(is_cut_orig, 0, theta_overlay, period_jump_overlay, voronoi_mesh.orig);
    comb_field_to_cut(voronoi_mesh.orig.m, is_cut_orig, theta, period_jump_halfedge);

    // average directions in original face with area weight
    spdlog::info("computing average field on original mesh");
    Eigen::MatrixXd frame_field_overlay = Field::generate_vector_field(V_o, F_o, reference_field_overlay, theta_overlay);

    Eigen::MatrixXd frame_field_orig = Eigen::MatrixXd::Zero(num_faces, 3);
    Eigen::VectorXd doublearea;
    igl::doublearea(V_o, F_o, doublearea);
    for (int f = 0; f < num_overlay_faces; ++f)
    {
        if (voronoi_mesh.is_face_deleted(f)) continue;
        int f_orig = voronoi_mesh.orig.face_parent[f];
        Scalar area = doublearea[f] / 2.;
        frame_field_orig.row(f_orig) += area * frame_field_overlay.row(f);
    }

    // compute angles in original mesh
    spdlog::info("computing angles on original mesh");
    Eigen::MatrixXd N;
    igl::per_face_normals(V, F, N);
    for (int f = 0; f < num_faces; ++f)
    {
        Eigen::Vector3d d = frame_field_orig.row(f).stableNormalized();
        theta[f] = -signed_angle<Eigen::Vector3d>(d, reference_field.row(f), N.row(f));
    }

    // set optimized field
    field_generator_int.set_halfedge_field(m, theta, kappa_halfedge, period_jump_halfedge);

    // smooth field directions
    field_generator_int.smooth_theta(m, smooth_iters);

    // get final field
    field_generator_int.get_field(m, vtx_reindex, F, face_map, reference_corner, theta, kappa, period_jump);
    reference_field = Field::generate_reference_field(V, F, reference_corner);

    return std::make_tuple(reference_field, theta, kappa, period_jump);
}

void lift_basis_loops(
    const VoronoiOverlayMesh& voronoi_mesh,    
    const MarkedPennerConeMetric& marked_metric,
    const VectorX& rotation_form,
    std::vector<std::unique_ptr<DualLoop>>& basis_loops,
    std::vector<Scalar>& kappa)
{
    std::vector<std::unique_ptr<DualLoop>> orig_basis_loops = {};
    basis_loops.clear();
    kappa.clear();
    if (!Field::is_trivial_torus(marked_metric))
    {
        // generate the homology basis on the overlay mesh
        Mesh<Scalar> m_ref = voronoi_mesh.generate_mesh();
        Holonomy::HomologyBasisGenerator homology_basis_generator(m_ref, 0, Holonomy::HomologyBasisGenerator::Weighting::minimal_homotopy);

        // project basis loops on the simplified and original mesh
        int num_homology_basis_loops = homology_basis_generator.n_homology_basis_loops();
        basis_loops.reserve(num_homology_basis_loops);
        orig_basis_loops.reserve(num_homology_basis_loops);
        auto [f_new2old, f_old2new] = voronoi_mesh.get_face_reductions();
        for (int i = 0; i < num_homology_basis_loops; ++i)
        {
            // compute the dual loop on the overlay mesh (in reduced indexing)
            std::vector<int> basis_loop = vector_compose(f_new2old, homology_basis_generator.construct_homology_basis_loop(i));
            spdlog::debug("basis loop with {} faces", basis_loop.size());

            // project dual loops to the simplified and original mesh
            std::vector<int> simp_basis_loop = voronoi_mesh.project_dual_loop(basis_loop, voronoi_mesh.simp);
            std::vector<int> orig_basis_loop = voronoi_mesh.project_dual_loop(basis_loop, voronoi_mesh.orig);
            assert(!simp_basis_loop.empty());
            assert(!orig_basis_loop.empty());

            // build the dual loop connectivity on the simplified mesh
            auto simp_dual_path = Holonomy::build_dual_path_from_face_sequence(voronoi_mesh.simp.m, simp_basis_loop);
            basis_loops.push_back(std::make_unique<Holonomy::DualLoopConnectivity>(Holonomy::DualLoopConnectivity(simp_dual_path)));
            assert(!simp_dual_path.empty());

            // build the dual loop connectivity on the original mesh
            auto orig_dual_path = Holonomy::build_dual_path_from_face_sequence(marked_metric, orig_basis_loop);
            orig_basis_loops.push_back(std::make_unique<Holonomy::DualLoopList>(Holonomy::DualLoopList(orig_dual_path)));
            assert(!orig_dual_path.empty());
        }

        // compute holonomy angles in the original mesh
        DiscreteMetric discrete_metric = generate_discrete_metric(marked_metric);
        kappa = compute_kappa(discrete_metric, rotation_form, orig_basis_loops);
        spdlog::info("target holonomy angles are {}", formatted_vector(kappa));
    }
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
refine_overlay_field(
    const VoronoiOverlayMesh& voronoi_mesh,    
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F,
    const std::vector<int>& fn_to_f_o,
    const Eigen::MatrixXi& hn_to_h_o,
    const Eigen::MatrixXi& is_edge_o,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump
)
{
    int num_faces = fn_to_f_o.size();
    Eigen::MatrixXd reference_field_r(num_faces, 3);
    Eigen::VectorXd theta_r(num_faces);
    Eigen::MatrixXd kappa_r(num_faces, 3);
    Eigen::MatrixXi period_jump_r(num_faces, 3);
    for (int fi = 0; fi < num_faces; ++fi)
    {
        int f_orig = fn_to_f_o[fi];
        reference_field_r.row(fi) = reference_field.row(f_orig);
        theta_r[fi] = theta[f_orig];

        for (int j = 0; j < 3; ++j)
        {
            // if e
            if (is_edge_o(fi, j) == 0)
            {
                kappa_r(fi, j) = 0.;
                period_jump_r(fi, j) = 0;
            }
            else
            {
                int Hij = hn_to_h_o(fi, j);
                int Vk = vtx_reindex[voronoi_mesh.orig.m.to[voronoi_mesh.orig.m.n[Hij]]];
                int i = 0;
                while (F(f_orig, i) != Vk)
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

    return std::make_tuple(reference_field_r, theta_r, kappa_r, period_jump_r);
}


} // namespace Resolution
} // namespace Penner
