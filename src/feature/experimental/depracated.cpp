
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi>
parameterize_components(
    const MarkedPennerConeMetric& embedding_metric,
    const MarkedPennerConeMetric& original_metric,
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::MatrixXd& V_cut,
    const std::vector<int>& vtx_reindex)
{
    // Separate mesh into components
    ComponentMesh component_mesh(marked_metric);
    const auto& mesh_components = component_mesh.get_mesh_components();
    const auto& he_maps = component_mesh.get_halfedge_maps();
    const auto& embedding_components = ComponentMesh(embedding_metric).get_mesh_components();
    const auto& init_components = ComponentMesh(original_metric).get_mesh_components();
    spdlog::info("Extracted {} components", mesh_components.size());

    // Parameterize each component
    int num_components = embedding_components.size();
    std::vector<Eigen::MatrixXd> V_components, uv_components;
    std::vector<Eigen::MatrixXi> F_components, FT_components;
    for (int i = 0; i < num_components; ++i) {
        const auto& embedding_component = embedding_components[i];
        const auto& init_component = init_components[i];
        const auto& mesh_component = mesh_components[i];
        const auto& he_index = he_maps[i];
        VectorX metric_init(mesh_component.n_halfedges());
        VectorX metric_coords(mesh_component.n_halfedges());
        for (int i = 0; i < mesh_component.n_halfedges(); ++i) {
            metric_init[i] = 2. * log(init_component.l[i]);
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
            embedding_component.v_rep[i] = vi;
            init_component.v_rep[i] = vi;
            component_metric.v_rep[i] = vi;
            V_component.row(vi) = V_cut.row(vtx_reindex[Vi]);
        }

        // undouble Th_hat
        for (int i = 0; i < n_ind_v_component; ++i) {
            embedding_component.Th_hat[i] /= 2.;
        }
        std::vector<int> vtx_reindex_component;
        arange(n_ind_v_component, vtx_reindex_component);

        // Generate overlay
        std::vector<bool> is_cut = {};
        auto vf_res = Optimization::generate_VF_mesh_from_halfedge_metric<Scalar>(
            V_component,
            embedding_component,
            vtx_reindex_component,
            component_init,
            component_metric.get_metric_coordinates(),
            is_cut,
            false);
        OverlayMesh<Scalar> m_o = std::get<0>(vf_res);
        Eigen::MatrixXd V_o = std::get<1>(vf_res);
        Eigen::MatrixXi F_o = std::get<2>(vf_res);
        Eigen::MatrixXd uv_o = std::get<3>(vf_res);
        Eigen::MatrixXi FT_o = std::get<4>(vf_res);
        std::vector<bool> is_cut_h_final = std::get<5>(vf_res);
        std::vector<bool> is_cut_o = std::get<6>(vf_res);
        std::vector<int> fn_to_f_o = std::get<7>(vf_res);
        std::vector<std::pair<int, int>> endpoints_o = std::get<8>(vf_res);

        // Refine mesh
        Optimization::RefinementMesh refinement_mesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o);
        auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r] = refinement_mesh.get_VF_mesh();

        // Add components to lists
        V_components.push_back(V_r);
        F_components.push_back(F_r);
        uv_components.push_back(uv_r);
        FT_components.push_back(FT_r);
    }

    // Combine components into single mesh
    auto [V_r, F_r] = combine_mesh_components(V_components, F_components);
    auto [uv_r, FT_r] = combine_mesh_components(uv_components, FT_components);

    return std::make_tuple(V_r, F_r, uv_r, FT_r);
}

    int _merge_segment(EdgeIterator& edge_iterator, int segment_base_vertex)
    {
        int v_base_hij = segment_base_vertex;
        int v_tip_hij = edge_iterator.get_segment_tip();
        int v_tip_hji = edge_iterator.get_opposite_segment_tip();
        int v_edge = v_tip_hji;
        int v_corner_hij = get_face_corner_index(refined_F, vv2f, v_base_hij, v_tip_hij);
        spdlog::info("merging triangle {}, {}, {}", v_corner_hij, v_base_hij, v_tip_hij);
        spdlog::info("merging {} into {}", v_tip_hij, v_tip_hji);

        // get adjacent face of the segment to split
        auto [f_old, f_old_corner] = get_face_corner(refined_F, vv2f, v_base_hij, v_tip_hij);

        // refine current face
        refined_F[f_old] = {v_corner_hij, v_base_hij, v_edge};

        // update edge to face map
        set_VV_edge_faces(v_corner_hij, v_base_hij, v_edge, f_old);
        vv2f.coeffRef(v_base_hij, v_tip_hij) = 0;
        vv2f.coeffRef(v_tip_hij, v_corner_hij) = 0;

        // iterate edges
        edge_iterator.iterate_segment();
        edge_iterator.iterate_opposite_segment();
        int v_next_hij = edge_iterator.get_segment_tip();
        int v_next_corner_hij = get_face_corner_index(refined_F, vv2f, v_tip_hij, v_next_hij);

        auto [f_next, f_next_corner] = get_face_corner(refined_F, vv2f, v_tip_hij, v_next_hij);

        // refine current face
        refined_F[f_next] = {v_next_corner_hij, v_edge, v_next_hij};

        // update edge to face map
        set_VV_edge_faces(v_next_corner_hij, v_edge, v_next_hij, f_next);
        vv2f.coeffRef(v_next_corner_hij, v_tip_hij) = 0;
        vv2f.coeffRef(v_tip_hij, v_next_hij) = 0;

        // TODO remove
        if (!is_valid_face(refined_F[f_old]))
            spdlog::error("Invalid old hij face {}", formatted_vector(refined_F[f_old]));

        return v_edge;
    }


        // TODO: this is generally not feasible; it can cause flipped triangles in the uv domain
        if (false) {
            int num_midpoints = edge_vertices.size();
            std::vector<double> uniform_coords = compute_uniform_coordinates(num_midpoints);
            int uv_base_hij = edge_iterator.get_start_uv_vertex();
            int uv_base_hji = edge_iterator.get_opposite_start_uv_vertex();
            int uv_tip_hij = edge_iterator.get_end_uv_vertex();
            int uv_tip_hji = edge_iterator.get_opposite_end_uv_vertex();
            for (int m = 0; m < num_midpoints; ++m) {
                double t = uniform_coords[m];
                for (int i : {0, 1}) {
                    refined_uv[hij_uv_vertices[m]][i] =
                        (1. - t) * refined_uv[uv_base_hij][i] + t * refined_uv[uv_tip_hij][i];
                    refined_uv[hji_uv_vertices[m]][i] =
                        (1. - t) * refined_uv[uv_base_hji][i] + t * refined_uv[uv_tip_hji][i];
                }
            }
        }
        
MatrixX compute_gluing_matrix(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    int num_glued_vertices = V_map.maxCoeff() + 1;
    int num_ind_vertices = m.n_ind_vertices();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_ind_vertices);

    // sum up angles, dividing by 2 to account for doubling
    for (int vi = 0; vi < num_ind_vertices; ++vi) {
        int wi = V_map[vtx_reindex[vi]];
        if (wi == (num_glued_vertices - 1)) continue;
        tripletList.push_back(T(wi, vi, 1.0));
    }

    // Create the matrix from the triplets
    MatrixX gluing_matrix;
    gluing_matrix.resize(num_glued_vertices - 1, num_ind_vertices);
    gluing_matrix.reserve(tripletList.size());
    gluing_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return gluing_matrix;
}

MatrixX compute_relaxed_angle_constraint_matrix(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const std::vector<bool>& is_relaxed_vertex)
{
    int num_glued_vertices = V_map.maxCoeff() + 1;
    int num_ind_vertices = m.n_ind_vertices();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_ind_vertices);

    // add unrelaxed vertex constraints
    int relaxed_count = 0;
    int unrelaxed_count = 0;
    std::vector<int> relaxed_map(num_glued_vertices, -1);
    std::vector<int> relaxed_counts(num_glued_vertices, 0);
    for (int vi = 0; vi < num_ind_vertices; ++vi) {
        int wi = V_map[vtx_reindex[vi]];
        if (is_relaxed_vertex[wi]) {
            ++relaxed_counts[wi];

            if (relaxed_map[wi] < 0) {
                relaxed_map[wi] = relaxed_count;
                ++relaxed_count;
            }

            // skip first two occurances of relaxed vertex
            if (relaxed_counts[wi] <= 2) continue;
            spdlog::info("{} seen {} times", wi, relaxed_counts[wi]);
        }

        tripletList.push_back(T(unrelaxed_count, vi, 1.0));
        ++unrelaxed_count;
    }

    for (int vi = 0; vi < num_ind_vertices; ++vi) {
        int wi = V_map[vtx_reindex[vi]];
        if (!is_relaxed_vertex[wi]) continue;
        if (relaxed_map[wi] == (relaxed_count - 1)) continue;

        spdlog::info("Adding constraint {}", relaxed_map[wi]);
        tripletList.push_back(T(unrelaxed_count + relaxed_map[wi], vi, 1.0));
    }

    // Create the matrix from the triplets
    MatrixX gluing_matrix;
    gluing_matrix.resize(unrelaxed_count + relaxed_count - 1, num_ind_vertices);
    gluing_matrix.reserve(tripletList.size());
    gluing_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return gluing_matrix;
}

// Measure the angle from a halfedge to a frame field in the face
Scalar compute_cross_field_halfedge_angle(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& N,
    int h)
{
    // Get halfedge direction
    int v0 = vtx_reindex[m.to[m.opp[h]]];
    int v1 = vtx_reindex[m.to[h]];
    Eigen::Vector3d h_direction = V.row(v1) - V.row(v0);

    // Get field reference direction and normal vector in the face
    int f = m.f[h];
    Eigen::Vector3d field_direction = R.row(f);
    Eigen::Vector3d face_normal = N.row(f);

    return signed_angle<Eigen::Vector3d>(h_direction, field_direction, face_normal);
}

// Measure the intrinsic angle between frame field vectors in two faces across an edge
Scalar compute_cross_field_edge_angle(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& N,
    int h)
{
    // Use consistent halfedge for stability
    if (!has_priority(m, vtx_reindex, h)) {
        return -compute_cross_field_edge_angle(m, vtx_reindex, V, R, N, m.opp[h]);
    }

    // Get halfedge direction
    int v0 = vtx_reindex[m.to[m.opp[h]]];
    int v1 = vtx_reindex[m.to[h]];
    Eigen::Vector3d h_direction = V.row(v1) - V.row(v0);

    // Get angle of rotation across the edge
    int f0 = m.f[h];
    int f1 = m.f[m.opp[h]];
    double d0 = signed_angle<Eigen::Vector3d>(h_direction, R.row(f0), N.row(f0));
    double d1 = signed_angle<Eigen::Vector3d>(h_direction, R.row(f1), N.row(f1));
    double alpha = (2 * M_PI) + (M_PI / 4) + d0 - d1;

    //return (alpha > 0) ? pos_fmod(alpha, M_PI / 2.0) : -pos_fmod(-alpha, M_PI / 2.0);
    return pos_fmod(alpha, M_PI / 2.0) - (M_PI / 4);
    //return pos_fmod(alpha - (M_PI/ 4), 2.0 * M_PI);
    //Scalar kij = pos_fmod(alpha - (M_PI / 4), 2 * M_PI);
    //if (kij > M_PI) {
    //    kij -= 2 * M_PI;
    //}
    //if (!(-M_PI < kij && kij <= M_PI)) {
    //    spdlog::error("{} out of range (-pi, pi])", kij);
    //}
    //return kij;
}

template <typename OverlayScalar>
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
layout_cut_mesh(
    const MarkedPennerConeMetric& embedding_metric,
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::MatrixXd& V_cut,
    const std::vector<int>& vtx_reindex)
{
#ifdef WITH_MPFR
    mpfr::mpreal::set_default_prec(100);
    mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
    mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

    // Separate mesh into components
    ComponentMesh component_embedding(embedding_metric);
    ComponentMesh component_mesh(marked_metric);
    const auto& embedding_components = component_embedding.get_mesh_components();
    const auto& mesh_components = component_mesh.get_mesh_components();
    const auto& he_maps = component_mesh.get_halfedge_maps();
    spdlog::info("Extracted {} components", mesh_components.size());

    // Parameterize each component
    int num_components = embedding_components.size();
    std::vector<Eigen::MatrixXd> V_components, uv_components;
    std::vector<Eigen::MatrixXi> F_components, FT_components;
    for (int i = 0; i < num_components; ++i) {
        const auto& embedding_component = embedding_components[i];
        const auto& mesh_component = mesh_components[i];
        VectorX metric_coords(mesh_component.n_halfedges());
        for (int i = 0; i < mesh_component.n_halfedges(); ++i) {
            metric_coords[i] = 2. * log(mesh_component.l[i]);
        }
        Optimization::PennerConeMetric component_metric(mesh_component, metric_coords);

        int n_v_component = component_metric.n_vertices();
        int n_ind_v_component = component_metric.n_ind_vertices();
        Eigen::MatrixXd V_component(n_ind_v_component, 3);
        const auto& he_index = he_maps[i];
        for (int i = 0; i < n_v_component; ++i) {
            // get vertex in component
            int hij = component_metric.out[i];
            int vi = component_metric.v_rep[i];

            // get vertex in total mesh
            int Hij = he_index[hij];
            int Vi = marked_metric.v_rep[marked_metric.to[marked_metric.opp[Hij]]];

            V_component.row(vi) = V_cut.row(vtx_reindex[Vi]);
        }

        // Fit conformal scale factors
        VectorX scale_factors;
        scale_factors.setZero(n_ind_v_component);

        // Compute interpolation overlay mesh
        Optimization::InterpolationMesh<OverlayScalar> interpolation_mesh, reverse_interpolation_mesh;
        interpolate_penner_coordinates(
            embedding_component,
            component_metric.get_metric_coordinates(),
            scale_factors,
            interpolation_mesh,
            reverse_interpolation_mesh);
        OverlayMesh<OverlayScalar> m_o = interpolation_mesh.get_overlay_mesh();
        auto [F_o, uv_o, FT_o] = Optimization::compute_layout_VF(m_o);
        //Optimization::view_mesh_topology(V_component, F_o);

        // Add components to lists
        V_components.push_back(V_component);
        F_components.push_back(F_o);
        uv_components.push_back(uv_o);
        FT_components.push_back(FT_o);
    }

    // Combine components into single mesh
    auto [V_r, F_r] = combine_mesh_components(V_components, F_components);
    auto [uv_r, FT_r] = combine_mesh_components(uv_components, FT_components);
    auto [V_rr, F_rr] = reindex_mesh(V_r, F_r, vtx_reindex);

    return std::make_tuple(V_rr, F_rr, uv_r, FT_r);
}
/**
 * @brief Glue the feature edges of a doubled cut mesh.
 *
 * @param m: mesh
 * @param vtx_reindex: reindexing of cut mesh vertices to VF index
 * @param cut_m: mesh with features cut
 * @param cut_vtx_reindex: reindexing of cut mesh vertices to cut VF index
 * @param V_map: map from cut VF indices to original VF indices
 * @return closed mesh with glued features
 */
Mesh<Scalar> glue_feature_edges(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& cut_m,
    const std::vector<int>& cut_vtx_reindex,
    const Eigen::VectorXi& V_map) const
{
    // compute map from vertex-vertex edges to halfedges
    int n_v = V_map.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);

    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int n_cut_he = cut_m.n_halfedges();
    for (int hij = 0; hij < n_cut_he; ++hij) {
        if (cut_m.type[hij] > 1) continue;

        int hji = cut_m.opp[hij];
        int vi = V_map[cut_vtx_reindex[cut_m.v_rep[cut_m.to[hji]]]];
        int vj = V_map[cut_vtx_reindex[cut_m.v_rep[cut_m.to[hij]]]];
        trips.push_back(Trip(vi, vj, hij + 1));
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    // compute new length vector
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> l(num_halfedges, -1.);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m.opp[hij];
        if (hij < hji) continue; // only process each edge once

        int vi = vtx_reindex[m.v_rep[m.to[hji]]];
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];
        int cut_hij = vv2he.coeffRef(vi, vj) - 1;
        assert(cut_hij >= 0);
        m.l[hij] = m.l[hji] = cut_m.l[cut_hij];
    }

    return m;
}

std::vector<bool> FeatureFinder::compute_dual_feature_forest() const
{
    const auto& m = get_mesh();

    // Initialize an array to keep track of vertices
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();
    std::vector<bool> is_processed_face(num_faces, false);
    std::vector<bool> is_dual_forest_halfedge(num_halfedges, false);

    // initialize the stack of vertices to process with all vertices
    std::queue<int> vertices_to_process = {};

    // initialize a dual spanning tree connecting components
    UnionFind component_faces = compute_component_faces();
    std::vector<int> face_labels = component_faces.index_sets();
    int num_components = component_faces.count_sets();
    std::vector<int> is_component_seen(num_components, false);
    std::queue<int> faces_to_process;
    faces_to_process.push(0);

    // do BFS on component with vl as root
    while (!faces_to_process.empty()) {
        // Get the next face to process
        int fi = faces_to_process.front();
        faces_to_process.pop();

        // Skip already processed faces 
        if (is_processed_face[fi]) continue;
        is_processed_face[fi] = true;
        is_component_seen[face_labels[fi]] = true;

        // Iterate over the face via halfedges
        int hij = m.h[fi];
        int hjk = hij;
        do {
            // get current edge data
            hjk = m.n[hjk];
            int hkj = m.opp[hjk];
            int fk = m.f[hkj];

            // check if the tip vertex has been seen yet
            if (is_processed_face[fk]) continue;

            // if opposite face is in unseen component, add to dual forest
            if (!is_component_seen[face_labels[fk]])
            {
                is_component_seen[face_labels[fk]] = true;
                is_dual_forest_halfedge[hjk] = true;
                is_dual_forest_halfedge[hkj] = true;
            }

            // add halfedge to tree
            faces_to_process.push(fk);
        } while (hjk != hij);
    }
    assert(std::find(is_component_seen.begin(), is_component_seen.end(), false) == is_component_seen.end());

    return is_dual_forest_halfedge;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> _refine_corner_feature_faces(const FeatureFinder& feature_finder)
{
    const Mesh<Scalar>& m = feature_finder.get_mesh();
    int num_faces = m.n_faces();
    IntrinsicRefinementMesh refinement_mesh(m);

    std::vector<bool> is_feature_vertex(m.n_vertices(), false);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        // count number of features in the face
        int num_feature_edges = 0;
        int hij = m.h[fijk];
        for (int h : {hij, m.n[hij], m.n[m.n[hij]]})
        {
            if (feature_finder.is_feature_halfedge(h))
            {
                is_feature_vertex[m.to[h]] = true;
                ++num_feature_edges;
            }
        }

        // refine faces with more than one feature
        if (num_feature_edges > 1)
        {
            refinement_mesh.refine_face(fijk);
        }
    }

    // refine faces with all boundary

    // generate VF mesh
    const auto& V = feature_finder.get_vertex_positions();
    const auto& vtx_reindex = feature_finder.get_vertex_reindex();
    return refinement_mesh.generate_mesh(V, vtx_reindex);
}


// TODO Broken
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> refine_strip_faces(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    // Convert VF mesh to halfedge with vertex reindexing
    int num_vertices = V.rows();
    bool fix_boundary = false;
    std::vector<Scalar> Th_hat(num_vertices, 0);
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, free_cones;
    auto m = FV_to_double<Scalar>(
        V,
        F,
        V,
        F,
        Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);

    IntrinsicRefinementMesh refinement_mesh(m);

    // ensure every component has an interior vertex
    std::vector<bool> is_boundary_vertex = compute_boundary_vertices(m);

    Eigen::VectorXi components = find_mesh_face_components(m);
    int num_components = components.maxCoeff() + 1;
    std::vector<bool> has_interior(num_components, false);
    std::vector<int> component_face(num_components, -1);
    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int fijk = m.f[hij];
        component_face[components[fijk]] = fijk;
        if (!is_boundary_vertex[m.to[hij]])
        {
            has_interior[components[fijk]] = true;
        }
    }

    for (int i = 0; i < num_components; ++i)
    {
        if (has_interior[i]) continue;
        spdlog::info("refining component {}", i);
        refinement_mesh.refine_face(component_face[i]);
    }

    // generate VF mesh
    return refinement_mesh.generate_mesh(V, vtx_reindex);
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> _refine_feature_components(const FeatureFinder& feature_finder)
{
    const Mesh<Scalar>& m = feature_finder.get_mesh();
    IntrinsicRefinementMesh refinement_mesh(m);

    // get face components
    UnionFind component_faces = feature_finder.compute_component_faces();
    std::vector<std::vector<int>> components = component_faces.build_sets();

    std::vector<int> feature_degrees = feature_finder.compute_feature_degrees();

    // refined edge on faces with no valence two edges
    for (const auto& component : components) {
        bool has_valence_two = false;
        int feature_h = -1;
        for (int f : component) {
            int hij = m.h[f];
            int hjk = m.n[hij];
            int hki = m.n[hjk];
            for (int h : {hij, hjk, hki}) {
                if (feature_degrees[m.v_rep[m.to[h]]] == 2)
                {
                    has_valence_two = true;
                }
                if (feature_finder.is_feature_halfedge(h))
                {
                    feature_h = h;
                }
            }
            if (has_valence_two) break;
        }
        if (!has_valence_two)
        {
            if (feature_h < 0)
            {
                spdlog::error("No feature edge for component");
                continue;
            }
            refinement_mesh.refine_halfedge(feature_h);
        }
    }

    // generate VF mesh
    const auto& V = feature_finder.get_vertex_positions();
    const auto& vtx_reindex = feature_finder.get_vertex_reindex();
    return refinement_mesh.generate_mesh(V, vtx_reindex);
}

/**
 * @brief Generate a Dirichlet metric for a cut mesh .
 * 
 * @param V_cut: cut mesh vertices
 * @param F_cut: cut mesh faces
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @param direction: per-face reference tangent direction matrix
 * @param is_fixed_direction: per-face mask for salient directions
 * @param marked_metric_params 
 * @param marked_corners 
 * @return std::tuple<
 * DirichletPennerConeMetric,
 * std::vector<int>,
 * std::vector<int>,
 * VectorX,
 * std::vector<Scalar>> 
 */
std::tuple<
    DirichletPennerConeMetric,
    std::vector<int>,
    std::vector<int>,
    VectorX,
    std::vector<Scalar>>
generate_aligned_metric(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXd& direction,
    const std::vector<bool>& is_fixed_direction,
    MarkedMetricParameters marked_metric_params,
    std::vector<std::pair<int, int>> marked_corners)
{
    CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, marked_corners);
    cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction);
    return cut_metric_generator.get_aligned_metric(V_map, marked_metric_params);
}

std::tuple<Mesh<Scalar>, std::vector<int>> generate_union_mesh(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut)
{
    auto [meshes, vertex_maps] = generate_component_meshes(V_cut, F_cut);

    // generate total mesh
    auto mesh = union_meshes(meshes);

    // reindex vertices
    Eigen::VectorXi vertex_map = union_vectors(vertex_maps);
    std::vector<int> vtx_reindex(vertex_map.data(), vertex_map.data() + vertex_map.size());

    return std::make_tuple(mesh, vtx_reindex);
}


std::tuple<MarkedPennerConeMetric, std::vector<int>, std::vector<int>, VectorX, std::vector<Scalar>>
generate_union_metric(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    const Eigen::MatrixXd& direction,
    const std::vector<bool>& is_fixed_direction,
    MarkedMetricParameters marked_metric_params,
    std::vector<std::pair<int, int>> marked_corners)
{
    // generate a cut metric with trivial gluing
    Eigen::VectorXi V_map(0);
    CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, marked_corners);
    cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction);
    return cut_metric_generator.get_union_metric(marked_metric_params);
}
