#include "layout.hh"

#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Layout.hh"
#include "embedding.hh"
#include "energies.hh"
#include "igl/edge_flaps.h"
#include "interpolation.hh"
#include "projection.hh"
#include "refinement.hh"
#include "translation.hh"
#include "viewers.hh"

namespace CurvatureMetric {

OverlayMesh<Scalar> add_overlay(const Mesh<Scalar>& m, const VectorX& reduced_metric_coords)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Build refl projection and embedding
    std::vector<int> proj;
    std::vector<int> embed;
    build_refl_proj(m, he2e, e2he, proj, embed);

    // Build overlay mesh from mesh m
    Mesh<Scalar> m_l = m;

    // Convert mesh Penner coordinates to a halfedge length array l for m
    int num_halfedges = he2e.size();
    for (int h = 0; h < num_halfedges; ++h) {
        m_l.l[h] = exp(reduced_metric_coords[proj[he2e[h]]] / 2.0);
    }

    OverlayMesh<Scalar> mo(m_l);

    return mo;
}

void make_tufted_overlay(
    OverlayMesh<Scalar>& mo,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Scalar>& Theta_hat)
{
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    FV_to_double(V, F, V, F, Theta_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    if (bnd_loops.size() != 0) {
        int n_v = V.rows();
        auto mc = mo.cmesh();
        create_tufted_cover(mo._m.type, mo._m.R, indep_vtx, dep_vtx, v_rep, mo._m.out, mo._m.to);
        mo._m.v_rep = range(0, n_v);
    }
}

bool check_areas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    Eigen::VectorXd areas;
    igl::doublearea(V, F, areas);
    Scalar min_area = areas.minCoeff() / 2.0;
    Scalar max_area = areas.maxCoeff() / 2.0;
    spdlog::info("Minimum face area: {}", min_area);
    spdlog::info("Maximum face area: {}", max_area);

    return (!float_equal(min_area, 0.0));
}

Scalar uv_length_squared(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    Eigen::Vector2d difference_vector = uv_1 - uv_0;
    Scalar length_sq = difference_vector.dot(difference_vector);
    return length_sq;
}

Scalar uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    return sqrt(uv_length_squared(uv_0, uv_1));
}

Scalar compute_uv_length_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    // Get the edge topology for the original uncut mesh
    Eigen::MatrixXi uE, EF, EI;
    Eigen::VectorXi EMAP;
    igl::edge_flaps(F, uE, EMAP, EF, EI);

    // Iterate over edges to check the length inconsistencies
    Scalar max_uv_length_error = 0.0;
    for (Eigen::Index e = 0; e < EF.rows(); ++e) {
        // Get face corners corresponding to the current edge
        int f0 = EF(e, 0);
        int f1 = EF(e, 1);

        // Check first face (if not boundary)
        if (f0 < 0) continue;
        int i0 = EI(e, 0); // corner vertex face index
        int v0n = F_uv(f0, (i0 + 1) % 3); // next vertex
        int v0p = F_uv(f0, (i0 + 2) % 3); // previous vertex

        // Check second face (if not boundary)
        if (f1 < 0) continue;
        int i1 = EI(e, 1); // corner vertex face index
        int v1n = F_uv(f1, (i1 + 1) % 3); // next vertex
        int v1p = F_uv(f1, (i1 + 2) % 3); // next vertex

        // Compute the length of each halfedge corresponding to the corner in the cut mesh
        Scalar l0 = uv_length(uv.row(v0n), uv.row(v0p));
        Scalar l1 = uv_length(uv.row(v1n), uv.row(v1p));

        // Determine if the max length inconsistency has increased
        max_uv_length_error = max(max_uv_length_error, abs(l0 - l1));
    }

    // Return the max uv length error
    return max_uv_length_error;
}

bool check_uv(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    int n_faces = F.rows();
    bool is_valid = true;

    // Check faces agree in number
    if (F_uv.rows() != n_faces) {
        spdlog::error("Mesh and uv faces are not in one to one correspondence");
        is_valid = false;
    }

    // Check length consistency
    if (!float_equal(compute_uv_length_error(F, uv, F_uv), 0.0)) {
        spdlog::warn("Inconsistent uv lengths across edges");
    }

    // Check mesh face areas
    if (!check_areas(V, F)) {
        spdlog::warn("Mesh face area is zero");
    }

    // Check uv face areas
    if (!check_areas(uv, F_uv)) {
        spdlog::warn("Mesh layout face area is zero");
    }

    // Return true if no issues found
    return is_valid;
}

void extract_embedded_mesh(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& corner_to_halfedge)
{
    // Get number of vertices and faces for the embedded mesh
    int num_faces = m.n_faces();
    int num_embedded_faces = 0;
    for (int f = 0; f < num_faces; ++f) {
        // Skip face if it is in the doubled mesh
        int hij = m.h[f];
        // TODO Hack for closed overlay meshes. A better solution would be to have
        // a separate method for overlay meshes
        if ((!m.type.empty()) && (m.type[hij] == 2)) continue;

        // Count embedded faces
        num_embedded_faces++;
    }

    // Build faces and halfedge lengths
    std::vector<std::array<int, 3>> F_vec, corner_to_halfedge_vec;
    F_vec.reserve(num_embedded_faces);
    corner_to_halfedge_vec.reserve(num_embedded_faces);
    for (int f = 0; f < num_faces; ++f) {
        // Get halfedges of face
        int hli = m.h[f];
        int hij = m.n[hli];
        int hjk = m.n[hij];

        // TODO Hack for closed overlay meshes. A better solution would be to have
        // a separate method for overlay meshes
        if ((!m.type.empty()) && (m.type[hij] == 2)) continue;

        // Get vertices of the fist face
        int vi = m.to[hli];
        int vj = m.to[hij];
        int vk = m.to[hjk];

        // Triangle case (l == k)
        if (m.n[hjk] == hli) {
            // Build vertex embedding for the face
            F_vec.push_back({vtx_reindex[vi], vtx_reindex[vj], vtx_reindex[vk]});

            // Build halfedge index map for the face
            corner_to_halfedge_vec.push_back({hjk, hli, hij});
        }
        // Polygon case
        else {
            // Build vertex embedding for the first face
            F_vec.push_back({vtx_reindex[vi], vtx_reindex[vj], vtx_reindex[vk]});

            // Build halfedge index map for the first face
            corner_to_halfedge_vec.push_back({hjk, -1, hij});

            int hkp = m.n[hjk];
            while (m.n[hkp] != hli) {
                // Get vertices of the interior face
                int vk = m.to[hjk];
                int vp = m.to[hkp];

                // Build vertex embedding for the interior face for halfedge hkp
                F_vec.push_back({vtx_reindex[vi], vtx_reindex[vk], vtx_reindex[vp]});

                // Build halfedge index map for the interior face for halfedge hkp
                corner_to_halfedge_vec.push_back({hkp, -1, -1});

                // Increment halfedges
                hkp = m.n[hkp];
                hjk = m.n[hjk];
            }

            // Get vertices of the final face (p == l)
            int vk = m.to[hjk];
            int vp = m.to[hkp];

            // Build vertex embedding for the final face
            F_vec.push_back({vtx_reindex[vi], vtx_reindex[vk], vtx_reindex[vp]});

            // Build halfedge index map for the final face
            corner_to_halfedge_vec.push_back({hkp, hli, -1});
        }
    }

    // Copy lists of lists to matrices
    int num_triangles = F_vec.size();
    F.resize(num_triangles, 3);
    corner_to_halfedge.resize(num_triangles, 3);
    for (int fi = 0; fi < num_triangles; ++fi) {
        for (int j = 0; j < 3; ++j) {
            F(fi, j) = F_vec[fi][j];
            corner_to_halfedge(fi, j) = corner_to_halfedge_vec[fi][j];
        }
    }
}

void compute_layout_faces(
    int origin_size,
    const OverlayMesh<Scalar>& m,
    const std::vector<bool>& is_cut_o,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& F_uv)
{
    int num_halfedges = m.n.size();
    std::vector<int> h_group(m.n.size(), -1);
    std::vector<int> to_map(origin_size, -1);
    std::vector<int> to_group(origin_size, -1);
    std::vector<int> which_to_group(m.out.size(), -1);
    for (int i = 0; i < origin_size; i++) {
        to_group[i] = i;
        which_to_group[i] = i;
    }
    for (int i = 0; i < num_halfedges; i++) {
        if (h_group[i] != -1) continue;
        if (which_to_group[m.to[i]] == -1) {
            which_to_group[m.to[i]] = to_group.size();
            to_group.push_back(m.to[i]);
        }
        if (m.to[i] < origin_size && to_map[m.to[i]] == -1) {
            h_group[i] = m.to[i];
            to_map[m.to[i]] = i;
        } else {
            h_group[i] = to_map.size();
            to_map.push_back(i);
        }
        int cur = m.n[i];
        while (is_cut_o[cur] == false && m.opp[cur] != i) {
            cur = m.opp[cur];
            h_group[cur] = h_group[i];
            cur = m.n[cur];
        }
        cur = m.opp[i];
        while (is_cut_o[cur] == false && m.prev[cur] != i) {
            cur = m.prev[cur];
            h_group[cur] = h_group[i];
            cur = m.opp[cur];
        }
    }

    std::vector<std::vector<int>> F_out;
    std::vector<std::vector<int>> FT_out;
    int num_faces = m.h.size();
    for (int i = 0; i < num_faces; i++) {
        int h0 = m.h[i];
        int hc = m.n[h0];
        std::vector<int> hf;
        hf.push_back(h0);
        while (hc != h0) {
            hf.push_back(hc);
            hc = m.n[hc];
        }
        for (size_t j = 0; j < hf.size() - 2; j++) {
            FT_out.push_back(std::vector<int>{h_group[h0], h_group[hf[j + 1]], h_group[hf[j + 2]]});
            F_out.push_back(std::vector<int>{
                which_to_group[m.to[h0]],
                which_to_group[m.to[hf[j + 1]]],
                which_to_group[m.to[hf[j + 2]]]});
        }
    }

    convert_std_to_eigen_matrix(F_out, F);
    convert_std_to_eigen_matrix(FT_out, F_uv);
}

std::vector<bool>
compute_layout_topology(const Mesh<Scalar>& m, const std::vector<bool>& is_cut_h, int start_h)
{
    // Determine if an initial cut is given
    bool cut_given = !is_cut_h.empty();
    auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);

    // set starting point - use a boundary edge if none given
    int h = 0;
    if (start_h == -1) {
        for (int i = 0; i < m.n_halfedges(); i++) {
            if (m.type[i] == 1 && m.type[m.opp[i]] == 2) {
                h = i;
                is_cut_h_gen[i] = true; // boundary edge should be cut
            }
        }
    }

    // Keep a record of faces that have been laid out
    auto done = std::vector<bool>(m.n_faces(), false);

    // discard part 2 faces by marking as done
    for (size_t i = 0; i < done.size(); i++) {
        int hh = m.h[i];
        if (m.type[hh] == 2 && m.type[m.n[hh]] == 2 && m.type[m.n[m.n[hh]]] == 2) {
            done[i] = true;
        }
    }

    // Set edge type 2 as cut
    for (size_t i = 0; i < is_cut_h.size(); i++) {
        if (m.type[i] == 2) {
            is_cut_h_gen[i] = true;
        }
    }

    // Initialize queue and record of faces to process
    std::queue<int> Q;
    Q.push(h);
    done[m.f[h]] = true;

    while (!Q.empty()) {
        // Get next halfedge to process
        h = Q.front();
        Q.pop();

        // Get other triangle edges
        int hn = m.n[h];
        int hp = m.n[hn];
        int hno = m.opp[hn];
        int hpo = m.opp[hp];
        int ho = m.opp[h];

        // Check if next edge triangle should be laid out
        if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn])) {
            done[m.f[hno]] = true;
            Q.push(hno);
        } else {
            is_cut_h_gen[hn] = true;
            is_cut_h_gen[m.opp[hn]] = true;
        }

        // Check if previous edge triangle should be laid out
        if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp])) {
            done[m.f[hpo]] = true;
            Q.push(hpo);
        } else {
            is_cut_h_gen[hp] = true;
            is_cut_h_gen[m.opp[hp]] = true;
        }

        // Check if current edge triangle should be laid out
        // NOTE: Should only be used once for original edge
        if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho])) {
            done[m.f[ho]] = true;
            Q.push(ho);
        }
    }

    // Check how many faces seen and edges cut
    // TODO Move to separate validity check function
    int num_done = std::count(done.begin(), done.end(), true);
    int num_cut = std::count(is_cut_h_gen.begin(), is_cut_h_gen.end(), true);
    spdlog::trace("{}/{} faces seen", num_done, m.n_faces());
    spdlog::trace("{}/{} halfedges cut", num_cut, is_cut_h_gen.size());
    auto is_found_vertex = std::vector<bool>(m.n_vertices(), false);
    for (int hi = 0; hi < m.n_halfedges(); ++hi) {
        int vi = m.to[hi];
        if (is_cut_h_gen[hi]) {
            is_found_vertex[vi] = true;
        }
    }
    int num_found_vertices = std::count(is_found_vertex.begin(), is_found_vertex.end(), true);
    spdlog::trace("{}/{} vertices seen", num_found_vertices, m.n_vertices());

    Eigen::MatrixXi F, F_uv;
    compute_layout_faces(m.n_vertices(), m, is_cut_h_gen, F, F_uv);
    int num_components = count_components(F_uv);
    if (num_components != 1) {
        spdlog::error("Layout connectivity has {} components", num_components);
    }

    return is_cut_h_gen;
};

// FIXME Remove once fix halfedge origin
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> compute_layout_components(
    Mesh<Scalar>& m,
    const std::vector<Scalar>& u,
    std::vector<bool>& is_cut_h,
    int start_h = -1)
{
    int num_halfedges = is_cut_h.size();

    auto _u = std::vector<Scalar>(m.n_halfedges(), 0.0);
    auto _v = std::vector<Scalar>(m.n_halfedges(), 0.0);

    bool cut_given = !is_cut_h.empty();
    auto is_cut_h_gen = std::vector<bool>(m.n_halfedges(), false);

    auto phi = std::vector<Scalar>(m.n_halfedges(), 0.0);
    auto xi = std::vector<Scalar>(m.n_halfedges(), 0.0);
    for (int i = 0; i < m.n_halfedges(); i++) {
        xi[i] = u[m.to[i]] - u[m.to[m.opp[i]]];
    }

    // set starting point - use a boundary edge
    int h = 0;
    if (start_h == -1) {
        for (int i = 0; i < m.n_halfedges(); i++) {
            if (m.type[i] == 1 && m.type[m.opp[i]] == 2) h = m.n[m.n[i]];
        }
    } else {
        assert(m.f[start_h] != -1);
        h = m.n[m.n[start_h]];
    }

    _u[h] = 0.0;
    _v[h] = 0.0;
    phi[h] = 0.0;
    h = m.n[h];
    assert(m.f[h] != -1);

    phi[h] = xi[h];
    _u[h] = m.l[h] * exp((u[m.to[h]] + u[m.to[m.opp[h]]]) / 2);
    _v[h] = 0.0;
    auto done = std::vector<bool>(m.n_faces(), false);

    // discard part 2
    for (int i = 0; i < m.n_faces(); i++) {
        int hh = m.h[i];
        if (m.type[hh] == 2 && m.type[m.n[hh]] == 2 && m.type[m.n[m.n[hh]]] == 2) {
            done[i] = true;
        }
    }
    // set edge type 2 as cut
    for (int i = 0; i < num_halfedges; i++) {
        if (m.type[i] == 2) {
            is_cut_h[i] = true;
        }
    }

    std::queue<int> Q;
    Q.push(h);
    done[m.f[h]] = true;

    auto perp = [](Eigen::Matrix<Scalar, 1, 2> a) {
        Eigen::Matrix<Scalar, 1, 2> b;
        b[0] = -a[1];
        b[1] = a[0];
        return b;
    };

    auto area_from_len = [](Scalar l1, Scalar l2, Scalar l3) {
        auto s = 0.5 * (l1 + l2 + l3);
        return sqrt(s * (s - l1) * (s - l2) * (s - l3));
    };

    auto square = [](Scalar x) { return x * x; };

    while (!Q.empty()) {
        h = Q.front();
        Q.pop();
        int hn = m.n[h];
        int hp = m.n[hn];
        phi[hn] = phi[h] + xi[hn];
        Eigen::Matrix<Scalar, 1, 2> p1;
        p1[0] = _u[hp];
        p1[1] = _v[hp];
        Eigen::Matrix<Scalar, 1, 2> p2;
        p2[0] = _u[h];
        p2[1] = _v[h];
        assert(m.l[h] != 0.0);
        Scalar l0 = Scalar(1.0);
        Scalar l1 = exp((phi[hn] - phi[hp]) / 2) * (m.l[hn] / m.l[h]);
        Scalar l2 = exp((phi[hn] - phi[h]) / 2) * (m.l[hp] / m.l[h]);
        Eigen::Matrix<Scalar, 1, 2> pn = p1 +
                                         (p2 - p1) * (1 + square(l2 / l0) - square(l1 / l0)) / 2 +
                                         perp(p2 - p1) * 2 * area_from_len(1.0, l1 / l0, l2 / l0);
        _u[hn] = pn[0];
        _v[hn] = pn[1];
        int hno = m.opp[hn];
        int hpo = m.opp[hp];
        int ho = m.opp[h];

        if (m.f[hno] != -1 && !done[m.f[hno]] && !(cut_given && is_cut_h[hn])) {
            done[m.f[hno]] = true;
            phi[hno] = phi[h];
            phi[m.n[m.n[hno]]] = phi[hn];
            _u[hno] = _u[h];
            _v[hno] = _v[h];
            _u[m.n[m.n[hno]]] = _u[hn];
            _v[m.n[m.n[hno]]] = _v[hn];
            Q.push(hno);
        } else {
            is_cut_h_gen[hn] = true;
            is_cut_h_gen[m.opp[hn]] = true;
        }

        if (m.f[hpo] != -1 && !done[m.f[hpo]] && !(cut_given && is_cut_h[hp])) {
            done[m.f[hpo]] = true;
            phi[hpo] = phi[hn];
            phi[m.n[m.n[hpo]]] = phi[hp];
            _u[hpo] = _u[hn];
            _v[hpo] = _v[hn];
            _u[m.n[m.n[hpo]]] = _u[hp];
            _v[m.n[m.n[hpo]]] = _v[hp];
            Q.push(hpo);
        } else {
            is_cut_h_gen[hp] = true;
            is_cut_h_gen[m.opp[hp]] = true;
        }

        if (m.f[ho] != -1 && !done[m.f[ho]] && !(cut_given && is_cut_h[ho])) {
            done[m.f[ho]] = true;
            phi[ho] = phi[hp];
            phi[m.n[m.n[ho]]] = phi[h];
            _u[ho] = _u[hp];
            _v[ho] = _v[hp];
            _u[m.n[m.n[ho]]] = _u[h];
            _v[m.n[m.n[ho]]] = _v[h];
            Q.push(ho);
        }

        // If Q is empty and there are faces left, continue layout with a new halfedge
        if (Q.empty()) {
            for (int fi = 0; fi < m.n_faces(); ++fi) {
                if (!done[fi]) {
                    int h = m.h[fi];
                    _u[h] = 0.0;
                    _v[h] = 0.0;
                    phi[h] = 0.0;
                    h = m.n[h];
                    assert(m.f[h] != -1);

                    phi[h] = xi[h];
                    _u[h] = m.l[h] * exp((u[m.to[h]] + u[m.to[m.opp[h]]]) / 2);
                    _v[h] = 0.0;

                    Q.push(h);
                    done[fi] = true;
                    break;
                }
            }
        }
    }

    return std::make_tuple(_u, _v, is_cut_h_gen);
};

// Pull back cut on the current mesh to the overlay
std::vector<bool> pullback_current_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h)
{
    int num_halfedges = m_o.n_halfedges();
    std::vector<bool> is_cut_o = std::vector<bool>(num_halfedges, false);
    for (int hi = 0; hi < num_halfedges; ++hi) {
        if (m_o.edge_type[hi] == CURRENT_EDGE) {
            is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
        } else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE) {
            is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
        }
        // Don't cut edges not in the current mesh
        else if (m_o.edge_type[hi] == ORIGINAL_EDGE) {
            continue;
        }
    }

    return is_cut_o;
}

std::vector<bool> pullback_original_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h)
{
    int num_halfedges = m_o.n_halfedges();
    std::vector<bool> is_cut_o = std::vector<bool>(num_halfedges, false);
    for (int hi = 0; hi < num_halfedges; ++hi) {
        // Don't cut edges not in the original mesh
        if (m_o.edge_type[hi] == CURRENT_EDGE) {
            continue;
        } else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE) {
            is_cut_o[hi] = is_cut_h[m_o.origin_of_origin[hi]];
        } else if (m_o.edge_type[hi] == ORIGINAL_EDGE) {
            is_cut_o[hi] = is_cut_h[m_o.origin[hi]];
        }
    }

    return is_cut_o;
}

std::vector<bool> pullback_cut_to_overlay(
    OverlayMesh<Scalar>& m_o,
    const std::vector<bool>& is_cut_h,
    bool is_original_cut)
{
    if (is_original_cut) {
        return pullback_original_cut_to_overlay(m_o, is_cut_h);
    } else {
        return pullback_current_cut_to_overlay(m_o, is_cut_h);
    }
}

// Helper function to determine if any faces in a triangle mesh are flipped
void check_if_flipped(Mesh<Scalar>& m, const std::vector<Scalar>& u, const std::vector<Scalar>& v)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> uv(u.size(), 2);
    Eigen::MatrixXi F_uv(m.n_faces(), 3);

    for (size_t i = 0; i < u.size(); ++i) {
        uv(i, 0) = static_cast<double>(u[i]);
        uv(i, 1) = static_cast<double>(v[i]);
    }

    for (int fi = 0; fi < m.n_faces(); ++fi) {
        int h = m.h[fi];
        F_uv(fi, 0) = h;
        F_uv(fi, 1) = m.n[h];
        F_uv(fi, 2) = m.n[m.n[h]];
    }

    Eigen::VectorXi flipped_f;
    igl::flipped_triangles(uv, F_uv, flipped_f);
    spdlog::info("{} flipped elements in mesh", flipped_f.size());
    for (int i = 0; i < flipped_f.size(); ++i) {
        int fi = flipped_f[i];
        spdlog::info("Face {} is flipped", F_uv.row(fi));
        spdlog::info(
            "Vertices {}, {}, {}",
            uv.row(F_uv(fi, 0)),
            uv.row(F_uv(fi, 1)),
            uv.row(F_uv(fi, 2)));
    }
}

std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>, std::vector<bool>>
get_consistent_layout(
    OverlayMesh<Scalar>& m_o,
    const std::vector<Scalar>& u_vec,
    std::vector<int> singularities,
    const std::vector<bool>& is_cut_orig,
    const std::vector<bool>& is_cut)
{
    // Get original overlay face labels
    auto f_labels = get_overlay_face_labels(m_o);

    // Compute layout of the underlying flipped mesh
    auto mc = m_o.cmesh();
    m_o.garbage_collection();
    if (!m_o.check(&mc)) {
        spdlog::error("Invalid overlay mesh for layout");
        return std::make_tuple(
            std::vector<Scalar>(),
            std::vector<Scalar>(),
            std::vector<bool>(),
            std::vector<bool>());
    }
    if (!overlay_has_all_original_halfedges(m_o)) {
        spdlog::error("Overlay mesh is missing an original edge");
        return std::make_tuple(
            std::vector<Scalar>(),
            std::vector<Scalar>(),
            std::vector<bool>(),
            std::vector<bool>());
    }

    mc.type = std::vector<char>(mc.n_halfedges(), 0);
    //std::vector<bool> _is_cut_place_holder; // TODO Remove
    // auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder, 0);
    std::vector<bool> _is_cut_place_holder = is_cut;
    auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder);
    auto _u_c = std::get<0>(layout_res);
    auto _v_c = std::get<1>(layout_res);
    auto is_cut_c = std::get<2>(layout_res);

    // Interpolate layout to the overlay mesh
    Eigen::Matrix<Scalar, -1, 1> u_eig;
    u_eig.resize(u_vec.size());
    for (size_t i = 0; i < u_vec.size(); i++) {
        u_eig(i) = u_vec[i];
    }
    m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);
    auto u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _u_c);
    auto v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _v_c);
    spdlog::trace("Interpolate on overlay mesh done.");

    // Build a new mesh directly from the triangulated overlay mesh
    Mesh<Scalar> m;
    m.n = m_o.n;
    m.opp = m_o.opp;
    m.f = m_o.f;
    m.h = m_o.h;
    m.out = m_o.out;
    m.to = m_o.to;
    m.l = std::vector<Scalar>(m.n.size(), 0.0);
    for (int i = 0; i < m.n_halfedges(); i++) {
        int h0 = i;
        int h1 = h0;
        do {
            if (m.n[h1] == h0) break;
            h1 = m.n[h1];
        } while (h0 != h1);
        if (m.to[m.opp[h0]] != m.to[h1]) {
            spdlog::error("h0 h1 picked wrong.");
            exit(0);
        }
        m.l[h0] = sqrt(
            (u_o[h0] - u_o[h1]) * (u_o[h0] - u_o[h1]) + (v_o[h0] - v_o[h1]) * (v_o[h0] - v_o[h1]));
    }
    triangulate_polygon_mesh(m, u_o, v_o, f_labels);
    m.type = std::vector<char>(m.n.size(), 0);
    m.type_input = m.type;
    m.R = std::vector<int>(m.n.size(), 0);
    m.v_rep = range(0, m.out.size());
    m.Th_hat = std::vector<Scalar>(m.out.size(), 0.0);
    OverlayMesh<Scalar> m_o_tri(m);
    for (int i = m_o.n_halfedges(); i < m_o_tri.n_halfedges(); i++) {
        m_o_tri.edge_type[i] = CURRENT_EDGE; // make sure do not use the new diagonal
    }
    bool view_layouts = false;
    if (view_layouts) {
        spdlog::info("Viewing layouts");
        view_halfedge_mesh_layout(m, u_o, v_o);
    }

    // Pullback cut on the original mesh to the overlay
    bool is_original_cut = !(is_cut_orig.empty());
    std::vector<bool> is_cut_poly;
    if (is_original_cut) {
        is_cut_poly = pullback_cut_to_overlay(m_o, is_cut_orig, true);
    } else {
        is_cut_poly = pullback_cut_to_overlay(m_o, is_cut, false);
    }

    // Check validity (not needed)
    Eigen::MatrixXi F_poly, F_uv_poly;
    compute_layout_faces(mc.n_vertices(), m_o, is_cut_poly, F_poly, F_uv_poly);
    int num_poly_components = count_components(F_uv_poly);
    if (num_poly_components != 1) {
        spdlog::error("Overlay connectivity has {} components", num_poly_components);
    }

    // Extend the overlay cut to the triangulated mesh
    // WARNING: Assumes triangulation halfedges added to the end
    std::vector<bool> is_cut_o = std::vector<bool>(m.n_halfedges(), false);
    for (int h = 0; h < m_o.n_halfedges(); ++h) {
        is_cut_o[h] = is_cut_poly[h];
    }

    // Check validity (not needed)
    Eigen::MatrixXi F_tri, F_uv_tri;
    compute_layout_faces(mc.n_vertices(), m, is_cut_o, F_tri, F_uv_tri);
    int num_tri_components = count_components(F_uv_tri);
    if (num_tri_components != 1) {
        spdlog::error("Triangulated overlay connectivity has {} components", num_tri_components);
    }

    // Now directly do layout on triangulated overlay mesh
    std::vector<Scalar> phi(m.n_vertices(), 0.0);
    auto overlay_layout_res = compute_layout_components(m, phi, is_cut_o);
    std::vector<Scalar> _u_o = std::get<0>(overlay_layout_res);
    std::vector<Scalar> _v_o = std::get<1>(overlay_layout_res);
    is_cut_o = std::get<2>(overlay_layout_res);
    if (view_layouts) {
        spdlog::info("Viewing layouts");
        view_halfedge_mesh_layout(m, _u_o, _v_o);
    }

    // Restrict back to original overlay
    // WARNING: Assumes triangulation halfedges added to the end
    _u_o.resize(m_o.n.size());
    _v_o.resize(m_o.n.size());
    is_cut_o.resize(m_o.n.size());

    // Trim unnecessary branches of the cut graph
    bool do_trim = true;
    if (do_trim) {
        trim_open_branch(m_o, f_labels, singularities, is_cut_o);
    }

    return std::make_tuple(_u_o, _v_o, is_cut_c, is_cut_o);
}

#ifdef PYBIND
std::
    tuple<
        Eigen::MatrixXi, // F
        Eigen::MatrixXi // corner_to_halfedge
        >
    extract_embedded_mesh_pybind(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex)
{
    Eigen::MatrixXi F;
    Eigen::MatrixXi corner_to_halfedge;
    extract_embedded_mesh(m, vtx_reindex, F, corner_to_halfedge);
    return std::make_tuple(F, corner_to_halfedge);
}

#endif

} // namespace CurvatureMetric
