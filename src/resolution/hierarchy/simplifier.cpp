#include "resolution/hierarchy/simplifier.h"
#include "resolution/core/common.h"
#include "metric/viewer.h"
#include "metric/constraint.h"
#include "parametrization/triangulation.h"
#include "util/union_find.h"
#include "util/map.h"
#include "util/vector.h"
#include "util/vf_mesh.h"
#include "holonomy/core/viewer.h"
#include "parametrization/layout.h"
#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Layout.hh"

#include "resolution/core/geodesics.h"
#include "resolution/core/geodesic_flipper.h"

#include <igl/remove_unreferenced.h>
#include <stack>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner
{
namespace Resolution
{

void Simplifier::refine_halfedges()
{
    int num_faces = m.n_faces();

    // get all invalid faces
    std::stack<int> invalid_faces;
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        if (is_refined_face[fijk]) continue;
        if (!is_face_self_overlapping(fijk)) invalid_faces.push(fijk);
    }
    spdlog::info("{} faces are invalid", invalid_faces.size());

    // Refine faces until all are valid
    int num_refined_halfedges = 0;
    while (!invalid_faces.empty()) {
        // Get invalid face
        int fijk = invalid_faces.top();
        invalid_faces.pop();
        spdlog::trace("Processing face {}", fijk);

        // Check if face already processed to prevent an infinite loop
        // NOTE: This should not happen, but may do to floating point error
        // or invalid inputs
        if (is_refined_face[fijk]) continue;

        int hij = m.h[fijk];
        int count = 0;
        for (int i = 0; i < 3; ++i)
        {
            // check if current face is self overlapping and refine if not
            if (is_face_self_overlapping(fijk)) break;
            is_refined_halfedge[hij] = true;
            is_refined_halfedge[m.opp[hij]] = true;
            ++count;
            ++num_refined_halfedges;

            // check if opposite face is self overlapping
            int fjil = m.f[m.opp[hij]];
            if (!is_face_self_overlapping(fjil)) invalid_faces.push(fjil);

            // iterate around face
            hij = m.n[hij];
        }

        // mark the face as refined if all halfedges are refined
        // TODO: make just a function
        is_refined_face[fijk] = ((count == 3) && (!is_face_self_overlapping(fijk)));
    }

    spdlog::info("{} halfedges were refined", num_refined_halfedges);
}

void Simplifier::simplify_vertices()
{
    std::vector<int> refined_he, unrefined_he, he_map;
    enumerate_boolean_array(is_refined_halfedge, refined_he, unrefined_he, he_map);
    int count = 0;

    // Iterate until no vertices can be removed
    bool vertex_removed = false;
    int max_iterations = 1;
    int iter = 0;
    do {
        // Loop over all inserted vertices to see if one can be removed
        vertex_removed = false;
        for (int hij : refined_he)
        {
            // get edge halfedges
            int f = m.f[hij];
            int fo = m.f[m.opp[hij]];
            std::vector<int> edge_vertices = get_halfedge_vertices(hij);
            int size = edge_vertices.size();
            for (int i = 0; i < size - 1; ++i)
            {
                int v = edge_vertices[i];
                if (is_removed_vertex[v]) continue;

                // try to remove vertex, and reset if not
                is_removed_vertex[v] = true;
                if ((is_face_self_overlapping(f)) && (is_face_self_overlapping(fo)))
                {
                    vertex_removed = true;
                    ++count;
                }
                else
                {
                    is_removed_vertex[v] = false;
                }
            }
        }
        spdlog::info("{} vertices were removed from refinement", count);

        ++iter;
        if (iter >= max_iterations) break;
    } while (vertex_removed);
 
}

std::vector<int> Simplifier::get_halfedge_vertices(int halfedge_index) const
{
    std::vector<int> halfedges = get_halfedges(halfedge_index);
    return vector_compose(to, halfedges);
}

void Simplifier::get_face_vertex_indices(
    int face_index,
    std::vector<int>& uv_vertices,
    std::vector<int>& vertices) const
{
    int hij = m.h[face_index];

    for (int i = 0; i < 3; ++i)
    {
        // add all vertices on edge if refined
        if (is_refined_halfedge[hij])
        {
            std::vector<int> edge_halfedges = get_halfedges(hij);
            for (int h : edge_halfedges)
            {
                int v = to[h];
                if (is_removed_vertex[v]) continue;
                vertices.push_back(v);
                uv_vertices.push_back(h);
            }
        }
        // add the uv coordiante for just the tip vertex if not refined
        else
        {
            int h = end_halfedge[hij];
            int v = to[h];
            vertices.push_back(v);
            uv_vertices.push_back(h);
        }

        // increment halfedge
        hij = m.n[hij];
    }

}

std::vector<int> SimplificationOverlaySimplifier::get_halfedges(int halfedge_index) const
{
    return parent.enumerate_halfedges(start_halfedge[halfedge_index], parent.orig);
}

bool SimplificationOverlaySimplifier::is_face_self_overlapping(int face_index) const
{
    if (is_refined_face[face_index]) return false;

    // get vertices for the face
    std::vector<Eigen::Vector2d> uv_vertices = {};
    std::vector<Eigen::Vector3d> vertices = {};
    get_face_vertices(face_index, uv_vertices, vertices);

    // Determine if the face is self overlapping
    std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
    std::vector<std::vector<int>> splitting_vertices;
    std::vector<std::vector<Scalar>> min_face_areas;
    return is_self_overlapping_polygon(
        uv_vertices,
        vertices,
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        area_threshold,
        use_angles);
}

void SimplificationOverlaySimplifier::get_face_vertices(
    int face_index,
    std::vector<Eigen::Vector2d>& uv_vertices,
    std::vector<Eigen::Vector3d>& vertices) const
{
    std::vector<int> uv_vertex_indices, vertex_indices;
    get_face_vertex_indices(face_index, uv_vertex_indices, vertex_indices);

    int num_vertices = vertex_indices.size();
    uv_vertices.resize(num_vertices);
    vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        int h = uv_vertex_indices[i];
        int v = vertex_indices[i];
        uv_vertices[i] = he_uv[h];
        vertices[i] = V_o.row(v);
    }
}

bool SimplificationOverlaySimplifier::triangulate_face(
    int face_index,
    std::vector<std::array<int, 3>>& face_triangles,
    std::vector<std::array<int, 3>>& uv_face_triangles,
    std::vector<std::array<int, 3>>& face_indices) const
{
    // Get uv vertices of the face
    std::vector<int> vertex_indices;
    std::vector<int> uv_vertex_indices;
    std::vector<Eigen::Vector2d> uv_vertices;
    std::vector<Eigen::Vector3d> vertices;
    get_face_vertex_indices(face_index, uv_vertex_indices, vertex_indices);
    get_face_vertices(face_index, uv_vertices, vertices);

    // Determine if the face is self overlapping
    std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
    std::vector<std::vector<int>> splitting_vertices;
    std::vector<std::vector<Scalar>> min_face_areas;
    bool is_self_overlapping = is_self_overlapping_polygon(
        uv_vertices,
        vertices,
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        area_threshold,
        use_angles);

    // Return false if the face is not self overlapping
    if (!is_self_overlapping) {
        //spdlog::warn("Face {} is not self overlapping", face_index);
        return false;
    }

    // Triangulate the face if it can be triangulated
    std::vector<std::array<int, 3>> polygon_faces;
    triangulate_self_overlapping_polygon(
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        polygon_faces);

    // Convert polygon face indices to global indices for V and uv
    int num_polygon_faces = polygon_faces.size();
    for (int fi = 0; fi < num_polygon_faces; ++fi) {
        face_triangles.push_back({});
        uv_face_triangles.push_back({});
        face_indices.push_back({});
        for (int j = 0; j < 3; ++j) {
            face_triangles.back()[j] = vertex_indices[polygon_faces[fi][j]];
            uv_face_triangles.back()[j] = uv_vertex_indices[polygon_faces[fi][j]];
            face_indices.back()[j] = polygon_faces[fi][j];
        }
    }

    return true;
}

std::tuple<
Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi,
std::vector<int>, Eigen::MatrixXi, Eigen::MatrixXi
>
SimplificationOverlaySimplifier::build_refined_mesh()
{
    int num_faces = m.n_faces();
    std::vector<std::array<int, 3>> face_triangles;
    std::vector<std::array<int, 3>> uv_face_triangles;
    std::vector<std::array<int, 3>> face_indices;
    std::vector<std::array<bool, 3>> is_triangle_edge;

    // get uv vertices
    auto [h_new2old, h_old2new] = parent.get_halfedge_reductions();
    UnionFind cut_vertices(h_new2old.size());
    for (int hij = 0; hij < parent.n.size(); ++hij)
    {
        if (parent.is_halfedge_deleted(hij)) continue;

        // union halfedges that are not on the layout cut
        if (parent.orig.edge_type[parent.h2e[hij]] == EdgeType::EE)
        {
            int h_o = parent.orig.halfedge_parent[hij];
            if (is_cut_orig[h_o])
                continue;
        } 

        // get halfedge in adjacent uncut face pointing to vj
        int hkj = parent.prev[parent.opp[hij]];
        cut_vertices.union_sets(h_old2new[hij], h_old2new[hkj]);
    }
    int num_cut_vertices = cut_vertices.count_sets();
    std::vector<int> cut_vertex_indices = cut_vertices.index_sets();

    // triangulate faces that are not completely refined
    std::vector<int> face_parent = {};
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_refined_face[f]) continue;
        is_refined_face[f] = !(triangulate_face(f, face_triangles, uv_face_triangles, face_indices));
        int curr_faces = face_parent.size();
        int size = face_triangles.size() - curr_faces + 2;
        for (int i = curr_faces; i < face_triangles.size(); ++i)
        {
            face_parent.push_back(f);
            is_triangle_edge.push_back({});
            for (int j = 0; j < 3; ++j)
            {
                if (face_indices[i][j] == (face_indices[i][(j + 2) % 3] + 1) % size)
                {
                    is_triangle_edge.back()[(j + 1) % 3] = true; 
                }
                else
                {
                    is_triangle_edge.back()[(j + 1) % 3] = false; 
                }
            }
        }
    }

    // add all refined faces from original mesh
    for (int f = 0; f < parent.f2h.size(); ++f)
    {
        if (parent.is_face_deleted(f)) continue;
        if (!is_refined_face[parent.orig.face_parent[f]]) continue;
        int hij = parent.f2h[f];
        face_parent.push_back(parent.orig.face_parent[f]);
        is_triangle_edge.push_back({});
        face_triangles.push_back({});
        uv_face_triangles.push_back({});
        face_indices.push_back({});
        for (int i = 0; i < 3; ++i)
        {
            face_triangles.back()[i] = parent.to[parent.n[hij]];
            uv_face_triangles.back()[i] = parent.n[hij];
            if (parent.orig.edge_type[parent.h2e[hij]] == EdgeType::EE)
            {
                is_triangle_edge.back()[i] = true; 
            }
            hij = parent.n[hij];
        }
    }

    // build eigen matrices
    int num_refined_faces = face_triangles.size();
    Eigen::MatrixXd uv_o(num_cut_vertices, 2);
    Eigen::MatrixXi F_o(num_refined_faces, 3);
    Eigen::MatrixXi F_uv_o(num_refined_faces, 3);
    Eigen::MatrixXi halfedge_parent(num_refined_faces, 3);
    Eigen::MatrixXi is_edge = Eigen::MatrixXi::Zero(num_refined_faces, 3);
    for (int f = 0; f < num_refined_faces; ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            int hij = uv_face_triangles[f][i];
            int hjk = uv_face_triangles[f][j];
            int hki = uv_face_triangles[f][k];
            //int hki = parent.orig.m.n[parent.orig.m.n[hij]];
            int uvi = cut_vertex_indices[h_old2new[hij]];
            uv_o.row(uvi) = he_uv[hij];
            F_o(f, i) = face_triangles[f][i];
            F_uv_o(f, i) = uvi;

            if (is_triangle_edge[f][i])
            {
                 is_edge(f, i) = 1;
                 halfedge_parent(f, i) = parent.orig.halfedge_parent[hki];
            }
        }
    }

    // Remove unreferenced vertices
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F, F_uv;
    Eigen::VectorXi I, J, I_uv, J_uv;
    igl::remove_unreferenced(V_o, F_o, V, F, I, J);
    igl::remove_unreferenced(uv_o, F_uv_o, uv, F_uv, I_uv, J_uv);

    return std::make_tuple(V, F, uv, F_uv, face_parent, halfedge_parent, is_edge);
}


void OverlaySimplifier::get_face_vertices(
    int face_index,
    std::vector<Eigen::Vector2d>& uv_vertices) const
{
    std::vector<int> uv_vertex_indices, vertex_indices;
    get_face_vertex_indices(face_index, uv_vertex_indices, vertex_indices);

    int num_vertices = vertex_indices.size();
    uv_vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        int h = uv_vertex_indices[i];
        uv_vertices[i] = he_uv[h];
    }
}

std::vector<int> OverlaySimplifier::get_halfedges(int halfedge_index) const
{
    int seg = start_halfedge[halfedge_index];
    std::vector<int> overlay_halfedges = {seg};
    while (m_o.vertex_type[m_o.to[seg]] != ORIGINAL_VERTEX)
    {
        seg = next_orig_segment(m_o, seg);
        overlay_halfedges.push_back(seg);
    }

    return overlay_halfedges;
}

bool OverlaySimplifier::is_face_self_overlapping(int face_index) const
{
    if (is_refined_face[face_index]) return false;

    // get vertices for the face
    std::vector<Eigen::Vector2d> uv_vertices = {};
    get_face_vertices(face_index, uv_vertices);

    // Determine if the face is self overlapping
    std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
    std::vector<std::vector<int>> splitting_vertices;
    std::vector<std::vector<Scalar>> min_face_areas;
    return is_self_overlapping_polygon(
        uv_vertices,
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        area_threshold,
        use_angles);
}

bool OverlaySimplifier::triangulate_face(
    int face_index,
    std::vector<std::array<int, 3>>& face_triangles,
    std::vector<std::array<int, 3>>& uv_face_triangles) const
{
    // Get uv vertices of the face
    std::vector<int> vertex_indices;
    std::vector<int> uv_vertex_indices;
    std::vector<Eigen::Vector2d> uv_vertices;
    get_face_vertex_indices(face_index, uv_vertex_indices, vertex_indices);
    get_face_vertices(face_index, uv_vertices);

    // Determine if the face is self overlapping
    std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
    std::vector<std::vector<int>> splitting_vertices;
    std::vector<std::vector<Scalar>> min_face_areas;
    bool is_self_overlapping = is_self_overlapping_polygon(
        uv_vertices,
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        area_threshold,
        use_angles);

    // Return false if the face is not self overlapping
    if (!is_self_overlapping) {
        //spdlog::warn("Face {} is not self overlapping", face_index);
        return false;
    }

    // Triangulate the face if it can be triangulated
    std::vector<std::array<int, 3>> polygon_faces;
    triangulate_self_overlapping_polygon(
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        polygon_faces);

    // Convert polygon face indices to global indices for V and uv
    int num_polygon_faces = polygon_faces.size();
    for (int fi = 0; fi < num_polygon_faces; ++fi) {
        face_triangles.push_back({});
        uv_face_triangles.push_back({});
        for (int j = 0; j < 3; ++j) {
            face_triangles.back()[j] = vertex_indices[polygon_faces[fi][j]];
            uv_face_triangles.back()[j] = uv_vertex_indices[polygon_faces[fi][j]];
        }
    }

    return true;
}

std::tuple<OverlayMesh<Scalar>, std::vector<int>, std::vector<Eigen::Vector2d>> OverlaySimplifier::get_refined_mesh()
{
    std::vector<int> fn_to_f;
    build_face_maps(m_o, fn_to_f);

    std::vector<std::array<int, 3>> face_triangles;
    std::vector<std::array<int, 3>> uv_face_triangles;
    for (int f = 0; f < m.n_faces(); ++f)
    {
        if (is_refined_face[f]) continue;
        is_refined_face[f] = !(triangulate_face(f, face_triangles, uv_face_triangles));
    }

    // add all refined faces from original mesh
    int num_faces_refined = 0;
    for (int f = 0; f < m_o.n_faces(); ++f)
    {
        int f_orig = fn_to_f[f];
        if (!is_refined_face[f_orig]) continue;
        int hij = m_o.h[f];
        face_triangles.push_back({});
        uv_face_triangles.push_back({});
        for (int i = 0; i < 3; ++i)
        {
            face_triangles.back()[i] = m_o.to[hij];
            uv_face_triangles.back()[i] = hij;
            hij = m_o.n[hij];
        }
        num_faces_refined++;
    }
    spdlog::info("{} faces refined", num_faces_refined);

    int num_corners = 3 * uv_face_triangles.size();
    int num_halfedges = m_o.n_halfedges();
    std::vector<int> halfedges(num_corners, -1);
    std::vector<int> n_r(num_corners, -1);
    std::vector<int> to_r(num_corners, -1);
    std::vector<int> f_r(num_corners, -1);
    std::vector<Eigen::Vector2d> uv_r(num_corners);
    Eigen::SparseMatrix<int> vv2he(num_halfedges, num_halfedges);
    std::vector<int> fn_to_f_o(uv_face_triangles.size(), -1);

    // build sparse matrix
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    for (int f = 0; f < uv_face_triangles.size(); ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            int hj = 3 * f + j;
            int hk = 3 * f + k;
            int vi = uv_face_triangles[f][i];
            int vj = uv_face_triangles[f][j];

            halfedges[hj] = vj;
            n_r[hj] = hk;
            f_r[hj] = f;
            to_r[hj] = m_o.to[vj];
            uv_r[hj] = he_uv[vj];
            fn_to_f_o[f] = fn_to_f[m_o.f[vj]];
            trips.push_back(Trip(vi, vj, hj + 1));
        }
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    // build map from original to reduced halfedges
    std::vector<int> he_map(num_halfedges, -1);
    for (int hij = 0; hij < num_corners; ++hij)
    {
        int hjk = n_r[hij];
        int hki = n_r[hjk];
        int vi = halfedges[hki];
        int vj = halfedges[hij];
        assert(vv2he.coeff(vi, vj) - 1 == hij);
        int hji = vv2he.coeff(vj, vi) - 1;

        // edge crossing faces
        if (hji < 0)
        {
            he_map[vj] = hij;
        }
    }

    // build opp maps
    std::vector<int> opp_r(num_corners, -1);
    std::vector<bool> is_inserted(num_corners, false);
    for (int hij = 0; hij < num_corners; ++hij)
    {
        int hjk = n_r[hij];
        int hki = n_r[hjk];
        int vi = halfedges[hki];
        int vj = halfedges[hij];
        assert(vv2he.coeff(vi, vj) - 1 == hij);
        int hji = vv2he.coeff(vj, vi) - 1;

        // edge crossing faces
        if (hji < 0)
        {
            is_inserted[hij] = false;
            int h = m_o.origin_of_origin[vj];
            if (m_o.edge_type[vj] == ORIGINAL_EDGE)
                h = m_o.origin[vj];
            if (is_refined_halfedge[h])
            {
                opp_r[hij] = he_map[m_o.opp[vj]];
                spdlog::debug("building refined edge {}, {} in {}, {}", hij, opp_r[hij], h, m.opp[h]);
            }
            else
            {
                opp_r[hij] = he_map[end_halfedge[m.opp[h]]];
            }
        }
        // face triangulation edge
        else
        {
            spdlog::debug("building inserted edge {}, {}", hij, hji);
            is_inserted[hij] = true;
            opp_r[hij] = hji;
        }
    }

    Mesh<Scalar> m_r;
    m_r.n = n_r;
    m_r.opp = opp_r;
    m_r.to = remove_unreferenced_indices(to_r); // need to remove unreferenced
    m_r.f = f_r;
    m_r.out = invert_map(vector_compose(m_r.to, m_r.opp));
    m_r.h = invert_map(m_r.f);

    // check validity
    std::vector<int> prev = invert_map(m_r.n);
    assert(are_polygon_mesh_halfedges_valid(m_r.n, prev));
    assert(are_polygon_mesh_vertices_valid(m_r.opp, prev, m_r.to, m_r.out));
    assert(are_polygon_mesh_faces_valid(m_r.n, m_r.f, m_r.h));

    m_r.l = std::vector<Scalar>(m_r.n.size(), 1.);
    m_r.type = std::vector<char>(m_r.n.size(), 0);
    m_r.type_input = std::vector<char>(m_r.n.size(), 0);
    m_r.R = std::vector<int>(m_r.n.size(), 0);
    m_r.v_rep = arange(m_r.out.size());
    m_r.Th_hat = std::vector<Scalar>(m_r.out.size(), 2 * PI);
    m_r.fixed_dof = std::vector<bool>(m_r.out.size(), false);
    m_r.fixed_dof[0] = true;

    for (int hij = 0; hij < num_corners; ++hij)
    {
        int hjk = m_r.n[hij];
        int hki = m_r.n[hjk];
        m_r.l[hij] = (uv_r[hij] - uv_r[hki]).norm();
    }

    OverlayMesh<Scalar> m_r_o(m_r);
    for (int hij = 0; hij < num_corners; ++hij)
    {
        if (is_inserted[hij])
        {
            m_r_o.edge_type[hij] = CURRENT_EDGE;
        }
        else
        {
            int h = halfedges[hij];
            assert(h >= 0);
            assert(h < m_o.n_halfedges());

            // transfer edge type data from original overlay
            //m_r_o.edge_type[hij] = m_o.edge_type[h];
            //m_r_o.origin_of_origin[hij] = m_o.origin_of_origin[h];
            //m_r_o.origin[hij] = m_o.origin[hij];

            m_r_o.edge_type[hij] = ORIGINAL_EDGE;
            m_r_o.origin[hij] = m_o.origin_of_origin[h];
            if (m_o.edge_type[h] == ORIGINAL_EDGE)
                m_r_o.origin[hij] = m_o.origin[h];

            // get vertex type as well
            int vj = m_r_o.to[hij];
            int Vj = m_o.to[h];
            assert (Vj >= 0);
            assert (Vj < m_o.n_vertices());
            m_r_o.vertex_type[vj] = m_o.vertex_type[Vj];
            assert((m_o.vertex_type[Vj] == ORIGINAL_VERTEX) || (m_o.vertex_type[Vj] == SEGMENT_VERTEX));
            if (m_r_o.vertex_type[vj] == SEGMENT_VERTEX)
            {
                spdlog::debug("inserted segment vertex {}, relabelled {}, on {}", Vj, vj, m_r_o.origin[hij]);
            }
        }
    }

    return std::make_tuple(m_r_o, fn_to_f_o, uv_r);
}

void SimplificationOverlaySimplifier::mark_cut_faces(const std::vector<bool>& is_cut_halfedge)
{
    int num_halfedges = parent.n.size();
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (parent.is_halfedge_deleted(hij)) continue;

        // check if seam edge
        if (parent.simp.edge_type[parent.h2e[hij]] != EdgeType::EE) continue;
        int h_o = parent.simp.halfedge_parent[hij];
        if (!is_cut_halfedge[h_o]) continue;

        // check if interior edge
        if (parent.orig.edge_type[parent.h2e[hij]] != EdgeType::EF) continue;
        int f = parent.orig.face_parent[parent.face[hij]];

        is_refined_face[f] = true;
    }

    // mark halfedges of refined faces as refined
    for (int f = 0; f < m.n_faces(); ++f)
    {
        if (!is_refined_face[f]) continue;

        int hij = m.h[f];
        for (int i = 0; i < 3; ++i)
        {
            is_refined_halfedge[hij] = true;
            is_refined_halfedge[m.opp[hij]] = true;
            hij = m.n[hij];
        }
    }
}

void SimplificationOverlaySimplifier::view_parameterization()
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();
    Eigen::MatrixXi F(num_faces, 3);
    Eigen::MatrixXi F_uv(num_faces, 3);
    Eigen::MatrixXd uv(num_halfedges, 2);
    std::vector<int> is_self_overlapping(num_faces, 0);
    for (int f = 0; f < num_faces; ++f)
    {
        int hij = m.h[f];
        for (int i = 0; i < 3; ++i)
        {
            F(f, i) = parent.vtx_reindex_orig[m.v_rep[m.to[hij]]];
            F_uv(f, i) = hij;
            uv.row(hij) = he_uv[end_halfedge[hij]];

            // increment halfedge
            hij = m.n[hij];
        }
        if (is_face_self_overlapping(f)) is_self_overlapping[f] = 1;
    }

    Eigen::MatrixXd& V = parent.V_orig;
//    auto [V_r, F_r, uv_r, FT_r] = build_refined_mesh();
    //Holonomy::view_seamless_parameterization(V, F, uv, F_uv, "simplification", false);
//    Holonomy::view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "refined");
    //polyscope::getSurfaceMesh("simplification")->addFaceScalarQuantity("is self overlapping", is_self_overlapping);
}

} // namespace Resolution
} // namespace Penner
