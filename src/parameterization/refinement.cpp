#include "refinement.hh"
#include <igl/boundary_facets.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/facet_components.h>
#include <igl/flipped_triangles.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <set>
#include <stack>
#include "area.hh"
#include "conformal_ideal_delaunay/Halfedge.hh"
#include "io.hh"
#include "triangulation.hh"
#include "vector.hh"
#include "vf_mesh.hh"
#include "viewers.hh"

#if ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

/// FIXME Do cleaning pass (Done through viewers)

namespace CurvatureMetric {

RefinementMesh::RefinementMesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints)
{
    clear();

    // Check for flipped elements in the overlay mesh
#if CHECK_VALIDITY
    Eigen::VectorXi flipped_f;
    igl::flipped_triangles(uv, F_uv, flipped_f);
    spdlog::trace("{} flipped elements in overlay mesh", flipped_f.size());
    if (flipped_f.size() > 0) {
        spdlog::warn("Refining a mesh with flipped elements in the overlay");
    }
#endif

    // Build initial topology with refinement data
    build_vertex_points(V, uv);
    build_connectivity(F, F_uv, Fn_to_F, endpoints);

    // Check for validity before refining and simplifying
#if CHECK_VALIDITY
    if (!is_valid_refinement_mesh()) {
        spdlog::error("Initial refinement mesh is invalid");
        clear();
        return;
    }
#endif

    // Refine the mesh as necessary and greedily simplify the refinement
    refine_mesh();
    simplify_mesh();

    // Check for validity after refining and simplifying
#if CHECK_VALIDITY
    if (!is_valid_refinement_mesh()) {
        spdlog::error("Final refinement mesh is invalid");
        clear();
        return;
    }
#endif
}

void RefinementMesh::get_VF_mesh(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& uv,
    Eigen::MatrixXi& F_uv,
    std::vector<int>& Fn_to_F,
    std::vector<std::pair<int, int>>& endpoints) const
{
    V.setZero(0, 0);
    F.setZero(0, 0);
    uv.setZero(0, 0);
    F_uv.setZero(0, 0);
    Fn_to_F.clear();
    endpoints.clear();

    // Build triangle faces
    std::vector<std::array<int, 3>> mesh_triangles(0);
    std::vector<std::array<int, 3>> uv_mesh_triangles(0);
    mesh_triangles.reserve(n_faces());
    uv_mesh_triangles.reserve(n_faces());
    Fn_to_F.reserve(n_faces());
    for (Index fi = 0; fi < n_faces(); ++fi) {
        // Skip boundary faces
        if (is_bnd_loop[fi]) continue;

        // Triangulate face
        std::vector<std::array<int, 3>> face_triangles;
        std::vector<std::array<int, 3>> uv_face_triangles;
        triangulate_face(fi, face_triangles, uv_face_triangles);
        mesh_triangles.insert(mesh_triangles.end(), face_triangles.begin(), face_triangles.end());
        uv_mesh_triangles.insert(
            uv_mesh_triangles.end(),
            uv_face_triangles.begin(),
            uv_face_triangles.end());
        Fn_to_F.insert(Fn_to_F.end(), face_triangles.size(), fi);
    }

    // Copy vectors of mesh triangles to the face matrices
    int num_triangles = mesh_triangles.size();
    Eigen::MatrixXi F_full(num_triangles, 3);
    Eigen::MatrixXi F_uv_full(num_triangles, 3);
    for (int fi = 0; fi < num_triangles; ++fi) {
        for (int j = 0; j < 3; ++j) {
            F_full(fi, j) = mesh_triangles[fi][j];
            F_uv_full(fi, j) = uv_mesh_triangles[fi][j];
            // TODO Check if valid indices with function in common.hh
        }
    }


    // Remove unreferenced vertices
    Eigen::VectorXi I, J, I_uv, J_uv;
    igl::remove_unreferenced(m_V, F_full, V, F, I, J);
    igl::remove_unreferenced(m_uv, F_uv_full, uv, F_uv, I_uv, J_uv);

    // Get endpoints from the overlay, removing unreferenced entries consistently
    // with the removed vertices
    endpoints.resize(V.rows());
    for (int vi = 0; vi < V.rows(); ++vi) {
        endpoints[vi] = m_endpoints[J[vi]];
    }
}

std::
    tuple<
        Eigen::MatrixXd, // V
        Eigen::MatrixXi, // F
        Eigen::MatrixXd, // uv
        Eigen::MatrixXi, // F_uv
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints
        >
    RefinementMesh::get_VF_mesh() const
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F_uv;
    std::vector<int> Fn_to_F;
    std::vector<std::pair<int, int>> endpoints;
    get_VF_mesh(V, F, uv, F_uv, Fn_to_F, endpoints);
    return std::make_tuple(V, F, uv, F_uv, Fn_to_F, endpoints);
}

int RefinementMesh::compute_face_size(RefinementMesh::Index face_index) const
{
    // Count number of edges by iterating around the face
    int face_size = 0;
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        ++face_size;
    }

    return face_size;
}

RefinementMesh::FaceIterator RefinementMesh::get_face_iterator(Index face_index) const
{
    return FaceIterator(*this, face_index);
}

Eigen::VectorXd RefinementMesh::get_vertex(RefinementMesh::Index vertex_index) const
{
    return m_V.row(vertex_index).transpose();
}

Eigen::VectorXd RefinementMesh::get_uv_vertex(RefinementMesh::Index vertex_index) const
{
    return m_uv.row(vertex_index).transpose();
}

void RefinementMesh::get_face_vertices(
    RefinementMesh::Index face_index,
    std::vector<Eigen::VectorXd>& vertices) const
{
    vertices.clear();
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hij = *iter;
        Index vj = to[hij];
        vertices.push_back(get_vertex(vj));
    }
}

void RefinementMesh::get_face_uv_vertices(
    RefinementMesh::Index face_index,
    std::vector<Eigen::VectorXd>& uv_vertices) const
{
    uv_vertices.clear();
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hij = *iter;
        Index uvij = uv_to[hij];
        uv_vertices.push_back(get_uv_vertex(uvij));
    }
}

void RefinementMesh::clear()
{
    n.clear();
    prev.clear();
    opp.clear();
    to.clear();
    f.clear();
    h.clear();
    is_bnd_loop.clear();
    uv_to.clear();
    m_V.setZero(0, 0);
    m_uv.setZero(0, 0);
    m_endpoints.clear();
    h_v_points.clear();
    h_uv_points.clear();
    m_inserted_vertex_halfedges.clear();
}

bool RefinementMesh::empty()
{
    return n.empty();
}


void RefinementMesh::view_refinement_mesh() const
{
#if ENABLE_VISUALIZATION
    polyscope::init();

    // Build mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F_uv;
    std::vector<int> Fn_to_F;
    std::vector<std::pair<int, int>> endpoints;
    get_VF_mesh(V, F, uv, F_uv, Fn_to_F, endpoints);

    // Get the flipped elements
    Eigen::VectorXi flipped_f;
    igl::flipped_triangles(uv, F_uv, flipped_f);
    spdlog::trace("{} flipped elements", flipped_f.size());

    // Cut mesh along seams
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, F_uv, V_cut);

    // Register refinement mesh with parameterization and layout
    polyscope::registerSurfaceMesh("refinement mesh", V_cut, F_uv)
        ->addVertexParameterizationQuantity("uv", uv);
    polyscope::registerSurfaceMesh2D("refinement layout", uv, F_uv);

    // Add Fn_to_F map
    polyscope::getSurfaceMesh("refinement mesh")->addFaceScalarQuantity("face map", Fn_to_F);
    polyscope::getSurfaceMesh("refinement layout")->addFaceScalarQuantity("face map", Fn_to_F);

    // Add face areas
    Eigen::VectorXd doublearea;
    igl::doublearea(V, F, doublearea);
    Eigen::VectorXd area = 0.5 * doublearea;
    polyscope::getSurfaceMesh("refinement mesh")->addFaceScalarQuantity("area", area);

    // Show meshes
    polyscope::show();

    // Remove meshes
    polyscope::removeStructure("refinement mesh");
    polyscope::removeStructure("refinement layout");
#endif // ENABLE_VISUALIZATION
}

void RefinementMesh::view_face(Index face_index) const
{
    spdlog::trace("Viewing face {}", face_index);

#if ENABLE_VISUALIZATION
    polyscope::init();

    // Register corner and halfedge vertices
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hi = *iter;
        Index vi = to[hi];
        polyscope::registerPointCloud(
            "corner vertex " + std::to_string(hi),
            get_vertex(vi).transpose());

        int num_h_points = h_v_points[hi].size();
        Eigen::MatrixXd h_V(num_h_points, 3);
        for (int k = 0; k < num_h_points; ++k) {
            Index vk = h_v_points[hi][k];
            h_V.row(k) = get_vertex(vk);
        }
        polyscope::registerPointCloud("halfedge vertices " + std::to_string(hi), h_V);
    }

    // Show meshes
    polyscope::show();

    // Remove meshes
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hi = *iter;
        polyscope::removeStructure("corner vertex " + std::to_string(hi));
        polyscope::removeStructure("halfedge vertices " + std::to_string(hi));
    }
#endif // ENABLE_VISUALIZATION
}


void RefinementMesh::view_uv_face(Index face_index) const
{
    spdlog::trace("Viewing uv face {}", face_index);

#if ENABLE_VISUALIZATION
    polyscope::init();

    // Register corner and halfedge vertices
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hi = *iter;
        Index uvi = uv_to[hi];
        polyscope::registerPointCloud(
            "uv corner vertex " + std::to_string(hi),
            get_uv_vertex(uvi).transpose());

        int num_h_points = h_v_points[hi].size();
        Eigen::MatrixXd h_uv(num_h_points, 3);
        for (int k = 0; k < num_h_points; ++k) {
            Index uvk = h_uv_points[hi][k];
            h_uv.row(k) = get_uv_vertex(uvk);
        }
        polyscope::registerPointCloud("halfedge uv vertices " + std::to_string(hi), h_uv);
    }

    // Show meshes
    polyscope::show();

    // Remove meshes
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hi = *iter;
        polyscope::removeStructure("uv corner vertex " + std::to_string(hi));
        polyscope::removeStructure("halfedge uv vertices " + std::to_string(hi));
    }
#endif // ENABLE_VISUALIZATION
}

void build_face(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& subfaces,
    const std::vector<std::pair<int, int>>& endpoints,
    std::array<int, 3>& vertices,
    std::array<std::vector<int>, 3>& edge_v_points,
    std::array<int, 3>& uv_vertices,
    std::array<std::vector<int>, 3>& edge_uv_points)
{
    // Build a VF mesh for the subdivided faces
    int num_subfaces = subfaces.size();
    Eigen::MatrixXi F_face(num_subfaces, 3);
    Eigen::MatrixXi F_uv_face(num_subfaces, 3);
    for (int fi = 0; fi < num_subfaces; ++fi) {
        F_face.row(fi) = F.row(subfaces[fi]);
        F_uv_face.row(fi) = F_uv.row(subfaces[fi]);
    }

    // Remove unreferenced vertices to get local face indices
    Eigen::MatrixXi F_local;
    std::vector<int> new_to_old_map;
    remove_unreferenced(F_face, F_local, new_to_old_map);

    // Get the (ordered) boundary loop of the faces
    std::vector<int> local_loop, loop, uv_loop;
    igl::boundary_loop(F_local, local_loop);
    if (static_cast<int>(local_loop.size()) != (F_local.rows() + 2)) {
        spdlog::error("Loop of size {} for face {}", local_loop.size(), F_local);
    }

    // Remap local boundary loop to global
    int num_boundary_vertices = local_loop.size();
    loop.resize(num_boundary_vertices);
    for (int i = 0; i < num_boundary_vertices; ++i) {
        loop[i] = new_to_old_map[local_loop[i]];
    }

    // Get the three original vertices of the loop
    int count = 0;
    std::array<int, 3> vertex_indices;
    for (int i = 0; i < num_boundary_vertices; ++i) {
        int vi = loop[i];

        // Check if original vertex
        spdlog::trace(
            "Endpoints for boundary vertex {} are {} and {}",
            vi,
            endpoints[vi].first,
            endpoints[vi].second);
        if ((endpoints[vi].first == -1) || (endpoints[vi].second == -1)) {
            // Exit with error if more than 3 vertices found
            if (count >= 3) {
                spdlog::error("More than 3 boundary vertices found in curved triangle {}", F_face);
                break;
            }

            // Add vertex and increment counter
            vertex_indices[count] = i;
            vertices[count] = vi;
            ++count;
        }
    }

    // Exit with error if less than 3 vertices found
    if (count < 3) {
        spdlog::error("Less than 3 vertices found in curved triangle {}", F_face);
        return;
    }

    // Build a map from boundary vertices to corners
    std::vector<std::pair<int, int>> loop_to_corner(num_boundary_vertices);
    for (int fi = 0; fi < num_subfaces; ++fi) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < num_boundary_vertices; ++k) {
                if (F_face(fi, j) == loop[k]) {
                    loop_to_corner[k] = std::make_pair(fi, j);
                }
            }
        }
    }

    // Get the three boundary vertices for the uv map
    for (int i = 0; i < 3; ++i) {
        int fi = loop_to_corner[vertex_indices[i]].first;
        int j = loop_to_corner[vertex_indices[i]].second;
        uv_vertices[i] = F_uv_face(fi, j);
    }

    // The edge opposite corner 0 begins ccw from corner 1
    int edge = 0;
    int i = (vertex_indices[1] + 1) % num_boundary_vertices;

    // Cycle around the boundary to find the other boundary edge vertices
    while (i != vertex_indices[1]) {
        // Increment the edge if one of the corner vertices is found
        if ((i == vertex_indices[0]) || (i == vertex_indices[1]) || (i == vertex_indices[2])) {
            ++edge;
            i = (i + 1) % num_boundary_vertices;
            continue;
        }

        // Add the 3D vertex to the corresponding edge
        int vi = loop[i];
        edge_v_points[edge].push_back(vi);

        // Add the uv vertex to the corresponding edge
        int fi = loop_to_corner[i].first;
        int j = loop_to_corner[i].second;
        int uvi = F_uv_face(fi, j);
        edge_uv_points[edge].push_back(uvi);

        // Increment the cyclic index
        i = (i + 1) % num_boundary_vertices;
    }
}

void build_faces(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::vector<int>>& F_to_Fn,
    const std::vector<std::pair<int, int>>& endpoints,
    Eigen::MatrixXi& F_orig,
    std::vector<std::array<std::vector<int>, 3>>& corner_v_points,
    Eigen::MatrixXi& F_uv_orig,
    std::vector<std::array<std::vector<int>, 3>>& corner_uv_points)
{
    int num_faces = F_to_Fn.size();
    F_orig.resize(num_faces, 3);
    corner_v_points.resize(num_faces);
    F_uv_orig.resize(num_faces, 3);
    corner_uv_points.resize(num_faces);
    for (int fi = 0; fi < num_faces; ++fi) {
        // Build face
        std::array<int, 3> vertices;
        std::array<int, 3> uv_vertices;
        build_face(
            F,
            F_uv,
            F_to_Fn[fi],
            endpoints,
            vertices,
            corner_v_points[fi],
            uv_vertices,
            corner_uv_points[fi]);

        // Copy vertices to face matrices
        for (int j = 0; j < 3; ++j) {
            F_orig(fi, j) = vertices[j];
            F_uv_orig(fi, j) = uv_vertices[j];
        }
    }
}

void RefinementMesh::build_connectivity(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints)
{
    m_inserted_vertex_halfedges.clear();

    // Check if uv faces are a single component
    int num_components = count_components(F_uv);
    if (num_components != 1) {
        spdlog::error("uv face connectivity has {} components", num_components);
    }

    // Make lists of new faces per original face
    int num_new_faces = Fn_to_F.size();

    // Get the number of faces
    int num_faces = 0;
    for (int i = 0; i < num_new_faces; ++i) {
        num_faces = std::max(num_faces, Fn_to_F[i] + 1);
    }

    // Get the map from faces to list of new faces
    std::vector<std::vector<int>> F_to_Fn(num_faces, std::vector<int>(0));
    for (int i = 0; i < num_new_faces; ++i) {
        int fi = Fn_to_F[i];
        F_to_Fn[fi].push_back(i);
    }

    // For each original face, get the overlay vertices corresponding to the face
    Eigen::MatrixXi F_orig, F_uv_orig;
    std::vector<std::array<std::vector<int>, 3>> corner_v_points, corner_uv_points;
    build_faces(F, F_uv, F_to_Fn, endpoints, F_orig, corner_v_points, F_uv_orig, corner_uv_points);

    // Record the original overlay triangulations
    m_overlay_face_triangles.resize(num_faces);
    m_overlay_uv_face_triangles.resize(num_faces);
    for (int fi = 0; fi < num_faces; ++fi) {
        m_overlay_face_triangles[fi].clear();
        m_overlay_uv_face_triangles[fi].clear();
        for (auto fij : F_to_Fn[fi]) {
            m_overlay_face_triangles[fi].push_back({F(fij, 0), F(fij, 1), F(fij, 2)});
            m_overlay_uv_face_triangles[fi].push_back({F_uv(fij, 0), F_uv(fij, 1), F_uv(fij, 2)});
        }
    }

    // Build halfedges for the faces
    std::vector<int> next_he;
    std::vector<int> opposite;
    std::vector<int> vtx_reindex;
    std::vector<int> bnd_loops;
    std::vector<std::vector<int>> corner_to_he;
    std::vector<std::pair<int, int>> he_to_corner;
    Connectivity C;
    FV_to_NOB(F_orig, next_he, opposite, bnd_loops, vtx_reindex, corner_to_he, he_to_corner);
    NOB_to_connectivity(next_he, opposite, bnd_loops, C);

    // Check if uv faces is manifold
    if (!igl::is_edge_manifold(F_uv_orig)) {
        spdlog::error("uv original face connectivity is not manifold");
    }

    // Check if uv faces are a single component
    int num_orig_components = count_components(F_uv_orig);
    if (num_orig_components != 1) {
        spdlog::error("uv original face connectivity has {} components", num_orig_components);
    }

    n = C.n;
    prev = C.prev;
    opp = C.opp;
    f = C.f;
    h = C.h;

    // Build boundary loop mask
    is_bnd_loop = std::vector<bool>(h.size(), false);
    for (auto fi : bnd_loops) {
        is_bnd_loop[fi] = true;
    }


    // Reindex the to arrays
    int num_halfedges = C.n.size();
    ;
    to.resize(num_halfedges);
    uv_to.resize(num_halfedges);
    h_v_points.resize(num_halfedges);
    h_uv_points.resize(num_halfedges);

    // TODO This currently cycles faces by 1; harmless, but may be confusing
    for (int hi = 0; hi < num_halfedges; ++hi) {
        // Reindex to with the vertex
        to[hi] = vtx_reindex[C.to[hi]];

        // Build corner vertex points for halfedges with known opposite corner vertices
        if (is_bnd_loop[f[hi]]) continue;
        std::pair<int, int> ci = he_to_corner[hi];
        if ((ci.first >= 0) && (ci.second >= 0)) {
            // TODO Check same instead of overwrite
            //to[hi] = F_orig(ci.first, (ci.second + 2)%3); // offset of 2 as corner is opposite edge
            uv_to[hi] =
                F_uv_orig(ci.first, (ci.second + 2) % 3); // offset of 2 as corner is opposite edge

            // Build halfedge refinement points
            h_v_points[hi].insert(
                h_v_points[hi].end(),
                corner_v_points[ci.first][ci.second].begin(),
                corner_v_points[ci.first][ci.second].end());
            h_uv_points[hi].insert(
                h_uv_points[hi].end(),
                corner_uv_points[ci.first][ci.second].begin(),
                corner_uv_points[ci.first][ci.second].end());
        }

        // FIXME
        // if (!corner_v_points[ci.first][ci.second].empty())
        //{
        //	spdlog::trace("Face {} has nontrivial refinement", fi);
        //}

        // Store endpoints
        m_endpoints = endpoints;
    }

    // FIXME
    // int num_vertices = C.out.size();;
    // out.resize(num_vertices);
    // for (int vi = 0; vi < num_vertices; ++vi)
    //{
    //	out[vtx_reindex[vi]] = C.out[vi];
    //}
}

void RefinementMesh::build_vertex_points(const Eigen::MatrixXd& V, const Eigen::MatrixXd& uv)
{
    // Simply copy the vertices
    m_V = V;
    m_uv = uv;
}

void RefinementMesh::refine_mesh()
{
    // Build list of inverted faces
    std::stack<Index> invalid_faces;
    int num_faces = n_faces();
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        // Skip boundary loops
        if (is_bnd_loop[fijk]) continue;

        std::array<Eigen::VectorXd, 3> triangle;
        Index hij = h[fijk];
        for (int l = 0; l < 3; ++l) {
            // Get vertex at tip of halfedge
            Index vj = uv_to[hij];
            triangle[l] = get_uv_vertex(vj);

            // Increment halfedge
            hij = n[hij];
        }

        // Check that face is a triangle
        if (hij != h[fijk]) {
            spdlog::error("Can only refine a triangle mesh");
            return;
        }

        // Check if the face is inverted
        if (is_inverted_triangle(triangle)) {
            spdlog::trace(
                "Inverted face {}, {}, {}",
                triangle[0],
                triangle[1],
                triangle[2]); // FIXME
            invalid_faces.push(fijk);
        }
    }

    // Refine faces until all are valid
    std::vector<bool> is_refined(num_faces, false);
    while (!invalid_faces.empty()) {
        // Get invalid face
        Index current_face = invalid_faces.top();
        invalid_faces.pop();
        spdlog::trace("Processing face {}", current_face);

        // Check if face already processed to prevent an infinite loop
        // NOTE: This should not happen, but may do to floating point error
        // or invalid inputs
        if (is_refined[current_face]) {
            spdlog::debug("Attempted to refine face {} twice", current_face);
            continue;
        }

        // Get (unique) list of adjacent faces
        std::set<Index> adjacent_faces;
        for (auto iter = get_face_iterator(current_face); !iter.done(); ++iter) {
            Index hij = *iter;
            Index hji = opp[hij];
            Index adjacent_face = f[hji];
            adjacent_faces.insert(adjacent_face);
        }

        // Refine the given face, marking if it is refined
        is_refined[current_face] = refine_face(current_face);

        // Check if the adjacent faces are valid
        // WARNING: Assumes the face indices are fixed
        for (auto face_index : adjacent_faces) {
            if ((compute_face_size(face_index) >= 50) || (!is_self_overlapping_face(face_index))) {
                invalid_faces.push(face_index);
            }
        }
    }
}

// Remove the vertex at the tip of the halfedge if possible
bool RefinementMesh::simplify_vertex(Index halfedge_index)
{
    // Local halfedges for vertex k
    Index hik = halfedge_index;
    Index hki = opp[hik];
    Index hkj = n[hik];
    Index hjk = opp[hkj];
    if (n[hjk] != hki) {
        spdlog::error("Cannot simplify a vertex that is not valence 2");
        return false;
    }
    Index h_prev = prev[hik];
    Index ho_next = n[hki];

    // Local vertices
    Index vi = to[hki];
    Index vk = to[hik];
    Index uvjk = uv_to[hjk];
    Index uvki = uv_to[hki];

    // Local faces
    Index f0 = f[hik];
    Index f1 = f[hki];

    // Do not simplify if the face is too large
    if ((compute_face_size(f0) >= 50) || (compute_face_size(f1) >= 50)) {
        return false;
    }

    // Check validity of current local halfedge information
    if ((prev[hkj] != hik) || (n[h_prev] != hik) || (prev[ho_next] != hki) || (n[hjk] != hki) ||
        (to[hjk] != vk) || (uv_to[hjk] != uvjk)) {
        spdlog::error("Invalid halfedge at vertex {}", vk);
        return false;
    }

    // Remove the vertex (replace k with j)
    prev[hkj] = h_prev;
    n[h_prev] = hkj;
    prev[ho_next] = hjk;
    n[hjk] = ho_next;
    to[hjk] = vi;
    uv_to[hjk] = uvki;

    // Ensure the face->halfedge map does not reference a removed halfedge
    h[f0] = hkj;
    h[f1] = hjk;

    // Test for self-overlapping adjacent faces
    if (!is_self_overlapping_face(f0) || !is_self_overlapping_face(f1)) {
        // Replace the vertex vk if the adjacent faces are no longer self overlapping
        prev[hkj] = hik;
        n[h_prev] = hik;
        prev[ho_next] = hki;
        n[hjk] = hki;
        to[hjk] = vk;
        uv_to[hjk] = uvjk;

        return false;
    } else {
        // Set all edge ik information to invalid (should not be necessary, but to be safe)
        n[hik] = n[hki] = -1;
        prev[hik] = prev[hki] = -1;
        to[hik] = to[hki] = -1;
        f[hik] = f[hki] = -1;
        uv_to[hik] = uv_to[hki] = -1;

        return true;
    }
}

// Simplify mesh by removing inserted vertices until none can be removed
void RefinementMesh::simplify_mesh()
{
    spdlog::trace("Simplifying mesh");

    // Initialize record of found vertices
    int num_inserted_vertices = m_inserted_vertex_halfedges.size();
    std::vector<bool> is_removed(num_inserted_vertices, false);
    bool vertex_removed = false;

    // Iterate until no vertices can be removed
    do {
        // Loop over all inserted vertices to see if one can be removed
        vertex_removed = false;
        for (int i = 0; i < num_inserted_vertices; ++i) {
            if (is_removed[i]) continue;

            // Attempt to remove a vertex
            Index hik = m_inserted_vertex_halfedges[i];
            bool vertex_k_removed = simplify_vertex(hik);

            // Update vertex found query
            if (vertex_k_removed) {
                is_removed[i] = true;
                vertex_removed = true;
            }
        }
    } while (vertex_removed);
}

// Mesh manipulation

// Refine a halfedge and return true iff it is fully refined
bool RefinementMesh::refine_halfedge(Index halfedge_index)
{
    spdlog::trace("Refining edge for halfedge {}", halfedge_index);

    // Get the number of vertices to add (and do nothing if none)
    int num_new_vertices = h_v_points[halfedge_index].size();
    if (num_new_vertices == 0) return true;
    spdlog::trace("Refining edge with {} vertices", num_new_vertices);

    // Get initial next and prev halfedges
    Index hij = halfedge_index;
    Index hji = opp[hij];
    Index next_h = n[hij];
    Index prev_ho = prev[hji];
    spdlog::trace("Refining edge with halfedges ({}, {})", hij, hji);

    // Get the faces adjacent to the edge
    Index f0 = f[hij];
    Index f1 = f[hji];

    // Get the endpoint vertices
    Index vi = to[hji];
    Index vj = to[hij];

    // Get the uv to vertices
    Index uvij = is_bnd_loop[f0] ? -1 : uv_to[hij];
    Index uvji = is_bnd_loop[f1] ? -1 : uv_to[hji];

    // TODO
    // Iteratively push new vertices and halfedges to the mesh
    Index hik = hij;
    Index hki = hji;
    Index uvkj = uvij;
    Index uvki = uvji;
    for (int k = 0; k < num_new_vertices; ++k) {
        // Terminate early if face containing the halfedge is small and self overlapping
        if ((compute_face_size(f0) < 50) && (is_self_overlapping_face(f0))) {
            return false;
        }

        // Get new vertex and halfedge indices
        Index hkj = n.size();
        Index hjk = hkj + 1;
        Index vk = h_v_points[hij].back(); // Get last vertex so hik can retain halfedge attributes
        if ((!is_bnd_loop[f1]) && (vk != h_v_points[hji].front())) {
            spdlog::warn("Inconsistent halfedge vertices {} and {}", vk, h_v_points[hji].front());
        }
        h_v_points[hij].pop_back();
        h_v_points[hji].pop_front();

        // Get new uv vertex indices (or -1 for boundary halfedges)
        Index uvik = -1;
        if (!is_bnd_loop[f0]) {
            uvik = h_uv_points[hij].back();
            h_uv_points[hij].pop_back();
        }
        Index uvjk = -1;
        if (!is_bnd_loop[f1]) {
            uvjk = h_uv_points[hji].front();
            h_uv_points[hji].pop_front();
        }

        // Update next
        n.push_back(0);
        n.push_back(0);
        n[hkj] = next_h;
        n[hjk] = hki;
        n[hik] = hkj;
        n[prev_ho] = hjk;

        // Update prev
        prev.push_back(0);
        prev.push_back(0);
        prev[hkj] = hik;
        prev[hjk] = prev_ho;
        prev[hki] = hjk;
        prev[next_h] = hkj;

        // Update opp
        opp.push_back(0);
        opp.push_back(0);
        opp[hkj] = hjk;
        opp[hjk] = hkj;

        // Update to
        to.push_back(0);
        to.push_back(0);
        to[hik] = vk;
        to[hkj] = vj;
        to[hjk] = vk;
        to[hki] = vi;

        // Update face
        f.push_back(0);
        f.push_back(0);
        f[hkj] = f0;
        f[hjk] = f1;

        // Update halfedge
        h[f0] = hik;
        h[f1] = hki;

        // Update uv to
        uv_to.push_back(0);
        uv_to.push_back(0);
        uv_to[hik] = uvik;
        uv_to[hkj] = uvkj;
        uv_to[hjk] = uvjk;
        uv_to[hki] = uvki;

        // Update halfedge points with trivial data
        h_v_points.push_back(std::deque<Index>(0));
        h_v_points.push_back(std::deque<Index>(0));
        h_uv_points.push_back(std::deque<Index>(0));
        h_uv_points.push_back(std::deque<Index>(0));

        // Record halfedge hjk pointing to inserted vk as inserted
        m_inserted_vertex_halfedges.push_back(hjk);

        // Iterate vertex j to k
        vj = vk;
        uvkj = uvik;
        next_h = hkj;
        prev_ho = hjk;
        // vi and uvki do not change
    }

    // Fully refined halfedge
    return true;
}

bool RefinementMesh::refine_face(Index face_index)
{
    // Get all edges of the face (before changing connectivity)
    std::vector<Index> face_halfedges(0);
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hij = *iter;
        face_halfedges.push_back(hij);
    }

    // Now refine all edges of the face
    // WARNING Assumes refine_halfedge only changes the halfedge and its opposite
    for (auto hij : face_halfedges) {
        bool is_halfedge_refined = refine_halfedge(hij);

        // If the halfedge is not fully refined, the face is already WSO
        if (!is_halfedge_refined) return false;
    }

    // The face is fully refined if all halfedges are refined
    return true;
}

// Mesh predicates

bool RefinementMesh::is_self_overlapping_face(RefinementMesh::Index face_index) const
{
    // Boundary faces are self overlapping
    if (is_bnd_loop[face_index]) return true;

    // Get vertices of the face
    std::vector<Eigen::VectorXd> uv_vertices, vertices;
    get_face_uv_vertices(face_index, uv_vertices);
    get_face_vertices(face_index, vertices);

    if (vertices.size() > 100) {
        spdlog::warn("Face of size {}", vertices.size());
        // view_face(face_index);
    }

    // Determine if the face is self overlapping
    std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
    std::vector<std::vector<int>> splitting_vertices;
    std::vector<std::vector<Scalar>> min_face_areas;
    return is_self_overlapping_polygon(
        uv_vertices,
        vertices,
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas);
}

bool RefinementMesh::triangulate_face(
    RefinementMesh::Index face_index,
    std::vector<std::array<int, 3>>& face_triangles,
    std::vector<std::array<int, 3>>& uv_face_triangles) const
{
    spdlog::trace("Triangulating face {}", face_index);
    face_triangles.clear();
    uv_face_triangles.clear();

    // Get uv vertices of the face
    std::vector<int> vertex_indices;
    std::vector<int> uv_vertex_indices;
    std::vector<Eigen::VectorXd> vertices;
    std::vector<Eigen::VectorXd> uv_vertices;
    for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter) {
        Index hij = *iter;
        Index vj = to[hij];
        Index uvij = uv_to[hij];
        vertex_indices.push_back(vj);
        uv_vertex_indices.push_back(uvij);
        vertices.push_back(get_vertex(vj));
        uv_vertices.push_back(get_uv_vertex(uvij));
    }

    // FIXME
    if (vertex_indices.size() > 3) {
        spdlog::trace("Triangulating face {} of size {}", face_index, vertex_indices.size());
        spdlog::trace("Face vertices are {}", formatted_vector(vertex_indices));
    }

    // Triangulate the face the same as the overlay if it is too large
    if (compute_face_size(face_index) >= 50) {
        face_triangles = m_overlay_face_triangles[face_index];
        uv_face_triangles = m_overlay_uv_face_triangles[face_index];
        return true;
    }

    // Determine if the face is self overlapping
    std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
    std::vector<std::vector<int>> splitting_vertices;
    std::vector<std::vector<Scalar>> min_face_areas;
    bool is_self_overlapping = is_self_overlapping_polygon(
        uv_vertices,
        vertices,
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas);

    // Return false if the face is not self overlapping
    if (!is_self_overlapping) {
        spdlog::warn("Face {} is not self overlapping", face_index);
        return false;
    }

    // Triangulate the face
    std::vector<std::array<int, 3>> polygon_faces;
    triangulate_self_overlapping_polygon(
        is_self_overlapping_subpolygon,
        splitting_vertices,
        min_face_areas,
        polygon_faces);

    // Convert polygon face indices to global indices for V and uv
    int num_polygon_faces = polygon_faces.size();
    face_triangles.resize(num_polygon_faces);
    uv_face_triangles.resize(num_polygon_faces);
    for (int fi = 0; fi < num_polygon_faces; ++fi) {
        for (int j = 0; j < 3; ++j) {
            face_triangles[fi][j] = vertex_indices[polygon_faces[fi][j]];
            uv_face_triangles[fi][j] = uv_vertex_indices[polygon_faces[fi][j]];
        }
    }

    return true;
}

bool RefinementMesh::is_valid_refinement_mesh() const
{
    // Check halfedge topology
    int num_halfedges = n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Skip invalidated halfedges if they are not corrupted
        if (n[hij] == -1) {
            if ((prev[hij] != -1) || (prev[hij] != -1) || (to[hij] != -1) || (f[hij] != -1)) {
                spdlog::error("invalid halfedge corrupted");
                return false;
            }
            continue;
        }

        // Check next and prev are inverse
        if ((n[prev[hij]] != hij) || (prev[n[hij]] != hij)) {
            spdlog::error("next and prev are not inverses");
            return false;
        }

        // Check opp is an order 2 involution
        if ((opp[hij] == hij) || (opp[opp[hij]] != hij)) {
            spdlog::error("opp is not an order 2 involution");
            return false;
        }

        // Check that to is invariant under vertex circulation
        if (to[opp[n[hij]]] != to[hij]) {
            spdlog::error("to is not invariant under vertex circulation");
            return false;
        }

        // Check that face is invariant under face circulation
        if (f[n[hij]] != f[hij]) {
            spdlog::error("f is not invariant under face circulation");
            return false;
        }
    }

    // Check face topology
    int num_faces = n_faces();
    for (int fi = 0; fi < num_faces; ++fi) {
        // Check that h is a left inverse for face
        if (f[h[fi]] != fi) {
            spdlog::error("h is not a left inverse for f");
            return false;
        }
    }

    // TODO Optional: check that to and face do not identify any distinct
    // orbits of the vertex and face circulators

    // TODO Check uv and halfedge refinement attributes

    // Valid otherwise
    return true;
}

} // namespace CurvatureMetric
