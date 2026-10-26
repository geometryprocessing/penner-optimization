#include "resolution/core/geodesics.h"
#include "util/union_find.h"
#include "util/vector.h"
#include "metric/constraint.h"
#include "parametrization/interpolation.h"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Layout.hh"


#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner
{
namespace Resolution
{





Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh)
{
    typedef geometrycentral::surface::Halfedge Halfedge;
    int num_vertices = mesh.nVertices();
    Eigen::SparseMatrix<int> vv2he(num_vertices, num_vertices);

    // build sparse matrix
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    for(Halfedge he : mesh.halfedges()) {
        int vj = he.tailVertex().getIndex();
        int vk= he.tipVertex().getIndex();
        trips.push_back(Trip(vj, vk, he.getIndex() + 1));
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

std::vector<std::vector<geometrycentral::surface::Halfedge>> generate_halfedge_paths(
    const std::vector<std::vector<int>>& paths,
    geometrycentral::surface::ManifoldSurfaceMesh& mesh)
{
    // create halfedge paths
    int num_paths = paths.size();
    typedef geometrycentral::surface::Halfedge Halfedge;
    std::vector<std::vector<Halfedge>> halfedge_paths(num_paths);
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(mesh);
    for (int i = 0; i < num_paths; ++i) 
    {
        int num_path_nodes = paths[i].size();
        halfedge_paths[i].reserve(num_path_nodes);
        for (int j = 0; j < num_path_nodes - 1; ++j)
        {
            int vi = paths[i][j];
            int vj = paths[i][j + 1];
            halfedge_paths[i].push_back(mesh.halfedge(vv2he.coeff(vi, vj) - 1));
        }
    }

    return halfedge_paths;
}


Eigen::MatrixXd build_overlay_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int> vtx_reindex,
    OverlayMesh<Scalar>& m_o)
{
    Mesh<Scalar>& m = m_o.cmesh();

    std::vector<std::vector<Scalar>> V_reindex(3);
    for (int i = 0; i < 3; i++) {
        V_reindex[i].resize(V.rows(), 0);
        for (int j = 0; j < V.rows(); j++) {
            V_reindex[i][j] = Scalar(V(vtx_reindex[j], i));
        }
    }

    // Interpolate vertex positions along the original edges
    std::vector<std::vector<Scalar>> V_overlay_rev(3);
    for (int j = 0; j < 3; j++) {
        V_overlay_rev[j] = m_o.interpolate_along_o_bc(
            m.opp,
            m.to,
            V_reindex[j]);
    }

    // Reindex vertices
    Eigen::MatrixXd V_overlay(V_overlay_rev[0].size(), 3);
    for (size_t j = 0; j < 3; j++) {
        for (size_t i = 0; i < V_overlay_rev[j].size(); i++) {
            V_overlay(i, j) = double(V_overlay_rev[j][i]);
        }
    }

    return V_overlay;
}

SurfacePoint convert_surface_point(const geometrycentral::surface::SurfacePoint& pt)
{
    SurfacePoint point;
    if (pt.type == geometrycentral::surface::SurfacePointType::Vertex)
    {
        point.index = pt.vertex.getIndex();
        point.type = SurfacePoint::Type::Vertex;
    }
    else if (pt.type == geometrycentral::surface::SurfacePointType::Edge)
    {
        point.type = SurfacePoint::Type::Edge;
        point.index = pt.edge.getIndex();
        point.edge_base = pt.edge.firstVertex().getIndex();
        point.edge_tip = pt.edge.secondVertex().getIndex();
        point.t = pt.tEdge;
    }
    else if (pt.type == geometrycentral::surface::SurfacePointType::Face)
    {
        point.type = SurfacePoint::Type::Face;
    }

    return point;
}

std::vector<std::vector<SurfacePoint>> generate_geodesic_path_points(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<int>>& paths)
{
    Eigen::MatrixXd V = overlay_mesh.generate_overlay_vertices(overlay_mesh.V_orig, overlay_mesh.vtx_reindex_orig, overlay_mesh.orig);
    Eigen::MatrixXi F = overlay_mesh.generate_overlay_faces_simple();

    geometrycentral::surface::ManifoldSurfaceMesh mesh(F);
    geometrycentral::surface::VertexPositionGeometry geometry(mesh, V);
    geometrycentral::surface::FlipEdgeNetwork edgeNetwork(mesh, geometry, generate_halfedge_paths(paths, mesh));

    // Make the path a geodesic
    edgeNetwork.straightenAroundMarkedVertices = false;
    edgeNetwork.iterativeShorten();

    // Extract the result as a polyline along the surface
    edgeNetwork.posGeom = &geometry;
    std::vector<std::vector<geometrycentral::surface::SurfacePoint>> polyline_pts = edgeNetwork.getPathPolyline();

    int num_paths = polyline_pts.size();
    std::vector<std::vector<SurfacePoint>> surface_pts(num_paths);
    for (int i = 0; i < num_paths; ++i)
    {
        int num_pts = polyline_pts[i].size();
        surface_pts[i].resize(num_pts);
        for (int j = 0; j < num_pts; ++j)
        {
            surface_pts[i][j] = convert_surface_point(polyline_pts[i][j]);
        }
    }

    return surface_pts;
}

PathStraightener::PathStraightener(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<SurfacePoint>>& paths
)
    : PathStraightener(overlay_mesh, paths, std::vector<bool>(paths.size(), false))
{}

PathStraightener::PathStraightener(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<SurfacePoint>>& paths,
    const std::vector<bool>& is_intersecting_path 
)
: V(overlay_mesh.generate_overlay_vertices(overlay_mesh.V_orig, overlay_mesh.vtx_reindex_orig, overlay_mesh.orig))
, F(overlay_mesh.generate_overlay_faces_simple())
, polyline_pts(paths)
, is_redundant_path(is_intersecting_path)
{
    std::tie(v_new2old, v_old2new) = overlay_mesh.get_vertex_reductions();
    vv2he_o = overlay_mesh.generate_VV_to_halfedge_map();
}

void PathStraightener::split_edges(VoronoiOverlayMesh& overlay_mesh)
{
    int num_paths = polyline_pts.size();
    int num_halfedges = overlay_mesh.n.size();
    std::vector<std::vector<std::array<int, 2>>> edges_to_split(num_halfedges);

    for (int i = 0; i < num_paths; ++i)
    {
        const auto& polyline_pt = polyline_pts[i];
        int num_pts = polyline_pt.size();
        for (int j = 0; j < num_pts; ++j)
        {
            const auto& pt = polyline_pt[j];
            if (pt.type == SurfacePoint::Type::Edge)
            {
                int vi = pt.edge_base;
                int vj = pt.edge_tip;
                int hij = vv2he_o.coeff(vi, vj) - 1;
                assert(hij >= 0);
                spdlog::debug("vertices {}, {} with halfedge {}", vi, vj, hij);
                edges_to_split[hij].push_back({i, j});
            }
        }
    }

    split_halfedges = std::vector<std::vector<int>>(num_paths);
    path_halfedges = std::vector<std::vector<int>>(num_paths);
    for (int i = 0; i < num_paths; ++i)
    {
        int num_pts = polyline_pts[i].size();
        split_halfedges[i] = std::vector<int>(num_pts, -1);
        path_halfedges[i] = std::vector<int>();
        path_halfedges[i].reserve(num_pts + 1);
        //is_reused[i] = std::vector<bool>(num_pts, false);
    }

    for (int hij = 0; hij < num_halfedges; ++hij)
    {

        const auto& edge_splits = edges_to_split[hij];
        int num_splits = edge_splits.size();
        if (num_splits == 0) continue;
        std::vector<SimplificationOverlayMesh::BaryCoords> bary_coords(num_splits);
        for (int k = 0; k < num_splits; ++k)
        {
            auto [i, j] = edge_splits[k];
            bary_coords[k] = overlay_mesh.convert_overlay_to_original_coordinates(hij, polyline_pts[i][j].t);
        }

        assert(edges_to_split[overlay_mesh.opp[hij]].empty());
        if (num_splits == 1)
        {
            auto [i, j] = edge_splits[0];
            int hik = overlay_mesh.split_edge(hij);
            split_halfedges[i][j] = hik;
            int vk = overlay_mesh.to[hik];
            overlay_mesh.orig.bary_coords[vk] = bary_coords[0];
        }
        else
        {
            spdlog::debug("splitting edge {} times", num_splits);
            std::vector<int> idx(num_splits);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                    [&](int i, int j) {
                        int ii = edge_splits[i][0];
                        int ij = edge_splits[i][1];
                        int ji = edge_splits[j][0];
                        int jj = edge_splits[j][1];
                        Scalar it = polyline_pts[ii][ij].t;
                        Scalar jt = polyline_pts[ji][jj].t;
                        //if (it == jt) return false;
                        return (it > jt);
                    });
            int hik = hij;
            Scalar t_prev = 1.;
            bool split_done = false;
            for (int k = 0; k < num_splits; ++k)
            {
                auto [i, j] = edge_splits[idx[k]];
                if (is_redundant_path[i]) continue;
                const auto& pt = polyline_pts[i][j];
                spdlog::debug("splitting point {} at {} in path {}", pt.index, pt.t, i);
                //if ((k > 0) && float_equal(pt.t, t_prev, 1e-6))
                //if ((split_done) && (pt.t == t_prev))
                if (false)
                {
                    spdlog::warn("intersecting geodesics {} at {} with delta {} in edge {}", i, t_prev, pt.t - t_prev, hij);
                    is_redundant_path[i] = true;
                    continue;
                }
                else
                {
                    hik = overlay_mesh.split_edge(hik);
                    int vk = overlay_mesh.to[hik];
                    overlay_mesh.orig.bary_coords[vk] = bary_coords[idx[k]];
                    split_done = true;
                }
                t_prev = pt.t;
                split_halfedges[i][j] = overlay_mesh.opp[overlay_mesh.n[hik]];
                assert(overlay_mesh.is_vertex_degree_two(overlay_mesh.to[split_halfedges[i][j]]));
            }
        }
    }
}

void PathStraightener::split_edges(
    VoronoiOverlayMesh& overlay_mesh,
    std::vector<std::vector<std::array<int, 2>>>& edges_to_split)
{
    int num_paths = polyline_pts.size();
    int num_halfedges = overlay_mesh.n.size();

    split_halfedges = std::vector<std::vector<int>>(num_paths);
    path_halfedges = std::vector<std::vector<int>>(num_paths);
    for (int i = 0; i < num_paths; ++i)
    {
        int num_pts = polyline_pts[i].size();
        split_halfedges[i] = std::vector<int>(num_pts, -1);
        path_halfedges[i] = std::vector<int>();
        path_halfedges[i].reserve(num_pts + 1);
        //is_reused[i] = std::vector<bool>(num_pts, false);
    }

    for (int hij = 0; hij < num_halfedges; ++hij)
    {

        const auto& edge_splits = edges_to_split[hij];
        int num_splits = edge_splits.size();
        if (num_splits == 0) continue;
        std::vector<SimplificationOverlayMesh::BaryCoords> bary_coords(num_splits);
        Scalar t_prev = 1.;
        for (int k = 0; k < num_splits; ++k)
        {
            auto [i, j] = edge_splits[k];
            Scalar t = polyline_pts[i][j].t;
            bool is_valid = true;

            // check for nans
            if (isnan(t))
            {
                spdlog::warn("barycentric coordinate is NaN");
                is_valid = false;
            }


            // check for consistent ordering
            if ((float_equal(t, t_prev)) || (t_prev < t) || (t < 1e-10))
            {
                spdlog::warn("intersecting geodesics {} at {}, {} in edge {}", i, t_prev, t, hij);
                is_valid = false;
            }

            if (!is_valid)
            {
                Scalar t_next = 0.;
                if (k + 1 < num_splits)
                {
                    auto [i_n, j_n] = edge_splits[k + 1];
                    t_next = polyline_pts[i_n][j_n].t;
                    if (isnan(t_next) || (t_next < 0.) || (t_prev <= t_next)) t_next = 0.;
                }
                t = 0.5 * (t_prev + t_next); // not a very satisfying fallback
            }
            bary_coords[k] = overlay_mesh.convert_overlay_to_original_coordinates(hij, t);
            t_prev = t;
        }

        assert(edges_to_split[overlay_mesh.opp[hij]].empty());
        if (num_splits == 1)
        {
            auto [i, j] = edge_splits[0];
            int hik = overlay_mesh.split_edge(hij);
            split_halfedges[i][j] = hik;
            int vk = overlay_mesh.to[hik];
            overlay_mesh.orig.bary_coords[vk] = bary_coords[0];
        }
        else
        {
            spdlog::debug("splitting edge {} times", num_splits);
            int hik = hij;
            for (int k = 0; k < num_splits; ++k)
            {
                auto [i, j] = edge_splits[k];
                if (is_redundant_path[i]) continue;

                hik = overlay_mesh.split_edge(hik);
                int vk = overlay_mesh.to[hik];
                overlay_mesh.orig.bary_coords[vk] = bary_coords[k];
                split_halfedges[i][j] = overlay_mesh.opp[overlay_mesh.n[hik]];
                assert(overlay_mesh.is_vertex_degree_two(overlay_mesh.to[split_halfedges[i][j]]));
            }
        }
    }
}

void PathStraightener::join_edges(VoronoiOverlayMesh& overlay_mesh)
{
    // write new edges
    int num_paths = polyline_pts.size();
    redundant_edges = std::vector<std::vector<int>>(num_paths, std::vector<int>());
    redundant_vertices = std::vector<std::vector<int>>(num_paths, std::vector<int>());

    // reset labels for simplified mesh, leaving vertex vertex tags
    overlay_mesh.simp.edge_type = std::vector<EdgeType>(overlay_mesh.e2h.size(), EdgeType::EF);
    for (int vi = 0; vi < overlay_mesh.out.size(); ++vi)
    {
        if (overlay_mesh.simp.vertex_type[vi] == VertexType::VE) overlay_mesh.simp.vertex_type[vi] = VertexType::VF;
    }

    for (int i = 0; i < num_paths; ++i)
    {
        // skip if already makred redundant
        if (is_redundant_path[i]) continue;

        int Hij = overlay_mesh.simp.e2h[i];
        int Hji = overlay_mesh.simp.m.opp[Hij];
        int curr = 0; // set to 1 to skip initial vertex
        int num_pts = split_halfedges[i].size();
        auto& curr_path_halfeges = path_halfedges[i];

        // lambda to mark edge halfedges in the overlay and build the halfedge sequence
        auto set_simp_edge = [&](int h) {
            curr_path_halfeges.push_back(h);
            if (overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] != EdgeType::EF)
            {
                return;
            }
            overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] = EdgeType::EE;
            overlay_mesh.simp.halfedge_parent[h] = Hij;
            overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[h]] = Hji;
        };

        // lambda to mark edge vertices in the overlay
        auto set_simp_vertex = [&](int v) {
            if (overlay_mesh.simp.vertex_type[v] != VertexType::VF)
            {
                return;
            }
            overlay_mesh.simp.vertex_type[v] = VertexType::VE;
            overlay_mesh.simp.vertex_parent[v] = i;
        };

        auto mark_redundant = [&](int h, int index) {
            redundant_edges[i].push_back(index);
            int h_parent = overlay_mesh.simp.halfedge_parent[h];
            int e_parent = overlay_mesh.simp.h2e[h_parent];
            is_redundant_path[i] = true;
            is_redundant_path[e_parent] = true;
            //spdlog::warn("edge already labeled in edge {}", i);
            //spdlog::debug("edge labeled with {}", e_parent);
        };

        // undo tracing and mark path to retrace
        bool aborted = false;
        auto abort_path = [&]() {
            spdlog::warn("aborting path {}", i);
            aborted = true;
            is_redundant_path[i] = true;
            assert(false);
        };

        // process sequences of split vertices between original vertices in the polyline
        while (curr < num_pts - 1)
        {
            int start = curr;
            std::vector<int> vertices = {};
            std::vector<int> halfedges = {};
            std::vector<std::array<int, 2>> faces = {};
            vertices.push_back(v_new2old[polyline_pts[i][curr].index]);
            curr++;

            // get sequence of split vertices until an original vertex
            while (curr < num_pts)
            {
                if (polyline_pts[i][curr].type == SurfacePoint::Type::Face)
                {
                    spdlog::warn("face in polyline {}", i);
                    abort_path();
                    break;
                }
                else if (polyline_pts[i][curr].type == SurfacePoint::Type::Vertex)
                {
                    int vk = v_new2old[polyline_pts[i][curr].index];
                    set_simp_vertex(vk);
                    vertices.push_back(vk);
                    break;
                }
                else
                {
                    int hik = split_halfedges[i][curr];
                    int hki = overlay_mesh.opp[hik];
                    int vk = overlay_mesh.to[hik];
                    set_simp_vertex(vk);
                    vertices.push_back(vk);
                    halfedges.push_back(hik);
                    faces.push_back({overlay_mesh.face[hik], overlay_mesh.face[hki]});
                }

                curr++;
            }
            if (aborted) break;

            // casework depends on how many splits are in the current chain
            int num_splits = faces.size();
            spdlog::debug("split {} edges", num_splits);

            // special case: existing edge between vertices
            if (num_splits == 0)
            {
                assert(vertices.size() == 2);
                int vi = v_old2new[vertices[0]];
                int vj = v_old2new[vertices[1]];
                int h = vv2he_o.coeff(vi, vj) - 1;
                if ((overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] == EdgeType::EE)
                    && (overlay_mesh.h2e[overlay_mesh.simp.halfedge_parent[h]] != i))
                {
                    //mark_redundant(h, start);
                    spdlog::warn("path intersects existing edges");
                    abort_path();
                    break;
                }
                if ((overlay_mesh.to[h] != vertices[1]) || (overlay_mesh.to[overlay_mesh.opp[h]] != vertices[0]))
                {
                    spdlog::warn("halfedge in path {} no longer exists", i);
                    //mark_redundant(h, start);
                    abort_path();
                    break;
                }
                set_simp_edge(h);
            }
            // special case: if only one split, the face chain is ambiguous
            else if (num_splits == 1)
            {
                if (!overlay_mesh.is_vertex_degree_two(vertices[1]))
                {
                    spdlog::error("split of quad already exists");
                    abort_path();
                }

                // TODO may be problematic if the endpoint vertices are the same, but this is very unlikely
                int h = overlay_mesh.halfedge_from_face_vertex(faces[0][0], vertices[0]);
                int g = overlay_mesh.halfedge_from_face_vertex(faces[0][0], vertices[1]);
                if (h < 0)
                {
                    h = overlay_mesh.halfedge_from_face_vertex(faces[0][1], vertices[0]);
                    g = overlay_mesh.halfedge_from_face_vertex(faces[0][1], vertices[1]);
                    overlay_mesh.split_face(h, g);
                    set_simp_edge(overlay_mesh.n[h]);

                    h = overlay_mesh.halfedge_from_face_vertex(faces[0][0], vertices[1]);
                    g = overlay_mesh.halfedge_from_face_vertex(faces[0][0], vertices[2]);
                    overlay_mesh.split_face(h, g);
                    set_simp_edge(overlay_mesh.n[h]);
                }
                else
                {
                    overlay_mesh.split_face(h, g);
                    set_simp_edge(overlay_mesh.n[h]);

                    h = overlay_mesh.halfedge_from_face_vertex(faces[0][1], vertices[1]);
                    g = overlay_mesh.halfedge_from_face_vertex(faces[0][1], vertices[2]);
                    overlay_mesh.split_face(h, g);
                    set_simp_edge(overlay_mesh.n[h]);

                    // check that order is unambiguous
                    assert(overlay_mesh.halfedge_from_face_vertex(faces[0][0], vertices[2]) < 0);
                }
            }
            // general case: sequence of edge splits in a face chain
            else
            {
                std::vector<int> face_chain = generate_face_chain(faces);
                for (int k = 0; k < num_splits + 1; ++k)
                {
                    int f = face_chain[k];
                    int vk = vertices[k];
                    int vj = vertices[k + 1];

                    // special case: face already split (excluding final face of chain)
                    if ((!overlay_mesh.is_vertex_degree_two(vj)) && (k < num_splits))
                    {
                        spdlog::error("face already split");
                        int h = overlay_mesh.n[halfedges[k]]; 
                        int hjk = rotate_halfedge_to_tip(overlay_mesh, h, vk);
                        if (hjk < 0)
                        {
                            abort_path();
                            break;
                        }
                        int hkj = overlay_mesh.opp[hjk];
                        assert(overlay_mesh.to[hkj] == vj);
                        assert(overlay_mesh.to[hjk] == vk);
                        //mark_redundant(hkj, start + k);
                        //set_simp_edge(hkj);
                        abort_path();
                        break;
                    }
                    // special case: final face of chain already split
                    else if ((k == num_splits) && (overlay_mesh.vertex_degree(vk) == 4))
                    {
                        spdlog::error("final face already split");
                        int h = overlay_mesh.n[halfedges.back()]; 
                        int hkj = rotate_halfedge_to_tip(overlay_mesh, h, vj);
                        if (hkj < 0)
                        {
                            abort_path();
                            break;
                        }
                        assert(overlay_mesh.to[hkj] == vj);
                        //mark_redundant(hkj, start + k);
                        //set_simp_edge(hkj);
                        abort_path();
                        break;
                    }
                    // general case: need to split face
                    else
                    {
                        int h = overlay_mesh.halfedge_from_face_vertex(f, vk);
                        int g = overlay_mesh.halfedge_from_face_vertex(f, vj);
                        if ((h < 0) || (g < 0))
                        {
                            spdlog::error("could not find halfedge for split");
                            abort_path();
                            break;
                        }
                        assert((k == 0) ||(overlay_mesh.simp.vertex_parent[vk] == i));
                        assert((k == num_splits) ||(overlay_mesh.simp.vertex_parent[vj] == i));
                        overlay_mesh.split_face(h, g);
                        set_simp_edge(overlay_mesh.n[h]);
                    }
                }
            }
            if (aborted) break;

            int v_end = overlay_mesh.to[curr_path_halfeges.back()];
            if ((curr + 1 < num_pts) && (overlay_mesh.simp.vertex_type[v_end] == VertexType::VV))
            {
                spdlog::warn("vertex {} intersected by edge {}", overlay_mesh.simp.vertex_parent[v_end], i);
                abort_path();
                //is_redundant_path[i] = true;
                //redundant_vertices[i].push_back(curr);
            }
            int v_parent = overlay_mesh.simp.vertex_parent[v_end];
            if ((curr + 1 < num_pts) && (v_parent != i))
            {
                spdlog::warn("vertex in edge {} already labeled in edge {}", i, v_parent);
                abort_path();
                //is_redundant_path[i] = true;
                //is_redundant_path[v_parent] = true;
                //redundant_vertices[i].push_back(curr);
            }
            if (aborted) break;
        }

        if (!aborted)
        {
            is_redundant_path[i] = false;
            assert(is_valid_path(overlay_mesh, i));
        }
    }

    // final triangulation
    overlay_mesh.triangulate();

    // ensure that the overlay is not topologically degenerate after tracing
    //overlay_mesh.make_three_connected(overlay_mesh.orig);
}

int PathStraightener::get_start_halfedge(const VoronoiOverlayMesh& overlay_mesh, int h)
{
    int e = overlay_mesh.simp.h2e[h];
    if (h == overlay_mesh.simp.e2h[e])
    {
        return path_halfedges[e].front();
    }
    else
    {
        return overlay_mesh.opp[path_halfedges[e].back()];
    }

}

bool PathStraightener::is_valid_path(VoronoiOverlayMesh& overlay_mesh, int e)
{
    int Hij = overlay_mesh.simp.e2h[e];
    int Hji = overlay_mesh.simp.m.opp[Hij];
    int Vi = overlay_mesh.simp.m.to[Hji];
    int Vj = overlay_mesh.simp.m.to[Hij];
    for (int i = 0; i < path_halfedges[e].size(); ++i)
    {
        int hij = path_halfedges[e][i];
        int hji = overlay_mesh.opp[hij];
        int eji = overlay_mesh.h2e[hij];
        int v = overlay_mesh.to[hij];
        if (overlay_mesh.simp.edge_type[eji] != EdgeType::EE) return errmess("halfedge not labeled edge");
        if (overlay_mesh.simp.halfedge_parent[hij] != Hij) return errmess("halfedge parent is inconsistent");
        if (overlay_mesh.simp.halfedge_parent[hij] != Hij) return errmess("halfedge parent is inconsistent");
        if (i != path_halfedges[e].size() - 1)
        { 
            if (overlay_mesh.simp.vertex_type[v] != VertexType::VE) return errmess("interior vertex not labeled edge");
            if (overlay_mesh.simp.vertex_parent[v] != e) return errmess("interior vertex parent is inconsistent");
        }
    }

    int vi = overlay_mesh.to[overlay_mesh.opp[path_halfedges[e].front()]];
    if (overlay_mesh.simp.vertex_type[vi] != VertexType::VV) return errmess("start vertex not labeled vertex");
    if (overlay_mesh.simp.vertex_parent[vi] != Vi) return errmess("start vertex parent is inconsistent");

    int vj = overlay_mesh.to[path_halfedges[e].back()];
    if (overlay_mesh.simp.vertex_type[vj] != VertexType::VV) return errmess("end vertex not labeled vertex");
    if (overlay_mesh.simp.vertex_parent[vj] != Vj) return errmess("end vertex parent is inconsistent");

    return true;
}

// need to first connect all vertices, then any holononomy handles, then faces
std::vector<int> PathStraightener::compute_spanning_paths(VoronoiOverlayMesh& overlay_mesh)
{
    auto& m = overlay_mesh.simp.m;

    // get components of already traced paths
    UnionFind vertex_components(m.out.size());
    for (int eij = 0; eij < is_redundant_path.size(); ++eij)
    {
        if (is_redundant_path[eij]) continue;

        // union vertices joined by edge
        int hij = overlay_mesh.simp.e2h[eij];
        int hji = m.opp[hij];
        int vi = m.to[hji];
        int vj = m.to[hij];
        vertex_components.union_sets(vi, vj);
    }

    std::vector<int> spanning_paths = {};
    for (int eij = 0; eij < is_redundant_path.size(); ++eij)
    {
        // only process redundant paths
        if (!is_redundant_path[eij]) continue;

        // check if edge joins separate components
        int hij = overlay_mesh.simp.e2h[eij];
        int hji = m.opp[hij];
        int vi = m.to[hji];
        int vj = m.to[hij];
        if (vertex_components.find_set(vi) != vertex_components.find_set(vj))
        {
            vertex_components.union_sets(vi, vj);
            spanning_paths.push_back(eij);
        }
    }

    return spanning_paths;
}

void PathStraightener::retrace_paths(VoronoiOverlayMesh& overlay_mesh)
{
    std::vector<int> redundant_paths = enumerate_boolean_array(is_redundant_path);
    int num_redundant = redundant_paths.size();
    if (num_redundant == 0) return;
    spdlog::info("{} redundant paths to retrace", num_redundant);
    auto& m = overlay_mesh.simp.m;

    for (int e : redundant_paths)
    {
        remove_path(overlay_mesh, e);
    }

    std::vector<int> vertex_locations(m.out.size(), -1);
    for (int vi = 0; vi < overlay_mesh.out.size(); ++vi)
    {
        if (overlay_mesh.simp.vertex_type[vi] == VertexType::VV)
        {
            vertex_locations[overlay_mesh.simp.vertex_parent[vi]] = vi;
        }

        // reset vertices in deleted path
        else if (overlay_mesh.simp.vertex_type[vi] == VertexType::VE)
        {
            if (is_redundant_path[overlay_mesh.simp.vertex_parent[vi]]) 
            {
                overlay_mesh.simp.vertex_type[vi] = VertexType::VF;
            }
        }
    }

    // check path consistency for remaining paths
    for (int e = 0; e < path_halfedges.size(); ++e)
    {
        if (is_redundant_path[e]) continue;
        if (!is_valid_path(overlay_mesh, e)) spdlog::error("path {} broken before tracing", e);
    }

    overlay_mesh.make_regular();


    // trace paths with no adjacent vertex
    for (int e : redundant_paths)
    {
        int Hij = overlay_mesh.simp.e2h[e];
        int Hji = m.opp[Hij];
        int Vi = m.to[Hji];
        int Vj = m.to[Hij];

        // get next clockwise nonredundant edge
        int Hik = Hij;
        do
        {
            if (!is_redundant_path[overlay_mesh.simp.h2e[Hik]]) break;
            Hik = m.n[m.opp[Hik]];
        } while (Hik != Hij);

        // get next clockwise nonredundant edge
        int Hjl = Hji;
        do
        {
            if (!is_redundant_path[overlay_mesh.simp.h2e[Hjl]]) break;
            Hjl = m.n[m.opp[Hjl]];
        } while (Hjl != Hji);
        if ((Hik != Hij) && (Hjl != Hji)) continue;

        // retrace path
        spdlog::debug("tracing {} = {}, {}", e, Hij, Hji);
        int v_start = vertex_locations[Vi];
        int v_end = vertex_locations[Vj];

        // trace new path
        std::vector<int> halfedges;
        overlay_mesh.trace_path(v_start, v_end, Hij, e, halfedges);
        for (int h : halfedges)
        {
            overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[h]] = Hji;
        }
        overlay_mesh.make_adjacent_regular(halfedges);

        path_halfedges[e] = reverse_vector(halfedges);
        is_redundant_path[e] = false;
    }

    // check path consistency for remaining paths
    for (int e = 0; e < path_halfedges.size(); ++e)
    {
        if (is_redundant_path[e]) continue;
        if (!is_valid_path(overlay_mesh, e))
        {
            spdlog::error("path {} broken after tracing isolated vertices", e);
            remove_path(overlay_mesh, e);
            is_redundant_path[e] = true;
            redundant_paths.push_back(e);
        }
    }

    // first, make sure all vertices are connected
    std::vector<int> spanning_paths = compute_spanning_paths(overlay_mesh);
    for (int e : spanning_paths)
    {
        if (!is_redundant_path[e]) continue;
        spdlog::debug("tracing spanning path {} with consistent orientation", e);
        retrace_consistent_path(overlay_mesh, e);
    }

    for (int e : redundant_paths)
    {
        if (!is_redundant_path[e]) continue;
        spdlog::debug("tracing remaining path {} with consistent orientation", e);
        retrace_consistent_path(overlay_mesh, e);
    }
}

void PathStraightener::retrace_consistent_path(VoronoiOverlayMesh& overlay_mesh, int e)
{
    auto& m = overlay_mesh.simp.m;
    int Hij = overlay_mesh.simp.e2h[e];
    int Hji = m.opp[Hij];
    int vi = m.to[Hji];
    int vj = m.to[Hij]; 
    int hij, hji;
    spdlog::debug("tracing path {} from {} to {}", e, vi, vj);

    // get next clockwise nonredundant edge
    int Hik = Hij;
    do
    {
        if (!is_redundant_path[overlay_mesh.simp.h2e[Hik]]) break;
        Hik = m.n[m.opp[Hik]];
    } while (Hik != Hij);
    assert(Hik != Hij);
    int hik = get_start_halfedge(overlay_mesh, Hik);
    hij = overlay_mesh.opp[overlay_mesh.prev[hik]];

    // get next clockwise nonredundant edge
    int Hjl = Hji;
    do
    {
        if (!is_redundant_path[overlay_mesh.simp.h2e[Hjl]]) break;
        Hjl = m.n[m.opp[Hjl]];
    } while (Hjl != Hji);
    assert(Hjl != Hji);
    int hjl = get_start_halfedge(overlay_mesh, Hjl);
    hji = overlay_mesh.opp[overlay_mesh.prev[hjl]];

    // rotate edges counterclockwise
    overlay_mesh.simp.halfedge_parent[hij] = Hij;
    overlay_mesh.simp.halfedge_parent[hji] = Hji;
    overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[hij]] = Hji;
    overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[hji]] = Hij;
    for (int h : {hij, hji})
    {
        assert(overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] == EdgeType::EF);
        overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] = EdgeType::EE;

        int v = overlay_mesh.to[h];
        assert((overlay_mesh.simp.vertex_type[v] == VertexType::VF) || (overlay_mesh.simp.vertex_parent[v] == e));
        overlay_mesh.simp.vertex_type[v] = VertexType::VE;
        overlay_mesh.simp.vertex_parent[v] = e;
    }

    int v_start = overlay_mesh.to[hij];
    int v_end = overlay_mesh.to[hji];

    // trace new path
    std::vector<int> halfedges;
    overlay_mesh.trace_path(v_start, v_end, Hij, e, halfedges);
    is_redundant_path[e] = false;
    path_halfedges[e] = {hij};
    for (int i = halfedges.size() - 1; i >= 0; --i)
    {
        int h = halfedges[i];
        overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[h]] = Hji;
        path_halfedges[e].push_back(h);
    }
    path_halfedges[e].push_back(overlay_mesh.opp[hji]);
    overlay_mesh.make_adjacent_regular(path_halfedges[e]);
}

/*
    return;
    int num_faces = m.n_faces();
    int num_traced = 0;
    do
    {
        num_traced = 0;
        for (int f = 0; f < num_faces; ++f)
        {
            int hij = m.h[f]; 
            int hjk = m.n[hij];
            int hki = m.n[hjk];
            int count = 0;
            int e = 0;
            int h_ref = 0;
            for (int h : {hij, hjk, hki})
            {
                if (!is_redundant_path[overlay_mesh.simp.h2e[h]])
                {
                    count++;
                    h_ref = h;
                }
                else
                {
                    e = overlay_mesh.simp.h2e[h];
                }

            }

            if (count == 2)
            {
                spdlog::info("tracing {} in face {}", e, f);
                for (int h = 0; h < overlay_mesh.n.size(); ++h)
                {
                    if (overlay_mesh.is_halfedge_deleted(h)) continue;
                    if (overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] != EdgeType::EE) continue;
                    if (overlay_mesh.simp.halfedge_parent[h] == h_ref)
                    {
                        spdlog::info("flooding {} in path {}", h, overlay_mesh.simp.h2e[h_ref]);
                        overlay_mesh.flood_face(h, f);
                        break;
                    }
                }
                retrace_path(overlay_mesh, e, f);
                is_redundant_path[e] = false;
                num_traced++;
            }
        }
    }
    while (num_traced > 0);


    //for (int e : redundant_paths)
    //{
    //    retrace_path(overlay_mesh, e);
    //}
}
*/

void PathStraightener::separate_paths(VoronoiOverlayMesh& overlay_mesh)
{
    std::vector<int> redundant_paths = enumerate_boolean_array(is_redundant_path);
    int num_redundant = redundant_paths.size();
    spdlog::info("{} redundant paths to separate", num_redundant);

    overlay_mesh.make_regular();
    overlay_mesh.view();
    halfedge_counts = std::vector<int>(overlay_mesh.e2h.size(), 0);
    std::vector<int> vertex_counts(overlay_mesh.out.size(), 0);
    for (int i : redundant_paths)
    {
        int vi = overlay_mesh.to[overlay_mesh.opp[path_halfedges[i].front()]];
        int Hij = overlay_mesh.simp.e2h[i];
        int Hji = overlay_mesh.simp.m.opp[Hij];
        vertex_counts[vi] += 1;
        for (int h : path_halfedges[i])
        {
            halfedge_counts[overlay_mesh.h2e[h]] += 1;
            vertex_counts[overlay_mesh.to[h]] += 1;
        }
    }

    for (int e : redundant_paths)
    {
        bool successs = separate_simple_path(overlay_mesh, e);
        if (!successs)
        {
            spdlog::warn("could not simplify {}", e);
        }
    }
}

void PathStraightener::remove_path(VoronoiOverlayMesh& overlay_mesh, int e)
{
    for (int h : path_halfedges[e])
    {
        int vi = overlay_mesh.to[h];
        if (overlay_mesh.simp.h2e[overlay_mesh.simp.halfedge_parent[h]] == e)
        {
            overlay_mesh.simp.edge_type[overlay_mesh.h2e[h]] = EdgeType::EF;
        }
        if ((overlay_mesh.simp.vertex_type[vi] == VertexType::VE) && (overlay_mesh.simp.vertex_parent[vi] == e)) 
        {
            overlay_mesh.simp.vertex_type[vi] = VertexType::VF;
        }
    }
}

void PathStraightener::retrace_path(VoronoiOverlayMesh& overlay_mesh, int e, int f)
{
    int Hij = overlay_mesh.simp.e2h[e];
    int Hji = overlay_mesh.simp.m.opp[Hij];
    int v_start = overlay_mesh.to[overlay_mesh.opp[path_halfedges[e].front()]];
    int v_end = overlay_mesh.to[path_halfedges[e].back()];

    // trace new path
    std::vector<int> halfedges;
    overlay_mesh.trace_path(v_start, v_end, Hij, e, halfedges, f);
    for (int h : halfedges)
    {
        overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[h]] = Hji;
    }
    overlay_mesh.make_adjacent_regular(halfedges);

    path_halfedges[e] = reverse_vector(halfedges);
}

bool PathStraightener::separate_simple_path(VoronoiOverlayMesh& overlay_mesh, int e)
{
    // check if starting from a simple halfedge
    int h_start = path_halfedges[e][0];
    if (halfedge_counts[overlay_mesh.h2e[h_start]] > 1) return false;
    int Hij = overlay_mesh.simp.e2h[e];
    int Hji = overlay_mesh.simp.m.opp[Hij];

    // get the first edge of the path and determine if it is simple
    bool was_simple = true;
    int prev = 0;
    int next = -1;
    int path_length = path_halfedges[e].size();
    std::vector<int> new_path = {h_start};
    for (int i = 1; i < path_length; ++i)
    {
        // make sure no intersections from the left hand side
        int hki = path_halfedges[e][i - 1];
        int hij = path_halfedges[e][i];
        if (overlay_mesh.get_clockwise_overlay_edge(hki, overlay_mesh.simp) != hij) return false;

        // check if current edge is simple
        bool is_simple = halfedge_counts[overlay_mesh.h2e[hij]] == 1;

        // if still traversing simple region, update previous halfedge and add to path
        if ((was_simple) && (is_simple))
        {
            new_path.push_back(hij);
            prev = 0;
        }
        // if entering complex region, remove previous path edge from path
        else if ((was_simple) && (!is_simple))
        {
            new_path.pop_back();
        }
        // if transitioning from complex to simple region, trace the new path
        else if ((!was_simple) && (is_simple))
        {
            next = i;
            int h_prev = path_halfedges[e][prev];
            int h_next = path_halfedges[e][next];
            int v_prev, v_next;
            if (prev > 0)
            {
                overlay_mesh.simp.edge_type[overlay_mesh.h2e[h_prev]] = EdgeType::EF;
                v_prev = overlay_mesh.to[overlay_mesh.opp[h_prev]];
            }
            else
            {
                new_path.push_back(h_prev);
                v_prev = overlay_mesh.to[h_prev];
            }
            if (next < (path_length - 1))
            {
                overlay_mesh.simp.edge_type[overlay_mesh.h2e[h_next]] = EdgeType::EF;
                v_prev = overlay_mesh.to[h_next];
            }
            else
            {
                v_next = overlay_mesh.to[overlay_mesh.opp[h_next]];
            }
            // TODO make local
            overlay_mesh.make_regular();

            // trace new path
            std::vector<int> halfedges;
            overlay_mesh.trace_path(v_prev, v_next, Hij, e, halfedges);
            for (int h : halfedges)
            {
                new_path.push_back(h_prev);
                overlay_mesh.simp.halfedge_parent[overlay_mesh.opp[h]] = Hji;
            }
            overlay_mesh.make_adjacent_regular(halfedges);
            for (int j = prev + 1; j < next; ++j)
            {
                int h = path_halfedges[e][j];
                halfedge_counts[overlay_mesh.h2e[h]] -= 1;
            }
            overlay_mesh.view();
        }
        // nothing to do if still in complex region

        was_simple = is_simple;
    }

    return true;
}

// rotate hij until it points to vk
int PathStraightener::rotate_halfedge_to_tip(const VoronoiOverlayMesh& overlay_mesh, int hij, int vk) const
{
    int h = hij;
    do
    {
        if (overlay_mesh.to[h] == vk) return h;
        h = overlay_mesh.n[overlay_mesh.opp[h]];
    }
    while (h != hij);

    return -1;
}

std::vector<int> PathStraightener::generate_face_chain(const std::vector<std::array<int, 2>>& faces) const
{
    std::vector<int> face_chain;
    int num_splits = faces.size();
    face_chain.reserve(num_splits + 1);
    int j = (vector_contains(faces[1], faces[0][0])) ? 1 : 0;

    // TODO: face chain is invalid in cases of shared edges; need to reconsider
    //assert((j == 1) || (vector_contains(faces[1], faces[0][1])));

    face_chain.push_back(faces[0][j]);
    for (int k = 0; k < num_splits - 1; ++k)
    {
        j = (faces[k][0] == face_chain.back()) ? 1 : 0;
        face_chain.push_back(faces[k][j]);
    }
    j = (faces.back()[0] == face_chain.back()) ? 1 : 0;
    face_chain.push_back(faces.back()[j]);
    spdlog::debug("built face chain of length {}", face_chain.size());

    return face_chain;
}

void PathStraightener::run(VoronoiOverlayMesh& overlay_mesh)
{
    //view(overlay_mesh);
    split_edges(overlay_mesh);
    join_edges(overlay_mesh);
    retrace_paths(overlay_mesh);
}

Eigen::Vector3d get_surface_point_position(
    Eigen::MatrixXd& V_o,
    VoronoiOverlayMesh& overlay_mesh,
    SurfacePoint& pt)
{
    if (pt.type == SurfacePoint::Type::Vertex)
    {
        return V_o.row(pt.index);
    }
    else if (pt.type == SurfacePoint::Type::Edge)
    {
        Eigen::Vector3d v0 = V_o.row(pt.edge_base);
        Eigen::Vector3d v1 = V_o.row(pt.edge_tip);
        double t = (double)(pt.t);
        return ((1. - t) * v0) + (t * v1);
    }
    else if (pt.type == SurfacePoint::Type::Face)
    {
        spdlog::error("faces not supported");
    }

    return Eigen::Vector3d();
}

std::vector<std::vector<Eigen::Vector3d>> get_surface_point_positions(
    VoronoiOverlayMesh& overlay_mesh,
    std::vector<std::vector<SurfacePoint>>& polyline_pts
)
{
    Eigen::MatrixXd V_o = overlay_mesh.generate_overlay_vertices(
        overlay_mesh.V_orig,
        overlay_mesh.vtx_reindex_orig,
        overlay_mesh.orig);
    int num_paths = polyline_pts.size();
    std::vector<std::vector<Eigen::Vector3d>> positions(num_paths);
    for (int i = 0; i < num_paths; ++i)
    {
        int path_size = polyline_pts[i].size();
        positions[i].resize(path_size);
        for (int j = 0; j < path_size; ++j)
        {
            positions[i][j] = get_surface_point_position(
                V_o,
                overlay_mesh,
                polyline_pts[i][j]);
        }
    }

    return positions;
}

void PathStraightener::view(
    VoronoiOverlayMesh& overlay_mesh
)
{
    std::vector<std::vector<Eigen::Vector3d>> polyline = get_surface_point_positions(overlay_mesh, polyline_pts);
    auto [nodes, edges, labels] = convert_polyline(polyline);
    //Eigen::MatrixXi F_c = edgeNetwork.tri->getFaceVertexMatrix();

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh("simplified mesh", V, F);
    polyscope::registerCurveNetwork("edges", nodes, edges);
    polyscope::getCurveNetwork("edges")->addEdgeScalarQuantity("edge parent", labels);
    polyscope::show();
#endif
}



void add_geodesic_paths(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<int>>& paths
) {
    std::vector<std::vector<int>> paths_init = overlay_mesh.enumerate_simplified_edges();
    std::vector<std::vector<SurfacePoint>> polyline_pts = generate_geodesic_path_points(overlay_mesh, paths_init);
    PathStraightener path_straightener(overlay_mesh, polyline_pts);
    path_straightener.split_edges(overlay_mesh);
    path_straightener.join_edges(overlay_mesh);
    path_straightener.retrace_paths(overlay_mesh);
}

}
}
