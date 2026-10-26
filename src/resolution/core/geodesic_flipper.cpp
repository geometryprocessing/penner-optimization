#include "resolution/core/geodesic_flipper.h"

#include "util/vector.h"
#include "util/union_find.h"
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

GeodesicPath::GeodesicPath(const std::vector<int>& path)
{
    // build initial chain
    int curr_segment = create_segment(path.front());
    start = curr_segment;
    for (int i = 1; i < path.size(); ++i)
    {
        curr_segment = create_next_segment(curr_segment, path[i]);
    }
    end = curr_segment;
}

std::vector<int> GeodesicPath::get_path_halfedges() const
{
    int i = start;
    std::vector<int> halfedges = {halfedge[i]};
    while (i != end)
    {
        i = next[i];
        halfedges.push_back(halfedge[i]);
    }

    return halfedges;
}

int GeodesicPath::create_segment(int h)
{
    int index = halfedge.size();
    halfedge.push_back(h);
    next.push_back(-1);
    prev.push_back(-1);
    to.push_back(-1);

    return index;
}

int GeodesicPath::create_wedge(int index)
{
    int wedge = m_is_wedge_valid.size();
    m_is_wedge_valid.push_back(true);
    in.push_back(index);
    to[index] = wedge;
    return wedge;
}

int GeodesicPath::create_next_segment(int index, int h)
{
    int next_index = create_segment(h);
    next[index] = next_index;
    prev[next_index] = index;
    int wedge = create_wedge(index);
    return next_index;
}

bool GeodesicPath::is_valid(Mesh<Scalar>& m)
{
    int i = start;
    int count = 0;
    while (i != end)
    {
        // check if in a loops
        if (count > halfedge.size()) return false;

        // check if next/prev are inverse
        int j = next[i];
        if (prev[j] != i) return false;

        // check if halfedges adjacent
        int hij = halfedge[i];
        int hjk = halfedge[j];
        if (m.to[hij] != m.to[m.opp[hjk]]) return false;

        i = j;
        ++count;
    }

    return true;
}

GeodesicFlipper::GeodesicFlipper(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<int>>& halfedge_paths,
    bool allow_intersections
)
  : GeodesicFlipper(overlay_mesh.generate_mesh(), halfedge_paths, allow_intersections)
{}

GeodesicFlipper::GeodesicFlipper(
    const Mesh<Scalar>& mesh,
    const std::vector<std::vector<int>>& halfedge_paths,
    bool allow_intersections
)
  : m_o(mesh)
  , edge_path_counts(m_o.n_edges(), 0)
{
    build_edge_maps(m_o.opp, h2e, e2h);
    keep_simple = !allow_intersections;

    // compute initial corner angles
    auto& m = m_o.cmesh();
    VectorX cot_angles;
    corner_angles(m, angles, cot_angles);

    // add paths to registry
    int num_paths = halfedge_paths.size();
    paths.reserve(num_paths);
    path_length = std::vector<Scalar>(num_paths, 0.);
    path_curvature = std::vector<Scalar>(num_paths, 0.);
    if (allow_intersections) adjacent_paths = std::vector<std::deque<std::pair<int, int>>>(m_o.n_edges(), std::deque<std::pair<int, int>>());
    for (int i = 0; i < num_paths; ++i)
    {
        const auto& path = halfedge_paths[i];
        paths.push_back(path);

        // add adjacent paths to record
        for (int j = 0; j < path.size(); ++j)
        {
            int h = paths[i].halfedge[j];
            ++edge_path_counts[h2e[h]];
            if (allow_intersections) adjacent_paths[h2e[h]].push_back({i, j});
            path_length[i] += m.l[h];
        }
        path_curvature[i] = compute_path_curvature(i);
        if (track_deltas)
        {
            paths[i].sizes.push_back(path.size());
            paths[i].lengths.push_back(path_length[i]);
        }
    }
    spdlog::info("initial path total length: {}", compute_total_sum(path_length));
    if (!is_simple()) spdlog::error("cannot straighten paths if not simple");
    assert(is_simple());
}

void GeodesicFlipper::flip_to_geodesics()
{
    // add all wedges to priority queue
    int num_paths = paths.size();

    wedges_to_process = PriorityQueue();

    // iterate until no more wedges to shorten
    int count = 1;
    while (count > 0)
    {
        count = 0;
        check_all_wedges();

        // iterate
        while (!wedges_to_process.empty())
        {
            if (!process_wedge()) continue;
            ++count;

            // check if exceed max iteration count
            if ((max_flipouts > 0) && (count > max_flipouts)) return;
        }
        spdlog::info("{} wedges shortened", count);
    }

    spdlog::info("final path total length: {}", compute_total_sum(path_length));
}

void GeodesicFlipper::check_wedge(int path_index, int wedge_index)
{
    const auto& path = paths[path_index];
    if (!path.is_wedge_valid(wedge_index)) return;
    if (path_curvature[path_index] < (path_curvature_threshold)) return;

    int j = path.in[wedge_index];
    int k = path.next[j];
    int hij = path.halfedge[j];
    int hjk = path.halfedge[k];

    // check if clockwise wedge is acute
    Scalar clockwise_wedge_angle = compute_clockwise_wedge_angle(hij, hjk);
    if (clockwise_wedge_angle < (flip_angle - wedge_eps))
    {
        wedges_to_process.push(std::make_tuple(clockwise_wedge_angle, path_index, wedge_index, true));
    }

    // check if ccw wedge is acute
    Scalar ccw_wedge_angle = compute_ccw_wedge_angle(hij, hjk);
    if (ccw_wedge_angle < (flip_angle - wedge_eps))
    {
        wedges_to_process.push(std::make_tuple(ccw_wedge_angle, path_index, wedge_index, false));
    }
}


void GeodesicFlipper::check_all_wedges()
{
    int num_paths = paths.size();
    for (int i = 0; i < num_paths; ++i)
    {
        const auto& path = paths[i];
        for (int j = path.get_start(); j != path.get_end(); j = path.next[j])
        {
            check_wedge(i, path.to[j]);
        }
    }
}

bool GeodesicFlipper::process_wedge()
{
    auto [wedge_angle, i, wedge_index, is_left_joint] = wedges_to_process.top();
    wedges_to_process.pop();
    if (!paths[i].is_wedge_valid(wedge_index)) return false;

    // get wedge halfedges and check if joint is still flexible
    int j = paths[i].in[wedge_index];
    int k = paths[i].next[j];
    int hij = paths[i].halfedge[j];
    int hjk = paths[i].halfedge[k];
    if (!is_joint_flexible(i, wedge_index, is_left_joint)) return false;

    // flip edges to make shorter path
    int num_flips = flip_out(hij, hjk, is_left_joint);

    // check if shorter joint path will result in undesired intersection
    if ((keep_simple) && (!is_wedge_simple(i, wedge_index, is_left_joint)))
    {
        undo_flips(num_flips);
        return false;
    }

    // shorten the wedge if possible
    shorten_wedge(i, wedge_index, is_left_joint);

    return true;
}

int GeodesicFlipper::flip_out(int hij, int hjk, bool is_left_joint)
{
    // flip opposite acute angles until none remain
    bool edge_flipped = true;
    int count = 0;
    while (edge_flipped)
    {
        if (is_left_joint)
        {
            edge_flipped = flip_left_acute_angles(hij, hjk);
        }
        else
        {
            edge_flipped = flip_right_acute_angles(hij, hjk);
        }
        if (edge_flipped) ++count;
        if (count == 1000) spdlog::warn("more than a thousand flips done in flip out");
    }

    return count;
}

bool GeodesicFlipper::flip_left_acute_angles(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    int hjl = m.n[hij];
    while (hjl != hjk)
    {
        // flip edge if opposite angle is less than pi
        Scalar beta = angles[m.n[m.n[hjl]]] + angles[m.n[m.opp[hjl]]];
        if (beta < (PI - wedge_eps))
        {
            flip_halfedge(hjl);
            return true;
        }

        // update halfedge index
        hjl = m.n[m.opp[hjl]];
    }

    return false;
}

bool GeodesicFlipper::flip_right_acute_angles(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    int hkj = m.opp[hjk];
    int hji = m.opp[hij];
    return flip_left_acute_angles(hkj, hji);
}

void GeodesicFlipper::flip_halfedge(int hij)
{
    // do flip in overlay mesh
    bool use_ptolemy = false;
    for (int i = 0; i < 3; ++i)
    {
        m_o.flip_ccw(hij, use_ptolemy);
    }

    // record flip
    flip_sequence.push_back(hij);

    // update adjacent angles triangle angles
    auto& m = m_o.cmesh();
    int hji = m.opp[hij];
    assert(hji >= 0);
    update_face_angles(m.f[hij]);
    update_face_angles(m.f[hji]);
}

void GeodesicFlipper::shorten_wedge(
    int path_index,
    int wedge_index,
    bool is_left_joint)
{
    auto& path = paths[path_index];
    auto& m = m_o.cmesh();
    assert(path.is_valid(m_o));

    // get relative indices
    int index = path.in[wedge_index];
    int prev_index = path.prev[index];
    int next_index = path.next[index];
    int hij = path.halfedge[index];
    int hjk = path.halfedge[next_index];
    Scalar old_length = m.l[hij] + m.l[hjk];
    Scalar new_length = 0.;
    Scalar new_curvature = 0.;
    int count = 0;

    // remove wedge from adjacent path
    --edge_path_counts[h2e[hij]];
    --edge_path_counts[h2e[hjk]];
    if (!keep_simple)
    {
        if (is_left_joint)
        {
            adjacent_paths[h2e[hij]].pop_front();
            adjacent_paths[h2e[hjk]].pop_front();
        }
        else
        {
            adjacent_paths[h2e[hij]].pop_back();
            adjacent_paths[h2e[hjk]].pop_back();
        }
    }

    // delete wedge being pushed
    path.mark_wedge_invalid(path.to[index]);

    // connect new path at start
    int hjl = (is_left_joint) ? m.n[hij] : m.n[m.n[m.opp[hij]]];
    int h_path = (is_left_joint) ? m.opp[m.n[hjl]] : m.n[m.n[hjl]];
    new_length += m.l[h_path];
    ++count;
    if (path.is_start(index))
    {
        int curr_index = path.create_segment(h_path);
        path.set_start(curr_index);
        add_adjacent_halfedge(path_index, curr_index, !is_left_joint);
        prev_index = curr_index;
    }
    else
    {
        path.mark_wedge_invalid(path.to[prev_index]);
        int curr_index = path.create_next_segment(prev_index, h_path);
        add_adjacent_halfedge(path_index, curr_index, !is_left_joint);
        check_wedge(path_index, path.to[prev_index]);
        prev_index = curr_index;
    }

    // set any interior paths
    int h_end = (is_left_joint) ? hjk : m.opp[hjk];
    while (hjl != h_end)
    {
        hjl = (is_left_joint) ? m.n[m.opp[hjl]] : m.n[m.n[m.opp[hjl]]];
        h_path = (is_left_joint) ? m.opp[m.n[hjl]] : m.n[m.n[hjl]];
        new_length += m.l[h_path];
        ++count;
        assert(m.to[path.halfedge[prev_index]] == m.to[m.opp[h_path]]);
        int curr_index = path.create_next_segment(prev_index, h_path);
        add_adjacent_halfedge(path_index, curr_index, !is_left_joint);
        check_wedge(path_index, path.to[prev_index]);
        prev_index = curr_index;
    }
    assert(m.to[path.halfedge[prev_index]] == m.to[hjk]);

    // connect new path at end
    if (path.is_end(next_index))
    {
        // mark new halfedge as end of path
        path.set_end(prev_index);
    }
    else
    {
        path.mark_wedge_invalid(path.to[next_index]);
        path.next[prev_index] = path.next[next_index];
        path.prev[path.next[next_index]] = prev_index;
        int wedge_index = path.create_wedge(prev_index);
        check_wedge(path_index, wedge_index);
    }

    path_length[path_index] += (new_length - old_length);
    path_curvature[path_index] = compute_path_curvature(path_index);
    if (track_deltas)
    {
        path.lengths.push_back(path.lengths.back() + new_length - old_length);
        path.sizes.push_back(path.sizes.back() + count - 2);
        deltas.push_back(old_length - new_length);
    }
    assert(path.is_valid(m_o));
    assert((keep_simple) || (is_simple()));
}

void GeodesicFlipper::undo_flips(int num_flips)
{
    // do flip in overlay mesh
    bool use_ptolemy = false;
    for (int i = 0; i < num_flips; ++i)
    {
        int hij = flip_sequence.back();
        m_o.flip_ccw(hij, use_ptolemy);
        flip_sequence.pop_back();

        // update adjacent angles triangle angles
        auto& m = m_o.cmesh();
        int hji = m.opp[hij];
        assert(hji >= 0);
        update_face_angles(m.f[hij]);
        update_face_angles(m.f[hji]);
    }
}

Scalar GeodesicFlipper::compute_clockwise_wedge_angle(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    assert(m.to[hij] == m.to[m.opp[hjk]]);

    int hjl = m.n[hij];
    Scalar wedge_angle = 0.;
    while (hjl != hjk)
    {
        wedge_angle += angles[m.n[hjl]];
        int hlj = m.opp[hjl];
        if (hlj < 0) return infinity;
        hjl = m.n[hlj];
    }
    wedge_angle += angles[m.n[hjk]];
        
    return wedge_angle;
}

Scalar GeodesicFlipper::compute_ccw_wedge_angle(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    int hkj = m.opp[hjk];
    int hji = m.opp[hij];
    if ((hji < 0) || (hkj < 0)) return infinity;
    return compute_clockwise_wedge_angle(hkj, hji);
}

Scalar GeodesicFlipper::compute_wedge_curvature(int hij, int hjk, bool use_signed)
{
    Scalar clockwise_angle = compute_clockwise_wedge_angle(hij, hjk);
    if (use_signed)
    {
        return (PI - clockwise_angle);
    }
    else
    {
        Scalar ccw_angle = compute_ccw_wedge_angle(hij, hjk);
        return abs(PI - min(clockwise_angle, ccw_angle));
    }
}

Scalar GeodesicFlipper::compute_path_curvature(int path_index, bool use_signed)
{
    Scalar angle = 0.;
    const auto& path = paths[path_index];
    for (int i = path.get_start(); i != path.get_end(); i = path.next[i])
    {
        int j = path.next[i];
        int hij = path.halfedge[i];
        int hjk = path.halfedge[j];
        angle += compute_wedge_curvature(hij, hjk, use_signed);
    }

    return angle;
}

std::vector<Scalar> GeodesicFlipper::compute_path_curvatures(bool use_signed)
{
    int num_paths = paths.size();
    std::vector<Scalar> angles(num_paths);
    for (int i = 0; i < num_paths; ++i)
    {
        angles[i] = compute_path_curvature(i, use_signed);
    }

    return angles;
}

void GeodesicFlipper::generate_surface_points()
{
    //m_o.garbage_collection();
    OverlayMesh<Scalar> m_rev(m_o.cmesh());
    for (auto iter = flip_sequence.rbegin(); iter != flip_sequence.rend(); ++iter) {
        int halfedge_index = *iter;
        m_rev.flip_ccw(halfedge_index, false);
    }
    m_rev.garbage_collection();
    auto& m = m_rev.cmesh();

    // store size statistic
    overlay_faces = m_rev.n_faces();

    std::vector<int> h2e_rev, e2h_rev;
    // TODO may be possible to reuse for forward overlay
    build_edge_maps(m.opp, h2e_rev, e2h_rev);

    // get list of first segments of the original edges in the overlay
    std::vector<int> first_segment(m.n_halfedges(), -1);
    for (int hij = 0; hij < m_rev.n_halfedges(); ++hij)
    {
        // only look for original edges
        if (m_rev.edge_type[hij] == CURRENT_EDGE)
            continue;


        int vi = m_rev.to[m_rev.opp[hij]];
        int Hij = m_rev.origin_of_origin[hij];
        if (m_rev.vertex_type[vi] == ORIGINAL_VERTEX)
            first_segment[Hij] = hij;
    }

    surface_points.clear();
    std::vector<int> vertex_path(m_rev.n_vertices(), -1);
    std::vector<int> vertex_index(m_rev.n_vertices(), -1);
    for (int i = 0; i < paths.size(); ++i)
    {
        // get halfedges in the path
        std::vector<int> path_halfedges = paths[i].get_path_halfedges();
        std::vector<int> overlay_halfedges = {};
        for (int h : path_halfedges)
        {
            // iterate over segments in the overlay of the edge
            int seg = first_segment[h];
            overlay_halfedges.push_back(seg);
            while (m_rev.vertex_type[m_rev.to[seg]] != ORIGINAL_VERTEX)
            {
                seg = next_orig_segment(m_rev, seg);
                overlay_halfedges.push_back(seg);
            }
        }

        // get vertices in the path
        std::vector<int> path_vertices = {m_rev.to[m_rev.opp[overlay_halfedges.front()]]};
        for (int h : overlay_halfedges)
        {
            path_vertices.push_back(m_rev.to[h]);
        }

        // build point data for each path vertex
        int path_size = path_vertices.size();
        surface_points.push_back(std::vector<SurfacePoint>(path_size));
        auto& path_points = surface_points.back();
        for (int j = 0; j < path_size; ++j)
        {
            int v = path_vertices[j];

            // path vertex is a vertex of the mesh
            if (m_rev.vertex_type[v] == ORIGINAL_VERTEX)
            {
                path_points[j].type = SurfacePoint::Type::Vertex;
                path_points[j].index = v;
            }
            // path vertex is an inserted vertex on an edge
            else
            {
                path_points[j].type = SurfacePoint::Type::Edge;
                int h = m_rev.opp[m_rev.out[v]];
                while (m_rev.edge_type[h] == ORIGINAL_EDGE)
                {
                    h = m_rev.opp[m_rev.n[h]];
                }

                path_points[j].index = h2e_rev[m_rev.origin[h]];
                int Hij = e2h_rev[path_points[j].index];
                path_points[j].edge_base = m.to[m.opp[Hij]];
                path_points[j].edge_tip = m.to[Hij];
                path_points[j].t = (Hij == m_rev.origin[h]) ? m_rev.seg_bcs[h][1] : m_rev.seg_bcs[h][0];
                vertex_path[v] = i;
                vertex_index[v] = j;
            }
        }
    }

    edge_splits = std::vector<std::vector<std::array<int, 2>>>(h2e_rev.size(), std::vector<std::array<int, 2>>(0));
    for (int eij = 0; eij < e2h_rev.size(); ++eij)
    {
        int Hij = e2h_rev[eij];
        int hij = m_rev.first_segment[m.opp[Hij]];
        while (m_rev.vertex_type[m_rev.to[hij]] != ORIGINAL_VERTEX)
        {
            int v = m_rev.to[hij];
            if (vertex_path[v] >= 0)
            {
                edge_splits[Hij].push_back({vertex_path[v], vertex_index[v]});
            }
            hij = m_rev.next_segment(hij);
        }
    }
}

std::vector<bool> GeodesicFlipper::get_intersecting_paths()
{
    int num_paths = paths.size();
    std::vector<bool> is_intersecting(num_paths, false);
    if (keep_simple) return is_intersecting;

    for (auto& paths : adjacent_paths)
    {
        if (paths.size() <= 1) continue;
        for (auto& path : paths)
        {
            int path_index = path.first;
            is_intersecting[path_index] = true;
        }
    }

    return is_intersecting;
}

void GeodesicFlipper::add_to_viewer(VoronoiOverlayMesh& overlay_mesh)
{
    Eigen::MatrixXd V = overlay_mesh.generate_overlay_vertices(overlay_mesh.V_orig, overlay_mesh.vtx_reindex_orig, overlay_mesh.orig);
    auto [F, F_halfedge] = overlay_mesh.generate_overlay_faces();
    std::vector<std::vector<Eigen::Vector3d>> polyline = get_surface_point_positions(overlay_mesh, surface_points);
    auto [nodes, edges, labels] = convert_polyline(polyline);
    auto shuffled_labels = shuffle_map_image(labels);
    //Eigen::MatrixXi F_c = edgeNetwork.tri->getFaceVertexMatrix();

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh("simplified mesh", V, F);
    polyscope::registerCurveNetwork("edges", nodes, edges)
        ->setRadius(0.002);
    polyscope::getCurveNetwork("edges")
        ->addEdgeScalarQuantity("edge parent", labels);
    polyscope::getCurveNetwork("edges")
        ->addEdgeScalarQuantity("shuffled edge parent", shuffled_labels)
        ->setEnabled(true);
#endif
}

void GeodesicFlipper::view(VoronoiOverlayMesh& overlay_mesh)
{
    add_to_viewer(overlay_mesh);
#ifdef ENABLE_VISUALIZATION
    polyscope::show();
#endif
}

void GeodesicFlipper::animate_flip_to_geodesics(VoronoiOverlayMesh& overlay_mesh, int frame_delta)
{
    // add all wedges to priority queue
    wedges_to_process = PriorityQueue();

    // iterate until no more wedges to shorten
    // TODO: replace with local wedge update
    int count = 1;
    while (count > 0)
    {
        count = 0;
        check_all_wedges();

        // iterate
        while (!wedges_to_process.empty())
        {
            if (!process_wedge()) continue;
            if ((count % frame_delta) == 0)
            {
                spdlog::info("saving screencap");
                add_to_viewer(overlay_mesh);
#ifdef ENABLE_VISUALIZATION
                polyscope::screenshot("flip_"+std::to_string(count)+".png");
#endif
            }
            ++count;
            if ((max_flipouts > 0) && (count > max_flipouts)) return;
        }
        spdlog::info("{} wedges shortened", count);
    }

    spdlog::info("final path total length: {}", compute_total_sum(path_length));
}

void GeodesicFlipper::add_adjacent_halfedge(int path_index, int index, bool on_left)
{
    auto& path = paths[path_index];
    int h = path.halfedge[index];
    int e = h2e[h];

    // add to the count of paths adjacent to the edge
    ++edge_path_counts[e];

    // if not ensuring path is simple, need to keep track of all paths on edge in order
    if (!keep_simple)
    {
        // check if new path is to the left of the oriented edge
        if (((on_left) && (e2h[e] == h)) || ((!on_left) && (e2h[e] != h)))
        {
            adjacent_paths[e].push_front({path_index, index});
        }
        // check if the new path is to the right of the oriented edge
        else
        {
            adjacent_paths[h2e[h]].push_back({path_index, index});
        }
    }
}


bool GeodesicFlipper::is_farthest_left(int path_index, int index)
{
    auto& path = paths[path_index];
    int h = path.halfedge[index];
    int e = h2e[h];
    if (keep_simple) return (edge_path_counts[e] == 1);

    int i, j;
    if (e2h[e] == h)
    {
        std::tie(i, j) = adjacent_paths[e].front();
    }
    else
    {
        std::tie(i, j) = adjacent_paths[e].back();
    }

    return ((i == path_index) && (j == index));
}

bool GeodesicFlipper::is_farthest_right(int path_index, int index)
{
    auto& path = paths[path_index];
    int h = path.halfedge[index];
    int e = h2e[h];
    if (keep_simple) return (edge_path_counts[e] == 1);

    int i, j;
    if (e2h[e] == h)
    {
        std::tie(i, j) = adjacent_paths[e].back();
    }
    else
    {
        std::tie(i, j) = adjacent_paths[e].front();
    }
    return ((i == path_index) && (j == index));
}

bool GeodesicFlipper::is_vertex_isolated(int vi)
{
    auto& m = m_o.cmesh();

    // check if any edge is adjacent to the wedge vertex
    int hij = m.out[vi];
    do
    {
        if (edge_path_counts[h2e[hij]] != 0) return false;

        // check if adjacent to boundary edge
        int hji = m.opp[hij];

        if (hji < 0) return false;
        hij = m.n[hji];
    } while (hij != m.out[vi]);
        
    return true;
}

bool GeodesicFlipper::is_left_wedge_empty(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    assert(m.to[hij] == m.to[m.opp[hjk]]);

    // check if any edge is adjacent to the wedge vertex
    int hjl = m.n[hij];
    while (hjl != hjk)
    {
        if (edge_path_counts[h2e[hjl]] != 0) return false;

        // check if adjacent to boundary edge
        int hlj = m.opp[hjl];

        if (hlj < 0) return false;
        hjl = m.n[hlj];
    }
        
    return true;
}

bool GeodesicFlipper::is_right_wedge_empty(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    int hkj = m.opp[hjk];
    int hji = m.opp[hij];

    // check if boundary edge
    if ((hji < 0) || (hkj < 0)) return false;

    return is_left_wedge_empty(hkj, hji);
}

bool GeodesicFlipper::is_left_wedge_simple(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    assert(m.to[hij] == m.to[m.opp[hjk]]);

    // check if any edge is adjacent to the left wedge path vertices 
    int hjl = m.n[hij];
    while (hjl != hjk)
    {
        if (!is_vertex_isolated(m.to[hjl])) return false;
        int hlj = m.opp[hjl];
        if (hlj < 0) return false;
        hjl = m.n[hlj];
    }
        
    return true;
}

bool GeodesicFlipper::is_right_wedge_simple(int hij, int hjk)
{
    auto& m = m_o.cmesh();
    int hkj = m.opp[hjk];
    int hji = m.opp[hij];

    // check if boundary edge
    if ((hji < 0) || (hkj < 0)) return false;

    return is_left_wedge_simple(hkj, hji);
}

bool GeodesicFlipper::is_wedge_simple(int path_index, int wedge_index, bool is_left_joint)
{
    int j = paths[path_index].in[wedge_index];
    int k = paths[path_index].next[j];
    int hij = paths[path_index].halfedge[j];
    int hjk = paths[path_index].halfedge[k];

    if (is_left_joint)
    {
        return is_left_wedge_simple(hij, hjk);
    }
    else 
    {
        return is_right_wedge_simple(hij, hjk);
    }
}

bool GeodesicFlipper::is_left_joint_flexible(int path_index, int wedge_index)
{
    int j = paths[path_index].in[wedge_index];
    int k = paths[path_index].next[j];
    int hij = paths[path_index].halfedge[j];
    int hjk = paths[path_index].halfedge[k];
    if (!is_farthest_left(path_index, j)) return false;
    if (!is_farthest_left(path_index, k)) return false;
    if (!is_left_wedge_empty(hij, hjk)) return false;
    return true;
}

bool GeodesicFlipper::is_right_joint_flexible(int path_index, int wedge_index)
{
    int j = paths[path_index].in[wedge_index];
    int k = paths[path_index].next[j];
    int hij = paths[path_index].halfedge[j];
    int hjk = paths[path_index].halfedge[k];
    if (!is_farthest_right(path_index, j)) return false;
    if (!is_farthest_right(path_index, k)) return false;
    if (!is_right_wedge_empty(hij, hjk)) return false;
    return true;
}

bool GeodesicFlipper::is_joint_flexible(int path_index, int wedge_index, bool is_left_joint)
{
    if (is_left_joint)
    {
        return is_left_joint_flexible(path_index, wedge_index);
    }
    else 
    {
        return is_right_joint_flexible(path_index, wedge_index);
    }
}

void GeodesicFlipper::update_face_angles(int face_index)
{
    auto& m = m_o.cmesh();
    int hi = m.h[face_index];
    int hj = m.n[hi];
    int hk = m.n[hj];
    angles[hi] = 0.0, angles[hj] = 0.0, angles[hk] = 0.0;
    Scalar l12 = m.l[hi], l23 = m.l[hj], l31 = m.l[hk];
    const Scalar t31 = +l12 + l23 - l31, t23 = +l12 - l23 + l31, t12 = -l12 + l23 + l31;
    // valid triangle
    if (t31 > 0 && t23 > 0 && t12 > 0) {
        const Scalar l123 = l12 + l23 + l31;
        const Scalar denom = sqrt(t12 * t23 * t31 * l123);
        angles[hj] = 2 * atan2(t12 * t31, denom); // a1 l23
        angles[hk] = 2 * atan2(t23 * t12, denom); // a2 l31
        angles[hi] = 2 * atan2(t31 * t23, denom); // a3 l12
    } else if (t31 <= 0)
        angles[hk] = PI;
    else if (t23 <= 0)
        angles[hj] = PI;
    else if (t12 <= 0)
        angles[hi] = PI;
    else
        angles[hj] = PI;
}



bool is_valid_mesh(Mesh<Scalar>& m)
{
    std::vector<int> prev = invert_map(m.n);
    if (!are_polygon_mesh_halfedges_valid(m.n, prev)) return errmess("mesh halfedges not valid");
    //if (!are_polygon_mesh_vertices_valid(m.opp, prev, m.to, m.out)) return errmess("mesh vertices not valid");
    if (!are_polygon_mesh_faces_valid(m.n, m.f, m.h)) return errmess("mesh faces not valid");

    return true;
}

bool is_valid_overlay(OverlayMesh<Scalar>& m_o)
{
    m_o.garbage_collection();
    if (!are_polygon_mesh_halfedges_valid(m_o.n, m_o.prev)) return errmess("mesh halfedges not valid");
    //if (!are_polygon_mesh_vertices_valid(m_o.opp, m_o.prev, m_o.to, m_o.out)) return errmess("mesh vertices not valid");
    if (!are_polygon_mesh_faces_valid(m_o.n, m_o.f, m_o.h)) return errmess("mesh faces not valid");

    return true;
}

bool GeodesicFlipper::is_valid()
{
    //if (!is_valid_mesh(m_o)) return false;

    return true;
}


bool GeodesicFlipper::is_simple()
{
    int num_paths = paths.size();
    for (int count : edge_path_counts)
    {
        if (count > 1) return false;
    }

    return true;
}

}
}