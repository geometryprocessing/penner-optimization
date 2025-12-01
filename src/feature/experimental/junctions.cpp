
#include "feature/experimental/junctions.h"
#include "feature/core/common.h"

#include "holonomy/core/viewer.h"
#include "feature/core/component_mesh.h"
#include "util/vf_corners.h"
#include "holonomy/holonomy/cones.h"
#include "util/vector.h"

#include <igl/per_face_normals.h>
#include <queue>
#include <random>
#include <algorithm>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Feature {


void JunctionFeatureFinder::view_junctions(bool show)
{
    const auto& faces = get_faces();
    const auto& vtx_reindex = get_vertex_reindex();
    const auto& vertex_positions = get_vertex_positions();

    // get junction vertices
    std::vector<int> junction_vertices = vector_reindex(vtx_reindex, compute_junctions());
    Eigen::MatrixXd junctions =
        Optimization::generate_subset_vertices(vertex_positions, junction_vertices);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle = "feature mesh";
    polyscope::registerSurfaceMesh(mesh_handle, vertex_positions, faces);
    polyscope::registerPointCloud("junctions", junctions);
    if (show) polyscope::show();
#endif
}

void remove_threshold_junctions(
    JunctionFeatureFinder& feature_finder,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::VectorXi& vertex_component,
    const VectorX& component_function,
    Scalar relative_threshold,
    bool do_remove_all_junctions)
{
    // get boolean mask for junctions
    std::vector<int> junctions = feature_finder.compute_junctions();
    std::vector<bool> is_junction;
    convert_index_vector_to_boolean_array(junctions, m.n_ind_vertices(), is_junction);

    // track relaxed components
    std::vector<bool> is_component_relaxed(component_function.size(), false);

    // prune junctions above relative threshold of maximum function value
    Scalar max_function = component_function.maxCoeff();
    for (int vi = 0; vi < m.n_vertices(); ++vi) {
        // skip vertices in components above threshold
        if (component_function[vertex_component[vi]] < relative_threshold * max_function) continue;

        // skip components already relaxed if not removing all
        if ((do_remove_all_junctions) || (!is_component_relaxed[vertex_component[vi]])) {
            // remove vertex edges if it is a junction
            int vi_root = V_map[vtx_reindex[m.v_rep[vi]]];
            if (!is_junction[vi_root]) continue;
            feature_finder.prune_junction(vi_root);

            // mark component as relaxed
            is_component_relaxed[vertex_component[vi]] = true;
        }
    }

    // remove any remaining closed loops
    feature_finder.prune_closed_loops();
}

void JunctionFeatureFinder::prune_closed_loops()
{
    const auto& m = get_mesh();

    // get features and the degrees of features
    UnionFind feature_unions = compute_feature_components();
    std::vector<std::vector<int>> features = feature_unions.build_sets();
    std::vector<int> feature_degrees = compute_feature_degrees();

    // iterate over features
    for (const auto& feature : features) {
        // check for junctions or darts
        bool is_closed_loop = true;
        for (int hij : feature) {
            int vj = m.v_rep[m.to[hij]];
            if (feature_degrees[vj] != 2) {
                is_closed_loop = false;
                break;
            }
        }

        // remove an edge from the loop feature
        if (is_closed_loop) {
            int hij = feature.front();
            set_feature_edge(hij, false);
        }
    }
}

// linear search for a halfedge vertex that maps to a given VF vertex
int compute_domain_vertex(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    int vertex_index)
{
    int num_vertices = m.n_vertices();
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (vtx_reindex[m.v_rep[vi]] == vertex_index) return vi;
    }

    return -1;
}

// compute distance from a start vertex without crossing cut halfedges
std::vector<Scalar>
find_vertex_weights(const Mesh<Scalar>& m, const std::vector<bool>& is_cut, int v_start)
{
    // Initialize an array to keep track of vertices
    int num_vertices = m.n_vertices();
    Scalar max_cost = vector_max(m.l);
    std::vector<bool> is_processed_vertex(num_vertices, false);
    std::vector<bool> is_boundary_vertex(num_vertices, false);

    // mark all cut vertices as processed
    int num_halfedges = is_cut.size();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (is_cut[hij]) {
            is_boundary_vertex[m.to[hij]] = true;
        }
    }

    // Initialize vertex cost with value above maximum possible weight
    Scalar inf_cost = std::max((num_vertices * max_cost) + 1.0, max_cost + 1.0);
    std::vector<Scalar> vertex_cost(num_vertices, inf_cost);

    // Mark root with 0 (or cost lower than other edges for all negative weights)
    Scalar root_cost = min(max_cost, 0.0);
    vertex_cost[v_start] = root_cost;
    is_processed_vertex[v_start] = false;
    is_boundary_vertex[v_start] = false;

    // Define a custom comparison function for the priority queue
    typedef std::pair<int, Scalar> WeightedVertex;
    auto vertex_compare = [](const WeightedVertex& left, const WeightedVertex& right) {
        return left.second > right.second;
    };

    // Initialize the stack of vertices to process with all vertices
    std::priority_queue<WeightedVertex, std::vector<WeightedVertex>, decltype(vertex_compare)>
        vertices_to_process(vertex_compare);
    for (int vi = 0; vi < num_vertices; ++vi) {
        vertices_to_process.push(std::make_pair(vi, vertex_cost[vi]));
    }

    // Perform Prim or Dijkstra algorithm
    while (!vertices_to_process.empty()) {
        // Get the next vertex to process
        auto [vi, vi_cost] = vertices_to_process.top();
        vertices_to_process.pop();

        // Skip already processed vertices and boundary vertices
        if (is_processed_vertex[vi]) continue;
        if (is_boundary_vertex[vi]) continue;
        is_processed_vertex[vi] = true;

        // Check if vertex has uninitialized cost and skip if so
        if (vertex_cost[vi] == inf_cost) continue;

        // Iterate over the vertex circulator via halfedges
        int h_start = m.out[vi];
        int hij = h_start;
        do {
            // Get the vertex in the one ring at the tip of the halfedge
            int vj = m.to[hij];

            // Get candidate edge cost (either path length or edge weight)
            Scalar candidate_cost = m.l[hij] + vertex_cost[vi];

            // Check if the edge to the tip vertex is the best seen so far
            if ((!is_processed_vertex[vj]) && (vertex_cost[vj] > candidate_cost)) {
                vertex_cost[vj] = candidate_cost;
                vertices_to_process.push(std::make_pair(vj, vertex_cost[vj]));
            }

            // stop at cut edges
            if (is_cut[hij]) break;

            // Progress to the next halfedge in the vertex circulator
            hij = m.n[m.opp[hij]];
        } while (hij != h_start);
        hij = m.opp[m.n[m.n[h_start]]];
        do {
            // Get the vertex in the one ring at the tip of the halfedge
            int vj = m.to[hij];

            // Get candidate edge cost (either path length or edge weight)
            Scalar candidate_cost = m.l[hij] + vertex_cost[vi];

            // Check if the edge to the tip vertex is the best seen so far
            if ((!is_processed_vertex[vj]) && (vertex_cost[vj] > candidate_cost)) {
                vertex_cost[vj] = candidate_cost;
                vertices_to_process.push(std::make_pair(vj, vertex_cost[vj]));
            }

            // stop at cut edges
            if (is_cut[hij]) break;

            // Progress to the next halfedge ccw in the vertex circulator
            hij = m.opp[m.n[m.n[hij]]];
        } while (hij != h_start);
    }

    return vertex_cost;
}

void JunctionFeatureFinder::prune_closest_junction(int vertex_index)
{
    // get index of the vertex in the halfedge mesh
    const auto& m = get_mesh();
    const auto& vtx_reindex = get_vertex_reindex();
    const auto& vertex_positions = get_vertex_positions();
    int root = compute_domain_vertex(m, vtx_reindex, vertex_index);

    // compute distance to input vertex with features cut
    const std::vector<bool>& is_feature_halfedge = get_feature_halfedges();
    std::vector<Scalar> vertex_weights = find_vertex_weights(m, is_feature_halfedge, root);
    Optimization::view_vertex_function(m, vtx_reindex, vertex_positions, vertex_weights);

    // get the junction closest to the root vertex
    std::vector<int> junctions = compute_junctions();
    std::vector<Scalar> junction_weights = vector_compose(vertex_weights, junctions);
    int farthest_cone_index = argmax(junction_weights);
    int closest_cone_index = argmin(junction_weights);
    if (closest_cone_index < 0) {
        spdlog::warn("no junctions remaining");
        return;
    }
    int closest_cone = junctions[closest_cone_index];
    spdlog::info("closest junction cone is {}", closest_cone);

    // remove features with noninfinite distance
    int num_junctions = junctions.size();
    for (int i = 0; i < num_junctions; ++i) {
        if (junction_weights[i] >= junction_weights[farthest_cone_index]) continue;
        prune_junction(junctions[i]);
        break;
    }
}

std::vector<int> JunctionFeatureFinder::compute_junctions() const
{
    // get degrees of all vertices
    std::vector<int> feature_degrees = compute_feature_degrees();

    // get vertices with degree greater than 2
    int num_vertices = feature_degrees.size();
    std::vector<int> vertex_indices = {};
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (feature_degrees[vi] > 2) {
            vertex_indices.push_back(vi);
        }
    }

    return vertex_indices;
}

void JunctionFeatureFinder::prune_junctions()
{
    // save current junctions
    const auto& junctions = compute_junctions();

    // remove all junctions
    for (int vi : junctions) {
        prune_junction(vi);
    }
}

void JunctionFeatureFinder::prune_junction(int vertex_index)
{
    // prune features adjacent to given vertex
    const auto& m = get_mesh();
    std::vector<int> one_ring = generate_vertex_one_ring(m, vertex_index);
    for (int hij : one_ring) {
        set_feature_edge(hij, false);
    }
}

std::tuple<std::vector<int>, std::vector<bool>> JunctionFeatureFinder::compute_feature_forest(
    const std::vector<int>& flood_seeds) const
{
    const auto& m = get_mesh();
    const auto& vtx_reindex_inverse = get_vertex_reindex_inverse();

    // Initialize an array to keep track of vertices
    int num_vertices = m.n_vertices();
    int num_halfedges = m.n_halfedges();
    std::vector<bool> is_processed_vertex(num_vertices, false);
    std::vector<bool> is_seen_halfedge(num_halfedges, false);
    std::vector<int> halfedge_from_vertex(num_vertices, -1);

    // initialize the stack of vertices to process with all vertices
    std::queue<int> vertices_to_process = {};

    // build spanning forest
    for (int vl_root : flood_seeds) {
        int vl = vtx_reindex_inverse[vl_root];

        // get unprocessed vertex vl
        if (is_processed_vertex[vl]) continue;
        vertices_to_process.push(vl);

        // do BFS on component with vl as root
        while (!vertices_to_process.empty()) {
            // Get the next vertex to process
            int vi = vertices_to_process.front();
            vertices_to_process.pop();

            // Skip already processed vertices
            if (is_processed_vertex[vi]) continue;
            is_processed_vertex[vi] = true;

            // Iterate over the vertex circulator via halfedges
            int hij = m.out[vi];
            int hik = hij;
            do {
                // get current edge data
                hik = m.n[m.opp[hik]];
                int hki = m.opp[hik];

                // only process feature halfedges
                if (!is_feature_halfedge(hik)) continue;

                // mark halfedges in current edge as seen
                is_seen_halfedge[hik] = true;
                is_seen_halfedge[hki] = true;

                // check if the tip vertex has been seen yet
                int vk = m.to[hik];
                if (is_processed_vertex[vk]) continue;

                // add halfedge to tree and propagte front to tip vertex
                halfedge_from_vertex[vk] = hik;
                vertices_to_process.push(vk);
            } while (hik != hij);
        }
    }

    return std::make_tuple(halfedge_from_vertex, is_seen_halfedge);
}

} // namespace Feature
} // namespace Penner
