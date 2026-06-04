
#include "feature/feature/features.h"
#include "feature/core/common.h"

#include "holonomy/core/viewer.h"
#include "feature/core/component_mesh.h"
#include "feature/core/vf_corners.h"
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


FeatureFinder::FeatureFinder(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
    : m_vertex_positions(V)
    , m_faces(F)
{
    // Convert VF mesh to halfedge with vertex reindexing
    int num_vertices = V.rows();
    bool fix_boundary = false;
    std::vector<Scalar> Th_hat(num_vertices, 0);
    std::vector<int> indep_vtx, dep_vtx, v_rep, bnd_loops, free_cones;
    m_mesh = FV_to_double<Scalar>(
        V,
        F,
        V,
        F,
        Th_hat,
        m_vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);

    // Compute face normals
    igl::per_face_normals(V, F, m_face_normals);

    // Initialize trivial features
    m_is_feature_halfedge = std::vector<bool>(m_mesh.n_halfedges(), false);

    // get inverse of vertex reindexing
    m_vtx_reindex_inverse = invert_map(m_vtx_reindex);
}


std::vector<VertexEdge> FeatureFinder::get_features() const
{
    const auto& m = get_mesh();
    const auto& vtx_reindex = get_vertex_reindex();

    std::vector<VertexEdge> feature_edges;
    int num_halfedges = m.n_halfedges();
    feature_edges.reserve(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        // only process feature edges, and process each edge once
        int hji = m.opp[hij];
        if (hij < hji) continue;
        if (!is_feature_halfedge(hij)) continue;

        // get vertex indices in the original mesh
        int vi = vtx_reindex[m.v_rep[m.to[hji]]];
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];

        // add edge to list
        feature_edges.push_back({vi, vj});
    }

    return feature_edges;
}


void FeatureFinder::reset_features()
{
    m_is_feature_halfedge = std::vector<bool>(m_mesh.n_halfedges(), false);
}


void FeatureFinder::mark_features(const std::vector<VertexEdge>& feature_edges)
{
    // get map from vertex endpoints to halfedges
    const auto& m = get_mesh();
    const auto& vtx_reindex = get_vertex_reindex();
    auto vv2he = generate_VV_to_halfedge_map(m, vtx_reindex);

    // mark all feature edges
    for (auto [vi, vj] : feature_edges) {
        int hij = vv2he.coeff(vi, vj) - 1;
        set_feature_edge(hij, true);
    }
}


void FeatureFinder::mark_dihedral_angle_features(Scalar feature_angle)
{
    int num_halfedges = m_mesh.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process each edge once
        int hji = m_mesh.opp[hij];
        if (hij < hji) continue;

        // Mark edge if dihedral angle is above threshold
        Scalar dihedral_angle = compute_dihedral_angle(hij);
        if (dihedral_angle > feature_angle) {
            set_feature_edge(hij, true);
        }
    }
}


void FeatureFinder::prune_small_components(int min_component_size)
{
    // get faces and vertices of feature cut components
    const auto& m = m_mesh;
    UnionFind component_faces = compute_face_components();
    UnionFind vertices = compute_vertices();

    // Get face components
    std::vector<std::vector<int>> components = component_faces.build_sets();
    std::vector<int> vertex_indices = vertices.index_sets();
    for (const auto& component : components) {
        // build set of vertices in component
        std::set<int> current_vertices = {};
        for (int f : component) {
            int hij = m.h[f];
            int hjk = m.n[hij];
            int hki = m.n[hjk];
            for (int h : {hij, hjk, hki}) {
                current_vertices.insert(vertex_indices[h]);
            }
        }

        // remove component cuts if component vertices too small
        int num_component_vertices = current_vertices.size();
        if (num_component_vertices < min_component_size) {
            for (int f : component) {
                int hij = m.h[f];
                int hjk = m.n[hij];
                int hki = m.n[hjk];
                for (int h : {hij, hjk, hki}) {
                    set_feature_edge(h, false);
                }
            }
        }
    }
}


void FeatureFinder::prune_small_features(int min_feature_size)
{
    // get connected feature structures
    UnionFind feature_unions = compute_feature_components();

    // Count the sizes of the feature sets (and also isolated non-feature halfedges)
    int num_halfedges = m_mesh.n_halfedges();
    int num_features = feature_unions.count_sets();
    std::vector<int> set_index = feature_unions.index_sets();
    Eigen::VectorXi feature_sizes = Eigen::VectorXi::Zero(num_features);
    for (int index : set_index) {
        feature_sizes[index] += 1;
    }

    // Prune features below the threshold (doubled since counting halfedges)
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_feature_halfedge(hij)) continue; // ignore non-features

        int feature_size = feature_sizes[set_index[hij]];
        if (feature_size < 2 * min_feature_size) {
            spdlog::debug("Removing halfedge from feature of size {}", feature_size / 2);
            set_feature_edge(hij, false);
        }
    }
}


void FeatureFinder::prune_cycles()
{
    // set forest as features
    m_is_feature_halfedge = compute_feature_forest_halfedges();
}


void FeatureFinder::prune_greedy()
{
    const auto& m = get_mesh();

    // get features and their degrees
    UnionFind feature_unions = compute_feature_components();
    std::vector<std::vector<int>> features = feature_unions.build_sets();
    std::vector<int> feature_degrees = compute_feature_degrees();

    // iterate while an edge is successfully pruned
    bool edge_pruned = true;
    while (edge_pruned)
    {
        edge_pruned = false;

        // iterate over features
        for (const auto& feature : features) {
            for (int hij : feature) {
                // only consider halfedges that are still features
                if (!is_feature_halfedge(hij)) continue;

                // skip if edge is adjacent to a valence 1 vertex
                int hji = m.opp[hij];
                int vi = m.v_rep[m.to[hji]];
                int vj = m.v_rep[m.to[hij]];
                if ((feature_degrees[vi] < 2) || (feature_degrees[vj] < 2)) continue;

                // remove feature edge
                set_feature_edge(hij, false);

                // update feature degrees
                feature_degrees[vi] -= 1;
                feature_degrees[vj] -= 1;

                // mark successful pruning
                edge_pruned = true;
            }
        }
    }
}

// helper function for greedy pruning
void make_minimal_forest(
    const Mesh<Scalar>& m,
    UnionFind& cut_feature_components,
    std::vector<bool>& is_spanning_halfedge)
{
    int num_halfedges = m.n_halfedges();

    std::vector<int> feature_component = cut_feature_components.index_sets();
    int num_cut_feature_components = cut_feature_components.count_sets();

    // compute initial vertex degrees and feature component sizes
    std::vector<int> vertex_degrees(m.n_vertices(), 0);
    std::vector<int> component_counts(num_cut_feature_components, 0);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (!is_spanning_halfedge[hij]) continue;

        // increment vertex degrees
        int vj = m.v_rep[m.to[hij]];
        ++vertex_degrees[vj];

        // increment number of component features
        int cij = feature_component[hij];
        ++component_counts[cij];
    }

    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (!is_spanning_halfedge[hij]) continue;

        // get edge endpoints and feature componeet
        int hji = m.opp[hij];
        int vi = m.v_rep[m.to[hji]];
        int vj = m.v_rep[m.to[hij]];
        int cij = feature_component[hij];
        int cji = feature_component[hji];

        // remove spanning edge if does not isolate vertex
        if ((vertex_degrees[vi] <= 1) || (vertex_degrees[vj] <= 1)) continue;
        if ((component_counts[cij] <= 1) || (component_counts[cji] <= 1)) continue;

        // unmark spanning edge
        is_spanning_halfedge[hij] = false;
        is_spanning_halfedge[hji] = false;

        // update vertex degrees
        --vertex_degrees[vi];
        --vertex_degrees[vj];

        // update component counts
        --component_counts[cij];
        --component_counts[cji];
    }
}

void FeatureFinder::prune_refined_greedy(UnionFind& cut_feature_components)
{
    make_minimal_forest(get_mesh(), cut_feature_components, get_feature_halfedges());
}

UnionFind FeatureFinder::compute_feature_components() const
{
    // start from disconnected halfedges
    int num_halfedges = m_mesh.n_halfedges();
    UnionFind feature_unions(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (is_feature_halfedge(hij)) {
            // union opposite halfedge
            int hji = m_mesh.opp[hij];
            feature_unions.union_sets(hij, hji);

            // union any features in circulator of the base vertex
            int hik = hij;
            do {
                hik = m_mesh.n[m_mesh.opp[hik]];
                if (is_feature_halfedge(hik)) {
                    feature_unions.union_sets(hij, hik);
                }
            } while (hik != hij);
        }
    }

    return feature_unions;
}


UnionFind FeatureFinder::compute_vertices() const
{
    // Start from triangle soup (indexed by halfedges) and union vertices of edges that are not cut
    int num_halfedges = m_mesh.n_halfedges();
    UnionFind vertex_unions(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process each edge once
        int hji = m_mesh.opp[hij];
        if (hij < hji) continue;

        // Union endpoints of edge if it is not cut
        if (!is_feature_halfedge(hij)) {
            int hjk = m_mesh.n[hij];
            int hki = m_mesh.n[hjk];
            int hil = m_mesh.n[hji];
            int hlj = m_mesh.n[hil];

            // Identify vertices (indexed by opposite halfedge)
            vertex_unions.union_sets(hjk, hlj); // vi
            vertex_unions.union_sets(hki, hil); // vj
        }
    }

    return vertex_unions;
}

UnionFind FeatureFinder::compute_face_components() const
{
    const auto& m = m_mesh;

    // Union component faces across edges
    int num_halfedges = m.n_halfedges();
    int num_faces = m.n_faces();
    UnionFind component_unions(num_faces);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (is_feature_halfedge(hij)) continue; // skip cut edges

        int hji = m.opp[hij];
        component_unions.union_sets(m.f[hij], m.f[hji]);
    }

    return component_unions;
}

UnionFind FeatureFinder::compute_cut_feature_components() const
{
    // get face component labels
    UnionFind component_faces = compute_face_components();
    std::vector<int> face_labels = component_faces.index_sets();

    // Union halfedges that are connected
    const auto& m = get_mesh();
    int num_halfedges = m.n_halfedges();
    UnionFind component_feature_unions(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_feature_halfedge(hij)) continue;

        // Union next feature in circulator
        int hik = hij;
        do {
            // circulate
            hik = m.opp[m.n[hik]];

            // keep circulating if not feature halfedge
            if (!is_feature_halfedge(hik)) continue;

            // union the two features in the same face component
            int hki = m.opp[hik];
            assert(face_labels[m.f[hki]] == face_labels[m.f[hij]]);
            component_feature_unions.union_sets(hij, hki);
            break;
        } while (hik != hij);
    }

    return component_feature_unions;
}


std::vector<int> FeatureFinder::compute_feature_degrees() const
{
    const auto& m = get_mesh();

    // count degrees of features
    int num_halfedges = m.n_halfedges();
    int num_vertices = m.n_ind_vertices();
    std::vector<int> feature_degrees(num_vertices, 0);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_feature_halfedge(hij)) continue; // skip interior edges

        // increase degree at the tip of the halfedge
        int vj = m.v_rep[m.to[hij]];
        feature_degrees[vj] += 1;
    }

    return feature_degrees;
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::VectorXi, Eigen::MatrixXi>
FeatureFinder::generate_feature_cut_mesh() const
{
    UnionFind vertex_unions = compute_vertices();

    // Build cut vertex set from unions
    int num_halfedges = m_mesh.n_halfedges();
    int num_faces = m_mesh.n_faces();
    std::vector<std::vector<int>> vertices = vertex_unions.build_sets();
    int num_vertex_unions = vertices.size();

    // build cut vertex positions and map to the original mesh vertices
    Eigen::MatrixXd V_cut(num_vertex_unions, 3);
    Eigen::VectorXi V_map(num_vertex_unions);
    std::vector<int> h2v(num_halfedges, -1); // utility map from haledges to opposite vertex
    for (int i = 0; i < num_vertex_unions; ++i)
    {
        // check that no trivial vertices (should not happen)
        if (vertices[i].empty())
        {
            spdlog::error("Trivial vertex built");
            continue;
        }

        // get halfedge pointing to vertex
        int hij = vertices[i][0]; //indexed by opposite halfedge
        int hjk = m_mesh.n[hij];

        // get vertex in the VF mesh
        int vk = m_vtx_reindex[m_mesh.v_rep[m_mesh.to[hjk]]];

        // map cut vertex to glued vertex and get position
        V_map[i] = vk;
        V_cut.row(i) = m_vertex_positions.row(vk);

        // map all vertex halfedges to the cut vertex
        for (const auto& h : vertices[i])
        {
            h2v[h] = i;
        }
    }

    // lambda to find halfedge of face opposite F(f, 0)
    auto find_face_halfedge = [&](int f) {
        int hij = m_mesh.h[f];

        // circulate over face until vertex found
        // WARNING: for speed, no safety shutoff for failure
        while (V_map[h2v[hij]] != m_faces(f, 0)) {
            hij = m_mesh.n[hij];
        }

        return hij;
    };

    // Build cut faces from unions and mark feature edges
    Eigen::MatrixXi F_cut(num_faces, 3);
    Eigen::MatrixXi F_is_cut(num_faces, 3);
    for (int f = 0; f < num_faces; ++f) {
        // get halfedge opposite F(f, 0) for index consistency
        int hij = find_face_halfedge(f);

        // iterate over three edges of triangle
        for (int i = 0; i < 3; ++i) {
            F_cut(f, i) = h2v[hij];
            F_is_cut(f, i) = is_feature_halfedge(hij) ? 1 : 0;
            hij = m_mesh.n[hij];
        }
    }

    return std::make_tuple(V_cut, F_cut, V_map, F_is_cut);
}


void FeatureFinder::view_features(bool show)
{
    // view the feature edges as a graph
    Optimization::view_primal_graph(
        m_vertex_positions,
        m_mesh,
        m_vtx_reindex,
        m_is_feature_halfedge,
        "features",
        show);

    // view the feature forest that would be produced by pruning cycles as a graph
    std::vector<bool> is_in_feature_forest = compute_feature_forest_halfedges();
    Optimization::view_primal_graph(
        m_vertex_positions,
        m_mesh,
        m_vtx_reindex,
        is_in_feature_forest,
        "forest",
        show);
}


void FeatureFinder::view_parametric_features(
    const Eigen::MatrixXd& uv,
    const std::vector<int>& vtx_reindex) const
{
    // add features
    Optimization::view_primal_graph(uv, m_mesh, vtx_reindex, m_is_feature_halfedge);
}

void FeatureFinder::set_feature_edge(int halfedge_index, bool is_feature)
{
    // get other edge halfedge
    const auto& m = get_mesh();
    int hij = halfedge_index;
    int hji = m.opp[hij];

    // set feature edge
    m_is_feature_halfedge[hij] = is_feature;
    m_is_feature_halfedge[hji] = is_feature;
}

std::vector<bool> FeatureFinder::compute_feature_forest_halfedges() const
{
    const auto& m = get_mesh();
    int num_halfedges = m.n_halfedges();

    // get spanning tree (either minimal or arbitrary)
    std::vector<int> halfedge_from_vertex = compute_feature_forest();

    // build mask for feature forest edges
    std::vector<bool> is_in_feature_forest(num_halfedges, false);
    for (int hij : halfedge_from_vertex) {
        if (hij < 0) continue;
        int hji = m.opp[hij];
        is_in_feature_forest[hij] = true;
        is_in_feature_forest[hji] = true;
    }

    return is_in_feature_forest;
}


// Compute signed dihedral angle for the given halfedge
Scalar FeatureFinder::compute_signed_dihedral_angle(int halfedge_index) const
{
    // Get adjacent face indices
    int hij = halfedge_index;
    int hji = m_mesh.opp[hij];
    int f0 = m_mesh.f[hij];
    int f1 = m_mesh.f[hji];

    // Compute dot product of face normals
    Eigen::RowVector3d N0 = m_face_normals.row(f0);
    Eigen::RowVector3d N1 = m_face_normals.row(f1);
    Scalar radian_angle = acos(std::min<Scalar>(std::max<Scalar>(N0.dot(N1), -1.0), 1.0));
    return radian_angle * (180. / M_PI);
}


// Compute unsigned dihedral angle for the given halfedge
Scalar FeatureFinder::compute_dihedral_angle(int halfedge_index) const
{
    return abs(compute_signed_dihedral_angle(halfedge_index));
}


std::vector<Scalar> FeatureFinder::compute_dihedral_weights() const
{
    const auto& m = get_mesh();
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> weights(num_halfedges, 0.);

    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_feature_halfedge(hij)) continue;

        // Only process each edge once
        int hji = m.opp[hij];
        if (hij < hji) continue;

        // Weight by dihedral angle (closer to flat is higher weight)
        weights[hij] = std::max<Scalar>(180. - compute_dihedral_angle(hij), 0.);
        weights[hji] = weights[hij];
    }

    return weights;
}
    
std::vector<Scalar> FeatureFinder::compute_interior_biased_weights() const
{
    const auto& m = get_mesh();
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> weights(num_halfedges, 0.);
    std::vector<int> feature_degrees = compute_feature_degrees();

    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_feature_halfedge(hij)) continue;

        // Only process each edge once
        int hji = m_mesh.opp[hij];
        if (hij < hji) continue;
        int vi = m.v_rep[m.to[hji]];
        int vj = m.v_rep[m.to[hij]];

        // Weight by dihedral angle with preference for interior edges
        weights[hij] = std::max<Scalar>(180. - compute_dihedral_angle(hij), 0.);
        if ((feature_degrees[vi] != 2) || (feature_degrees[vj] != 2))
        {
            spdlog::trace("Increasing weight for {}", hij);
            weights[hij] += 2. * M_PI;
        }
        weights[hji] = weights[hij];
    }

    return weights;
}

// random forest weights
std::vector<Scalar> FeatureFinder::compute_random_weights() const
{
    const auto& m = get_mesh();
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> weights(num_halfedges, 0.);

    // initialize randomization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(0.0, 1.0);

    // generate weights
    std::generate(weights.begin(), weights.end(), [&]() {return distrib(gen);} );

    return weights;
}


std::vector<Scalar> FeatureFinder::compute_forest_weights() const
{
    return compute_dihedral_weights();
    // TOOD: other weighting options
}

std::vector<int> FeatureFinder::compute_feature_forest() const
{
    const auto& m = get_mesh();

    // Initialize an array to keep track of vertices
    int num_vertices = m.n_vertices();
    std::vector<bool> is_processed_vertex(num_vertices, false);
    std::vector<int> halfedge_from_vertex(num_vertices, -1);

    // get edge weights
    std::vector<Scalar> weights = compute_forest_weights();

    // Initialize vertex cost with value above maximum possible weight
    Scalar max_cost = vector_max(weights);
    Scalar inf_cost = max_cost + 1.0;
    std::vector<Scalar> vertex_cost(num_vertices, inf_cost);

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

    // build spanning forest
    while (!vertices_to_process.empty()) {
        // Get the next vertex to process
        auto [vi, vi_cost] = vertices_to_process.top();
        vertices_to_process.pop();

        // Skip already processed vertices
        if (is_processed_vertex[vi]) continue;

        // mark vertex as processed now
        is_processed_vertex[vi] = true;

        // Check if vertex has uninitialized cost and give it the root cost of 0 if so
        // NOTE: not actually necessary due to processed vertex flag
        if (vertex_cost[vi] == inf_cost) {
            vertex_cost[vi] = 0.;
        }

        // Iterate over the vertex circulator via halfedges
        int hij = m.out[vi];
        int hik = hij;
        do {
            // get current edge data
            hik = m.n[m.opp[hik]];

            // only process feature halfedges
            if (!is_feature_halfedge(hik)) continue;

            // check if the tip vertex has been seen yet
            int vk = m.to[hik];
            if (is_processed_vertex[vk]) continue;

            // add halfedge to tree and propagte front to tip vertex
            if (vertex_cost[vk] >= weights[hij]) {
                halfedge_from_vertex[vk] = hik;
                vertex_cost[vk] = weights[hij];
                vertices_to_process.push(std::make_pair(vk, vertex_cost[vk]));
            }
        } while (hik != hij);
    }

    return halfedge_from_vertex;
}

} // namespace Feature
} // namespace Penner
