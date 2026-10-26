#include "resolution/experimental/simplification_overlay.h"
#include "metric/viewer.h"
#include "util/union_find.h"
#include "util/map.h"
#include "util/vf_mesh.h"
#include <queue>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner {
namespace Resolution {

IntrinsicSimplificationOverlayMesh::IntrinsicSimplificationOverlayMesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& mesh)
: SimplificationOverlayMesh(V, vtx_reindex, mesh)
, M(mesh)
{
}


// TODO: replace with local version that trims paths above the minimum found length
bool IntrinsicSimplificationOverlayMesh::find_edge(int edge_index)
{
    const auto& m = M.get_mesh();
    int halfedge_index = M.get_e2h()[edge_index];
    int Vi = m.to[m.opp[halfedge_index]];
    int Vj = m.to[halfedge_index];
    int first_vertex = M.vertex_parent(Vi);
    int second_vertex = M.vertex_parent(Vj);
    spdlog::info("tracing path from {} to {}", Vi, Vj);

    // Initialize an array to keep track of vertices
    int num_vertices = out.size();
    std::vector<bool> is_processed_vertex(num_vertices, false);
    std::vector<int> halfedge_from_vertex(num_vertices, -1);

    // Initialize vertex cost with value above maximum possible weight
    //Scalar inf_cost = std::numeric_limits<double>::max();
    Scalar inf_cost = 1e20;
    std::vector<Scalar> vertex_cost(num_vertices, inf_cost);

    // Mark root with 0 (or cost lower than other edges for all negative weights)
    vertex_cost[first_vertex] = 0.;

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

    // Perform Dijkstra algorithm
    bool vertex_found = false;
    while (!vertices_to_process.empty()) {
        // Get the next vertex to process
        auto [vi, vi_cost] = vertices_to_process.top();
        vertices_to_process.pop();

        // Skip already processed vertices
        if (is_processed_vertex[vi]) continue;
        is_processed_vertex[vi] = true;

        // Iterate over the vertex circulator via halfedges
        int h_start = out[vi];
        int hij = h_start;
        do {
            // Get the vertex in the one ring at the tip of the halfedge
            int vj = to[hij];

            // Get candidate edge cost (either path length or edge weight)
            Scalar candidate_cost = halfedge_length(hij) + vertex_cost[vi];

            bool is_invalid = false;
            is_invalid = (is_invalid) || (is_processed_vertex[vj]); // don't process same vertex twice
            is_invalid = (is_invalid) || (simp.edge_type[h2e[hij]] != EdgeType::EF); // don't use traced edges
            is_invalid = (is_invalid) || (vertex_cost[vj] < candidate_cost); // don't use longer paths

            // if processing edge from the first vertex, check for consistent cyclic orientation
            if (to[opp[hij]] == first_vertex)
            {
                is_invalid = (is_invalid) || (!is_start_cyclic_orientation(hij, halfedge_index)); // maintain consistent orientation
            }

            // if processing edge to the second vertex, check for consistent cyclic orientation
            if (vj == second_vertex)
            {
                is_invalid = (is_invalid) || (!is_end_cyclic_orientation(hij, halfedge_index)); // maintain consistent orientation
            }
            // don't intersect any other vertex
            else
            {
                is_invalid = (is_invalid) || (simp.vertex_type[vj] != VertexType::VF); // don't use traced vertices
            }

            // add current candidate to record if it is valid
            if (!is_invalid)
            {
                halfedge_from_vertex[vj] = hij;
                vertex_cost[vj] = candidate_cost;
                vertices_to_process.push(std::make_pair(vj, vertex_cost[vj]));
                if (vj == second_vertex)
                {
                    vertex_found = true;
                }
            }

            // Progress to the next halfedge in the vertex circulator
            hij = opp[prev[hij]];
        } while (hij != h_start);

        if (vertex_found) break;
    }

    if (vertex_cost[second_vertex] == inf_cost)
    {
        //spdlog::error("could not find path from {} to {}", Vi, Vj);

        // visualize costs
        bool show_failure = false;
        if (show_failure)
        {
#ifdef ENABLE_VISUALIZATION
            view();
            std::string mesh_handle = "overlay mesh";
            auto [v_new2old, v_old2new] = get_vertex_reductions();
            std::vector<Scalar> cost = vector_compose(vertex_cost, v_new2old);
            polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("cost", cost);
            polyscope::show();
#endif
        }

        return false;
    }

    // trace back path and mark all edges and vertex parents
    int hjk = halfedge_from_vertex[second_vertex];
    while (true)
    {
        // mark edge parent
        int ejk = h2e[hjk];
        simp.edge_type[ejk] = EdgeType::EE;
        simp.halfedge_parent[hjk] = edge_index; // TODO broken

        // check if at end
        int vj = to[opp[hjk]];
        if (vj == first_vertex)
        {
            break;
        }
        // TODO: barycentric coordinates

        // iterate if still interior
        simp.vertex_type[vj] = VertexType::VE;
        simp.vertex_parent[vj] = edge_index;
        hjk = halfedge_from_vertex[vj];
    }

    return true;
}

void IntrinsicSimplificationOverlayMesh::reset()
{
const auto& m = M.get_mesh();
int num_vertices = out.size();
int num_edges = e2h.size();
int num_faces = f2h.size();
orig = OverlayStructure(num_vertices, num_edges, num_faces);

simp.vertex_parent = std::vector<int>(num_vertices, -1);
simp.halfedge_parent = std::vector<int>(2*num_edges, -1);
simp.face_parent = std::vector<int>(num_faces, -1);
simp.vertex_type = std::vector<VertexType>(num_vertices, VertexType::VF);
simp.edge_type = std::vector<EdgeType>(num_edges, EdgeType::EF);
simp.bary_coords = std::vector<BaryCoords>(num_vertices, {0., 0.});
}

void IntrinsicSimplificationOverlayMesh::mark_vertices()
{
    const auto& m = M.get_mesh();
    int num_vertices = m.n_vertices();
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        int v = M.vertex_parent(vi);
        simp.vertex_type[v] = VertexType::VV;
        simp.vertex_parent[v] = vi;
    }
}
} // namespace Resolution 
} // namespace Penner

/*
bool SimplificationOverlayMesh::collapse_edge(int halfedge_index)
{
    auto& m = M.get_mesh();

    // get first clockwise non collapsed edge 
    int h_cw = m.n[m.opp[halfedge_index]];
    h_cw = m.n[m.opp[h_cw]];

    // get first counterclockwise non collapsed edge 
    int h_ccw = m.opp[m.n[m.n[halfedge_index]]];
    h_ccw = m.opp[m.n[m.n[h_ccw]]];

    // determine the degree of the collapsing vertex
    int degree = M.compute_degree(m.to[m.opp[halfedge_index]]);

    // get edge in the overlay mesh
    int hij = start_halfedge[halfedge_index];
    spdlog::debug("collapsing edge {} = ({}, {})", h2e[hij], hij, opp[hij]);

    // get vertices
    int vi = to[opp[hij]];
    int vj = to[hij];
    spdlog::debug("collapsing a degree {} vertex {} to {}", degree, vi, vj);
    if (!M.collapse_edge(halfedge_index)) return false;

    // determine how many edges are in the clockwise and counterclockwise sides
    int half_edges = (degree % 2) ? (degree + 1) / 2 : (degree / 2) - 1;

    // get the farthest edge in the counterclockwise side
    int H_ccw = h_ccw;
    for (int i = 0; i < half_edges - 2; ++i)
    {
        H_ccw = m.opp[m.n[m.n[H_ccw]]];
    }

    // get the farthest edge in the clockwise side
    int H_cw = h_cw;
    for (int i = 0; i < half_edges - 2; ++i)
    {
        H_cw = m.n[m.opp[H_cw]];
    }

    // if odd degree vertex, mark central vertex as VF and collapsed edge as EF
    if (degree % 2)
    {
        spdlog::debug("collapsing odd degree vertex");
        vertex_type[vi] = VertexType::VF;
        vertex_parent[vi] = m.f[H_ccw];
        bary_coords[vi][0] = 1. / 3.;
        bary_coords[vi][1] = 1. / 3.;
        edge_type[h2e[hij]] = EdgeType::EF;
        assert(m.n[m.opp[H_cw]] == H_ccw); // check rotated start edges
    }
    // if even degree verte, mark central vertex as VE and collapsed edge as EE
    else
    {
        spdlog::debug("collapsing even degree vertex");
        vertex_type[vi] = VertexType::VE;
        vertex_parent[vi] = M.get_h2e()[m.n[m.opp[H_cw]]];
        bary_coords[vi][0] = 0.5;
        edge_type[h2e[hij]] = EdgeType::EE;
        assert(m.n[m.opp[H_cw]] == m.opp[m.n[m.n[H_ccw]]]); // check rotated start edges
    }
    edge_parent[h2e[hij]] = vertex_parent[vi];

    // mark counterclockwise edges as EF
    int hik = hij;
    for (int i = 0; i < half_edges; ++i)
    {
        hik = opp[prev[hik]];
        edge_type[h2e[hik]] = EdgeType::EF;
    }
    // if even degree, mark opposite edge as EE
    if ((degree % 2) == 0)
    {
        hik = opp[prev[hik]];
        edge_type[h2e[hik]] = EdgeType::EE;
    }
    // continue counterclockwise to mark clockwise edges as EF
    for (int i = 0; i < half_edges; ++i)
    {
        hik = opp[prev[hik]];
        edge_type[h2e[hik]] = EdgeType::EF;
    }
    assert(opp[prev[hik]] == hij); // ensure we made a complete loop

    // split counterclockwise edges
    spdlog::debug("splitting counterclockwise edges");
    hik = hij;
    for (int i = 0; i < half_edges; ++i)
    {
        hik = opp[prev[hik]];
        for (int j = i; j < half_edges - 1; ++j)
        {
            split_edge(hik);
        }
    }

    // split clockwise edges
    spdlog::debug("splitting clockwise edges");
    hik = hij;
    for (int i = 0; i < half_edges; ++i)
    {
        hik = n[opp[hik]];
        for (int j = i; j < half_edges - 1; ++j)
        {
            split_edge(hik);
        }
    }

    hik = opp[prev[hij]];
    for (int i = 0; i < half_edges - 1; ++i)
    {
        int Hik = H_ccw;
        int h = hik;
        for (int j = i; j < half_edges - 1; ++j)
        {
            int vk = to[h];
            spdlog::debug("setting coordinates for VE edge {}", vk);
            vertex_type[vk] = VertexType::VE;
            vertex_parent[vk] = M.get_h2e()[Hik];
            bary_coords[vk][0] = (1. * i + 1.) / (1. * half_edges);

            h = n[h];
            Hik = m.n[m.opp[Hik]];
        }
        hik = opp[prev[hik]];
    }

    hik = n[opp[hij]];
    for (int i = 0; i < half_edges - 1; ++i)
    {
        int Hik = H_cw;
        int h = hik;
        for (int j = i; j < half_edges - 1; ++j)
        {
            int vk = to[h];
            spdlog::debug("setting coordinates for VE edge {}", vk);
            vertex_type[vk] = VertexType::VE;
            vertex_parent[vk] = M.get_h2e()[Hik];
            bary_coords[vk][0] = (1. * i + 1.) / (1. * half_edges);

            h = opp[prev[opp[h]]];
            Hik = m.opp[m.n[m.n[Hik]]];
        }
        hik = n[opp[hik]];
    }

    hik = opp[prev[hij]];
    for (int i = 0; i < half_edges; ++i)
    {
        int h = hik;
        for (int j = max(0, i - 1); j < half_edges - 1; ++j)
        {
            split_face(prev[opp[h]], n[opp[h]]); 
            h = n[h];
        }
        hik = opp[prev[hik]];
    }


    hik = n[opp[hij]];
    for (int i = 0; i < half_edges; ++i)
    {
        int h = hik;
        for (int j = max(0, i - 1); j < half_edges - 1; ++j)
        {
            split_face(prev[prev[h]], h); 
            h = opp[prev[opp[h]]];
        }
        hik = n[opp[hik]];
    }

    return true;
}
*/
