#include "resolution/hierarchy/voronoi.h"
#include "resolution/core/layout.h"
#include "metric/viewer.h"
#include "metric/constraint.h"
#include "parametrization/triangulation.h"
#include "util/union_find.h"
#include "util/map.h"
#include "util/vf_mesh.h"
#include "holonomy/core/viewer.h"
#include "holonomy/core/dual_lengths.h"
#include "field/frame_field.h"
#include "parametrization/layout.h"
#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "metric/area.h"
#include "conformal_ideal_delaunay/Layout.hh"

#include "resolution/core/geodesics.h"
#include "resolution/core/geodesic_flipper.h"

#include <igl/remove_unreferenced.h>
#include <igl/local_basis.h>
#include <stack>

#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "main_helper.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner
{
namespace Resolution
{


VoronoiOverlayMesh::VoronoiOverlayMesh(
    const Eigen::MatrixXd &V,
    const std::vector<int> &vtx_reindex,
    const Mesh<Scalar> &mesh)
    : SimplificationOverlayMesh(V, vtx_reindex, mesh), voronoi(out.size(), e2h.size(), f2h.size()), path_tracer(*this)
{
}

// -----------------
// Atomic Operations
// -----------------

int VoronoiOverlayMesh::create_face()
{
    int f = SimplificationOverlayMesh::create_face();
    voronoi.face_region.push_back(-1);
    return f;
}

int VoronoiOverlayMesh::create_edge()
{
    int e = SimplificationOverlayMesh::create_edge();

    voronoi.edge_region.push_back(-1);
    voronoi.edge_type.push_back(EdgeType::EF);

    for (int i : {0, 1})
    {
        //voronoi.dual_halfedge_next.push_back(-1);
        voronoi.dual_halfedge.push_back(-1);
    }

    return e;
}

int VoronoiOverlayMesh::create_vertex()
{
    int v = SimplificationOverlayMesh::create_vertex();

    voronoi.distance.push_back(infinity);
    voronoi.vertex_region.push_back(-1);
    voronoi.vertex_type.push_back(VertexType::VF);

    return v;
}

int VoronoiOverlayMesh::split_edge(int halfedge_index)
{
    int edge_index = h2e[halfedge_index];
    int hij = SimplificationOverlayMesh::split_edge(halfedge_index);
    split_edge_voronoi(halfedge_index, edge_index, voronoi);

    return hij;
}

int VoronoiOverlayMesh::split_face(int h, int g)
{
    int face_index = face[h];
    int h_split = SimplificationOverlayMesh::split_face(h, g);
    int e = h2e[h_split];
    split_face_voronoi(e, face_index, voronoi);

    return h_split;
}

int VoronoiOverlayMesh::add_center_to_face(int face_index)
{
    int vi = SimplificationOverlayMesh::add_center_to_face(face_index);
    add_center_to_face_voronoi(vi, face_index, voronoi);

    return vi;
}

bool VoronoiOverlayMesh::is_regular_face(int face_index) const
{
    if (is_face_deleted(face_index))
        return false;
    int hij = f2h[face_index];
    int hkl = hij;
    int num_embedded_edges = 0;
    do
    {
        if ((simp.edge_type[h2e[hkl]] == EdgeType::EE) || (voronoi.edge_type[h2e[hkl]] == EdgeType::EE))
        {
            ++num_embedded_edges;
        }
        hkl = n[hkl];
    } while (hij != hkl);

    return (num_embedded_edges < 2);
}

bool VoronoiOverlayMesh::is_regular_edge(int edge_index) const
{
    if (is_edge_deleted(edge_index))
        return false;
    if (simp.edge_type[edge_index] == EdgeType::EE)
        return true;
    if (voronoi.edge_type[edge_index] == EdgeType::EE)
        return true;
    int hij = e2h[edge_index];
    int hji = opp[hij];
    if ((simp.vertex_type[to[hij]] == VertexType::VF) && (voronoi.vertex_type[to[hij]] == VertexType::VF))
        return true;
    if ((simp.vertex_type[to[hji]] == VertexType::VF) && (voronoi.vertex_type[to[hji]] == VertexType::VF))
        return true;
    return false;
}

// ---------------
// Voronoi Regions
// ---------------

// TODO: replace with local version that trims paths above the minimum found length
bool VoronoiOverlayMesh::grow_voronoi_regions(const std::vector<int>& seeds)
{
    // Initialize an array to keep track of vertices
    int num_vertices = out.size();
    //std::vector<int> halfedge_from_vertex(num_vertices, -1);

    // Mark seeds with 0 distance
    int num_dual_vertices = voronoi.dual_vertices.size();
    for (int vi : seeds)
    {
        voronoi.distance[vi] = 0.;
        voronoi.vertex_region[vi] = voronoi.dual_vertices.size();;
        voronoi.vertex_type[vi] = VertexType::VF;
        voronoi.dual_vertices.push_back(vi);
    }

    // Define a custom comparison function for the priority queue
    typedef std::pair<int, Scalar> WeightedVertex;
    auto vertex_compare = [](const WeightedVertex &left, const WeightedVertex &right)
    {
        return left.second > right.second;
    };

    // Initialize the stack of vertices to process with all vertices
    std::priority_queue<WeightedVertex, std::vector<WeightedVertex>, decltype(vertex_compare)>
        vertices_to_process(vertex_compare);
    // for (int vi = 0; vi < num_vertices; ++vi) {
    for (int vi : seeds)
    {
        vertices_to_process.push(std::make_pair(vi, voronoi.distance[vi]));
    }

    // Perform Dijkstra algorithm
    while (!vertices_to_process.empty())
    {
        // Get the next vertex to process
        auto [vi, vi_cost] = vertices_to_process.top();
        vertices_to_process.pop();

        // Skip vertices with lower cost
        if (vi_cost > voronoi.distance[vi])
            continue;

        // Iterate over the vertex circulator via halfedges
        int h_start = out[vi];
        int hij = h_start;
        do
        {
            // Get the vertex in the one ring at the tip of the halfedge
            int vj = to[hij];

            // Get candidate edge cost (either path length or edge weight)
            Scalar candidate_cost = halfedge_length(hij) + voronoi.distance[vi];

            // add current candidate to record if it is valid
            if (voronoi.distance[vj] > candidate_cost)
            {
                int hji = opp[hij];
                int hjk = hji;
                bool is_base_region = true;
                int num_regions = 0;
                do
                {
                    hjk = opp[prev[hjk]];
                    int vk = to[hjk];

                    if ((voronoi.vertex_region[vk] == voronoi.vertex_region[vi]) && (!is_base_region))
                    {
                        num_regions++;
                    }

                    is_base_region = (voronoi.vertex_region[vk] == voronoi.vertex_region[vi]);
                } while (hjk != hji);

                // if vertex would destroy disk topology or exceed max distance, add as new seed 
                if ((num_regions > 1) || (candidate_cost > voronoi.max_distance))
                {
                    voronoi.distance[vj] = 0.;
                    voronoi.vertex_region[vj] = voronoi.dual_vertices.size();
                    voronoi.vertex_type[vj] = VertexType::VF;
                    voronoi.dual_vertices.push_back(vj);
                    vertices_to_process.push(std::make_pair(vj, voronoi.distance[vj]));
                }
                // add vertex otherwise
                else
                {
                    //halfedge_from_vertex[vj] = hij;
                    voronoi.distance[vj] = candidate_cost;
                    voronoi.vertex_region[vj] = voronoi.vertex_region[vi];
                    vertices_to_process.push(std::make_pair(vj, voronoi.distance[vj]));
                }
            }

            // Progress to the next halfedge in the vertex circulator
            hij = opp[prev[hij]];
        } while (hij != h_start);
    }

    return true;
}

void VoronoiOverlayMesh::set_max_voronoi_distance(Scalar max_distance)
{
    spdlog::info("setting max voronoi distance {}", max_distance);

    // if maximally distant vertex too far away from nearest neighbor, add as seed
    int num_vertices = out.size();
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (voronoi.distance[vi] < max_distance) continue;

        // grow voronoi regions with new seed
        grow_voronoi_regions({vi});
    }
}

void VoronoiOverlayMesh::set_constained_max_voronoi_distance(
    Scalar max_distance,
    const std::vector<bool>& is_allowed)
{
    spdlog::info("setting max voronoi distance {}", max_distance);

    // sort vertices by distance
    int num_vertices = out.size();
    std::vector<int> idx = arange(num_vertices);
    std::sort(idx.begin(), idx.end(),
            [&](int i, int j) {
                return (voronoi.distance[i] > voronoi.distance[j]);
            });
    spdlog::info("maximum initial distance is {}", voronoi.distance[idx[0]]);

    // if maximally distant vertex too far away from nearest neighbor, add as seed
    for (int i = 0; i < num_vertices; ++i)
    {
        int vi = idx[i];
        if (!is_allowed[vi]) continue;
        if (voronoi.distance[vi] < max_distance) continue;

        // grow voronoi regions with new seed
        grow_voronoi_regions({vi});
    }
}

void VoronoiOverlayMesh::set_weighted_max_voronoi_distance(
    Scalar max_distance,
    const VectorX& weights)
{
    spdlog::info("setting max voronoi distance {}", max_distance);

    // sort vertices by distance
    int num_vertices = out.size();
    std::vector<int> idx = arange(num_vertices);
    std::sort(idx.begin(), idx.end(),
            [&](int i, int j) {
                return (voronoi.distance[i] > voronoi.distance[j]);
            });
    spdlog::info("maximum initial distance is {}", voronoi.distance[idx[0]]);

    // if maximally distant vertex too far away from nearest neighbor, add as seed
    for (int i = 0; i < num_vertices; ++i)
    {
        int vi = idx[i];
        if ((weights[vi] * voronoi.distance[vi]) < max_distance) continue;

        // grow voronoi regions with new seed
        grow_voronoi_regions({vi});
    }
}

void VoronoiOverlayMesh::make_voronoi_uniform(Scalar dual_length_ratio)
{
    int num_faces = f2h.size();
    std::vector<int> dual_faces = generate_dual_faces();
    std::vector<bool> is_dual_face;
    convert_index_vector_to_boolean_array(dual_faces, num_faces, is_dual_face);
    int num_simp_faces = dual_faces.size();
    int num_simp_halfedges = 3 * num_simp_faces;

    // need to temporarily store a lot of connectivity data to compute geometric quantities
    // TODO: make this a standalone utility function
    int num_simp_vertices = voronoi.dual_vertices.size();
    voronoi.mesh.n.resize(num_simp_halfedges);
    voronoi.mesh.opp.resize(num_simp_halfedges);
    voronoi.mesh.prev.resize(num_simp_halfedges);
    voronoi.mesh.to.resize(num_simp_halfedges);
    voronoi.mesh.f.resize(num_simp_halfedges);
    voronoi.mesh.h.resize(num_simp_faces);
    voronoi.mesh.out.resize(num_simp_vertices);
    std::vector<int> from(num_simp_halfedges);
    voronoi.dual_halfedge_starts.resize(num_simp_halfedges);

    std::vector<std::vector<int>> dual_edges(num_simp_halfedges, std::vector<int>());
    int F = 0;
    typedef Eigen::Triplet<int> T;
    std::vector<T> triplet_list;
    for (int f = 0; f < num_faces; ++f)
    {
        if (!is_dual_face[f])
            continue;

        int hij = f2h[f];
        int I = 0;
        do
        {
            int hji = opp[hij];
            int vi = to[hji];
            int vj = to[hij];
            int Vi = voronoi.vertex_region[vi];
            int Vj = voronoi.vertex_region[vj];

            // mark edge
            triplet_list.push_back(T(Vi, Vj, 1));

            int H = 3 * F + I;
            int Hn = 3 * F + ((I + 1) % 3);
            assert(F < num_simp_faces);
            assert(H < num_simp_halfedges);
            voronoi.mesh.n[H] = Hn;
            voronoi.mesh.prev[Hn] = H;
            voronoi.mesh.f[H] = F;
            voronoi.mesh.h[F] = H;
            from[H] = Vi; // store from to avoid needing opp
            voronoi.mesh.to[H] = Vj;
            voronoi.mesh.out[Vj] = Hn;
            int h = hji;
            dual_edges[H].push_back(hji);
            while (!is_dual_face[face[h]])
            {
                // iterate to next crossing edge
                h = n[h];
                if (voronoi.vertex_region[to[h]] == voronoi.vertex_region[to[opp[h]]]) h = n[h];
                h = opp[h];

                // add dual edge
                dual_edges[H].push_back(h);
            }
            I = (I + 1) % 3;
            hij = n[hij];
        } while (hij != f2h[f]);
        F++;
    }
    
    // get halfedge lengths
    std::vector<Scalar> dual_lengths = Holonomy::compute_dual_edge_lengths(orig.m);

    // sum up lengths around perimeter
    int num_dual_faces = voronoi.dual_vertices.size();
    std::vector<Scalar> lengths(num_simp_halfedges, 0.);
    std::vector<Scalar> perimeters(num_simp_vertices, 0.);
    for (int Hij = 0; Hij < num_simp_halfedges; ++Hij)
    {
        int Vj = voronoi.mesh.to[Hij];
        for (int h : dual_edges[Hij])
        {
            lengths[Hij] += dual_lengths[h];
            perimeters[Vj] += dual_lengths[h];
        }
    }

    // check for short vertex joints relative to perimeter
    std::vector<bool> is_new_seed(out.size(), false);
    for (int Hij = 0; Hij < num_simp_halfedges; ++Hij)
    {
        int Vj = voronoi.mesh.to[Hij];
        int Hjk = voronoi.mesh.n[Hij];
        if ((lengths[Hij] + lengths[Hjk]) < (dual_length_ratio * perimeters[Vj]))
        {
            int hkj = dual_edges[Hjk].front();
            int vj = to[hkj];
            assert(voronoi.vertex_region[vj] == Vj);

            // only add if not already root
            if (voronoi.distance[vj] != 0.) is_new_seed[vj] = true;
        }
    }

    // Build adjacency matrix from triplets
    int num_entries = triplet_list.size();
    int num_dual_vertices = voronoi.dual_vertices.size();
    Eigen::SparseMatrix<int> adjacency(num_dual_vertices, num_dual_vertices);
    adjacency.reserve(num_entries);
    adjacency.setFromTriplets(triplet_list.begin(), triplet_list.end());

    // check for repeated edges and add a vertex to remove
    int num_vertices = out.size();
    std::vector<bool> is_dual_vertex;
    convert_index_vector_to_boolean_array(voronoi.dual_vertices, num_vertices, is_dual_vertex);
    for (int Hij = 0; Hij < num_simp_halfedges; ++Hij)
    {
        int Vi = from[Hij];
        int Vj = voronoi.mesh.to[Hij];
        if (Vi < Vj) continue; // only process each edge once
        if (adjacency.coeffRef(Vi, Vj) == 1) continue; // skip simple edges
        spdlog::debug("Adding seed for repeated edge {} = {}, {}", Hij, Vi, Vj);

        // add new seed near dual edge
        bool is_seed_found = false;
        for (int h : dual_edges[Hij])
        {
            // try adding seed at tip
            int vi = to[h];
            if (!is_dual_vertex[vi])
            {
                is_new_seed[vi] = true;
                is_seed_found = true;
                break;
            }

            // try vertex at base if fails
            int vj = to[opp[h]];
            if (!is_dual_vertex[vj])
            {
                is_new_seed[vj] = true;
                is_seed_found = true;
                break;
            }
        }
    }

    // get new seeds
    std::vector<int> new_seeds;
    convert_boolean_array_to_index_vector(is_new_seed, new_seeds);

    if (!new_seeds.empty())
    {
        spdlog::info("Adding {} seeds", new_seeds.size());
        // grow voronoi region for new seeds
        grow_voronoi_regions(new_seeds);

        // recurse in case not uniform
        make_voronoi_uniform(dual_length_ratio);
    }
    else
    {
        // seed insertion can introduce non-disk regions; regrow voronoi regions from seeds to remove
        std::vector<int> seeds = voronoi.dual_vertices;

        // ensure have sufficiently many seeds
        if (seeds.size() < 4) seeds = arange(num_vertices);

        voronoi.reset();
        grow_voronoi_regions(seeds);
    }
}


void VoronoiOverlayMesh::grow_uniform_voronoi_regions(
    std::vector<int>& seeds,
    Scalar ratio,
    Scalar normal_scale,
    Scalar dual_length_ratio)
{
    if (ratio > 0.)
    {
        std::vector<Scalar> l = orig.m.l;
        if (normal_scale > 0.) orig.m.l = compute_curvature_adaptive_lengths(V_orig, F_orig, orig.m, vtx_reindex_orig, normal_scale);

        // compute intrinsic mesh area
        VectorX mesh_areas = areas(orig.m);
        Scalar mesh_area = mesh_areas.sum() / 3.; // divide by 3 as per halfedge area

        // compute target edge length to achieve density defined by ratio
        int num_vertices = orig.m.n_ind_vertices();
        Scalar target_length = sqrt((2. * mesh_area) / (sqrt(3.) * ratio * num_vertices));
        Scalar avg_length = Eigen::Map<Eigen::VectorXd>(orig.m.l.data(), orig.m.l.size()).mean();
        spdlog::info("average length of original mesh is {}", avg_length);
        spdlog::info("target length is {}", target_length);

        // adaptively sample vertices by adding max edge length
        grow_voronoi_regions(seeds);
        set_max_voronoi_distance(target_length);

        // set lengths back to uniform and regrow regions
        if (normal_scale > 0.) orig.m.l = l; // reset lengths if changed
        seeds = voronoi.dual_vertices;
        voronoi.reset();
    }

    // grow region from new seeds and make uniform
    grow_voronoi_regions(seeds);
    make_voronoi_uniform(dual_length_ratio);
}

// ----------------
// Voronoi Pipeline
// ----------------

void VoronoiOverlayMesh::build_voronoi_connectivity()
{
    split_voronoi_edges();
    split_voronoi_faces();
    find_voronoi_dual_edges();
    propagate_voronoi_vertices_to_faces();
}

void VoronoiOverlayMesh::split_voronoi_edges()
{
    int num_edges = e2h.size();
    //typedef Eigen::Triplet<int> T;
    //std::vector<T> triplet_list;
    for (int e = 0; e < num_edges; ++e)
    {
        if (is_edge_deleted(e))
            continue;

        // determine whether edge separates two distinct voronoi regions
        int hij = e2h[e];
        int vj = to[hij];
        int vi = to[opp[hij]];
        if (voronoi.vertex_type[vi] != VertexType::VF)
            continue;
        if (voronoi.vertex_type[vj] != VertexType::VF)
            continue;
        if (voronoi.vertex_region[vi] == voronoi.vertex_region[vj])
            continue;

        // split edge, and label inserted vertex as an edge vertex
        int hik = split_edge(hij);
        int vk = to[hik];
        voronoi.vertex_type[vk] = VertexType::VE;

        // set distance to vertex as the maximum distance to a neighboring region
        voronoi.distance[vk] = max(voronoi.distance[vi], voronoi.distance[vj]);

        //int Vi = voronoi.vertex_region[vi];
        //int Vj = voronoi.vertex_region[vj];
        //triplet_list.push_back(T(min(Vi, Vj), max(Vi, Vj), -vk - 1));
    }

#ifdef ENABLE_VISUALIZATION
    if (false)
    {
        view();
        std::string mesh_handle = "overlay mesh";
        auto [v_new2old, v_old2new] = get_vertex_reductions();
        std::vector<int> vertex_region = vector_compose(voronoi.vertex_region, v_new2old);
        polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("vertex region", vertex_region);
        polyscope::show();
    }
#endif
}


std::vector<int> VoronoiOverlayMesh::generate_dual_faces() const
{
    int num_faces = f2h.size();
    std::vector<int> dual_faces = {};
    dual_faces.reserve(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f))
            continue;
        int hij = f2h[f];
        int vi = to[hij];
        int vj = to[n[hij]];
        int vk = to[n[n[hij]]];

        // check if all vertices are distinct regions
        if (voronoi.vertex_region[vi] == voronoi.vertex_region[vj]) continue;
        if (voronoi.vertex_region[vj] == voronoi.vertex_region[vk]) continue;
        if (voronoi.vertex_region[vk] == voronoi.vertex_region[vi]) continue;

        // mark dual vertex
        dual_faces.push_back(f);
    }

    return dual_faces;
}

void VoronoiOverlayMesh::split_voronoi_faces()
{
    // count the number of faces where three voronoi regions meet
    int num_faces = f2h.size();
    int num_simp_faces = 0;
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f))
            continue;

        // for a split triangle mesh, the number of edges is 6 iff three voronoi regions meet here
        if (face_degree(f) == 6)
            num_simp_faces++;
    }

    // initialize halfedge data structure array sizes
    int num_simp_halfedges = 3 * num_simp_faces;
    int num_simp_vertices = voronoi.dual_vertices.size();
    voronoi.mesh.n.resize(num_simp_halfedges);
    voronoi.mesh.opp.resize(num_simp_halfedges);
    voronoi.mesh.prev.resize(num_simp_halfedges);
    voronoi.mesh.to.resize(num_simp_halfedges);
    voronoi.mesh.f.resize(num_simp_halfedges);
    voronoi.mesh.h.resize(num_simp_faces);
    voronoi.mesh.out.resize(num_simp_vertices);
    voronoi.dual_halfedge_starts.resize(num_simp_halfedges);
    std::vector<int> circ(num_simp_halfedges);

    int F = 0;
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f))
            continue;
        int degree = face_degree(f);

        // case: unrefined face contained in a voronoi region, so nothing to do
        if (degree <= 3)
        {
            continue; // face not refined
        }
        // case: dual face where three voronoi regions meet
        else if (degree == 6)
        {
            // add center to face
            spdlog::debug("adding center to face {}", f);
            int vi = add_center_to_face(f);
            voronoi.vertex_type[vi] = VertexType::VV;
            voronoi.distance[vi] = 0.;
            int hij = out[vi];
            int I = 0;
            do
            {
                int vj = to[hij];

                // case: halfedge is part of the voronoi region boundary
                if (voronoi.vertex_type[vj] != VertexType::VF)
                {
                    // mark adjacent edges in the voronoi boundary and mark halfedge start
                    int H = 3 * F + I;
                    voronoi.edge_type[h2e[hij]] = EdgeType::EE;
                    voronoi.dual_halfedge_starts[H] = hij;
                }
                // case: halfedge is interior to a voronoi vertex region
                else
                {
                    // set distance to vertex as maximum distance of adjacent regions
                    voronoi.distance[vi] = max(voronoi.distance[vi], voronoi.distance[vj]);

                    // set connectivity data in this region (except opp relations)
                    int Vj = voronoi.vertex_region[vj];
                    int H = 3 * F + I;
                    int Hn = 3 * F + ((I + 1) % 3);
                    assert(F < num_simp_faces);
                    assert(H < num_simp_halfedges);
                    voronoi.mesh.n[H] = Hn;
                    voronoi.mesh.prev[Hn] = H;
                    voronoi.mesh.f[H] = F;
                    voronoi.mesh.h[F] = H;
                    voronoi.mesh.to[H] = Vj;
                    voronoi.mesh.out[Vj] = Hn;
                    I = (I + 1) % 3;
                }
                hij = opp[prev[hij]];
            } while (hij != out[vi]);
            F++;
        }
        // case: boundary face where two voronoi regions meet
        else
        {
            // find first halfedge with split vertex at the tip
            int hij = f2h[f];
            while (voronoi.vertex_type[to[hij]] != VertexType::VE)
                hij = n[hij];

            // find second halfedge with split vertex at the tip
            int hjk = n[hij];
            while (voronoi.vertex_type[to[hjk]] != VertexType::VE)
                hjk = n[hjk];

            // split the face between the two vertices and mark it as voronoi region edge
            int h_split = split_face(hij, hjk);
            voronoi.edge_type[h2e[h_split]] = EdgeType::EE;
        }
    }

    // ensure mesh is a triangulation
    triangulate(); // TODO Remove
    assert(are_polygon_mesh_halfedges_valid(voronoi.mesh.n, voronoi.mesh.prev));
    assert(are_polygon_mesh_faces_valid(voronoi.mesh.n, voronoi.mesh.f, voronoi.mesh.h));

#ifdef ENABLE_VISUALIZATION
    if (false)
    {
        view();
        std::string mesh_handle = "overlay mesh";
        auto [v_new2old, v_old2new] = get_vertex_reductions();
        std::vector<int> vertex_region = vector_compose(voronoi.vertex_region, v_new2old);
        polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("vertex region", vertex_region);
        polyscope::show();
    }
#endif
}

void VoronoiOverlayMesh::find_voronoi_dual_edges()
{
    int num_dual_halfedges = voronoi.dual_halfedge_starts.size();
    spdlog::info("tracing {} dual halfedges", num_dual_halfedges);

    for (int Hij = 0; Hij < num_dual_halfedges; ++Hij)
    {
        // get start of embedded boundary chain (already computed) and label it
        int hij = voronoi.dual_halfedge_starts[Hij];
        int min_node = to[hij];
        voronoi.dual_halfedge[hij] = Hij;
        int length = 1;

        // traverse Voronoi halfedge, labelling the overlay edges and searching for
        // a minimal distance vertex on the chain to set as the midpoint
        while (voronoi.vertex_type[to[hij]] != VertexType::VV)
        {
            int vj = to[hij];
            int hjk = opp[hij];

            // get next dual halfedge in chain
            do
            {
                hjk = opp[prev[hjk]];
                int ejk = h2e[hjk];
                if (voronoi.edge_type[ejk] == EdgeType::EE)
                {
                    break;
                }
            } while (hjk != opp[hij]);

            // label overlay halfedge
            voronoi.dual_halfedge[hjk] = Hij;

            // check if shorter maximum distance found
            if (voronoi.distance[vj] < voronoi.distance[min_node])
                min_node = vj;

            //iterate
            hij = hjk;
            length++;
        }
        if (length == 1)
            spdlog::warn("encountered single edge voronoi segment");

        // set the minimum distance node as the midpoint
        voronoi.dual_edge_midpoints.push_back(min_node);
    }

    // build opp maps by propagating overlay opposite maps
    for (int Hij = 0; Hij < num_dual_halfedges; ++Hij)
    {
        int hij = voronoi.dual_halfedge_starts[Hij];
        int hji = opp[hij];
        int Hji = voronoi.dual_halfedge[hji];
        voronoi.mesh.opp[Hij] = Hji;
        assert(voronoi.mesh.to[Hji] == voronoi.mesh.to[voronoi.mesh.prev[Hij]]);

        // ensure consistency
        voronoi.dual_edge_midpoints[Hij] = voronoi.dual_edge_midpoints[Hji];
    }
    assert(are_polygon_mesh_vertices_valid(voronoi.mesh.opp, voronoi.mesh.prev, voronoi.mesh.to, voronoi.mesh.out));

    // copy voronoi dual connectivity to the simplified mesh
    simp.m.n = voronoi.mesh.n;
    simp.m.opp = voronoi.mesh.opp;
    simp.m.f = voronoi.mesh.f;
    simp.m.h = voronoi.mesh.h;
    simp.m.to = voronoi.mesh.to;
    simp.m.out = voronoi.mesh.out;
    simp.m.opp = voronoi.mesh.opp;
    simp.m.v_rep = arange(simp.m.out.size());

    // set arbitrary metric and constraints
    simp.m.l = std::vector<Scalar>(simp.m.n.size(), 1.);
    simp.m.Th_hat = std::vector<Scalar>(simp.m.out.size(), 2 * PI);
    simp.m.fixed_dof = std::vector<bool>(simp.m.out.size(), false);
    simp.m.fixed_dof[0] = true;

    // set trivial reflection structure
    simp.m.type = std::vector<char>(simp.m.n.size(), 0);
    simp.m.type_input = std::vector<char>(simp.m.n.size(), 0);
    simp.m.R = std::vector<int>(simp.m.n.size(), 0);

    build_edge_maps(simp.m.opp, simp.h2e, simp.e2h);
}

void VoronoiOverlayMesh::propagate_voronoi_vertices_to_faces()
{
    int num_faces = f2h.size();
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f))
            continue;

        // each face contains at least one interior vertex: use to label face
        int hij = f2h[f];
        do
        {
            int vj = to[hij];

            // find interior vertex
            if (voronoi.vertex_type[vj] == VertexType::VF)
            {
                voronoi.face_region[f] = voronoi.vertex_region[vj];
                break;
            }
            hij = n[hij];
        } while (hij != f2h[f]);
    }
}

// -----------------
// Delaunay Pipeline
// -----------------

void VoronoiOverlayMesh::trace_delaunay_edges()
{
    // enusre regular at the start
    make_regular();

    int num_halfedges = voronoi.dual_edge_midpoints.size();
    int num_regions = voronoi.dual_vertices.size();
    simp.vertex_index = voronoi.dual_vertices;
    std::vector<bool> is_traced(num_halfedges, false);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        // mark voronoi seed vertex as an embedded vertex
        int Vi = simp.m.to[simp.m.opp[hij]];
        int vi = voronoi.dual_vertices[Vi];
        simp.vertex_type[vi] = VertexType::VV;
        simp.vertex_parent[vi] = Vi;

        // mark voronoi boundary as an embedded edge vertex
        int vk = voronoi.dual_edge_midpoints[hij];
        simp.vertex_type[vk] = VertexType::VE;
        simp.vertex_parent[vk] = simp.h2e[hij];

        // trace path from the seed to the boundary
        std::vector<int> halfedges = {};
        if (!trace_path(vi, vk, hij, simp.h2e[hij], halfedges))
        {
            spdlog::error("could not trace path from {} to {}", vi, vk);
        }

        // set partent of opposite halfedges 
        for (int h : halfedges)
        {
            simp.halfedge_parent[opp[h]] = simp.m.opp[hij];
        }

        // check locally whether the connectivity is still regular in regions
        // where tracing is still necessary
        if (!is_traced[simp.m.opp[simp.m.n[simp.m.n[hij]]]])
        {
            make_oriented_adjacent_regular(reverse_vector(halfedges));
        }
        if (!is_traced[simp.m.n[simp.m.opp[hij]]])
        {
            make_oriented_adjacent_regular(vector_compose(opp, halfedges));
        }

        // mark the halfedge as traced
        is_traced[hij] = true;
    }
}

void VoronoiOverlayMesh::retrace_edges()
{
    spdlog::info("retracing edges");
    std::vector<int> overlay_vertices = find_overlay_vertices(simp);
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    int num_edges = simp.e2h.size();
    use_consistency_check = true;
    for (int e = 0; e < num_edges; ++e)
    {
        int hij = simp.e2h[e];
        int hji = simp.m.opp[hij];
        int Vi = simp.m.to[hji];
        int Vj = simp.m.to[hij];
        int vi = overlay_vertices[Vi];
        int vj = overlay_vertices[Vj];

        // remove the current path
        remove_path(start_halfedge[hij], simp);

        // retrace the path
        std::vector<int> halfedges = {};
        if (!trace_path(vi, vj, hij, e, halfedges))
        {
            spdlog::error("could not trace path from {} to {}", vi, vj);
        }
        for (int h : halfedges)
        {
            simp.halfedge_parent[opp[h]] = hji;
        }
        //make_adjacent_regular(halfedges);
    }
    use_consistency_check = false;
}

void VoronoiOverlayMesh::straighten_edges()
{
    std::vector<std::vector<int>> halfedge_path = enumerate_simplified_halfedges();

    // get straightened edges
    GeodesicFlipper geodesic_flipper(*this, halfedge_path);
    geodesic_flipper.flip_to_geodesics();
    geodesic_flipper.generate_surface_points();
    std::vector<std::vector<SurfacePoint>>& surface_points = geodesic_flipper.surface_points;
    std::vector<bool> is_intersecting = geodesic_flipper.get_intersecting_paths();

    // insert straightened paths
    PathStraightener path_straightener(*this, surface_points, is_intersecting);
    path_straightener.split_edges(*this, geodesic_flipper.edge_splits);
    path_straightener.join_edges(*this);
    path_straightener.retrace_paths(*this);

    // create new embedded face data
    flood_faces();

    // remove excess refinement from original paths
    remove_excess_refinement();
}

// -----------
// Frame Field
// -----------

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXi>
VoronoiOverlayMesh::sample_field_from_overlay(
    const Eigen::MatrixXd& reference_field_overlay,
    const Eigen::VectorXd& thetas_overlay,
    const Eigen::VectorXi& period_jumps_overlay,
    const OverlayStructure& overlay) const
{
    // sample theta and reference direction on each face
    int num_faces = f2h.size();
    Eigen::MatrixXd reference_field = Eigen::MatrixXd::Zero(num_faces, 3);
    Eigen::VectorXd thetas = Eigen::VectorXd::Zero(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f)) continue;
        int F = overlay.face_parent[f];
        reference_field.row(f) = reference_field_overlay.row(F);
        thetas[f] = thetas_overlay[F];
    }

    // sample period jump across overlay edges
    int num_halfedges = n.size();
    Eigen::VectorXi period_jumps = Eigen::VectorXi::Zero(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h)
    {
        if (is_halfedge_deleted(h)) continue;

        // skip interior face edges
        if (overlay.edge_type[h2e[h]] == EdgeType::EF) continue;

        period_jumps[h] = period_jumps_overlay[overlay.halfedge_parent[h]];
    }

    return std::make_tuple(reference_field, thetas, period_jumps);
}

void VoronoiOverlayMesh::comb_field_to_overlay_seam(
    const std::vector<bool> is_overlay_cut,
    int start_halfedge_index,
    Eigen::VectorXd& thetas,
    Eigen::VectorXi& period_jumps,
    const OverlayStructure& overlay) const
{
    // initialize queue and markings with first halfedge
    int h = start_halfedge_index;
    while (is_halfedge_deleted(h)) ++h; // make sure not deleted
    std::queue<int> Q;
    std::vector<bool> done(f2h.size());
    Q.push(h);
    done[face[h]] = true;

    while (!Q.empty()) {
        h = Q.front();
        Q.pop();
        int hn = n[h];
        int hp = n[hn];

        for (int h_curr : {h, hn, hp})
        {
            int e = h2e[h_curr];
            int ho = opp[h_curr];
            int fo = face[ho];

            // stop at visited edges and explicitly cut edges
            if (done[face[ho]]) continue;
            if ((overlay.edge_type[e] == EdgeType::EE) && (is_overlay_cut[overlay.halfedge_parent[h_curr]])) continue;

            // bake period jump across edge into the face angle
            int pj = period_jumps[ho];
            thetas[fo] += pj * (PI / 2.);

            // push period jumps through face to opposite wedge
            period_jumps[ho] = period_jumps[opp[ho]] = 0;
            period_jumps[n[ho]] -= pj;
            period_jumps[prev[ho]] -= pj;
            period_jumps[opp[n[ho]]] += pj;
            period_jumps[opp[prev[ho]]] += pj;

            // mark opposite face as seen and queue
            done[fo] = true;
            Q.push(ho);
        }
    }
}

Eigen::MatrixXd VoronoiOverlayMesh::compute_overlay_principal_direction(
    const Eigen::MatrixXd& V_overlay,
    const Eigen::MatrixXi& F_overlay,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& thetas,
    const OverlayStructure& overlay) const
{
    int num_faces = reference_field.rows();
    Eigen::MatrixXd PD1(num_faces, 3);

    // get local basis for rotations in overlay mesh
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V_overlay, F_overlay, B1, B2, B3);

    // rotate reference field in overlay faces
    for (int f = 0; f < num_faces; ++f)
    {
        int f_orig = overlay.face_parent[f];
        PD1.row(f) = Field::rotate_vector(reference_field.row(f), thetas(f), B1.row(f_orig), B2.row(f_orig));
    }

    return PD1;
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
VoronoiOverlayMesh::average_overlay_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& field_direction,
    const OverlayStructure& overlay) const
{
    int num_overlay_faces = F.rows();

    // get local basis for rotations in overlay mesh
    Eigen::MatrixXd B1, B2, N;
    igl::local_basis(V, F, B1, B2, N);

    // compute area weighted average directions
    int num_faces = f2h.size();
    std::vector<Scalar> areas = compute_face_areas();
    Eigen::MatrixXd weighted_direction = Eigen::MatrixXd::Zero(num_overlay_faces, 3);
    for (int f = 0; f < num_faces; ++f)
    {
        if (is_face_deleted(f)) continue;
        int f_orig = overlay.face_parent[f];
        weighted_direction.row(f_orig) += areas[f] * field_direction.row(f);
    }

    // normalize directions and generate rotated frame direction
    Eigen::MatrixXd PD1(num_overlay_faces, 3);
    Eigen::MatrixXd PD2(num_overlay_faces, 3);
    for (int f = 0; f < num_overlay_faces; ++f)
    {
        PD1.row(f) = project_orthogonal_to(weighted_direction.row(f), N.row(f)).stableNormalized();
        PD2.row(f) = Field::rotate_vector(PD1.row(f), PI / 2., B1.row(f), B2.row(f));
    }

    return std::make_tuple(PD1, PD2);
}

Eigen::MatrixXd VoronoiOverlayMesh::transfer_original_field_to_overlay() const
{
    // sample field on overlay mesh
    auto [reference_field, thetas, period_jumps] = sample_field_from_overlay(
        reference_field_orig,
        thetas_orig,
        period_jumps_orig,
        orig);

    // comb field on overlay to match simplified cut
    int h_start = 0;
    comb_field_to_overlay_seam(is_cut_simp, h_start, thetas, period_jumps, simp);

    // compute combed principal field direction
    Eigen::MatrixXd field_direction = compute_overlay_principal_direction(
        V_orig,
        F_orig,
        reference_field,
        thetas,
        orig);

    bool view_sampled_field = false;
    if (view_sampled_field)
    {
        Eigen::MatrixXd frame_field_orig = Field::generate_vector_field(V_orig, F_orig, reference_field_orig, thetas_orig);
        Holonomy::view_vector_field(V_orig, F_orig, frame_field_orig, "original");

        // generate frame field geometry from cross field
        Eigen::MatrixXd V_o = generate_overlay_vertices(V_orig, vtx_reindex_orig, orig);
        auto [F_o, F_halfedge] = generate_overlay_faces();
        auto [f_new2old, f_old2new] = get_face_reductions();
        Eigen::MatrixXd reference_field_o = subset_vertices(reference_field, f_new2old);
        Eigen::MatrixXd theta_o = subset_vertices(thetas, f_new2old);
        Eigen::MatrixXd frame_field_o = Field::generate_vector_field(V_o, F_o, reference_field_o, theta_o);
        Holonomy::view_vector_field(V_o, F_o, frame_field_o, "overaly");
    }

    return field_direction;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
VoronoiOverlayMesh::transfer_overlay_field_to_simplified(
    const Eigen::MatrixXd& field_direction
) const
{
    // compute averaged field on the simplified mesh
    auto [F_simp, F_halfedge] = generate_connectivity_faces(simp.m.n, simp.m.h, simp.m.to);
    auto [PD1, PD2] = average_overlay_field(simplified_V, F_simp, field_direction, simp);

    bool view_sampled_field = false;
    if (view_sampled_field)
    {
        // view coarse mesh 
        Holonomy::view_vector_field(simplified_V, F_simp, PD1, "simplified");
    }

    return std::make_tuple(PD1, PD2);
}

// ---------------
// Parametrization
// ---------------

std::vector<bool> VoronoiOverlayMesh::compute_layout()
{
    std::vector<bool> is_cut_place_holder;
    std::vector<bool> is_cut = compute_layout_topology(orig.m, is_cut_place_holder);
    std::vector<bool> is_cone(orig.m.n_vertices(), false);
    for (int vi = 0; vi < orig.m.n_vertices(); ++vi)
    {
        is_cone[vi] = (!float_equal(orig.m.Th_hat[orig.m.v_rep[vi]] , 2 * PI));
    }
    trim_topology(orig.m, is_cone, is_cut);

    // set starting point - use a boundary edge
    int h = 0;
    while (is_halfedge_deleted(h)) ++h;
    spdlog::debug("Using edge {} as layout start", h);
    auto done = std::vector<bool>(f2h.size(), false);

    Eigen::MatrixXd V_o = generate_overlay_vertices(V_orig, vtx_reindex_orig, orig);
    std::vector<Eigen::Vector3d> V_r(V_o.rows());
    // TODO make this an option
    for (int vi = 0; vi < V_o.rows(); ++vi)
    {
        V_r[vi] = V_o.row(vi);
    }

    std::queue<int> Q;
    Q.push(h);
    done[face[h]] = true;

    spdlog::debug("beginning layout");
    while (!Q.empty()) {
        h = Q.front();
        Q.pop();
        int hn = n[h];
        int hp = n[hn];

        for (int h_curr : {h, hn, hp})
        {
            int e = h2e[h_curr];
            int ho = opp[h_curr];
            if (done[face[ho]]) continue;
            if ((orig.edge_type[e] == EdgeType::EE) && (is_cut[orig.halfedge_parent[h_curr]])) continue;

            done[face[ho]] = true;
            if (PD1_o.empty() || PD2_o.empty())
            {
                transform_opposite_face(*this, he_uv, h_curr);
            }
            else
            {
                transform_opposite_face(*this, he_uv, V_r, PD1_o, PD2_o, h_curr);
            }
            Q.push(ho);
        }
    }

    return is_cut;
};


std::tuple<Mesh<Scalar>, std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>,
std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>>
get_consistent_layout(Mesh<Scalar> &m_orig, OverlayMesh<Scalar> &m_o)
{
    std::vector<bool> is_cut = {};
    std::vector<bool> is_cut_orig = compute_layout_topology(m_orig, is_cut);

    // Get original overlay face labels
    auto f_labels = get_overlay_face_labels(m_o);

    // Compute layout of the underlying flipped mesh
    auto mc = m_o.cmesh();
    m_o.garbage_collection();

    spdlog::info("computing layout");
    auto mc_type = mc.type;
    mc.type = std::vector<char>(mc.n_halfedges(), 0);
    std::vector<Scalar> u_vec(mc.n_ind_vertices(), 0.);
    // std::vector<bool> _is_cut_place_holder; // TODO Remove
    //  auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder, 0);
    std::vector<bool> _is_cut_place_holder = is_cut;
    auto layout_res = compute_layout(mc, u_vec, _is_cut_place_holder);
    auto _u_c = std::get<0>(layout_res);
    auto _v_c = std::get<1>(layout_res);
    auto is_cut_c = std::get<2>(layout_res);
    mc.type = mc_type;

    std::vector<Scalar> u_o, v_o, u_unif, v_unif;

    // Interpolate layout to the overlay mesh
    spdlog::info("interpolating layout");
    Eigen::Matrix<Scalar, -1, 1> u_eig;
    u_eig.resize(u_vec.size());
    for (size_t i = 0; i < u_vec.size(); i++)
    {
        u_eig(i) = u_vec[i];
    }
    m_o.bc_eq_to_scaled(mc.n, mc.to, mc.l, u_eig);

    // check alignment
    for (int hij = 0; hij < mc.n_halfedges(); ++hij)
    {
        if (m_o.first_segment[hij] == m_o.last_segment(hij))
            continue;
        int hji = mc.opp[hij];

        // get the vertices on the _h side
        std::vector<Scalar> lambdas;
        int current_seg = m_o.first_segment[hij];
        while (m_o.vertex_type[m_o.to[current_seg]] != ORIGINAL_VERTEX)
        {
            std::array<Scalar, 2> tmp = m_o.seg_bcs[current_seg];
            lambdas.push_back(tmp[1]);
            current_seg = m_o.next_segment(current_seg);
        }
        // check the vertices on the _hopp side

        current_seg = m_o.first_segment[hji];
        int cnt = lambdas.size() - 1;
        while (m_o.vertex_type[m_o.to[current_seg]] != ORIGINAL_VERTEX)
        {
            std::array<Scalar, 2> tmp = m_o.seg_bcs[current_seg];
            if (abs(lambdas[cnt] - tmp[0]) > 1e-10)
            {
                spdlog::error("alignment problem: {}, {}", lambdas[cnt], tmp[0]);
            }
            cnt--;
            current_seg = m_o.next_segment(current_seg);
        }
    }

    spdlog::info("interpolating on overlay mesh");
    u_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _u_c);
    v_o = m_o.interpolate_along_c_bc(mc.n, mc.f, _v_c);

    // Build a new mesh directly from the triangulated overlay mesh
    spdlog::info("creating new mesh with overlay connectivity");
    Mesh<Scalar> m;
    m.n = m_o.n;
    m.opp = m_o.opp;
    m.f = m_o.f;
    m.h = m_o.h;
    m.out = m_o.out;
    m.to = m_o.to;
    m.l = std::vector<Scalar>(m.n.size(), 0.0);
    for (int i = 0; i < m.n_halfedges(); i++)
    {
        int h0 = i;
        int h1 = h0;
        do
        {
            if (m.n[h1] == h0)
                break;
            h1 = m.n[h1];
        } while (h0 != h1);
        if (m.to[m.opp[h0]] != m.to[h1])
        {
            spdlog::error("h0 h1 picked wrong.");
            exit(0);
        }
        m.l[h0] = sqrt(max(
            (u_o[h0] - u_o[h1]) * (u_o[h0] - u_o[h1]) + (v_o[h0] - v_o[h1]) * (v_o[h0] - v_o[h1]), 0.));
        if (std::isnan(double(m.l[h0])))
            spdlog::error("length computation produced nan.");
    }
    m.type = std::vector<char>(m.n.size(), 0);
    int num_m_halfedges = m.n.size();
    for (int hi = 0; hi < num_m_halfedges; ++hi)
    {
        if (!float_equal(m.l[hi], m.l[m.opp[hi]]))
        {
            spdlog::error("inconsistent edge lengths");
        }
        if (m_o.edge_type[hi] == CURRENT_EDGE)
        {
            m.type[hi] = 4;
        }
        else if (m_o.edge_type[hi] == ORIGINAL_AND_CURRENT_EDGE)
        {
            m.type[hi] = m_orig.type[m_o.origin_of_origin[hi]];
        }
        else if (m_o.edge_type[hi] == ORIGINAL_EDGE)
        {
            m.type[hi] = m_orig.type[m_o.origin[hi]];
        }
    }
    m.type_input = m.type;

    spdlog::info("triangulating overlay mesh");
    triangulate_polygon_mesh(m, u_o, v_o, f_labels);
    m.R = std::vector<int>(m.n.size(), 0);
    m.type.resize(m.n.size(), 4);
    OverlayMesh<Scalar> m_o_tri(m);
    for (int i = m_o.n_halfedges(); i < m_o_tri.n_halfedges(); i++)
    {
        m_o_tri.edge_type[i] = CURRENT_EDGE; // make sure do not use the new diagonal
    }

    // map angles to mesh
    m.v_rep = range(0, m.out.size());
    m.Th_hat = std::vector<Scalar>(m.out.size(), 2 * PI);
    for (int vi = 0; vi < m_o.n_vertices(); vi++)
    {
        if (m_o.vertex_type[vi] == ORIGINAL_VERTEX)
        {
            m.Th_hat[vi] = mc.Th_hat[mc.v_rep[vi]];
        }
    }

    // Pullback cut on the original mesh to the overlay
    spdlog::info("pulling back cut");
    bool is_original_cut = !(is_cut_orig.empty());
    std::vector<bool> is_cut_poly;
    is_cut_poly = pullback_cut_to_overlay(m_o, is_cut_orig, is_original_cut);

    // Extend the overlay cut to the triangulated mesh
    // WARNING: Assumes triangulation halfedges added to the end
    std::vector<bool> is_cut_o = std::vector<bool>(m.n_halfedges(), false);
    for (int h = 0; h < m_o.n_halfedges(); ++h)
    {
        is_cut_o[h] = is_cut_poly[h];
    }

    // Now directly do layout on triangulated overlay mesh
    spdlog::info("creating new layout for triangulated overlay");
    std::vector<Scalar> phi(m.n_vertices(), 0.0);
    auto overlay_layout_res = compute_layout_components(m, phi, is_cut_o);
    std::vector<Scalar> _u_o = std::get<0>(overlay_layout_res);
    std::vector<Scalar> _v_o = std::get<1>(overlay_layout_res);
    is_cut_o = std::get<2>(overlay_layout_res);

    return std::make_tuple(m, _u_o, _v_o, is_cut_o, _u_c, _v_c, is_cut_c);
}

// WARNING: Only works if all simplified vertices are original vertices
std::tuple<VectorX, VectorX> VoronoiOverlayMesh::integrate_simplified_function_onto_original(
    const VectorX& simplified_function)
{
    std::vector<int> start_halfedge = find_start_halfedges(orig);
    int num_halfedges = orig.m.n.size();

    // get integrated values between halfedges rotated at vertices
    VectorX corner_integral = VectorX::Zero(num_halfedges);
    for (int Hij = 0; Hij < num_halfedges; ++Hij)
    {
        int Hji = orig.m.opp[Hij];
        int hij = start_halfedge[Hij];
        do
        {
            if (simp.edge_type[h2e[hij]] == EdgeType::EE)
            {
                int Hij_simp = simp.halfedge_parent[hij];
                corner_integral[Hij] += simplified_function[Hij_simp];
                corner_integral[Hji] -= simplified_function[Hij_simp];
            }

            hij = opp[prev[hij]];
        }
        while (orig.edge_type[h2e[hij]] != EdgeType::EE);
    }

    // get integrated values along edges
    VectorX edge_integral = VectorX::Zero(num_halfedges);
    for (int Hij = 0; Hij < num_halfedges; ++Hij)
    {
        int Hji = orig.m.opp[Hij];
        if (Hji < Hij) continue;

        int hij = start_halfedge[Hij];
        while (orig.vertex_type[to[hij]] != VertexType::VV)
        {
            if (simp.vertex_type[to[hij]] == VertexType::VE)
            {
                int hjk = get_next_overlay_edge(hij, simp);
                int Hjk_simp = simp.halfedge_parent[hjk];
                edge_integral[Hij] += simplified_function[Hjk_simp];
                edge_integral[Hji] -= simplified_function[Hjk_simp];
            }
            else if (simp.vertex_type[to[hij]] == VertexType::VV)
            {
                spdlog::error("cannot integrate if simplified vertices intersect edges");
            }

            hij = get_next_overlay_edge(hij, orig);
        }
    }

    return std::make_tuple(corner_integral, edge_integral);
}

SymDir::Parameters read_parameters(nlohmann::json& config)
{
    SymDir::Parameters param;
    param.max_iters = config["max_iters"]; // iterations
    param.max_time = config["max_time"]; // time in seconds
    param.smooth_only_iters = config["smooth_only_iters"];
    param.E_target = config["E_target"]; // Energy target
    param.ls_iters = config["ls_iters"]; // param for linesearch in smoothing operation
    param.do_newton = config["do_newton"]; // do newton/gd steps for smoothing operation
    param.local_smooth = config["local_smooth"];
    param.global_smooth = config["global_smooth"];
    param.elen_alpha = config["elen_alpha"];
    param.do_projection = config["do_projection"];
    param.with_cons = config["with_cons"];
    param.Lp = config["Lp"];
    param.save_meshes = config["save_meshes"];
    param.do_feature_alignment = config["do_feature_alignment"]; // align feature edges
    param.symdir_weight = config["symdir_weight"];
    param.alignment_weight = config["alignment_weight"];
    param.degenerate_weight = config["degenerate_weight"];
    param.fix_misaligned = config["fix_misaligned"];
    param.use_rref = config["use_rref"];
    // param.solver_type = config["solver_type"];
    param.cg_rel_err = config["cg_rel_err"];
    param.cg_iters = config["cg_iters"];
    
    param.percentages = config["percentages"].get<std::vector<double>>();
    param.percentage_target = config["percentage_target"];
    param.percentage_target_value = config["percentage_target_value"];
    param.save_percentages_meshes = config["save_percentages_meshes"];

    param.E_abs_err = config["E_abs_err"];
    param.E_rel_err = config["E_rel_err"];
    param.diff_err = config["diff_err"];
    param.grad_abs_err = config["grad_abs_err"];
    param.grad_rel_err = config["grad_rel_err"];
    
    param.precompute_seamless = config["precompute_seamless"];
    param.projected_newton = config["projected_newton"];
    param.soft_max = config["soft_max"];
    param.t = config["t"];
    param.precompute_seamless = config["precompute_seamless"];
    
    param.percentage_target_converge = config["percentage_target_converge"];
    param.max_grad_abs_converge = config["max_grad_abs_converge"];
    param.max_grad_rel_converge = config["max_grad_rel_converge"];
    param.energy_diff_converge = config["energy_diff_converge"];
    param.use_worst_n_energy_in_ls = config["use_worst_n_energy_in_ls"];
    param.E_abs_converge = config["E_abs_converge"];
    param.E_rel_converge = config["E_rel_converge"];

    param.last_screenshot_after_optimization = config["last_screenshot_after_optimization"];
    param.screenshot_interval = config["screenshot_interval"];
    param.output_dir_for_screenshots = config["output_dir_for_screenshots"];
    param.uv_scale_for_screenshots = config["uv_scale_for_screenshots"];
    param.angle_to_rotate_model_for_screenshots = config["angle_to_rotate_model_for_screenshots"];
    param.screenshot_during_optimization = config["screenshot_during_optimization"];

    param.degenerate_vertices_preconditioner = config["degenerate_vertices_preconditioner"];
    param.precond_dim = config["precond_dim"];
    param.triangle_threshold = config["triangle_threshold"];

    return param;
}   

void VoronoiOverlayMesh::optimize_simplified_parameterization()
{
    const Mesh<Scalar>& m_tri = simp.m;
    auto [F_tri, F_halfedge] = generate_connectivity_faces(m_tri.n, m_tri.h, m_tri.to);
    //Holonomy::view_seamless_parameterization(simplified_V, F_tri, uv_s, F_halfedge);

    // trim cut topology to just cones
    std::vector<bool> is_cone(simp.m.n_vertices(), false);
    for (int vi = 0; vi < simp.m.n_vertices(); ++vi)
    {
        is_cone[vi] = (!float_equal(simp.m.Th_hat[simp.m.v_rep[vi]] , 2 * PI));
    }
    trim_topology(m_tri, is_cone, is_cut_simp);

    int num_halfedges = m_tri.n_halfedges();
    int num_faces = m_tri.n_faces();
    UnionFind cut_vertices(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (is_cut_simp[hij]) continue;

        // get halfedge in adjacent uncut face pointing to vj
        int hkj = m_tri.n[m_tri.n[m_tri.opp[hij]]];
        cut_vertices.union_sets(hij, hkj);
    }
    int num_cut_vertices = cut_vertices.count_sets();
    std::vector<int> cut_vertex_indices = cut_vertices.index_sets();

    Eigen::MatrixXd V_tri(num_cut_vertices, 3);
    Eigen::MatrixXi FT_tri(num_faces, 3);
    Eigen::MatrixXd uv_tri(num_cut_vertices, 2);
    std::vector<std::array<int, 4>> edge_pairings;
    edge_pairings.reserve(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int hij = F_halfedge(f, i);
            uv_tri(cut_vertex_indices[hij], 0) = u_simp[hij];
            uv_tri(cut_vertex_indices[hij], 1) = v_simp[hij];
            V_tri.row(cut_vertex_indices[hij]) = simplified_V.row(m_tri.to[hij]);
            FT_tri(f, i) = cut_vertex_indices[hij];

            // check for seam edges
            if (is_cut_simp[hij])
            {
                int hji = m_tri.opp[hij];
                if (hij < hji) continue; // only process edge once
                int uvj = cut_vertex_indices[hij];
                int uvi = cut_vertex_indices[m_tri.n[m_tri.n[hij]]];
                int uvi_opp = cut_vertex_indices[hji];
                int uvj_opp = cut_vertex_indices[m_tri.n[m_tri.n[hji]]];
                edge_pairings.push_back({ uvi, uvj, uvj_opp, uvi_opp });
            }
        }
    }
    //Holonomy::view_seamless_parameterization(simplified_V, F_tri, uv_tri, FT_tri, "cut");
    //Holonomy::view_seamless_parameterization(V_tri, FT_tri, uv_tri, FT_tri, "cut");

    Eigen::MatrixXi EE(edge_pairings.size(), 4);
    for (int i = 0; i < edge_pairings.size(); ++i) {
        EE.row(i) << edge_pairings[i][0], edge_pairings[i][1], edge_pairings[i][2], edge_pairings[i][3];
    }

    if (output_dir != "")
    {
        std::string output_filename = join_path(output_dir, name + "_initial_coarse_mesh.obj");
        write_obj_with_uv(output_filename, simplified_V, F_tri, uv_tri, FT_tri);
    }

    //MeshCutter meshcutter(simplified_V, uv_tri, F_tri, FT_tri);
    //auto [V_tri, EE] = meshcutter.cut_mesh();
    spdlog::info("Initializing optimization");
    SymDir::ExtremeOpt extremeopt(V_tri, FT_tri);

    spdlog::info("Reading parameters from file");
    std::ifstream js_in(symdir_json);
    nlohmann::json config = nlohmann::json::parse(js_in);
    SymDir::Parameters param;
    param = read_parameters(config);
    //param.diff_err = 1e-4;
    //param.grad_abs_err = 1e-4;
    //param.energy_diff_converge = false;
    extremeopt.m_params = param;

    spdlog::info("Creating optimization mesh");
    extremeopt.create_mesh(V_tri, FT_tri, uv_tri);
    extremeopt.set_v_map(F_tri, FT_tri);

    spdlog::info("Initializing constraints");
    std::vector<std::vector<int>> EE_e = transform_EE(FT_tri, EE);
    extremeopt.init_constraints(EE_e);
    extremeopt.EE = EE;

    // optionally align layout to field
    int phase = 0;
    bool do_align_layout_to_field = true;
    if (do_align_layout_to_field)
    {
        Eigen::MatrixXd field_direction = transfer_original_field_to_overlay();

        bool optimize_coarse_alignment = true;
        if (optimize_coarse_alignment)
        {
            // causes consistency issues with fine field rotation
            auto [PD1, PD2] = transfer_overlay_field_to_simplified(field_direction);
            std::tie(phase, extremeopt.PD1, extremeopt.PD2) = Field::maximize_combed_frame_alignment(
                simplified_V, F_tri, uv_tri, FT_tri, PD1, PD2);
        }
        else
        {
            std::tie(extremeopt.PD1, extremeopt.PD2) = transfer_overlay_field_to_simplified(field_direction);
        }
        
        // save combed field
        // generate frame field geometry from cross field
        Eigen::MatrixXd V_o = generate_overlay_vertices(V_orig, vtx_reindex_orig, orig);
        std::vector<Eigen::Vector3d> V_r(V_o.rows());
        for (int vi = 0; vi < V_o.rows(); ++vi)
        {
            V_r[vi] = V_o.row(vi);
        }
        int num_faces = field_direction.rows();
        PD1_o.resize(num_faces);
        PD2_o.resize(num_faces);
        Eigen::Matrix2d R = generate_ccw_rotation();
        for (int fi = 0; fi < num_faces; ++fi)
        {
            // get initial direction
            PD1_o[fi] = field_direction.row(fi);
            
            // align phase with combed field
            for (int i = 0; i < phase; ++i)
            {
                PD1_o[fi] = rotate_in_face_plane(*this, V_r, R, PD1_o[fi], fi);
            }
            
            // v gradient target is one more rotation
            PD2_o[fi] = rotate_in_face_plane(*this, V_r, R, PD1_o[fi], fi);
        }

        std::string field_filename = join_path(output_dir, name + "_coarse.cfield");
        spdlog::info("writing field to {}", field_filename);
        Field::write_combed_field(field_filename, extremeopt.PD1, extremeopt.PD2);
    }
    else
    {
        param.alignment_weight = 0.;
    }

    spdlog::info("Optimizing layout");
    extremeopt.do_optimization_without_log();
    extremeopt.export_mesh(V_tri, FT_tri, uv_tri);

    //Holonomy::view_seamless_parameterization(V_tri, FT_tri, uv_tri, FT_tri, "optimized");
    if (output_dir != "")
    {
        std::string output_filename = join_path(output_dir, name + "_coarse.obj");
        write_obj_with_uv(output_filename, simplified_V, F_tri, uv_tri, FT_tri);
    }

    // copy back optimized uv coordinates
    for (int f = 0; f < num_faces; ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int hij = F_halfedge(f, i);
            u_simp[hij] = uv_tri(cut_vertex_indices[hij], 0);
            v_simp[hij] = uv_tri(cut_vertex_indices[hij], 1);
        }
    }
}

void VoronoiOverlayMesh::parameterize_with_overlay(
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa
)
{
    assert(is_valid());

    // create parametrization for the simplified mesh with Penner coordinates
    timer.start();
    auto [V_overlay, m_o] = generate_simplified_parameterization(homology_basis_loops, kappa);
    simplified_V = V_overlay;
    log["4.1-param-penner-time"] = timer.getElapsedTime();

    // find coarse mesh layout
    timer.start();
    spdlog::info("generating coarse layout");
    auto &mc = m_o.cmesh();
    std::vector<Scalar> u_vec(mc.n_vertices(), 0.);
    auto layout_res = get_consistent_layout(simp.m, m_o);
    auto m_tri = std::get<0>(layout_res);
    u_simp = std::get<1>(layout_res);
    v_simp = std::get<2>(layout_res);
    is_cut_simp = std::get<3>(layout_res);

    // convert layout to a matrix
    Eigen::MatrixXd uv_s(u_simp.size(), 2);
    for (int i = 0; i < u_simp.size(); ++i)
    {
        uv_s(i, 0) = u_simp[i];
        uv_s(i, 1) = v_simp[i];
        if (isnan(u_simp[i])) spdlog::error("nan in layout");
        if (isnan(v_simp[i])) spdlog::error("nan in layout");
    }

    // check layout error
    auto [length_error, angle_error] = compute_seamless_error(m_tri, uv_s);
    spdlog::info("coarse length error is {}", length_error.maxCoeff());
    spdlog::info("coarse angle error is {}", angle_error.maxCoeff());

    // get face map from overlay to original mesh
    std::vector<int> fn_to_f;
    build_face_maps(m_o, fn_to_f);

    // (optionally) simplify the parametrization connectivity before refining
    // WARNING: currently does not support field transfer
    if (simplify_param_overlay)
    {
        spdlog::warn("not implemented for field transfer; need to transfer cut information");

        // copy parametrization to vector of points
        std::vector<Eigen::Vector2d> uv_simp(u_simp.size());
        for (int i = 0; i < u_simp.size(); ++i)
        {
            uv_simp[i] = {u_simp[i], v_simp[i]};
        }

        // build simplifier data structure and use to simplify
        OverlaySimplifier simplifier(m_o, simp.m, is_cut_simp, uv_simp);
        simplifier.refine_halfedges();
        std::tie (m_o, fn_to_f, uv_simp) = simplifier.get_refined_mesh();
        
        // copy simplified connectivity back
        m_tri = m_o.cmesh();
        u_simp.resize(uv_simp.size());
        v_simp.resize(uv_simp.size());
        for (int i = 0; i < uv_simp.size(); ++i)
        {
            u_simp[i] = uv_simp[i][0];
            v_simp[i] = uv_simp[i][1];
        }
    }

    // get edge map for overlay triangulation
    std::vector<int> h2e_tri, e2h_tri;
    build_edge_maps(m_tri.opp, h2e_tri, e2h_tri);
    if (false)
        view_halfedge_mesh_layout(m_tri, u_simp, v_simp);

    // check some basic diagnostics
    bool check_validity = false;
    if (check_validity)
    {
        VectorX he2angle, he2cot;
        corner_angles(m_tri, he2angle, he2cot);
        spdlog::info("min angle in coarse overlay: {}", he2angle.minCoeff());
    }
    spdlog::info("original coarse mesh has {} halfedges", simp.m.n_halfedges());
    spdlog::info("param overlay mesh has {} halfedges", m_o.n_halfedges());
    spdlog::info("triangulated param mesh has {} halfedges", m_tri.n_halfedges());

    // build map from vertices to original halfedge originating from them
    std::vector<int> first_segment(simp.m.n_halfedges(), -1);
    for (int hij = 0; hij < m_o.n_halfedges(); ++hij)
    {
        // only look for original edges
        if (m_o.edge_type[hij] == CURRENT_EDGE)
            continue;

        // only look for segment vertices
        int vi = m_o.to[m_o.opp[hij]];
        int vj = m_o.to[hij];

        int Hij = m_o.origin_of_origin[hij];
        if (m_o.edge_type[hij] == ORIGINAL_EDGE)
            Hij = m_o.origin[hij];
        if (m_o.vertex_type[vi] == ORIGINAL_VERTEX)
            first_segment[Hij] = hij;
    }

    // step 1: refine simplified mesh halfedges at intersection points
    // the vertices are inserted (approximately) uniformly on the simplified edge path
    spdlog::info("splitting original overlay edges");
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    std::vector<int> v2v(m_o.n_vertices(), -1);
    simp.vertex_index.resize(m_o.n_vertices());
    for (int hij = 0; hij < simp.m.n_halfedges(); ++hij)
    {
        // only process simplified edge once
        if (hij < simp.m.opp[hij]) continue;
        spdlog::debug("tracing original edge {}, {}", hij, simp.m.opp[hij]);

        // get the length of the edge and the total length of the path on the overlay
        int h_curr = start_halfedge[hij];
        int h = first_segment[hij];
        Scalar edge_length = m_tri.l[h];
        int num_segments = 1;
        while (m_o.vertex_type[m_o.to[h]] != ORIGINAL_VERTEX)
        {
            h = next_orig_segment(m_o, h);
            edge_length += m_tri.l[h];
            ++num_segments;
        }
        h = first_segment[hij];
        Scalar delta = 1. / num_segments;
        Scalar path_length = compute_simplified_halfedge_length(h_curr);

        // iterate over overlay edges to insert new vertices on the correct halfedge
        v2v[m_o.to[m_o.opp[h]]] = to[opp[h_curr]];
        simp.vertex_index[m_o.to[m_o.opp[h]]] = to[opp[h_curr]];
        Scalar edge_tip_pos = m_tri.l[h]; // FIXME risky to use inconsistent m_o and m_tri
        Scalar path_base_pos = 0.;
        int count = 0;
        while (m_o.vertex_type[m_o.to[h]] != ORIGINAL_VERTEX)
        {
            // uniform distributes intersections, but increases parameterization distortion
            bool use_uniform_intersections = false;

            // compute target edge coordinate
            Scalar edge_coord;
            if (use_uniform_intersections)
            {
                edge_coord = (count + 1) * delta;
            }
            else
            {
                edge_coord = edge_tip_pos / edge_length;
            }
            spdlog::debug("tracing overlay edge {}, {} with target coordinate {}", h, m_o.opp[h], edge_coord);

            // increment path halfedge until approximately at the right halfedge
            while (path_base_pos / path_length  < edge_coord)
            {
                // stop if at last halfedge in path
                if (simp.vertex_type[to[h_curr]] == VertexType::VV)
                    break;

                // set parent data for current edge and its tip
                path_base_pos += halfedge_length(h_curr);
                simp.halfedge_parent[h_curr] = h;
                simp.halfedge_parent[opp[h_curr]] = m_o.opp[h];
                simp.vertex_parent[to[h_curr]] = h2e_tri[h];

                // increment to next overlay edge
                h_curr = get_next_overlay_edge(h_curr, simp);
                assert(simp.edge_type[h2e[h_curr]] == EdgeType::EE);
            }

            // split edge (while preserving triangular faces)
            h_curr = split_edge(h_curr);
            int h_next = n[h_curr]; // need to get before splitting
            int h_opp = opp[n[h_curr]];
            split_face(h_curr, n[n[h_curr]]);
            split_face(h_opp, n[n[h_opp]]);

            // set parent data for current edge 
            simp.halfedge_parent[h_curr] = h;
            simp.halfedge_parent[opp[h_curr]] = m_o.opp[h];

            // mark the vertex tip as an embedded vertex and update the vertex map
            simp.vertex_type[to[h_curr]] = VertexType::VV;
            simp.vertex_parent[to[h_curr]] = m_o.to[h];
            v2v[m_o.to[h]] = to[h_curr];
            simp.vertex_index[m_o.to[h]] = to[h_curr];
            spdlog::debug("new overlay vertex {} at {}", m_o.to[h], to[h_curr]);

            // increment both halfedges and lengths
            path_base_pos += halfedge_length(h_curr);
            h_curr = h_next;
            h = next_orig_segment(m_o, h);
            assert(m_o.edge_type[h] != CURRENT_EDGE);
            edge_tip_pos += m_tri.l[h];
            ++count;
        }

        // label parent of remaining edges
        spdlog::debug("tracing remaining overlay edge {}, {}", h, m_o.opp[h]);
        std::vector<int> remaining_halfedges = enumerate_halfedges(h_curr, simp);
        int num_remaining_halfedges = remaining_halfedges.size();
        for (int h_path : remaining_halfedges)
        {
            simp.halfedge_parent[h_path] = h;
            simp.halfedge_parent[opp[h_path]] = m_o.opp[h];
        }
        for (int i = 0; i < num_remaining_halfedges - 1; ++i)
        {
            int h_path = remaining_halfedges[i];
            simp.vertex_parent[to[h_path]] = h2e_tri[h]; // TODO not for last
        }
    }

    // restore regular triangulation
    triangulate();
    make_regular();

    // step 2: trace overlay edges not in the original mesh
    // note that the overlay mesh is generally not a triangulation
    spdlog::info("tracing current overlay edges");
    for (int hij = 0; hij < m_o.n_halfedges(); ++hij)
    {
        if (hij < m_o.opp[hij])
            continue; // only process once
        if (m_o.edge_type[hij] != CURRENT_EDGE)
            continue; // only add current edges that were not original edges

        // get endpoints of path to trace
        spdlog::debug("tracing overlay edge {}, {}", hij, m_o.opp[hij]);
        int vi = m_o.to[m_o.opp[hij]];
        int vj = m_o.to[hij];
        int Vi = v2v[vi];
        int Vj = v2v[vj];

        // get embedded face to stay inside
        int f = fn_to_f[m_o.f[hij]];

        // trace path while staying in the correct embedded face
        std::vector<int> halfedges;
        trace_path(Vi, Vj, hij, h2e_tri[hij], halfedges, f);
        for (int h : halfedges)
        {
            simp.halfedge_parent[opp[h]] = m_o.opp[hij];
            assert(simp.face_parent[face[h]] == f);
        }

        // maintain regularity
        // TODO set the start halfedge of the opposite path
        make_adjacent_regular(halfedges);
    }

    // step 3: trace additional edges added to make the overlay a triangulation
    spdlog::info("tracing current overlay edges {} to {}", m_o.n_halfedges(), m_tri.n_halfedges());
    for (int hij = m_o.n_halfedges(); hij < m_tri.n_halfedges(); ++hij)
    {
        if (hij < m_tri.opp[hij])
            continue; // only process once

        // get endpoints of path
        spdlog::debug("tracing additional edge {}, {}", hij, m_tri.opp[hij]);
        int vi = m_tri.to[m_tri.opp[hij]];
        int vj = m_tri.to[hij];
        int Vi = v2v[vi];
        int Vj = v2v[vj];

        // get halfedge in triangulated face that is in the overlay mesh
        int hjk = m_tri.n[hij];
        while (hjk >= m_o.n_halfedges())
        {
            hjk = m_tri.n[hjk];
        }

        // get halfedge in the same overlay face that is an original edge
        while (m_o.edge_type[hjk] == CURRENT_EDGE)
        {
            hjk = m_o.n[hjk];
        }

        // get face of embedded original mesh to trace path in
        // TODO: should actually update face data and use overlay face labels
        int f = simp.m.f[m_o.origin_of_origin[hjk]];
        if (m_o.edge_type[hjk] == ORIGINAL_EDGE)
            f = simp.m.f[m_o.origin[hjk]];

        // trace path while staying in the correct embedded face
        std::vector<int> halfedges;
        trace_path(Vi, Vj, hij, h2e_tri[hij], halfedges, f);
        for (int h : halfedges)
        {
            simp.halfedge_parent[opp[h]] = m_tri.opp[hij];
            assert(simp.face_parent[face[h]] == f);
        }

        // maintain regularity
        // TODO set the start halfedge of the opposite path
        make_adjacent_regular(halfedges);
        // TODO refine
    }

    // set simplified mesh to overlay
    simp.m = m_tri;
    simp.h2e = h2e_tri;
    simp.e2h = e2h_tri;
    log["4.2-param-refine-time"] = timer.getElapsedTime();
}

std::tuple<
    Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi,
    std::vector<int>, Eigen::MatrixXi, Eigen::MatrixXi
>
VoronoiOverlayMesh::transfer_simplified_parameterization()
{
    // first, ensure no redundancy
    timer.start();
    flood_faces();
    remove_excess_refinement();
    log["4.3-param-simplify-time"] = timer.getElapsedTime();

    // optionally retrace paths with geodesics
    // TODO: move to the initial tracing
    if (use_geodesic_refinement)
    {
        assert(HalfedgeMesh::is_valid());
        assert(is_simplicial());

        timer.start();
        straighten_edges();
        log["4.4-param-retrace-time"] = timer.getElapsedTime();
    }

    timer.start();
    // perform additional refinement to the mesh so that the Tutte maps are not degenerate
    if (make_overlay_three_connected) make_three_connected(orig);
    if (make_simplified_three_connected) make_three_connected(simp);

    // (optionally) solve for barycentric coordiantes on the original mesh
    if (use_uniform_overlay_edges) set_uniform_edge_coordinates(orig);
    if (use_harmonic_overlay_vertices) find_tutte_coordinates(orig);
    log["4.5-orig-tutte-time"] = timer.getElapsedTime();

    // solve for barycentric coordiantes on the simplified mesh
    spdlog::info("finding harmonic coordiantes");
    timer.start();
    set_uniform_edge_coordinates(simp);
    find_tutte_coordinates(simp);
    log["4.5-param-tutte-time"] = timer.getElapsedTime();
    
    // (optionally) optimize the parametrization
    if (optimize_layout) optimize_simplified_parameterization();

    // lift layout to overlay
    timer.start();
    he_uv.resize(n.size());
    assert(is_valid_edge_network());
    for (int hij = 0; hij < n.size(); ++hij)
    {
        he_uv[hij] = {0., 0.};
        if (is_halfedge_deleted(hij))
            continue;
        int fijk = simp.face_parent[face[hij]];
        int hj = simp.m.h[fijk];
        int hk = simp.m.n[hj];
        int hi = simp.m.n[hk];
        assert(hj >= 0);
        assert(hj < u_simp.size());

        // create uv face vertices for the simplified triangle
        BaryCoords Vi = {u_simp[hi], v_simp[hi]};
        BaryCoords Vj = {u_simp[hj], v_simp[hj]};
        BaryCoords Vk = {u_simp[hk], v_simp[hk]};

        // get barycentric coordinates for the tip of the halfedge in the simplified triangle
        BaryCoords bary_coords = tip_face_coordinates(hij, fijk, simp);
        Scalar u = bary_coords[0];
        Scalar v = bary_coords[1];
        Scalar w = 1. - (u + v);
        assert(u >= 0.);
        assert(v >= 0.);
        assert(w >= 0.);

        // set halfedge tip uv coordinates
        he_uv[hij] = u * Vi + v * Vj + w * Vk;
        assert(!isnan(he_uv[hij][0]));
        assert(!isnan(he_uv[hij][1]));
        assert(abs(he_uv[hij][0]) < 1e10);
        assert(abs(he_uv[hij][1]) < 1e10);
    }

    if (simplify_parameterization)
    {
        // create parametrized mesh that only refines edges where this is necessary
        spdlog::info("Simplifying parameterization");
        std::vector<bool> is_cut_orig = compute_layout();
        SimplificationOverlaySimplifier simplifier(*this, is_cut_orig, he_uv);
        simplifier.refine_halfedges();
        auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, hn_to_h_r, is_edge] = simplifier.build_refined_mesh();

        log["4.6-param-vf-time"] = timer.getElapsedTime();

        return std::make_tuple(V_r, F_r, uv_r, FT_r, fn_to_f_r, hn_to_h_r, is_edge);
    }
    else
    {
        spdlog::info("Skipping parameterization simplification");

        // get reduced indices
        auto [h_new2old, h_old2new] = get_halfedge_reductions();
        auto he_uv_r = vector_compose(he_uv, h_new2old);
        auto [opp_r, n_r, prev_r] = reduce_halfedge_maps();
        auto [face_r, f2h_r] = reduce_face_maps();
        auto [to_r, out_r] = reduce_vertex_maps();
        int num_faces = f2h_r.size();

        // find vertices for the uv topology with cut to disk
        UnionFind cut_vertices(h_new2old.size());
        for (int hij = 0; hij < h_new2old.size(); ++hij)
        {
            // union halfedges that are not on the layout cut
            if (simp.edge_type[h2e[h_new2old[hij]]] == EdgeType::EE)
            {
                int h_o = simp.halfedge_parent[h_new2old[hij]];
                if (is_cut_simp[h_o])
                    continue;
            } 

            // get halfedge in adjacent uncut face pointing to vj
            int hkj = prev_r[opp_r[hij]];
            cut_vertices.union_sets(hij, hkj);
        }
        int num_cut_vertices = cut_vertices.count_sets();
        std::vector<int> cut_vertex_indices = cut_vertices.index_sets();

        // build VF connectivity
        Eigen::MatrixXd V_o = generate_overlay_vertices(V_orig, vtx_reindex_orig, orig);
        Eigen::MatrixXi F(num_faces, 3);
        Eigen::MatrixXi F_uv(num_faces, 3);
        Eigen::MatrixXd uv(num_cut_vertices, 2);
        for (int f = 0; f < num_faces; ++f)
        {
            int hij = f2h_r[f];
            int hjk = n_r[hij];
            for (int i = 0; i < 3; ++i)
            {
                uv(cut_vertex_indices[hij], 0) = (double)(he_uv_r[hij][0]);
                uv(cut_vertex_indices[hij], 1) = (double)(he_uv_r[hij][1]);
                F(f, i) = to_r[hij];
                F_uv(f, i) = cut_vertex_indices[hij];

                // increment halfedge
                hij = n_r[hij];
            }
        }
        log["4.6-param-vf-time"] = timer.getElapsedTime();
        std::vector<int> face_parent_o;
        Eigen::MatrixXi halfedge_parent, is_edge;

        return std::make_tuple(V_o, F, uv, F_uv, face_parent_o, halfedge_parent, is_edge);
    }
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
VoronoiOverlayMesh::generate_voronoi_boundary(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    int num_edges = e2h.size();
    std::vector<std::array<int, 2>> edges;
    edges.reserve(count_edges());
    for (int eij = 0; eij < num_edges; ++eij)
    {
        // only add edges contained in simpified mesh edges
        if (is_edge_deleted(eij)) continue;
        if (voronoi.edge_type[eij] == EdgeType::EF) continue;

        int hij = e2h[eij];
        int vi = v_old2new[to[opp[hij]]];
        int vj = v_old2new[to[hij]];
        edges.push_back({vi, vj});
    }
    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    auto [V_s, edges_s] = remove_unreferenced_vertices(V_o, edges);
    Eigen::MatrixXi E_s(edges_s.size(), 2);
    for (int i = 0; i < edges_s.size(); ++i)
    {
        E_s(i, 0) = edges_s[i][0];
        E_s(i, 1) = edges_s[i][1];
    }
    
    return std::make_tuple(V_s, E_s);
}

void VoronoiOverlayMesh::PathTracerList::initialize()
{
    // Initialize an array to keep track of vertices
    int num_vertices = parent.out.size();
    m_is_processed_vertex = std::vector<bool>(num_vertices, false);
    m_halfedge_from_vertex = std::vector<int>(num_vertices, -1);

    // Initialize vertex cost with value above maximum possible weight
    m_vertex_cost = std::vector<Scalar>(num_vertices, inf_cost);
    m_vertices_to_process = PriorityQueue();
}

void VoronoiOverlayMesh::PathTracerCommonList::initialize()
{
    // Initialize an array to keep track of vertices
    int num_vertices = parent.out.size();
    m_is_processed_vertex.resize(num_vertices, false);
    m_halfedge_from_vertex.resize(num_vertices, -1);

    // Initialize vertex cost with value above maximum possible weight
    m_vertex_cost.resize(num_vertices, inf_cost);
    m_vertices_to_process = PriorityQueue();
    m_modified_vertices.clear();
    assert(m_vertex_cost[argmin(m_vertex_cost)] == inf_cost);
    assert(m_halfedge_from_vertex[argmax(m_halfedge_from_vertex)] == -1);
}

void VoronoiOverlayMesh::PathTracerCommonList::clean()
{
    // Initialize an array to keep track of vertices
    for (int vi : m_modified_vertices)
    {
        m_is_processed_vertex[vi] = false;
        m_halfedge_from_vertex[vi] = -1;
        m_vertex_cost[vi] = inf_cost;
    }
}

bool VoronoiOverlayMesh::PathTracer::trace_path(
    int first_vertex,
    int second_vertex,
    int halfedge_index,
    int edge_index,
    std::vector<int> &halfedges,
    int face_index)
{
    // nothing to do if same vertex
    if (first_vertex == second_vertex) return true;

    // check if valid start
    assert(parent.is_valid_vertex_index(first_vertex));
    assert(parent.is_valid_vertex_index(second_vertex));
    if (parent.is_vertex_deleted(first_vertex))
    {
        spdlog::error("tracing from deleted vertex");
        return false;
    }

    initialize();

    // Mark root with 0 (or cost lower than other edges for all negative weights)
    set_vertex_cost(first_vertex, 0.);
    add_vertex_to_queue(first_vertex);

    // Perform Dijkstra algorithm
    bool vertex_found = false;
    bool terminate_early = false;
    Scalar min_cost = inf_cost; // keep track of best cost so far
    while (!is_vertex_queue_empty())
    {
        // Get the next vertex to process
        auto [vi, vi_cost] = get_vertex_to_process();

        // Skip already processed vertices
        if (is_processed_vertex(vi))
            continue;
        mark_processed_vertex(vi);

        // Iterate over the vertex circulator via halfedges
        int h_start = parent.out[vi];
        int hij = h_start;
        do
        {
            // Get the vertex in the one ring at the tip of the halfedge
            int vj = parent.to[hij];

            // Get candidate edge cost (either path length or edge weight)
            Scalar candidate_cost = parent.halfedge_length(hij) + get_vertex_cost(vi);

            bool is_invalid = false;
            is_invalid = (is_invalid) || (is_processed_vertex(vj));                                   // don't process same vertex twice
            is_invalid = (is_invalid) || (parent.simp.edge_type[parent.h2e[hij]] != EdgeType::EF);    // don't use traced edges
            is_invalid = (is_invalid) || (parent.voronoi.edge_type[parent.h2e[hij]] != EdgeType::EF); // don't cross voronoi boundaries
            is_invalid = (is_invalid) || (get_vertex_cost(vj) < candidate_cost);                      // don't use longer paths
            is_invalid = (is_invalid) || (min_cost < candidate_cost);                                 // don't use longer paths

            if (face_index >= 0)
            {
                is_invalid = (is_invalid) || (parent.simp.halfedge_parent[hij] != face_index); // stay in the target face
            }

            // if processing edge from the first vertex, check for consistent cyclic orientation
            if (vi == first_vertex)
            {
                is_invalid = (is_invalid) || (!parent.is_start_cyclic_orientation(hij, halfedge_index)); // maintain consistent orientation
            }

            // if processing edge to the second vertex, check for consistent cyclic orientation
            if (vj == second_vertex)
            {
                is_invalid = (is_invalid) || (!parent.is_end_cyclic_orientation(hij, halfedge_index)); // maintain consistent orientation
            }
            // only allow intersections at first and last vertex
            if (vj != second_vertex)
            {
                is_invalid = (is_invalid) || (parent.simp.vertex_type[vj] != VertexType::VF);    // don't use traced vertices
                is_invalid = (is_invalid) || (parent.voronoi.vertex_type[vj] != VertexType::VF); // don't use voronoi vertices
            }

            // add current candidate to record if it is valid
            if (!is_invalid)
            {
                set_halfedge_from_vertex(vj, hij);
                set_vertex_cost(vj, candidate_cost);
                add_vertex_to_queue(vj);
                if (vj == second_vertex)
                {
                    vertex_found = true;
                    min_cost = min(min_cost, candidate_cost);
                }
            }

            // Progress to the next halfedge in the vertex circulator
            hij = parent.opp[parent.prev[hij]];
        } while (hij != h_start);

        if ((vertex_found) && terminate_early)
            break;
    }

    if (get_vertex_cost(second_vertex) == inf_cost)
    {
        spdlog::error("could not find path from {} to {}", first_vertex, second_vertex);

        if (show_failure)
        {
#ifdef ENABLE_VISUALIZATION
            parent.view();
            std::string mesh_handle = "overlay mesh";
            auto [v_new2old, v_old2new] = parent.get_vertex_reductions();
            std::vector<Scalar> cost(v_new2old.size());
            for (int i = 0; i < v_new2old.size(); ++i)
            {
                cost[i] = get_vertex_cost(v_new2old[i]);
            }
            std::vector<Scalar> traced_vertices(v_new2old.size(), 0.);
            traced_vertices[v_old2new[first_vertex]] = 1.;
            traced_vertices[v_old2new[second_vertex]] = 2.;
            polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("cost", cost);
            polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("path vertices", traced_vertices);
            polyscope::show();
#endif
        }

        clean();
        return false;
    }

    // trace back path and mark all edges and vertex parents
    int hjk = get_halfedge_from_vertex(second_vertex);
    // start_halfedge[m.opp[halfedge_index]] = opp[hjk];
    while (true)
    {
        // mark edge parent
        int ejk = parent.h2e[hjk];
        parent.simp.edge_type[ejk] = EdgeType::EE;
        parent.simp.halfedge_parent[hjk] = halfedge_index;
        // TODO Make proper edge index

        // add edges to list
        halfedges.push_back(hjk);

        // check if at end
        int vj = parent.to[parent.opp[hjk]];
        if (vj == first_vertex)
        {
            // start_halfedge[halfedge_index] = hjk;
            break;
        }
        // TODO: barycentric coordinates

        // iterate if still interior
        parent.simp.vertex_type[vj] = VertexType::VE;
        parent.simp.vertex_parent[vj] = edge_index;
        hjk = get_halfedge_from_vertex(vj);
    }

    clean();
    return true;
}

bool VoronoiOverlayMesh::trace_path(
    int first_vertex,
    int second_vertex,
    int halfedge_index,
    int edge_index,
    std::vector<int> &halfedges,
    int face_index)
{
    path_tracer.show_failure = show_trace_failure;
    return path_tracer.trace_path(first_vertex, second_vertex, halfedge_index, edge_index, halfedges, face_index);
}

void VoronoiOverlayMesh::add_voronoi_edges_to_viewer(
    const Eigen::MatrixXd &V,
    const std::vector<int> &vtx_reindex) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    int num_edges = e2h.size();
    std::vector<std::array<int, 2>> edges;
    std::vector<int> parent;
    edges.reserve(count_edges());
    for (int eij = 0; eij < num_edges; ++eij)
    {
        // only add edges contained in simpified mesh edges
        if (is_edge_deleted(eij))
            continue;
        if (voronoi.edge_type[eij] != EdgeType::EE)
            continue;

        int hij = e2h[eij];
        int vi = v_old2new[to[opp[hij]]];
        int vj = v_old2new[to[hij]];
        edges.push_back({vi, vj});
        parent.push_back(orig.halfedge_parent[hij]); // TODO: get edge in original mesh
    }
    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    auto [V_s, edges_s] = remove_unreferenced_vertices(V_o, edges);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle;
    mesh_handle = "voronoi edges";
    polyscope::registerCurveNetwork(mesh_handle, V_s, edges_s)
        ->setColor(BLACK)
        ->setRadius(0.0005);
    polyscope::getCurveNetwork(mesh_handle)->addEdgeScalarQuantity("edge parent", parent);
#endif
}

void VoronoiOverlayMesh::add_voronoi_points_to_viewer(
    const Eigen::MatrixXd &V,
    const std::vector<int> &vtx_reindex) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    std::vector<int> dual_vertices_new = vector_compose(v_old2new, voronoi.dual_vertices);
    std::vector<int> dual_edge_midpoints_new = vector_compose(v_old2new, voronoi.dual_edge_midpoints);
    Eigen::MatrixXd V_o = generate_overlay_vertices(V, vtx_reindex, orig);
    Eigen::MatrixXd V_vertex = subset_vertices(V_o, dual_vertices_new);
    Eigen::MatrixXd V_edge = subset_vertices(V_o, dual_edge_midpoints_new);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerPointCloud("dual vertices", V_vertex);
    polyscope::registerPointCloud("dual edge midpoints", V_edge);
#endif
}

void VoronoiOverlayMesh::add_voronoi_to_viewer(
    const std::string &mesh_handle,
    const VoronoiStructure &voronoi_structure) const
{
    auto [v_new2old, v_old2new] = get_vertex_reductions();
    auto [f_new2old, f_old2new] = get_face_reductions();
    std::vector<int> vertex_region = vector_compose(voronoi_structure.vertex_region, v_new2old);
    std::vector<int> face_region = vector_compose(voronoi_structure.face_region, f_new2old);
    std::vector<VertexType> vertex_type = vector_compose(voronoi_structure.vertex_type, v_new2old);
    std::vector<int> vertex_enum = enumerate_vertex_type(vertex_type);
    std::vector<Scalar> distance = vector_compose(voronoi_structure.distance, v_new2old);
    std::vector<int> shuffled_face_region = shuffle_map_image(face_region);
#ifdef ENABLE_VISUALIZATION
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("distance", distance);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("vertex region", vertex_region);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("vertex type", vertex_enum);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("face region", face_region);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("shuffled face region", shuffled_face_region);
#endif
}

void VoronoiOverlayMesh::view() const
{
    std::string mesh_handle = "overlay mesh";
    add_original_to_viewer(V_orig, vtx_reindex_orig);
    add_mesh_to_viewer(V_orig, vtx_reindex_orig);
    add_simplified_vertices_to_viewer(V_orig, vtx_reindex_orig);
    add_simplified_edges_to_viewer(V_orig, vtx_reindex_orig);
    if (voronoi.mesh.n.size() > 0) // TODO
    {
        add_voronoi_edges_to_viewer(V_orig, vtx_reindex_orig);
        add_voronoi_points_to_viewer(V_orig, vtx_reindex_orig);
        add_voronoi_to_viewer(mesh_handle, voronoi);
    }
#ifdef ENABLE_VISUALIZATION
    polyscope::show();
#endif
}

void VoronoiOverlayMesh::add_center_to_face_voronoi(
    int vertex_index,
    int face_index,
    VoronoiStructure &voronoi_structure)
{
    int f_parent = voronoi_structure.face_region[face_index];
    voronoi_structure.vertex_type[vertex_index] = VertexType::VF;
    voronoi_structure.vertex_region[vertex_index] = f_parent;
    int hij = out[vertex_index];
    voronoi_structure.distance[vertex_index] = voronoi_structure.distance[to[hij]];
    int hik = hij;
    do
    {
        int eik = h2e[hik];
        voronoi_structure.edge_type[eik] = EdgeType::EF;
        voronoi_structure.edge_region[eik] = f_parent;
        voronoi_structure.face_region[face[hik]] = f_parent;
        voronoi_structure.distance[vertex_index] = max(
            voronoi_structure.distance[to[hij]],
            voronoi_structure.distance[vertex_index]);
        hik = opp[prev[hik]];
    } while (hik != hij);
}

void VoronoiOverlayMesh::split_edge_voronoi(
    int halfedge_index,
    int edge_index,
    VoronoiStructure &voronoi_structure)
{
    int hij = halfedge_index;
    EdgeType e_type = voronoi_structure.edge_type[edge_index];
    int e_parent = voronoi_structure.edge_region[edge_index];
    voronoi_structure.edge_type[h2e[hij]] = e_type;
    voronoi_structure.edge_type[h2e[n[hij]]] = e_type;
    voronoi_structure.edge_region[h2e[hij]] = e_parent;
    voronoi_structure.edge_region[h2e[n[hij]]] = e_parent;
    int vj = to[hij];
    int vi = to[opp[hij]];
    int vk = to[n[hij]];
    voronoi_structure.vertex_region[vj] = e_parent;
    voronoi_structure.vertex_type[vj] = (e_type == EdgeType::EE) ? VertexType::VE : VertexType::VF;
    voronoi_structure.distance[vj] = max(voronoi_structure.distance[vi], voronoi_structure.distance[vk]);
}

int VoronoiOverlayMesh::split_face_voronoi(
    int e,
    int face_index,
    VoronoiStructure &voronoi_structure)
{
    int f_parent = voronoi_structure.face_region[face_index];
    voronoi_structure.edge_type[e] = EdgeType::EF;
    int hij = e2h[e];
    int hji = opp[hij];
    voronoi_structure.face_region[face[hij]] = f_parent;
    voronoi_structure.face_region[face[hji]] = f_parent;

    return e;
}

void VoronoiOverlayMesh::collect_garbage()
{
    if (skip_garbage_collection) return;

    // get element reindex maps
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [v_new2old, v_old2new] = get_vertex_reductions();
    auto [e_new2old, e_old2new] = get_edge_reductions();
    auto [f_new2old, f_old2new] = get_face_reductions();

    // collect voronoi garbage
    voronoi.collect_garbage(he_new2old, v_new2old, e_new2old, f_new2old, he_old2new, v_old2new);

    // do here to avoid redundant map computation
    orig.collect_garbage(he_new2old, v_new2old, e_new2old, f_new2old);
    simp.collect_garbage(he_new2old, v_new2old, e_new2old, f_new2old);
    HalfedgeMesh::collect_garbage(
        he_new2old,
        e_new2old,
        v_new2old,
        f_new2old,
        he_old2new,
        e_old2new,
        v_old2new,
        f_old2new);
}

// WARNING: not debugged yet
void VoronoiOverlayMesh::parameterize_with_layout(
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa
)
{
    assert(is_valid());
    timer.start();
    if (!use_zero_lengths)
    {
        set_simplified_halfedge_extrinsic_lengths();
        correct_simplified_halfedge_lengths();
    }
    auto [V_overlay, m_o] = generate_simplified_parameterization(homology_basis_loops, kappa);
    log["4.1-param-penner-time"] = timer.getElapsedTime();

    // find coarse mesh layout
    timer.start();
    spdlog::info("generating coarse layout");
    auto &mc = m_o.cmesh();
    std::vector<Scalar> u_vec(mc.n_vertices(), 0.);
    auto layout_res = get_consistent_layout(simp.m, m_o);
    auto m_tri = std::get<0>(layout_res);
    u_simp = std::get<4>(layout_res);
    v_simp = std::get<5>(layout_res);
    is_cut_simp = std::get<6>(layout_res);
    assert(u_simp.size() == mc.n_halfedges());
    assert(v_simp.size() == mc.n_halfedges());
    assert(is_cut_simp.size() == mc.n_halfedges());

    // get face map
    std::vector<int> fn_to_f;
    build_face_maps(m_o, fn_to_f);

    // get halfedge maps for current mesh
    std::vector<int> h2e_tri, e2h_tri;
    build_edge_maps(mc.opp, h2e_tri, e2h_tri);

    if (false)
        view_halfedge_mesh_layout(mc, u_simp, v_simp);

    bool check_validity = true;
    if (check_validity)
    {
        VectorX he2angle, he2cot;
        corner_angles(m_tri, he2angle, he2cot);
        spdlog::info("min angle in coarse overlay: {}", he2angle.minCoeff());
    }
    spdlog::info("original coarse mesh has {} halfedges", simp.m.n_halfedges());
    spdlog::info("param overlay mesh has {} halfedges", m_o.n_halfedges());
    spdlog::info("triangulated param mesh has {} halfedges", m_tri.n_halfedges());

    // build map from vertices to original halfedge pointing to them
    std::vector<int> first_segment(simp.m.n_halfedges(), -1);
    for (int hij = 0; hij < m_o.n_halfedges(); ++hij)
    {
        // only look for original edges
        if (m_o.edge_type[hij] == CURRENT_EDGE)
            continue;

        // only look for segment vertices
        int vi = m_o.to[m_o.opp[hij]];
        int vj = m_o.to[hij];

        int Hij = m_o.origin_of_origin[hij];
        if (m_o.edge_type[hij] == ORIGINAL_EDGE)
            Hij = m_o.origin[hij];
        if (m_o.vertex_type[vi] == ORIGINAL_VERTEX)
            first_segment[Hij] = hij;
    }

    // Refine halfedges
    std::vector<int> start_halfedge = find_start_halfedges(simp);
    spdlog::info("splitting original overlay edges");
    std::vector<int> v2v(m_o.n_vertices(), -1);
    std::vector<int> original_edges = {}; // record of edges to delete
    std::vector<int> original_vertices = {}; // record of edges to delete
    for (int hij = 0; hij < simp.m.n_halfedges(); ++hij)
    {
        if (hij < simp.m.opp[hij])
            continue; // only process once
        spdlog::debug("tracing original edge {}, {}", hij, simp.m.opp[hij]);

        // get the length of the edge and the total length of the path on the overlay
        int h_curr = start_halfedge[hij];
        int h = first_segment[hij];
        Scalar edge_length = m_tri.l[h];
        int num_segments = 1;
        while (m_o.vertex_type[m_o.to[h]] != ORIGINAL_VERTEX)
        {
            h = next_orig_segment(m_o, h);
            edge_length += m_tri.l[h];
            ++num_segments;
        }
        h = first_segment[hij];
        Scalar delta = 1. / num_segments;
        Scalar path_length = compute_simplified_halfedge_length(h_curr);

        // iterate over overlay edges to insert new vertices on the correct halfedge
        v2v[m_o.to[m_o.opp[h]]] = to[opp[h_curr]];
        Scalar edge_tip_pos = m_tri.l[h]; // FIXME risky to use inconsistent m_o and m_tri
        Scalar path_base_pos = 0.;
        int count = 0;
        while (m_o.vertex_type[m_o.to[h]] != ORIGINAL_VERTEX)
        {
            // uniform distributes intersections, but increases parameterization distortion
            bool use_uniform_intersections = false;

            // compute target edge coordinate
            Scalar edge_coord;
            if (use_uniform_intersections)
            {
                edge_coord = (count + 1) * delta;
            }
            else
            {
                edge_coord = edge_tip_pos / edge_length;
            }
            spdlog::debug("tracing overlay edge {}, {} with target coordinate {}", h, m_o.opp[h], edge_coord);

            // increment path halfedge until approximately at the right halfedge
            while (path_base_pos / path_length  < edge_coord)
            {
                // stop if at last halfedge in path
                if (simp.vertex_type[to[h_curr]] == VertexType::VV)
                    break;

                // increment to next overlay edge otherwise
                path_base_pos += halfedge_length(h_curr);
                original_edges.push_back(h2e[h_curr]);
                original_vertices.push_back(to[h_curr]);
                assert(m_o.edge_type[h] == ORIGINAL_EDGE);

                h_curr = get_next_overlay_edge(h_curr, simp);
                assert(simp.edge_type[h2e[h_curr]] == EdgeType::EE);
            }

            // split edge and set endpoint vertex
            h_curr = split_edge(h_curr);
            int h_next = n[h_curr]; // need to get before splitting
            int h_opp = opp[n[h_curr]];
            split_face(h_curr, n[n[h_curr]]);
            split_face(h_opp, n[n[h_opp]]);

            original_edges.push_back(h2e[h_curr]);
            assert(m_o.edge_type[h] == ORIGINAL_EDGE);
            v2v[m_o.to[h]] = to[h_curr];
            spdlog::debug("new overlay vertex {} at {}", m_o.to[h], to[h_curr]);

            // increment both halfedges and lengths
            path_base_pos += halfedge_length(h_curr);
            h_curr = h_next;
            h = next_orig_segment(m_o, h);
            assert(m_o.edge_type[h] != CURRENT_EDGE);
            edge_tip_pos += m_tri.l[h];
            ++count;
        }
        // start_halfedge[h] = h_curr;
        // start_halfedge[m_o.opp[h]] = -1; // get last halfedge in chain

        // label parent of remaining edges
        spdlog::debug("tracing remaining overlay edge {}, {}", h, m_o.opp[h]);
        std::vector<int> remaining_halfedges = enumerate_halfedges(h_curr, simp);
        int num_remaining_halfedges = remaining_halfedges.size();
        for (int h_path : remaining_halfedges)
        {
            if (m_o.edge_type[h] == ORIGINAL_EDGE)
            {
                original_edges.push_back(h2e[h_path]);
            }
            else
            {
                simp.halfedge_parent[h_path] = m_o.origin[h];
                simp.halfedge_parent[opp[h_path]] = m_o.origin[m_o.opp[h]];
            }
        }
        for (int i = 0; i < num_remaining_halfedges - 1; ++i)
        {
            int h_path = remaining_halfedges[i];
            if (m_o.edge_type[h] == ORIGINAL_EDGE)
            {
                original_vertices.push_back(to[h_path]);
            }
            else
            {
                simp.vertex_parent[to[h_path]] = h2e_tri[m_o.origin[h]]; // TODO not for last
            }
        }
    }

    // restore regular triangulation
    triangulate();
    make_regular();

    // trace current edges not in the original mesh
    spdlog::info("tracing current overlay edges");
    for (int hij = 0; hij < m_o.n_halfedges(); ++hij)
    {
        if (hij < m_o.opp[hij])
            continue; // only process once
        if (m_o.edge_type[hij] != CURRENT_EDGE)
            continue; // only add current edges
        spdlog::debug("tracing overlay edge {}, {}", hij, m_o.opp[hij]);
        int vi = m_o.to[m_o.opp[hij]];
        int vj = m_o.to[hij];
        int Vi = v2v[vi];
        int Vj = v2v[vj];

        int f = fn_to_f[m_o.f[hij]];

        std::vector<int> halfedges;
        trace_path(Vi, Vj, m_o.origin[hij], h2e_tri[m_o.origin[hij]], halfedges, f);
        for (int h : halfedges)
        {
            simp.halfedge_parent[opp[h]] = m_o.origin[m_o.opp[hij]];
            assert(simp.face_parent[face[h]] == f);
            // if (simp.face_parent[face[h]] != f) spdlog::error("traced edge in wrong face");
        }

        // set vertices if not set yet
        if (simp.vertex_type[Vi] == VertexType::VE)
        {
            simp.vertex_parent[Vi] = h2e_tri[m_o.origin[hij]];
        }
        if (simp.vertex_type[Vj] == VertexType::VE)
        {
            simp.vertex_parent[Vj] = h2e_tri[m_o.origin[hij]];
        }
        // TODO set the start halfedge of the opposite path
        make_adjacent_regular(halfedges);
        // TODO refine
    }

    // remove original edges
    for (int e : original_edges)
    {
        simp.edge_type[e] = EdgeType::EF;
    }
    for (int v : original_vertices)
    {
        simp.vertex_type[v] = VertexType::VF;
    }

    // set simplified mesh to overlay
    simp.m = mc;
    simp.h2e = h2e_tri;
    simp.e2h = e2h_tri;
    log["4.2-param-refine-time"] = timer.getElapsedTime();
}


} // namespace Resolution
} // namespace Penner
