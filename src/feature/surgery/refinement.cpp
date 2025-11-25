#include "feature/surgery/refinement.h"

#include "util/boundary.h"
#include "util/vector.h"
#include "holonomy/core/viewer.h"
#include "feature/core/component_mesh.h"

namespace Penner {
namespace Feature {

IntrinsicRefinementMesh::IntrinsicRefinementMesh(const Mesh<Scalar>& m)
    : m_mesh(m)
    , m_endpoints({})
{}

// Refine a single face, disregarding symmetry structure and independent vertices
//       k
//     / | \ .
//    /  |  \ .
//   /   l   \ .
//  /  /   \  \ .
// i ---------- j
int IntrinsicRefinementMesh::refine_single_face(int face_index)
{
    Mesh<Scalar>& m = get_mesh();

    // get halfedges and vertices of triangle by clockwise circulation
    int hij = m.h[face_index];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];

    // get new vertex, face and hafledge indices
    int vl = get_new_vertex();
    int fijl = face_index;
    int fjkl = get_new_face();
    int fkil = get_new_face();
    auto [hil, hli] = get_new_edge();
    auto [hjl, hlj] = get_new_edge();
    auto [hkl, hlk] = get_new_edge();

    // set connectivity for faces
    auto set_face = [&m](int h0, int h1, int h2, int f) {
        m.n[h0] = h1;
        m.n[h1] = h2;
        m.n[h2] = h0;
        m.f[h0] = f;
        m.f[h1] = f;
        m.f[h2] = f;
        m.h[f] = h0;
    };
    set_face(hij, hjl, hli, fijl);
    set_face(hjk, hkl, hlj, fjkl);
    set_face(hki, hil, hlk, fkil);

    // set connectivity for vertices
    m.to[hil] = vl;
    m.to[hli] = vi;
    m.to[hjl] = vl;
    m.to[hlj] = vj;
    m.to[hkl] = vl;
    m.to[hlk] = vk;
    m.out[vl] = hli;

    // set length of all edges to 1 and type to that of the face
    // TODO Use midpoint length; only topology used currently
    int type = m.type[hij];
    int type_input = m.type_input[hij];
    for (int h : {hil, hli, hjl, hlj, hkl, hlk}) {
        m.l[h] = 1.;
        m.type[h] = type;
        m.type_input[h] = type_input;
    }

    return vl;
}

int IntrinsicRefinementMesh::refine_face(int face_index)
{
    Mesh<Scalar>& m = get_mesh();

    // if closed mesh, just refine the face and return the new vertex index
    if (m.type[0] == 0)
    {
        int vl = refine_single_face(face_index);

        // set trivial reflection
        int h = m.n[m.out[vl]];
        for (int i = 0; i < 3; ++i) {
            int prev_Rh = m.n[m.n[m.R[h]]];
            int next_h = m.n[h];
            m.R[prev_Rh] = 0;
            m.R[next_h] = 0;
            m.R[m.opp[prev_Rh]] = 0;
            m.R[m.opp[next_h]] = 0;

            // iterate to next outer halfedge
            h = m.n[m.opp[m.n[h]]];
        }

        return get_new_independent_vertex(vl);
    }
    // handle doubled mesh by refining both paired faces and fix symmetry structure
    else
    {
        // refine primal and reflected face
        int vl = refine_single_face(face_index);
        int Rvl = refine_single_face(m.f[m.R[m.h[face_index]]]);

        // patch up halfedge symmetry structure
        int h = m.n[m.out[vl]];
        for (int i = 0; i < 3; ++i) {
            int prev_Rh = m.n[m.n[m.R[h]]];
            int next_h = m.n[h];
            m.R[prev_Rh] = next_h;
            m.R[next_h] = prev_Rh;
            m.R[m.opp[prev_Rh]] = m.opp[next_h];
            m.R[m.opp[next_h]] = m.opp[prev_Rh];

            // iterate to next outer halfedge
            h = m.n[m.opp[m.n[h]]];
        }

        return get_new_independent_vertex(vl, Rvl);
    }
}

// refine halfedge, without considering symmetry
int IntrinsicRefinementMesh::refine_single_halfedge(int halfedge_index)
{
    Mesh<Scalar>& m = get_mesh();

    // get halfedges and vertices of triangle by clockwise circulation
    int hij = halfedge_index;
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hji = m.opp[hij];
    int hil = m.n[hji];
    int hlj = m.n[hil];
    int fijk = m.f[hij];
    int fjil = m.f[hji];
    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];
    int vl = m.to[hil];

    // get new vertex, face and hafledge indices
    int vm = get_new_vertex();
    int fmjk = fijk;
    int fmki = get_new_face();
    int fmil = fjil;
    int fmlj = get_new_face();
    int him = hij;
    int hmi = hji;
    auto [hmj, hjm] = get_new_edge();
    auto [hmk, hkm] = get_new_edge();
    auto [hml, hlm] = get_new_edge();

    // set connectivity for faces
    auto set_face = [&m](int h0, int h1, int h2, int f) {
        m.n[h0] = h1;
        m.n[h1] = h2;
        m.n[h2] = h0;
        m.f[h0] = f;
        m.f[h1] = f;
        m.f[h2] = f;
        m.h[f] = h0;
    };
    set_face(hmj, hjk, hkm, fmjk);
    set_face(hmk, hki, him, fmki);
    set_face(hmi, hil, hlm, fmil);
    set_face(hml, hlj, hjm, fmlj);

    // set connectivity for vertices
    m.to[hmi] = vi;
    m.to[hmj] = vj;
    m.to[hmk] = vk;
    m.to[hml] = vl;
    m.to[him] = vm;
    m.to[hjm] = vm;
    m.to[hkm] = vm;
    m.to[hlm] = vm;
    m.out[vm] = hmi;
    m.out[vi] = him;
    m.out[vj] = hjm;
    m_endpoints[vm] = {vi, vj};

    // set length of all edges to 1 and type to that of the face
    //int type_0 = m.type[hij];
    //int type_input = m.type_input[hij];
    //Scalar l = 1.;
    // TODO Use midpoint length

    return vm;
}

int IntrinsicRefinementMesh::refine_halfedge(int halfedge_index)
{
    Mesh<Scalar>& m = get_mesh();

    // for closed mesh, just refine edge
    if (m.type[0] == 0)
    {
        int vl = refine_single_halfedge(halfedge_index);
        return get_new_independent_vertex(vl);
    }
    // TODO: double case
    else
    {
        spdlog::error("Halfedge refinement for double mesh not implemented");
        return -1;
    }
}


void IntrinsicRefinementMesh::refine_spanning_faces()
{
    // get mask for boundary vertices
    const auto& m = get_mesh();
    std::vector<bool> is_bd_vertex = compute_boundary_vertices(m);

    // check all faces
    for (int f = 0; f < m.n_faces(); ++f) {
        // skip double faces
        if (m.type[m.h[f]] == 2) continue;

        // get face vertices
        const auto& to = m.to;
        const auto& next = m.n;
        int h = m.h[f];
        int vi = to[h];
        int vj = to[next[h]];
        int vk = to[next[next[h]]];

        // refie face if all vertices are boundary
        if ((is_bd_vertex[vi]) && (is_bd_vertex[vj]) && (is_bd_vertex[vk])) {
            refine_face(f);
        }
    }
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> IntrinsicRefinementMesh::generate_mesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    const Mesh<Scalar>& m = get_mesh();

    // copy original vertices
    // WARNING: assumes new vertices added to back
    int num_orig_vertices = V.rows();
    int num_ref_vertices = m.n_ind_vertices();
    Eigen::MatrixXd V_ref(num_ref_vertices, 3);
    std::vector<bool> is_vertex_defined(num_ref_vertices, false);
    for (int vi = 0; vi < num_orig_vertices; ++vi) {
        V_ref.row(vi) = V.row(vtx_reindex[vi]);
        is_vertex_defined[vi] = true;
    }

    // generate new vertices at midpoints of faces
    int num_vertices = m.n_vertices();
    int all_vertices_defined = false;
    while (!all_vertices_defined) {
        all_vertices_defined = true; // assume all defined until proven otherwise

        // define any vertices possible
        for (int vm = 0; vm < num_vertices; ++vm) {
            // skip already defined vertices
            if (is_vertex_defined[m.v_rep[vm]]) continue;
            all_vertices_defined = false;

            // find neighbors of the vertex
            std::vector<int> neighbors = {};
            std::vector<int> defined_neighbors = {};
            int h_start = m.out[vm];
            int hij = h_start;
            do {
                int vj = m.to[hij];
                neighbors.push_back(vj);

                // check if defined neighbor position
                if (is_vertex_defined[m.v_rep[vj]]) defined_neighbors.push_back(vj);

                // circulate
                hij = m.n[m.opp[hij]];
            } while (hij != h_start);

            // check for vertices with three endpoints defined and use the face midpoint
            // WARNING: this assumes particular midpoint refinement scheme
            if ((defined_neighbors.size() == 3) && (neighbors.size() == 3)) {
                V_ref.row(m.v_rep[vm]) << 0., 0., 0.;
                for (int vj : defined_neighbors) {
                    V_ref.row(m.v_rep[vm]) += V_ref.row(m.v_rep[vj]) / 3.;
                }
                is_vertex_defined[m.v_rep[vm]] = true;
            }

            // check for vertices with four endpoints defined and use an edge midpoint determined by out
            // WARNING: this assumes particular midpoint refinement scheme
            if (neighbors.size() == 4) {
                auto [vi, vj] = m_endpoints.find(vm)->second;
                if (!is_vertex_defined[m.v_rep[vi]]) continue; 
                if (!is_vertex_defined[m.v_rep[vj]]) continue; 
                V_ref.row(m.v_rep[vm]) << 0., 0., 0.;
                V_ref.row(m.v_rep[vm]) += V_ref.row(m.v_rep[vi]) / 2.;
                V_ref.row(m.v_rep[vm]) += V_ref.row(m.v_rep[vj]) / 2.;
                is_vertex_defined[m.v_rep[vm]] = true;
            }
        }
    }

    // generate faces for original copy
    int num_ref_faces = m.n_faces();
    int num_faces = (m.type[0] == 0) ? num_ref_faces : num_ref_faces / 2;
    Eigen::MatrixXi F(num_faces, 3);
    int face_count = 0;
    for (int fijk = 0; fijk < num_ref_faces; ++fijk) {
        // skip double faces
        if (m.type[m.h[fijk]] == 2) continue;

        // Get halfedges of face
        int hij = m.h[fijk];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Get vertices of face
        int vj = m.v_rep[m.to[hij]];
        int vk = m.v_rep[m.to[hjk]];
        int vi = m.v_rep[m.to[hki]];

        // Write face with opposite vertex data
        F(face_count, 0) = vi;
        F(face_count, 1) = vj;
        F(face_count, 2) = vk;
        ++face_count;
    }

    return std::make_tuple(V_ref, F);
}

void IntrinsicRefinementMesh::view_refined_mesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [V_double, F] = generate_mesh(V, vtx_reindex);

#ifdef ENABLE_VISUALIZATION
    std::string mesh_handle = "refined mesh";
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F);
    polyscope::show();
#endif
}


// add new face
int IntrinsicRefinementMesh::get_new_face()
{
    Mesh<Scalar>& m = get_mesh();

    // get new face index
    int face_index = m.n_faces();

    // add null entries for face attributes
    m.h.push_back(-1);

    return face_index;
}

// add new vertex
int IntrinsicRefinementMesh::get_new_vertex()
{
    Mesh<Scalar>& m = get_mesh();

    // get new vertex index
    int vertex_index = m.n_vertices();

    // add null entries for vertex attributes
    m.out.push_back(-1);
    m.v_rep.push_back(-1);

    return vertex_index;
}

// add new edge with paired halfedges
std::pair<int, int> IntrinsicRefinementMesh::get_new_edge()
{
    Mesh<Scalar>& m = get_mesh();

    // get new halfedge indices
    int hij = m.n_halfedges();
    int hji = hij + 1;

    // add null entries for halfedges attributes
    for (int i = 0; i < 2; ++i) {
        m.n.push_back(-1);
        m.opp.push_back(-1);
        m.to.push_back(-1);
        m.f.push_back(-1);
        m.R.push_back(-1);
        m.type.push_back(-1);
        m.type_input.push_back(-1);
        m.l.push_back(-1);
    }

    // set opp directly
    m.opp[hij] = hji;
    m.opp[hji] = hij;

    return std::make_pair(hij, hji);
}

// add new independent vertex with flat angle and free dof corresponding to a closed mesh vertex
int IntrinsicRefinementMesh::get_new_independent_vertex(int vi)
{
    Mesh<Scalar>& m = get_mesh();

    // get new independent vertex index
    int Vi = m.n_ind_vertices();

    // set attributes for new vertex
    // assume flat and free dof
    m.v_rep[vi] = Vi;
    m.Th_hat.push_back(2. * M_PI);
    m.fixed_dof.push_back(false);

    return Vi;
}

// add new independent vertex with flat angle and free dof corresponding to paired double mesh vertices
int IntrinsicRefinementMesh::get_new_independent_vertex(int vi, int Rvi)
{
    Mesh<Scalar>& m = get_mesh();

    // get new independent vertex index
    int Vi = m.n_ind_vertices();

    // set attributes for new vertex
    // assume flat and free dof in doubled mesh
    m.v_rep[vi] = Vi;
    m.v_rep[Rvi] = Vi;
    m.Th_hat.push_back(4. * M_PI);
    m.fixed_dof.push_back(false);

    return Vi;
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<VertexEdge>> refine_corner_feature_faces(const FeatureFinder& feature_finder)
{
    const Mesh<Scalar>& m = feature_finder.get_mesh();
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();
    IntrinsicRefinementMesh refinement_mesh(m);

    // tag feature vertices
    std::vector<bool> is_feature_vertex(m.n_vertices(), false);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (feature_finder.is_feature_halfedge(hij))
        {
            is_feature_vertex[m.to[hij]] = true;
        }
    }

    // refine feature faces
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        // count number of feature vertices in face
        int num_feature_vertices = 0;
        int hij = m.h[fijk];
        for (int h : {hij, m.n[hij], m.n[m.n[hij]]})
        {
            if (is_feature_vertex[m.to[h]])
            {
                ++num_feature_vertices;
            }
        }

        // refine if all vertices are features
        if (num_feature_vertices == 3)
        {
            refinement_mesh.refine_face(fijk);
        }
    }

    // build list of feature edges
    std::vector<VertexEdge> feature_edges;
    feature_edges.reserve(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hji = m.opp[hij];
        if (hij < hji) continue;
        if (!feature_finder.is_feature_halfedge(hij)) continue;
        int vi = m.v_rep[m.to[hji]];
        int vj = m.v_rep[m.to[hij]];
        feature_edges.push_back({vi, vj});
    }

    // generate refined VF mesh
    const auto& V = feature_finder.get_vertex_positions();
    const auto& vtx_reindex = feature_finder.get_vertex_reindex();
    auto [V_ref, F_ref] = refinement_mesh.generate_mesh(V, vtx_reindex);

    return std::make_tuple(V_ref, F_ref, feature_edges);
}


void make_minimal_forest(const Mesh<Scalar>& m, std::vector<bool>& is_spanning_halfedge)
{
    int num_halfedges = m.n_halfedges();

    // compute initial vertex degrees
    std::vector<int> vertex_degrees(m.n_vertices(), 0);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (!is_spanning_halfedge[hij]) continue;
        int vj = m.v_rep[m.to[hij]];
        ++vertex_degrees[vj];
    }

    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (!is_spanning_halfedge[hij]) continue;

        // get edge endpoints
        int hji = m.opp[hij];
        int vi = m.v_rep[m.to[hji]];
        int vj = m.v_rep[m.to[hij]];

        // remove spanning edge if does not isolate vertex
        if ((vertex_degrees[vi] > 1) && (vertex_degrees[vj] > 1))
        {
            is_spanning_halfedge[hij] = false;
            is_spanning_halfedge[hji] = false;

            // update vertex degrees
            --vertex_degrees[vi];
            --vertex_degrees[vj];
        }
    }
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<VertexEdge>, std::vector<VertexEdge>> refine_feature_components(const FeatureFinder& feature_finder, bool use_minimal_forest)
{
    // initialize mesh for refinement
    const Mesh<Scalar>& m = feature_finder.get_mesh();
    IntrinsicRefinementMesh refinement_mesh(m);

    // get spanning forest
    std::vector<bool> is_spanning_halfedge = feature_finder.compute_feature_forest_halfedges();

    // if desired, use a smaller forest without isolated vertices
    int num_halfedges = m.n_halfedges();
    if (use_minimal_forest) make_minimal_forest(m, is_spanning_halfedge);

    // get cut mesh feature components
    UnionFind cut_features = feature_finder.compute_cut_feature_components();
    std::vector<std::vector<int>> components = cut_features.build_sets();
    std::vector<int> feature_labels = cut_features.index_sets();

    // make list of spanning features and check if all face components seen
    int num_components = cut_features.count_sets();
    std::vector<int> is_component_seen(num_components, false);
    std::vector<VertexEdge> spanning_edges = {};
    spanning_edges.reserve(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        // only consider spanning halfedges
        if (!is_spanning_halfedge[hij]) continue;

        // add spanning edge to list
        int hji = m.opp[hij];
        int vj = m.v_rep[m.to[hij]];
        int vi = m.v_rep[m.to[hji]];
        spanning_edges.push_back({vi, vj});

        // label component as seen
        is_component_seen[feature_labels[hij]] = true;
    }

    // refine edge on components with no adjacent spanning tree edges
    const auto& vtx_reindex = feature_finder.get_vertex_reindex();
    std::vector<bool> is_refined_halfedge(num_halfedges, false);
    std::vector<VertexEdge> feature_edges = {};
    feature_edges.reserve(num_halfedges);
    for (int ci = 0; ci < num_components; ++ci)
    {
        if (is_component_seen[ci]) continue;
        for (int h : components[ci])
        {
            if (!feature_finder.is_feature_halfedge(h)) continue; // skip nonfeature components
            spdlog::warn("No feature edge for component.");

            // refine arbitrary halfedge in feature
            int vk = refinement_mesh.refine_halfedge(h);

            // add spanning edge and feature edges for refined edge
            int vi = m.v_rep[m.to[m.opp[h]]];
            int vj = m.v_rep[m.to[h]];
            spanning_edges.push_back({vj, vk});
            feature_edges.push_back({vi, vk});
            feature_edges.push_back({vj, vk});
            spdlog::info("Refining {} with spanning edge {}, {}", h, vtx_reindex[vj], vk);

            // mark component and opposite component as seen
            is_component_seen[feature_labels[h]] = true;
            is_component_seen[feature_labels[m.opp[h]]] = true;
            is_refined_halfedge[h] = true;
            is_refined_halfedge[m.opp[h]] = true;
            break;
        }
    }

    // add feature edges that were not refined
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        // only process each unrefined feature edge once
        int hji = m.opp[hij];
        if (hij < hji) continue;
        if (!feature_finder.is_feature_halfedge(hij)) continue;
        if (is_refined_halfedge[hij]) continue;

        // add feature edge
        int vj = m.v_rep[m.to[hij]];
        int vi = m.v_rep[m.to[hji]];
        feature_edges.push_back({vi, vj});
    }

    // generate refined VF mesh
    const auto& V = feature_finder.get_vertex_positions();
    auto [V_ref, F_ref] = refinement_mesh.generate_mesh(V, vtx_reindex);
    
    return std::make_tuple(V_ref, F_ref, feature_edges, spanning_edges);
}



} // namespace Feature
} // namespace Penner