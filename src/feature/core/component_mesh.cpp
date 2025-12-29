
#include "feature/core/component_mesh.h"

#include "holonomy/core/dual_lengths.h"
#include "holonomy/holonomy/cones.h"

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/remove_unreferenced.h>

#include "util/vector.h"

namespace Penner {
namespace Feature {

ComponentMesh::ComponentMesh(const Mesh<Scalar>& m)
 : ComponentMesh(m, find_mesh_face_components(m))
{
}

ComponentMesh::ComponentMesh(
    const Mesh<Scalar>& m,
    const Eigen::VectorXi& components)
    : m_mesh_components({})
    , m_v_maps({})
    , m_f_maps({})
    , m_he_maps({})
    , m_f_proj({})
    , m_v_proj({})
    , m_he_proj({})
    , m_num_vertices(m.n_ind_vertices())
{
    // construct each component individually
    int num_components = components.maxCoeff() + 1;
    for (int i = 0; i < num_components; ++i) {
        build_mesh_component(m, components, i);
    }
}

bool ComponentMesh::is_valid_component_mesh() const
{
    // TODO

    return true;
}

// build component halfedge arrays
void ComponentMesh::build_halfedge_data(
    Mesh<Scalar>& component_mesh,
    const Mesh<Scalar>& m,
    const std::vector<bool>& he_in_component,
    const std::vector<int>& component_he,
    const std::vector<int>& he_map,
    const std::vector<int>& v_map,
    const std::vector<int>& f_map)
{
    int n_component_he = component_he.size();
    component_mesh.n.resize(n_component_he);
    component_mesh.to.resize(n_component_he);
    component_mesh.f.resize(n_component_he);
    component_mesh.opp.resize(n_component_he);
    component_mesh.R.resize(n_component_he);
    component_mesh.type.resize(n_component_he);
    component_mesh.type_input.resize(n_component_he);
    component_mesh.l.resize(n_component_he);
    for (int chij = 0; chij < n_component_he; ++chij) {
        int hij = component_he[chij]; // halfedge index in original mesh
        int hji = m.opp[hij];
        component_mesh.n[chij] = he_map[m.n[hij]];
        component_mesh.to[chij] = v_map[m.to[hij]];
        component_mesh.f[chij] = f_map[m.f[hij]];
        component_mesh.opp[chij] = (he_in_component[hji]) ? he_map[m.opp[hij]] : -1;
        component_mesh.R[chij] = he_map[m.R[hij]];
        component_mesh.type[chij] = m.type[hij];
        component_mesh.type_input[chij] = m.type_input[hij];
        component_mesh.l[chij] = m.l[hij];
    }
}

// build component face arrays
void ComponentMesh::build_face_data(
    Mesh<Scalar>& component_mesh,
    const Mesh<Scalar>& m,
    const std::vector<int>& component_f,
    const std::vector<int>& he_map)
{
    int n_component_f = component_f.size();
    component_mesh.h.resize(n_component_f);
    for (int cfijk = 0; cfijk < n_component_f; ++cfijk) {
        int fijk = component_f[cfijk]; // face index in original mesh
        component_mesh.h[cfijk] = he_map[m.h[fijk]];
    }
}

// build component vertex arrays
void ComponentMesh::build_vertex_data(
    Mesh<Scalar>& component_mesh,
    const Mesh<Scalar>& m,
    const std::vector<int>& component_v,
    const std::vector<int>& he_map,
    const std::vector<int>& ind_v_map)
{
    int n_component_v = component_v.size();
    component_mesh.out.resize(n_component_v);
    component_mesh.v_rep.resize(n_component_v);
    for (int cvi = 0; cvi < n_component_v; ++cvi) {
        int vi = component_v[cvi]; // vertex index in original mesh
        component_mesh.out[cvi] = he_map[m.out[vi]];
        component_mesh.v_rep[cvi] = ind_v_map[m.v_rep[vi]];
    }
}

void ComponentMesh::build_independent_vertex_data(
    Mesh<Scalar>& component_mesh,
    const Mesh<Scalar>& m,
    const std::vector<int>& component_ind_v)
{
    // Build component independent vertex arrays
    int n_component_v_ind = component_ind_v.size();
    component_mesh.Th_hat.resize(n_component_v_ind);
    component_mesh.fixed_dof.resize(n_component_v_ind);
    for (int cvi = 0; cvi < n_component_v_ind; ++cvi) {
        int vi = component_ind_v[cvi]; // independent vertex index in original mesh
        component_mesh.Th_hat[cvi] = m.Th_hat[vi];
        component_mesh.fixed_dof[cvi] = m.fixed_dof[vi];
    }
}

// build component mesh for component with the given index
void ComponentMesh::build_mesh_component(
    const Mesh<Scalar>& m,
    const Eigen::VectorXi& components,
    int component_index)
{
    Mesh<Scalar> component_mesh;

    // Build lists of attribute counts
    int n_he = m.n_halfedges();
    int n_v = m.n_vertices();
    int n_f = m.n_faces();
    int n_ind_v = m.n_ind_vertices();
    std::vector<bool> he_in_component(n_he, false);
    std::vector<bool> v_in_component(n_v, false);
    std::vector<bool> f_in_component(n_f, false);
    std::vector<bool> ind_v_in_component(n_ind_v, false);
    for (int hij = 0; hij < n_he; ++hij) {
        // Ignore halfedges not in face component
        if (components[m.f[hij]] != component_index) continue;

        // Add mesh elements associated to halfedge
        he_in_component[hij] = true;
        v_in_component[m.to[hij]] = true;
        f_in_component[m.f[hij]] = true;
        ind_v_in_component[m.v_rep[m.to[hij]]] = true;
    }

    // Build index arrays and maps for the mesh components
    std::vector<int> component_he, component_v, component_f, component_ind_v;
    convert_boolean_array_to_index_vector(he_in_component, component_he);
    convert_boolean_array_to_index_vector(v_in_component, component_v);
    convert_boolean_array_to_index_vector(f_in_component, component_f);
    convert_boolean_array_to_index_vector(ind_v_in_component, component_ind_v);
    std::vector<int> he_map = index_subset(n_he, component_he);
    std::vector<int> v_map = index_subset(n_v, component_v);
    std::vector<int> f_map = index_subset(n_f, component_f);
    std::vector<int> ind_v_map = index_subset(n_ind_v, component_ind_v);

    // build component data
    build_halfedge_data(component_mesh, m, he_in_component, component_he, he_map, v_map, f_map);
    build_face_data(component_mesh, m, component_f, he_map);
    build_vertex_data(component_mesh, m, component_v, he_map, ind_v_map);
    build_independent_vertex_data(component_mesh, m, component_ind_v);

    // add data to lists
    m_mesh_components.push_back(component_mesh);
    m_he_maps.push_back(component_he);
    m_f_maps.push_back(component_f);
    m_v_maps.push_back(component_ind_v);
    m_he_proj.push_back(he_map);
    m_f_proj.push_back(f_map);
    m_v_proj.push_back(v_map);
}

// partial validity check
// TODO: augment with full checks and make more global
bool is_valid_overlay_mesh(
    const Eigen::MatrixXi& F,
    const std::vector<int>& fn_to_f,
    const std::vector<std::pair<int, int>>& endpoints)
{
    // check element counts
    int num_faces = F.rows();
    int num_vertices = F.maxCoeff() + 1;
    if (static_cast<int>(fn_to_f.size()) != num_faces) {
        spdlog::error("Inconsistent number of faces in overlay mesh");
        return false;
    }
    if (static_cast<int>(endpoints.size()) != num_vertices) {
        spdlog::error("Inconsistent number of vertices in overlay mesh");
        return false;
    }

    // check each face seen at least once
    int num_orig_faces = vector_max(fn_to_f) + 1;
    std::vector<bool> is_face_seen(num_orig_faces, false);
    for (int f : fn_to_f) {
        is_face_seen[f] = true;
    }
    for (int f = 0; f < num_orig_faces; ++f) {
        if (!is_face_seen[f]) {
            spdlog::error("Original face {} not seen", f);
            return false;
        }
    }

    return true;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<int>, std::vector<std::pair<int, int>>>
ComponentMesh::combine_refined_components(
    const std::vector<int>& face_reindex,
    std::vector<Eigen::MatrixXd>& mesh_vertex_components,
    std::vector<Eigen::MatrixXi>& mesh_face_components,
    std::vector<std::vector<int>>& fn_to_f_components,
    std::vector<std::vector<std::pair<int, int>>>& endpoint_components) const
{
    // get total mesh sizes
    int dim = mesh_vertex_components.front().cols();
    int num_vertices = count_total_vertices(mesh_vertex_components);
    int num_faces = count_total_faces(mesh_face_components);

    // build combined vertex positions
    // TODO: Can separate this complete method into five modular methods (vn_to_v separate)
    Eigen::MatrixXd V(num_vertices, dim);
    int num_components = mesh_vertex_components.size();
    int original_vertex_count = m_num_vertices;
    int refined_vertex_count = 0;
    std::vector<std::vector<int>> vn_to_v(num_components);
    for (int i = 0; i < num_components; ++i) {
        const auto& V_c = mesh_vertex_components[i];
        const auto& endpoints_component = endpoint_components[i];
        const auto& v_map = m_v_maps[i];

        // build global vertex position matrix with local to global index maps
        int num_component_vertices = V_c.rows();
        vn_to_v[i] = std::vector<int>(num_component_vertices, -1);
        for (int vi = 0; vi < num_component_vertices; ++vi) {
            const auto& endpoints = endpoints_component[vi];

            // case: inserted vertex; add new global vertex index
            if ((endpoints.first >= 0) || (endpoints.second >= 0)) {
                vn_to_v[i][vi] = original_vertex_count + refined_vertex_count;
                ++refined_vertex_count;
            }
            // case: original vertex; find existing global vertex index
            else {
                vn_to_v[i][vi] = v_map[vi];
            }

            // set global vertex position from local component position
            V.row(vn_to_v[i][vi]) = V_c.row(vi);
        }
    }

    // combine component faces (reindexed to global indices)
    Eigen::MatrixXi F(num_faces, 3);
    int face_count = 0;
    for (int i = 0; i < num_components; ++i) {
        const auto& F_c = mesh_face_components[i];
        int num_component_faces = F_c.rows();
        for (int f = 0; f < num_component_faces; ++f) {
            for (int j = 0; j < 3; ++j) {
                int vj = F_c(f, j);
                F(face_count, j) = vn_to_v[i][vj];
            }
            ++face_count;
        }
    }

    // build global refined to original face mappings
    std::vector<int> fn_to_f = {};
    fn_to_f.reserve(num_faces);
    for (int i = 0; i < num_components; ++i) {
        const auto& fn_to_f_component = fn_to_f_components[i];
        for (int fijk : fn_to_f_component) {
            int f_global = m_f_maps[i][fijk]; // global mesh face index
            fn_to_f.push_back(face_reindex[f_global]);
        }
    }

    // build global remapped endpoints
    std::vector<std::pair<int, int>> endpoints(num_vertices);
    for (int i = 0; i < num_components; ++i) {
        const auto& endpoints_component = endpoint_components[i];
        int num_component_vertices = endpoints_component.size();
        for (int vi = 0; vi < num_component_vertices; ++vi) {
            // get local component endpoints
            const auto& endpoints_pair = endpoints_component[vi];

            // lambda to do local to global vertex remapping
            auto remap_vertex = [&](int v) {
                if (v < 0) return v;
                return vn_to_v[i][v];
            };

            // set global vertex endpoints with remapping
            int e0 = remap_vertex(endpoints_pair.first);
            int e1 = remap_vertex(endpoints_pair.second);
            endpoints[vn_to_v[i][vi]] = std::make_pair(e0, e1);
        }
    }

#ifdef CHECK_VALIDITY
    if (!is_valid_overlay_mesh(F, fn_to_f, endpoints)) {
        spdlog::error("Could not combine components");
        return std::make_tuple(
            Eigen::MatrixXd(),
            Eigen::MatrixXi(),
            std::vector<int>(),
            std::vector<std::pair<int, int>>());
    }
#endif

    return std::make_tuple(V, F, fn_to_f, endpoints);
}

Eigen::VectorXi find_mesh_face_components(const Mesh<Scalar>& m)
{
    int num_halfedges = m.n_halfedges();
    int num_faces = m.n_faces();

    // Union component faces across edges
    UnionFind component_unions(num_faces);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m.opp[hij];
        int fi = m.f[hij];
        int fj = m.f[hji];
        component_unions.union_sets(fi, fj);
    }

    // Index the components
    std::vector<int> index = component_unions.index_sets();
    return Eigen::VectorXi::Map(&index[0], index.size());
}

Eigen::VectorXi find_mesh_vertex_components(const Mesh<Scalar>& m)
{
    int num_halfedges = m.n_halfedges();
    int num_vertices = m.n_vertices();

    // Union component vertices along edges 
    UnionFind component_unions(num_vertices);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m.opp[hij];
        int vi = m.to[hji];
        int vj = m.to[hij];
        component_unions.union_sets(vi, vj);
    }

    // Index the components
    std::vector<int> index = component_unions.index_sets();
    return Eigen::VectorXi::Map(&index[0], index.size());
}

VectorX compute_vertex_component_max(
    const Mesh<Scalar>& m,
    const Eigen::VectorXi& vertex_component,
    const VectorX& v)
{
    // initialize component max vector with global minimum value
    Scalar min_val = v.minCoeff();
    int num_components = vertex_component.maxCoeff() + 1;
    VectorX component_max = VectorX::Constant(num_components, min_val);

    // compute maximums over each component
    int num_vertices = m.n_vertices();
    for (int vi = 0; vi < num_vertices; ++vi) {
        int id = vertex_component[vi];
        component_max[id] = max(component_max[id], v[vi]);
    }

    return component_max;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<int>, Eigen::VectorXi> build_component(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& components,
    int component_index)
{
    // get faces in component
    int num_faces = F.rows();
    std::vector<int> component_f;
    component_f.reserve(num_faces);
    for (int f = 0; f < num_faces; ++f) {
        if (components[f] == component_index) component_f.push_back(f);
    }

    // Build new restricted face subset
    int num_component_faces = component_f.size();
    Eigen::MatrixXi F_comp(num_component_faces, 3);
    for (int i = 0; i < num_component_faces; ++i) {
        int f = component_f[i];
        F_comp.row(i) = F.row(f);
    }

    // Remove unreferenced vertices for component
    Eigen::MatrixXd V_new;
    Eigen::MatrixXi F_new;
    Eigen::VectorXi I, J;
    igl::remove_unreferenced(V, F_comp, V_new, F_new, I, J);

#ifdef CHECK_VALIDITY
    // Check manifold structure
    if (!igl::is_edge_manifold(F_new)) {
        spdlog::error("Mesh is not edge manifold");
    }
    Eigen::VectorXi B;
    if (!igl::is_vertex_manifold(F_new, B)) {
        spdlog::error("Mesh is not edge manifold");
    }
#endif

    return std::make_tuple(V_new, F_new, component_f, J);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> combine_mesh_components(
    std::vector<Eigen::MatrixXd>& mesh_vertex_components,
    std::vector<Eigen::MatrixXi>& mesh_face_components)
{
    // Get total mesh sizes
    int dim = mesh_vertex_components.front().cols();
    int num_vertices = count_total_vertices(mesh_vertex_components);
    int num_faces = count_total_faces(mesh_face_components);

    // Build combined mesh block by component block
    Eigen::MatrixXd V(num_vertices, dim);
    Eigen::MatrixXi F(num_faces, 3);
    int num_components = mesh_vertex_components.size();
    int vertex_count = 0;
    int face_count = 0;
    for (int i = 0; i < num_components; ++i) {
        const auto& V_c = mesh_vertex_components[i];
        const auto& F_c = mesh_face_components[i];

        // Add component blocks, offsetting vertex indices in the face list accordingly
        int dimension = V_c.cols();
        int num_component_vertices = V_c.rows();
        int num_component_faces = F_c.rows();
        V.block(vertex_count, 0, num_component_vertices, dimension) = V_c;
        F.block(face_count, 0, num_component_faces, 3) = F_c.array() + vertex_count;

        // Increment counts
        vertex_count += num_component_vertices;
        face_count += num_component_faces;
    }

    return std::make_pair(V, F);
}

void add_component_cone_pair(Mesh<Scalar>& m, Eigen::VectorXi vertex_component, int component, bool only_interior)
{
    Scalar angle_delta = M_PI;

    int num_vertices = m.n_vertices();
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (vertex_component[vi] != component) continue;

        // check if vertex is valid
        int Vi = m.v_rep[vi];
        if (m.Th_hat[Vi] < 2. * angle_delta) continue;
        if ((only_interior) && (!Holonomy::is_interior(m, vi))) continue;

        // check if adjacent vertex is valid
        int vj = m.to[m.out[vi]];
        int Vj = m.v_rep[vj];
        if ((only_interior) && (!Holonomy::is_interior(m, vj))) continue;

        // add cones
        spdlog::debug("Adding negative cone at {} with angle {}", Vi, m.Th_hat[Vi]);
        spdlog::debug("Adding positive cone at {} with angle {}", Vj, m.Th_hat[Vj]);
        m.Th_hat[Vi] -= angle_delta;
        m.Th_hat[Vj] += angle_delta;
        return;
    }

    // try again with interior if fails
    if (only_interior)
    {
        spdlog::warn("Cannot add cone pair in interior");
        add_component_cone_pair(m, vertex_component, component, false);
    } else {
        spdlog::error("Cannot add cone pair");
    }
}

} // namespace Feature
} // namespace Penner