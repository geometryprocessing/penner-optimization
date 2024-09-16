
#include "optimization/core/viewer.h"

#include "util/vf_mesh.h"

#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/edge_flaps.h>
#include <igl/facet_components.h>
#include <igl/local_basis.h>
#include <igl/rotate_vectors.h>

#include <random>

#include "util/vector.h"
#include "util/map.h"
#include "util/vf_mesh.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {

#ifdef ENABLE_VISUALIZATION
glm::vec3 BEIGE(0.867, 0.765, 0.647);
glm::vec3 BLACK_BROWN(0.125, 0.118, 0.125);
glm::vec3 TAN(0.878, 0.663, 0.427);
glm::vec3 MUSTARD(0.890, 0.706, 0.282);
glm::vec3 FOREST_GREEN(0.227, 0.420, 0.208);
glm::vec3 TEAL(0., 0.375, 0.5);
glm::vec3 DARK_TEAL(0., 0.5*0.375, 0.5*0.5);
#endif

namespace Optimization {

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> generate_mesh_faces(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex)
{
    int num_faces = m.n_faces();
    Eigen::MatrixXi F(num_faces, 3);
    Eigen::MatrixXi F_halfedge(num_faces, 3);

    for (int fijk = 0; fijk < num_faces; ++fijk) {
        // Get halfedges of face
        int hij = m.h[fijk];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Get vertices of face
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];
        int vk = vtx_reindex[m.v_rep[m.to[hjk]]];
        int vi = vtx_reindex[m.v_rep[m.to[hki]]];

        // Write face with halfedge and opposite vertex data
        F_halfedge(fijk, 0) = hij;
        F_halfedge(fijk, 1) = hjk;
        F_halfedge(fijk, 2) = hki;
        F(fijk, 0) = vi;
        F(fijk, 1) = vj;
        F(fijk, 2) = vk;
    }

    return std::make_tuple(F, F_halfedge);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> generate_doubled_mesh(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex)
{
    int num_vertices = m.n_vertices();
    int num_faces = m.n_faces();

    // Copy and double vertices
    Eigen::MatrixXd V_double(num_vertices, 3);
    for (int vi = 0; vi < num_vertices; ++vi) {
        V_double.row(vi) = V.row(vtx_reindex[m.v_rep[vi]]);
    }

    // Generate doubled faces and halfedge map
    Eigen::MatrixXi F(num_faces, 3);
    Eigen::MatrixXi F_halfedge(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        // Get halfedges of face
        int hij = m.h[fijk];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Get vertices of face
        int vj = m.to[hij];
        int vk = m.to[hjk];
        int vi = m.to[hki];


        // Write face with halfedge and opposite vertex data
        F_halfedge(fijk, 0) = hij;
        F_halfedge(fijk, 1) = hjk;
        F_halfedge(fijk, 2) = hki;
        F(fijk, 0) = vi;
        F(fijk, 1) = vj;
        F(fijk, 2) = vk;
    }

    // Inflate vertices so they can be distinguished
    bool do_inflate_mesh = false;
    if (do_inflate_mesh) {
        V_double = inflate_mesh(V_double, F, 0.001);
    }

    return std::make_tuple(V_double, F, F_halfedge);
}

VectorX generate_FV_halfedge_data(const Eigen::MatrixXi& F_halfedge, const VectorX& halfedge_data)
{
    int num_faces = F_halfedge.rows();
    int num_halfedges = halfedge_data.size();
    VectorX FV_halfedge_data(num_halfedges);
    for (int f = 0; f < num_faces; ++f) {
        for (int i : {0, 1, 2}) {
            int h = F_halfedge(f, i);
            FV_halfedge_data(3 * f + i) = halfedge_data(h);
        }
    }

    return FV_halfedge_data;
}

Eigen::Vector3d generate_dual_vertex(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    int f)
{
    // generate face vertices
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int vi = m.v_rep[m.to[hki]];
    int vj = m.v_rep[m.to[hij]];
    int vk = m.v_rep[m.to[hjk]];

    // get face corner positions
    Eigen::Vector3d Vi = V.row(vtx_reindex[vi]);
    Eigen::Vector3d Vj = V.row(vtx_reindex[vj]);
    Eigen::Vector3d Vk = V.row(vtx_reindex[vk]);

    // compute midpoint
    Eigen::Vector3d midpoint = (Vi + Vj + Vk) / 3.;

    return midpoint;
}

Eigen::Vector3d generate_dual_edge_midpoint(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    int hij)
{
    // generate edge vertices
    int hji = m.opp[hij];
    int vi = m.v_rep[m.to[hji]];
    int vj = m.v_rep[m.to[hij]];

    // get edge endpoint positions
    Eigen::Vector3d Vi = V.row(vtx_reindex[vi]);
    Eigen::Vector3d Vj = V.row(vtx_reindex[vj]);

    // compute midpoint
    Eigen::Vector3d midpoint = (Vi + Vj) / 2.;

    return midpoint;
}

void view_dual_graph(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const std::vector<bool> is_edge)
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();

    // Get dual vertices (i.e., faces) in the dual graph
    std::vector<bool> is_dual_vertex(num_faces, false);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_edge[hij]) continue; // skip edges not in graph
        is_dual_vertex[m.f[hij]] = true;
    }
    std::vector<int> dual_vertices, dual_halfedges;
    convert_boolean_array_to_index_vector(is_dual_vertex, dual_vertices);
    convert_boolean_array_to_index_vector(is_edge, dual_halfedges);

    // Add all dual face vertices at endpoints of edges
    int num_dual_vertices = dual_vertices.size();
    int num_dual_halfedges = dual_halfedges.size();
    std::vector<int> dual_vertex_map(num_faces, -1);
    Eigen::MatrixXd dual_vertex_positions(num_dual_vertices, 3);
    Eigen::MatrixXd dual_node_positions(num_dual_vertices + num_dual_halfedges, 3);
    for (int i = 0; i < num_dual_vertices; ++i) {
        int f = dual_vertices[i];
        dual_vertex_map[f] = i;
        dual_vertex_positions.row(i) = generate_dual_vertex(V, m, vtx_reindex, f);
        dual_node_positions.row(i) = dual_vertex_positions.row(i);
    }

    // Generate dual graph network
    Eigen::MatrixXi edges(num_dual_halfedges, 2);
    for (int i = 0; i < num_dual_halfedges; ++i) {
        int hij = dual_halfedges[i];
        int f = m.f[hij];
        edges(i, 0) = dual_vertex_map[f];
        edges(i, 1) = num_dual_vertices + i;
        dual_node_positions.row(num_dual_vertices + i) =
            generate_dual_edge_midpoint(V, m, vtx_reindex, hij);
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerPointCloud("dual vertices", dual_vertex_positions)->setPointRadius(0.00025);
    polyscope::registerCurveNetwork("dual tree", dual_node_positions, edges)->setRadius(0.00015);
#endif
}

void view_primal_graph(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const std::vector<bool> is_edge,
    std::string handle,
    bool show)
{
    int num_vertices = m.n_ind_vertices();
    int num_halfedges = m.n_halfedges();

    if (show) {
        spdlog::info("Viewing primal graph {}", handle);
    }

    // Get vertices in the graph
    std::vector<bool> is_vertex(num_vertices, false);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (!is_edge[hij]) continue; // skip edges not in graph
        is_vertex[vtx_reindex[m.v_rep[m.to[hij]]]] = true;
    }
    std::vector<int> graph_vertices, graph_halfedges;
    convert_boolean_array_to_index_vector(is_vertex, graph_vertices);
    convert_boolean_array_to_index_vector(is_edge, graph_halfedges);

    // Add all vertices at endpoints of edges
    int num_graph_vertices = graph_vertices.size();
    int num_graph_halfedges = graph_halfedges.size();
    std::vector<int> graph_vertex_map(num_vertices, -1);
    Eigen::MatrixXd graph_vertex_positions(num_graph_vertices, 3);
    for (int i = 0; i < num_graph_vertices; ++i) {
        int v = graph_vertices[i];
        graph_vertex_map[v] = i;
        graph_vertex_positions.row(i) = V.row(v);
    }

    // Generate graph network
    Eigen::MatrixXi edges(num_graph_halfedges, 2);
    for (int i = 0; i < num_graph_halfedges; ++i) {
        int hij = graph_halfedges[i];
        int hji = m.opp[hij];
        int vi = vtx_reindex[m.v_rep[m.to[hji]]];
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];
        edges(i, 0) = graph_vertex_map[vi];
        edges(i, 1) = graph_vertex_map[vj];
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerPointCloud(handle + " graph vertices", graph_vertex_positions)
        ->setPointRadius(0.00025);
    polyscope::registerCurveNetwork(handle + " graph tree", graph_vertex_positions, edges)
        ->setRadius(0.00015);
    if (show) polyscope::show();
#endif
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& Th_hat)
{
    int num_vertices = Th_hat.size();
    std::vector<int> cone_indices;
    cone_indices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (float_equal(Th_hat[vi], 2 * M_PI)) continue;
        cone_indices.push_back(vi);
    }

    int num_cones = cone_indices.size();
    Eigen::MatrixXd cone_positions(num_cones, 3);
    Eigen::VectorXd cone_values(num_cones);
    for (int i = 0; i < num_cones; ++i) {
        int vi = cone_indices[i];
        cone_positions.row(i) = V.row(vi);
        cone_values[i] = (double)(Th_hat[vi]) - (2 * M_PI);
    }

    return std::make_tuple(cone_positions, cone_values);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& m)
{
    bool is_closed = (m.type[0] == 0);
    int num_vertices = m.n_ind_vertices();
    std::vector<int> cone_indices;
    cone_indices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if ((is_closed) && (float_equal(m.Th_hat[vi], 2 * M_PI))) continue;
        if ((!is_closed) && (float_equal(m.Th_hat[vi], 4 * M_PI))) continue;
        cone_indices.push_back(vi);
    }

    int num_cones = cone_indices.size();
    Eigen::MatrixXd cone_positions(num_cones, 3);
    Eigen::VectorXd cone_values(num_cones);
    for (int i = 0; i < num_cones; ++i) {
        int vi = cone_indices[i];
        cone_positions.row(i) = V.row(vtx_reindex[vi]);
        cone_values[i] = (double)(m.Th_hat[vi]) - (2 * M_PI);
    }

    return std::make_tuple(cone_positions, cone_values);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_closed_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& Th_hat)
{
    // get cone indices
    int num_vertices = V.rows();
    std::vector<int> cone_indices;
    cone_indices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (float_equal(Th_hat[vi], 2 * M_PI)) continue;
        cone_indices.push_back(vi);
    }

    // build cone positions and values
    int num_cones = cone_indices.size();
    Eigen::MatrixXd cone_positions(num_cones, 3);
    Eigen::VectorXd cone_values(num_cones);
    for (int i = 0; i < num_cones; ++i) {
        int vi = cone_indices[i];
        cone_positions.row(i) = V.row(vi);
        cone_values[i] = (double)(Th_hat[vi]) - (2 * M_PI);
    }

    return std::make_tuple(cone_positions, cone_values);
}

Eigen::MatrixXd generate_subset_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vertex_indices)
{
    int num_vertices = vertex_indices.size();
    Eigen::MatrixXd vertex_positions(num_vertices, 3);
    for (int i = 0; i < num_vertices; ++i) {
        int vi = vertex_indices[i];
        vertex_positions.row(i) = V.row(vi);
    }

    return vertex_positions;
}

void view_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle,
    bool show)
{
    if (show) {
        spdlog::info("Viewing {}", mesh_handle);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);

    // Generate cones
    auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V, vtx_reindex, m);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::registerPointCloud(mesh_handle + " cones", cone_positions);
    polyscope::getPointCloud(mesh_handle + " cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-M_PI, M_PI})
        ->setEnabled(true);
    if (show) polyscope::show();
#endif
}

void view_mesh_quality(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle,
    bool show)
{
    int num_vertices = V.rows();
    if (show) {
        spdlog::info("Viewing mesh {} with {} vertices", mesh_handle, num_vertices);
    }

    Eigen::VectorXd double_area;
    igl::doublearea(V, F, double_area);


#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    if (mesh_handle == "") {
        mesh_handle = "quality_analysis_mesh";
        polyscope::registerSurfaceMesh(mesh_handle, V, F);
        polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    }
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("double_area", double_area);
    if (show) polyscope::show();
#endif
}

void view_mesh_topology(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle,
    bool show)
{
    // Get components and boundary
    Eigen::MatrixXi bd;
    Eigen::VectorXi components;
    igl::boundary_facets(F, bd);
    igl::facet_components(F, components);

    if (mesh_handle == "") {
        mesh_handle = "topology_analysis_mesh";
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("component", components);
    if (show) polyscope::show();
#else
    if (show) {
        int num_vertices = V.rows();
        int num_faces = F.rows();
        spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
    }
#endif
}

void view_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "cut_mesh";
    }

    // Cut mesh along seams
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, FT, V_cut);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V_cut, FT);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexParameterizationQuantity("uv", uv)
        ->setStyle(polyscope::ParamVizStyle::GRID)
        ->setGridColors(std::make_pair(DARK_TEAL, TEAL))
        ->setEnabled(true);

    if (show) polyscope::show();
#else
    if (show) {
        int num_vertices = V.rows();
        int num_faces = F.rows();
        spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
    }
#endif
}

void view_triangulation(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& fn_to_f,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "triangulation_mesh";
    }
    spdlog::info("Viewing triangulation for map with {} faces", fn_to_f.size());

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // add mesh with permuted face map
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceScalarQuantity("face_map", fn_to_f)
        ->setColorMap("spectral")
        ->setEnabled(true);

    if (show) polyscope::show();
#else
    if (show) {
        int num_vertices = V.rows();
        int num_faces = F.rows();
        spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
    }
#endif
}

void view_layout(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "layout";
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    // Add layout
    polyscope::registerSurfaceMesh2D(mesh_handle, uv, FT)->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexParameterizationQuantity("uv", uv)
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)->setEdgeWidth(1.);

    if (show) polyscope::show();
#else
    if (show) {
        int num_vertices = uv.rows();
        int num_faces = FT.rows();
        spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
    }
#endif
}

void view_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& vertex_function,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "vertex function";
    }

    if (show) {
        spdlog::info("Viewing {} mesh", mesh_handle);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);
    int num_vertices = m.n_vertices();
    if (num_vertices != vertex_function.size())
    {
        spdlog::error("Inconsistent number of vertices and function size");
        return;
    }

#ifdef ENABLE_VISUALIZATION
    spdlog::debug("Initializing mesh");
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity(
            "function value",
            convert_scalar_to_double_vector(vertex_function))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    if (show) polyscope::show();
#endif
}

void view_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& vertex_function,
    std::string mesh_handle,
    bool show)
{
    VectorX vertex_function_eig;
    convert_std_to_eigen_vector(vertex_function, vertex_function_eig);
    view_vertex_function(m, vtx_reindex, V, vertex_function_eig, mesh_handle, show);
}

void view_independent_vertex_function(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& vertex_function,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "independent vertex function mesh";
    }

    int num_vertices = V.rows();
    if (show) {
        spdlog::info("Viewing {} mesh with {} vertices", mesh_handle, num_vertices);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);

    VectorX double_vertex_constraint = vector_compose(vertex_function, m.v_rep);

#ifdef ENABLE_VISUALIZATION
    spdlog::debug("Initializing mesh");
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity(
            "function value",
            convert_scalar_to_double_vector(double_vertex_constraint))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    if (show) polyscope::show();
#endif
}

} // namespace Optimization
} // namespace Penner
