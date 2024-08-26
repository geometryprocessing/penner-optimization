
#include "holonomy/core/viewer.h"

#include "util/vf_mesh.h"

#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/edge_flaps.h>
#include <igl/facet_components.h>
#include <igl/local_basis.h>
#include <igl/rotate_vectors.h>

#include <random>

#include "util/vector.h"
#include "util/vf_mesh.h"
#include "holonomy/holonomy/constraint.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

#ifdef ENABLE_VISUALIZATION
glm::vec3 BEIGE(0.867, 0.765, 0.647);
glm::vec3 BLACK_BROWN(0.125, 0.118, 0.125);
glm::vec3 TAN(0.878, 0.663, 0.427);
glm::vec3 MUSTARD(0.890, 0.706, 0.282);
glm::vec3 FOREST_GREEN(0.227, 0.420, 0.208);
glm::vec3 TEAL(0., 0.375, 0.5);
glm::vec3 DARK_TEAL(0., 0.5*0.375, 0.5*0.5);
#endif

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

Eigen::MatrixXd rotate_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field)
{
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V, F, B1, B2, B3);
    return igl::rotate_vectors(frame_field, Eigen::VectorXd::Constant(1, M_PI / 2), B1, B2);
}

void view_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle)
{
    spdlog::info("Viewing mesh {}", mesh_handle);
    int num_vertices = V.rows();
    int num_faces = F.rows();
    int num_field = frame_field.rows();
    int num_cones = Th_hat.size();

    // Check size consistency
    if (num_field != num_faces) return;
    if (num_cones != num_vertices) return;

    // Generate cones
    auto [cone_positions, cone_values] = generate_cone_vertices(V, Th_hat);

    // Generate rotated cross field vectors from reference
    std::array<Eigen::MatrixXd, 4> cross_field;
    cross_field[0] = frame_field;
    for (int i : {1, 2, 3}) {
        cross_field[i] = rotate_frame_field(V, F, cross_field[i - 1]);
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    if (mesh_handle == "") {
        mesh_handle = "frame_field_mesh";
        polyscope::registerSurfaceMesh(mesh_handle, V, F);
        polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    }
    for (int i : {0, 1, 2, 3}) {
        polyscope::getSurfaceMesh(mesh_handle)
            ->addFaceVectorQuantity("field_" + std::to_string(i), cross_field[i])
            ->setVectorColor((i == 0) ? FOREST_GREEN : BLACK_BROWN)
            ->setVectorRadius(0.0005)
            ->setVectorLengthScale(0.005)
            ->setEnabled(true);
    }
    polyscope::registerPointCloud("cross_field_cones", cone_positions);
    polyscope::getPointCloud("cross_field_cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-M_PI, M_PI})
        ->setEnabled(true);


    polyscope::show();
#else
    spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
#endif
}

void view_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const VectorX& rotation_form,
    const std::vector<Scalar>& Th_hat,
    std::string mesh_handle,
    bool show)
{
    int num_vertices = V.rows();
    if (show) {
        spdlog::info("Viewing mesh {} with {} vertices", mesh_handle, num_vertices);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);
    VectorX rotation_form_mesh = generate_FV_halfedge_data(F_halfedge, rotation_form);

    // Generate cones
    spdlog::info("{} vertices", Th_hat.size());
    auto [cone_positions, cone_values] = generate_cone_vertices(V, vtx_reindex, m);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    if (mesh_handle == "") {
        mesh_handle = "rotation_form_mesh";
    }
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "rotation_form",
            convert_scalar_to_double_vector(rotation_form_mesh))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    polyscope::registerPointCloud(mesh_handle + "_cones", cone_positions);
    polyscope::getPointCloud(mesh_handle + "_cones")
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
    if (mesh_handle == "") {
        polyscope::registerSurfaceMesh(mesh_handle, V, F);
        polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    }
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

Scalar uv_length_squared(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    Eigen::Vector2d difference_vector = uv_1 - uv_0;
    Scalar length_sq = difference_vector.dot(difference_vector);
    return length_sq;
}

Scalar uv_length(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    return sqrt(uv_length_squared(uv_0, uv_1));
}

Scalar uv_cos_angle(const Eigen::Vector2d& uv_0, const Eigen::Vector2d& uv_1)
{
    Scalar dot_01 = uv_0.dot(uv_1);
    Scalar norm_0 = sqrt(uv_0.dot(uv_0));
    Scalar norm_1 = sqrt(uv_1.dot(uv_1));
    Scalar norm = norm_0 * norm_1;
    if (norm == 0.) return 0.;
    return (dot_01 / norm);
}

std::tuple<VectorX, VectorX, VectorX> compute_seamless_error(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    // Get the edge topology for the original uncut mesh
    Eigen::MatrixXi uE, EF, EI;
    Eigen::VectorXi EMAP;
    igl::edge_flaps(F, uE, EMAP, EF, EI);

    // Iterate over edges to check the length inconsistencies
    VectorX uv_length_error(F.size());
    VectorX uv_angle_error(F.size());
    VectorX uv_angle(F.size());
    for (Eigen::Index e = 0; e < EF.rows(); ++e) {
        // Get face corners corresponding to the current edge
        int f0 = EF(e, 0);
        int f1 = EF(e, 1);

        // Check first face (if not boundary)
        if (f0 < 0) continue;
        int i0 = EI(e, 0); // corner vertex face index
        int v0n = F_uv(f0, (i0 + 1) % 3); // next vertex
        int v0p = F_uv(f0, (i0 + 2) % 3); // previous vertex

        // Check second face (if not boundary)
        if (f1 < 0) continue;
        int i1 = EI(e, 1); // corner vertex face index
        int v1n = F_uv(f1, (i1 + 1) % 3); // next vertex
        int v1p = F_uv(f1, (i1 + 2) % 3); // next vertex

        // Compute the length of each halfedge corresponding to the corner in the cut mesh
        Eigen::Vector2d uv_00 = uv.row(v0n);
        Eigen::Vector2d uv_01 = uv.row(v0p);
        Eigen::Vector2d uv_10 = uv.row(v1n);
        Eigen::Vector2d uv_11 = uv.row(v1p);
        Scalar l0 = uv_length(uv_00, uv_01);
        Scalar l1 = uv_length(uv_10, uv_11);
        Scalar cos_angle = uv_cos_angle(uv_01 - uv_00, uv_11 - uv_10);
        Scalar length_error = abs(l0 - l1);
        Scalar angle_error = min(abs(cos_angle), abs(abs(cos_angle) - 1));

        // set length error for the given edge
        i0 = (i0 + 1) % 3;
        i1 = (i1 + 1) % 3;
        uv_length_error(3 * f0 + i0) = length_error;
        uv_length_error(3 * f1 + i1) = length_error;
        uv_angle_error(3 * f0 + i0) = angle_error;
        uv_angle_error(3 * f1 + i1) = angle_error;
        uv_angle(3 * f0 + i0) = cos_angle;
        uv_angle(3 * f1 + i1) = cos_angle;
    }

    return std::make_tuple(uv_length_error, uv_angle_error, uv_angle);
}

// TODO MAke separate layout method

void view_quad_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "quad mesh";
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V, F);

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
    auto [uv_length_error, uv_angle_error, uv_angle] = compute_seamless_error(F, uv, FT);
    spdlog::info("Max uv length error: {}", uv_length_error.maxCoeff());
    spdlog::info("Max uv angle error: {}", uv_angle_error.maxCoeff());

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V_cut, FT);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexParameterizationQuantity("uv", uv)
        ->setStyle(polyscope::ParamVizStyle::GRID)
        ->setGridColors(std::make_pair(DARK_TEAL, TEAL))
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "uv length error",
            convert_scalar_to_double_vector(uv_length_error));
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "uv angle error",
            convert_scalar_to_double_vector(uv_angle_error));
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "uv angle",
            convert_scalar_to_double_vector(uv_angle));

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

void view_constraint_error(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "constraint error";
    }

    int num_vertices = V.rows();
    if (show) {
        spdlog::info("Viewing {} mesh with {} vertices", mesh_handle, num_vertices);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, marked_metric, vtx_reindex);

    // Make mesh into discrete metric
    spdlog::debug("Making metric discrete");
    MarkedPennerConeMetric marked_metric_copy = marked_metric;
    marked_metric_copy.make_discrete_metric();

    // Generate corner angles
    spdlog::debug("Computing corner angles");
    VectorX he2angle;
    VectorX cotangents;
    marked_metric_copy.get_corner_angles(he2angle, cotangents);

    // Generate cones and cone errors
    spdlog::debug("Computing cones and errors");
    VectorX vertex_constraint = compute_vertex_constraint(marked_metric_copy, he2angle);
    auto [cone_positions, cone_values] = generate_cone_vertices(V, vtx_reindex, marked_metric);
    VectorX cone_error = vector_compose(vertex_constraint, marked_metric.v_rep);

#ifdef ENABLE_VISUALIZATION
    spdlog::debug("Initializing mesh");
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity(
            "cone error",
            convert_scalar_to_double_vector(cone_error))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    polyscope::registerPointCloud(mesh_handle + "_cones", cone_positions);
    polyscope::getPointCloud(mesh_handle + "_cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-M_PI, M_PI})
        ->setEnabled(true);
    if (show) polyscope::show();
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

    int num_vertices = V.rows();
    if (show) {
        spdlog::info("Viewing {} mesh with {} vertices", mesh_handle, num_vertices);
    }
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);
    if (num_vertices != vertex_function.size()) return;

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

} // namespace Holonomy
} // namespace Penner
