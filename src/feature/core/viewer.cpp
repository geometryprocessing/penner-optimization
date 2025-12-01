#include "feature/core/viewer.h"

#include "util/vf_corners.h"
#include "feature/feature/error.h"
#include "feature/feature/features.h"
#include "feature/feature/gluing.h"
#include "holonomy/field/facet_field.h"
#include "holonomy/field/frame_field.h"
#include "holonomy/field/cross_field.h"
#include "feature/core/quads.h"
#include "optimization/core/projection.h"
#include "holonomy/field/intrinsic_field.h"
#include "optimization/core/viewer.h"

#include "holonomy/core/viewer.h"
#include "util/vf_mesh.h"

#include <igl/principal_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/bounding_box_diagonal.h>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Feature {

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_glued_cone_vertices(
    const Eigen::MatrixXd& V,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    // get the cone angle defects for the glued mesh
    std::vector<Scalar> glued_angle_defects = compute_glued_angle_defects(m, vtx_reindex, V_map);

    // get cone indices with nontrivial defect
    int num_vertices = glued_angle_defects.size();
    std::vector<int> cone_indices;
    cone_indices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (float_equal<Scalar>(glued_angle_defects[vi], 0.)) continue;
        cone_indices.push_back(vi);
    }

    // build cone positions and values
    int num_cones = cone_indices.size();
    Eigen::MatrixXd cone_positions(num_cones, 3);
    Eigen::VectorXd cone_values(num_cones);
    for (int i = 0; i < num_cones; ++i) {
        int vi = cone_indices[i];
        cone_positions.row(i) = V.row(vi);
        cone_values[i] = (double)(glued_angle_defects[vi]);
    }

    return std::make_tuple(cone_positions, cone_values);
}

void view_glued_mesh_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXd& V,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "glued mesh cones";
    }
    if (show) {
        spdlog::info("Viewing {}", mesh_handle);
    }

    // build vertex identification
    int num_vertices = vtx_reindex.size();
    std::vector<int> vtx_gluing(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        vtx_gluing[i] = V_map[vtx_reindex[i]];
    }

    // get doubled mesh to view from halfedge
    auto [V_double, F_mesh, F_halfedge] = Optimization::generate_doubled_mesh(V, m, vtx_gluing);

    // Generate cone geometry
    spdlog::info("Generating cone info");
    auto [cone_positions, cone_values] = generate_glued_cone_vertices(V, m, vtx_reindex, V_map);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    polyscope::registerPointCloud(mesh_handle + "_cones", cone_positions);
    polyscope::getPointCloud(mesh_handle + "_cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-M_PI, M_PI})
        ->setEnabled(true);
    if (show) polyscope::show();
#endif
}


void view_uv_alignment(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_aligned,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "uv alignment";
    }
    if (show) {
        spdlog::info("Viewing {}", mesh_handle);
    }

    // cut mesh along seams
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, F_uv, V_cut);

    // generate alignment
    Eigen::MatrixXd F_uv_alignment = compute_mask_uv_alignment(uv, F_uv, F_is_aligned);
    Eigen::VectorXd uv_alignment = compute_polyscope_halfedge_from_corner_function(F_uv_alignment);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V_cut, F_uv);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexParameterizationQuantity("uv", uv, polyscope::ParamCoordsType::WORLD)
        ->setCheckerColors(std::make_pair(DARK_TEAL, TEAL))
        ->setEnabled(false);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity("angle alignment", uv_alignment)
        ->setColorMap("reds")
        ->setEnabled(true);

    if (show) polyscope::show();
#endif
}


void view_feature_edges(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "feature edges";
    }
    int num_vertices = V.rows();
    int num_edges = E.rows();
    if (show) {
        spdlog::info("Viewing {} with {} vertices and {} edges", mesh_handle, num_vertices, num_edges);
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerCurveNetwork(mesh_handle, V, E);

    if (show) polyscope::show();
#endif
}

// find cones and defect of a quad mesh
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_quad_cones(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    // get valences of all vertices
    std::vector<int> valences = compute_valences(F);

    // find irregular vertices
    int num_vertices = valences.size();
    std::vector<int> cone_indices;
    cone_indices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (valences[vi] == 4) continue;
        cone_indices.push_back(vi);
    }

    // generate cone positions and values
    int num_cones = cone_indices.size();
    Eigen::MatrixXd cone_positions(num_cones, 3);
    Eigen::VectorXd cone_values(num_cones);
    for (int i = 0; i < num_cones; ++i) {
        int vi = cone_indices[i];
        cone_positions.row(i) = V.row(vi);
        cone_values[i] = 4 - valences[vi];
    }

    return std::make_tuple(cone_positions, cone_values);
}

void view_quad_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "quad mesh";
    }

    // get quad mesh cones
    auto [cone_positions, cone_values] = generate_quad_cones(V, F);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(BEIGE);
    polyscope::registerPointCloud(mesh_handle + "_cones", cone_positions);
    polyscope::getPointCloud(mesh_handle + "_cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-2, 2})
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


void view_cut(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    const Eigen::MatrixXi& F_is_cut,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "cut";
    }
    if (show) {
        int num_vertices = V_cut.rows();
        int num_faces = F_cut.rows();
        spdlog::info("Viewing {} of size ({}, {})", mesh_handle, num_vertices, num_faces);
    }

    // move cut data to viewer halfedge indexing
    Eigen::VectorXi h_is_cut = compute_polyscope_halfedge_from_corner_function(F_is_cut);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V_cut, F_cut);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity("is cut", h_is_cut)
        ->setEnabled(true);

    if (show) polyscope::show();
#endif
}


void view_endpoints(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& endpoints,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "endpoints";
    }
    if (show) {
        int num_vertices = V.rows();
        int num_faces = F.rows();
        spdlog::info("Viewing {} of size ({}, {})", mesh_handle, num_vertices, num_faces);
    }

    // extract individual endpoint masks from pairs
    int num_vertices = endpoints.size();
    Eigen::VectorXi first_endpoint(num_vertices);
    Eigen::VectorXi second_endpoint(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        first_endpoint[vi] = endpoints[vi].first;
        second_endpoint[vi] = endpoints[vi].second;
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity("first_endpoint", first_endpoint)
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity("second_endpoint", second_endpoint)
        ->setEnabled(false);

    if (show) polyscope::show();
#endif
}


void view_direction_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& direction_field,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "direction field mesh";
    }
    int num_vertices = V.rows();
    int num_faces = F.rows();
    if (show) {
        spdlog::info("Viewing {} with {} vertices and {} faces", mesh_handle, num_vertices, num_faces);
    }
    if (direction_field.rows() != num_faces){
        spdlog::info("Inconsistent number of faces");
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("direction", direction_field)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    if (show) polyscope::show();
#endif
}


void view_principal_curvature(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "principal_curvature_mesh";
    }

    // compute principal curvature and directions
    Eigen::MatrixXd PD1,PD2;
    Eigen::VectorXd PV1,PV2;
    std::vector<int> bad_vertices;
    igl::principal_curvature(V,F,PD1,PD2,PV1,PV2, bad_vertices, radius);
    auto[face_max_direction, face_min_direction, face_max_curvature, face_min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, radius);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("max curvature", PV1);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexScalarQuantity("min curvature", PV2);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexVectorQuantity("max curvature direction", PD1);
    polyscope::getSurfaceMesh(mesh_handle)->addVertexVectorQuantity("min curvature direction", PD2);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceVectorQuantity("max face direction", face_max_direction);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceVectorQuantity("min face direction", face_min_direction);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("max face curvature", face_max_curvature);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("min face curvature", face_min_curvature);
    if (show) polyscope::show();
#else
    if (show) {
        int num_vertices = V.rows();
        int num_faces = F.rows();
        spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
    }
#endif
}

void view_fixed_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& direction,
    const std::vector<bool>& is_fixed_direction,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "fixed_direction_mesh";
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceVectorQuantity("direction", direction);
    polyscope::getSurfaceMesh(mesh_handle)->addFaceScalarQuantity("is fixed", is_fixed_direction);
    if (show) polyscope::show();
#else
    if (show) {
        int num_vertices = V.rows();
        int num_faces = F.rows();
        spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
    }
#endif
}


// compute the best fit conformal scaling for the parameterization metric
VectorX best_fit_conformal_vf(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    std::vector<Scalar> Th_hat(V.rows());

    std::vector<int> vtx_reindex;
    auto C_v = Optimization::generate_initial_mesh(V, F, V, F, Th_hat, vtx_reindex);
    auto C_uv = Optimization::generate_initial_mesh(V, F, uv, F_uv, Th_hat, vtx_reindex);
    VectorX metric_coords = C_uv->get_metric_coordinates();
    VectorX r_perm = Optimization::best_fit_conformal(*C_v, metric_coords);
    VectorX r = vector_inverse_reindex(r_perm, vtx_reindex);

    return r;
}

void view_conformal_scaling(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    std::string mesh_handle,
    bool show)
{
    if (mesh_handle == "") {
        mesh_handle = "conformal scaling";
    }
    if (show) {
        spdlog::info("Viewing {}", mesh_handle);
    }

    // generate best fit vertex scaling function for parameterization metric
    VectorX conformal_scaling = best_fit_conformal_vf(V, F, uv, F_uv);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity("conformal scaling", convert_scalar_to_double_vector(conformal_scaling))
        ->setColorMap("coolwarm")
        ->setEnabled(true);

    if (show) polyscope::show();
#endif
}

Eigen::MatrixXd displace_cut_faces(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Scalar displacement
) {
    // compute normals for displacement
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);
    Scalar abs_disp = displacement * igl::bounding_box_diagonal(V);

    return V + abs_disp * N;
}

void view_feature_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    Scalar displacement,
    std::string mesh_handle)
{
    if (mesh_handle == "") {
        mesh_handle = "cross_field";
    }

    // generate frame field geometry from cross field
    Eigen::MatrixXd frame_field = Holonomy::generate_frame_field(V, F, reference_field, theta);
    int num_faces = F.rows();

    // transfer kappa and period jump to viewer halfedge indexing
    Eigen::VectorXd halfedge_kappa(3 * num_faces);
    Eigen::VectorXd halfedge_period_jump(3 * num_faces);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            halfedge_kappa[3* fijk + j] = kappa(fijk, i);
            halfedge_period_jump[3* fijk + j] = period_jump(fijk, i);
        }
    }

    // displace the cut faces for visualization
    Eigen::MatrixXd V_disp = displace_cut_faces(V, F, displacement);

    // get cone angles
    std::vector<Scalar> Th_hat = Holonomy::compute_cone_angle(V, F, kappa, period_jump);
    auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V_disp, Th_hat);


#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_disp, F);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceScalarQuantity(
            "theta",
            theta)
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("reference", reference_field)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("frame", frame_field)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity("kappa", halfedge_kappa)
        ->setEnabled(false);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity("period jump", halfedge_period_jump)
        ->setEnabled(false);
        
    polyscope::registerPointCloud("cross_field_cones", cone_positions);
    polyscope::getPointCloud("cross_field_cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-M_PI, M_PI})
        ->setEnabled(true);

    polyscope::show();
#endif
}



} // namespace Feature
} // namespace Penner
