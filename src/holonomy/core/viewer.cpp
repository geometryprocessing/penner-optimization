
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
    auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V, Th_hat);

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
    auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V, vtx_reindex, m);

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
    auto [V_double, F_mesh, F_halfedge] = Optimization::generate_doubled_mesh(V, marked_metric, vtx_reindex);

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
    auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V, vtx_reindex, marked_metric);
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

void view_seamless_parameterization(
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

} // namespace Holonomy
} // namespace Penner
