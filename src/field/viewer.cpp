// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "field/viewer.h"

#include "util/vector.h"
#include "util/vf_mesh.h"

#include <igl/local_basis.h>
#include <igl/rotate_vectors.h>

// cross field generation for viewer
#include "field/frame_field.h"
#include "field/vector_field.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Field {

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
    std::string mesh_handle,
    Scalar scale)
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
    }
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    for (int i : {0, 1, 2, 3}) {
        polyscope::getSurfaceMesh(mesh_handle)
            ->addFaceVectorQuantity("field_" + std::to_string(i), cross_field[i])
            ->setVectorColor((i == 0) ? BLACK_BROWN : BLACK_BROWN)
            //->setVectorColor((i == 0) ? FOREST_GREEN : BLACK_BROWN)
            ->setVectorRadius(scale)
            ->setVectorLengthScale(scale / 10.)
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

void view_vector_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& vector_field,
    std::string mesh_handle)
{
    spdlog::info("Viewing mesh {}", mesh_handle);
    int num_vertices = V.rows();
    int num_faces = F.rows();
    int num_field = vector_field.rows();

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    if (mesh_handle == "") {
        mesh_handle = "vector_field_mesh";
    }
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
    polyscope::getSurfaceMesh(mesh_handle)->setSurfaceColor(MUSTARD);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("field", vector_field)
        ->setEnabled(true);

    polyscope::show();
#else
    spdlog::info("Viewer disabled for mesh (|V|={}, |F|={})", num_vertices, num_faces);
#endif
}

void view_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    std::string mesh_handle)
{
    if (mesh_handle == "") {
        mesh_handle = "cross_field";
    }

    // generate frame field geometry from cross field
    Eigen::MatrixXd frame_field = Field::generate_vector_field(V, F, reference_field, theta);
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

    // get cone angles
    std::vector<Scalar> Th_hat = Field::compute_frame_field_cones(V, F, kappa, period_jump);
    auto [cone_positions, cone_values] = generate_cone_vertices(V, Th_hat);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V, F);
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
    //auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V, vtx_reindex, m);
    auto [cone_positions, cone_values] = generate_cone_vertices(V, Th_hat);

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


} // namespace Field
} // namespace Penner
