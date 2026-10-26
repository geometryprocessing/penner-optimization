
#include "resolution/core/viewer.h"
#include "metric/viewer.h"
#include "parametrization/layout.h"
#include "util/vector.h"
#include "util/vf_mesh.h"
#include "SYMDIR_NEW.h"
#include "igl/grad.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Resolution {

void view_layout(const MarkedPennerConeMetric& marked_metric, bool show)
{
    auto [F, uv, FT] = compute_layout_VF(marked_metric);

    int num_faces = FT.rows();
    Eigen::MatrixXi F_halfedge(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i : {0, 1, 2})
        {
            int j = (i + 1) % 3;
            F_halfedge(fijk, i) = FT(fijk, j);
        }
    }

    int num_halfedges = marked_metric.n_halfedges();
    VectorX opposite_vertex(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        opposite_vertex[hij] = marked_metric.v_rep[marked_metric.to[marked_metric.n[hij]]];
    }

    VectorX length;
    convert_std_to_eigen_vector(marked_metric.l, length);
    VectorX length_mesh = generate_FV_halfedge_data(F_halfedge, length);
    VectorX opposite_vertex_mesh = generate_FV_halfedge_data(F_halfedge, opposite_vertex);

    std::string mesh_handle = "layout";
#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh2D(mesh_handle, uv, FT);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "length",
            convert_scalar_to_double_vector(length_mesh))
        ->setColorMap("coolwarm")
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "opposite vertex",
            convert_scalar_to_double_vector(opposite_vertex_mesh))
        ->setEnabled(true);

    for (int i = 0; i < marked_metric.n_homology_basis_loops(); ++i) {
        const auto& homology_basis_loops = marked_metric.get_homology_basis_loops();
        std::vector<int> dual_loop_faces =
            homology_basis_loops[i]->generate_face_sequence(marked_metric);
        spdlog::info("{} faces in loop", dual_loop_faces.size());

        VectorX is_dual_loop_halfedge = VectorX::Constant(num_halfedges, 0.5);
        for (auto itr = homology_basis_loops[i]->begin(); !itr.is_end(); ++itr)
        {
            Holonomy::DualSegment segment = *itr; 
            is_dual_loop_halfedge(segment[0]) = 0;
            is_dual_loop_halfedge(segment[1]) = 0;
        }

        Eigen::VectorXd is_dual_loop_face = Eigen::VectorXd::Constant(num_faces, 0.5);
        for (const auto& dual_loop_face : dual_loop_faces) {
            is_dual_loop_face(dual_loop_face) = (double)(marked_metric.kappa_hat[i]);
        }

        VectorX dual_loop_mesh = generate_FV_halfedge_data(F_halfedge, is_dual_loop_halfedge);
        polyscope::getSurfaceMesh(mesh_handle)
            ->addFaceScalarQuantity("dual_loop_" + std::to_string(i), is_dual_loop_face);
        polyscope::getSurfaceMesh(mesh_handle)
            ->addHalfedgeScalarQuantity(
                "halfedge_dual_loop_" + std::to_string(i),
                convert_scalar_to_double_vector(dual_loop_mesh));
    }
    
    if (show) polyscope::show();
#endif
} 

void view_combed_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2,
    std::string mesh_handle)
{
    if (mesh_handle == "") {
        mesh_handle = "combed_field";
    }

    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, FT, V_cut);

    Eigen::SparseMatrix<double> Grad;
    igl::grad(V_cut, FT, Grad);
    Eigen::MatrixXd Guv = Grad * uv;
    Eigen::MatrixXd G_u = Eigen::Map<const Eigen::MatrixXd>(Guv.col(0).data(), F.rows(), 3);
    Eigen::MatrixXd G_v = Eigen::Map<const Eigen::MatrixXd>(Guv.col(1).data(), F.rows(), 3);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_cut, FT);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexParameterizationQuantity("uv", uv)
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("PD1", PD1)
        ->setEnabled(false);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("PD2", PD2)
        ->setEnabled(false);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("grad_u", G_u)
        ->setEnabled(false);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addFaceVectorQuantity("grad_v", G_v)
        ->setEnabled(false);
    polyscope::show();
#endif
}

}
}
