/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#include "optimization/util/viewers.h"

#include "optimization/core/vector.h"
#include "optimization/core/vf_mesh.h"

#include <igl/flipped_triangles.h>
#if ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif // ENABLE_VISUALIZATION

namespace CurvatureMetric {

void view_flipped_triangles(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    // Cut the mesh along the parametrization seams
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, F_uv, V_cut);

    // Get the flipped elements
    Eigen::VectorXi flipped_f;
    igl::flipped_triangles(uv, F_uv, flipped_f);
    spdlog::info("{} flipped elements", flipped_f.size());

    // Convert flipped element list to a scalar function
    int num_faces = F_uv.rows();
    Eigen::VectorXd is_flipped;
    is_flipped.setZero(num_faces);
    for (int i = 0; i < flipped_f.size(); ++i) {
        int fi = flipped_f[i];
        is_flipped[fi] = 1.0;
        spdlog::info("Face {}: {} is flipped", fi, F_uv.row(fi));
    }

#if ENABLE_VISUALIZATION
    // Add the mesh and uv mesh with flipped element maps
    polyscope::init();
    polyscope::registerSurfaceMesh("mesh", V_cut, F_uv)
        ->addFaceScalarQuantity("is_flipped", is_flipped);
    polyscope::getSurfaceMesh("mesh")->addVertexParameterizationQuantity("uv", uv);
    polyscope::registerSurfaceMesh2D("uv mesh", uv, F_uv)
        ->addFaceScalarQuantity("is_flipped", is_flipped);
    polyscope::show();
#endif // ENABLE_VISUALIZATION
}

void view_halfedge_mesh_layout(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& u_vec,
    const std::vector<Scalar>& v_vec)
{
    Eigen::VectorXd u, v;
    convert_std_to_eigen_vector(u_vec, u);
    convert_std_to_eigen_vector(v_vec, v);
    Eigen::MatrixXd uv(u.size(), 3);
    uv.col(0) = u;
    uv.col(1) = v;
    Eigen::MatrixXi F(m.n_faces(), 3);
    for (int f = 0; f < m.n_faces(); ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];
        F(f, 0) = hij;
        F(f, 1) = hjk;
        F(f, 2) = hki;
    }
#if ENABLE_VISUALIZATION
    spdlog::info("Viewing layout");
    polyscope::init();
    polyscope::registerSurfaceMesh2D("layout", uv, F);
    polyscope::show();
#endif
}

void view_parameterization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT)
{
    // Cut mesh along seams
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, FT, V_cut);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle = "cut_mesh";

    // Add cut mesh with
    polyscope::registerSurfaceMesh(mesh_handle, V_cut, FT);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexParameterizationQuantity("uv", uv)
        ->setEnabled(true);

    polyscope::show();
#else
    spdlog::info("Visualization disabled");
#endif
}

} // namespace CurvatureMetric
