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
#include "optimization/core/vf_mesh.h"

#include <igl/facet_components.h>

namespace CurvatureMetric {

int count_components(const Eigen::MatrixXi& F)
{
    Eigen::VectorXi face_components;
    igl::facet_components(F, face_components);
    return face_components.maxCoeff() + 1;
}

void remove_unreferenced(
    const Eigen::MatrixXi& F,
    Eigen::MatrixXi& FN,
    std::vector<int>& new_to_old_map)
{
    int num_faces = F.rows();

    // Iterate over faces to find all referenced vertices in sorted order
    std::vector<int> referenced_vertices;
    for (int fi = 0; fi < num_faces; ++fi) {
        for (int j = 0; j < 3; ++j) {
            int vk = F(fi, j);
            referenced_vertices.push_back(vk);
        }
    }

    // Make the list of referenced vertices sorted and unique
    std::sort(referenced_vertices.begin(), referenced_vertices.end());
    auto last_sorted = std::unique(referenced_vertices.begin(), referenced_vertices.end());

    // Get the new to old map from the sorted referenced vertices list
    new_to_old_map.assign(referenced_vertices.begin(), last_sorted);

    // Build a (compact) map from old to new vertices
    int num_vertices = new_to_old_map.size();
    std::unordered_map<int, int> old_to_new_map;
    for (int k = 0; k < num_vertices; ++k) {
        int vk = new_to_old_map[k];
        old_to_new_map[vk] = k;
    }

    // Reindex the vertices in the face list
    FN.resize(num_faces, 3);
    for (int fi = 0; fi < num_faces; ++fi) {
        for (int j = 0; j < 3; ++j) {
            int vk = F(fi, j);
            int k = old_to_new_map[vk];
            FN(fi, j) = k;
        }
    }
}

void cut_mesh_along_parametrization_seams(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    Eigen::MatrixXd& V_cut)
{
    int num_uv_vertices = uv.rows();
    int num_uv_faces = FT.rows();

    // Check input validity
    if (F.rows() != num_uv_faces) {
        spdlog::error("F and FT have a different number of faces");
        return;
    }

    // Copy by face index correspondences
    V_cut.resize(num_uv_vertices, 3);
    for (int f = 0; f < num_uv_faces; ++f) {
        for (int i = 0; i < 3; ++i) {
            int vi = F(f, i);
            int uvi = FT(f, i);
            V_cut.row(uvi) = V.row(vi);
        }
    }
}

} // namespace CurvatureMetric
