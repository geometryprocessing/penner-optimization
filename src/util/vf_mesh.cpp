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
#include "util/vf_mesh.h"

#include "util/vector.h"

#include <igl/boundary_facets.h>
#include <igl/facet_components.h>
#include <igl/per_vertex_normals.h>
#include <igl/remove_unreferenced.h>
#include <igl/unique.h>

namespace Penner {

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
    int cols = F.cols();

    // Iterate over faces to find all referenced vertices in sorted order
    std::vector<int> referenced_vertices;
    for (int fi = 0; fi < num_faces; ++fi) {
        for (int j = 0; j < cols; ++j) {
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
    FN.resize(num_faces, cols);
    for (int fi = 0; fi < num_faces; ++fi) {
        for (int j = 0; j < cols; ++j) {
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

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_seams(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& FT)
{
    // get boundary edges of the uv map
    Eigen::MatrixXi uv_edges;
    Eigen::VectorXi J, K;
    igl::boundary_facets(FT, uv_edges, J, K);

    // get 3D vertices on seam
    int num_seam_edges = J.size();
    Eigen::MatrixXi edges(num_seam_edges, 2);
    for (int i = 0; i < num_seam_edges; ++i) {
        edges(i, 0) = F(J[i], (K[i] + 1) % 3);
        edges(i, 1) = F(J[i], (K[i] + 2) % 3);
    }

    // reindex to remove redundant
    Eigen::MatrixXi seam_edges;
    std::vector<int> new_to_old_map;
    remove_unreferenced(edges, seam_edges, new_to_old_map);

    // get new vertices
    int num_seam_vertices = new_to_old_map.size();
    Eigen::MatrixXd seam_vertices(num_seam_vertices, 3);
    for (int i = 0; i < num_seam_vertices; ++i) {
        seam_vertices.row(i) = V.row(new_to_old_map[i]);
    }

    return std::make_tuple(seam_vertices, seam_edges);
}

std::vector<int> find_boundary_vertices(const Eigen::MatrixXi& F)
{
    // Get the boundary edges
    Eigen::MatrixXi B;
    igl::boundary_facets(F, B);

    // Get all unique vertex indices in the boundary
    Eigen::VectorXi bd_vertices;
    igl::unique(B, bd_vertices);

    // Convert Eigen vector to standard vector
    return convert_vector<int>(bd_vertices);
}

std::vector<bool> compute_boundary_vertices(const Eigen::MatrixXi& F, int num_vertices)
{
    // Get the boundary vertices
    auto bd_vertices = find_boundary_vertices(F);

    // Make list of boundary vertices into boolean mask
    std::vector<bool> is_boundary_vertex(num_vertices, false);
    int num_bd_vertices = bd_vertices.size();
    for (int i = 0; i < num_bd_vertices; ++i) {
        int vi = bd_vertices[i];
        is_boundary_vertex[vi] = true;
    }

    return is_boundary_vertex;
}

Eigen::MatrixXd
inflate_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double inflation_distance)
{
    // Get vertex normals
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);

    // Displace mesh vertices along normal
    int num_vertices = V.rows();
    Eigen::MatrixXd V_inflated(num_vertices, 3);
    for (int vi = 0; vi < num_vertices; ++vi) {
        // Add displacement along the vertex normal (if the normal is well defined)
        Eigen::RowVector3d ni = Eigen::RowVector3d::Zero();
        if (!(isnan(N(vi, 0)) || isnan(N(vi, 1)) || isnan(N(vi, 2)))) {
            ni = N.row(vi);
        }
        V_inflated.row(vi) = V.row(vi) + inflation_distance * ni;
    }

    return V_inflated;
}

// compute map from vertex-vertex edges to halfedges for mesh
Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& he_to_corner)
{
    int num_vertices = F.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv2he(num_vertices, num_vertices);

    // build sparse matrix
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int num_halfedges = he_to_corner.size();
    for (int hjk = 0; hjk < num_halfedges; ++hjk) {
        const auto& corner = he_to_corner[hjk];
        int fijk = corner.first;
        int i = corner.second;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;
        int vj = F(fijk, j);
        int vk = F(fijk, k);
        trips.push_back(Trip(vj, vk, hjk + 1));
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

Eigen::SparseMatrix<int> generate_VV_to_face_map(const Eigen::MatrixXi& F)
{
    int num_vertices = F.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv2f(num_vertices, num_vertices);

    // build sparse matrix
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int num_faces = F.rows();
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int vi = F(fijk, i);
            int vj = F(fijk, j);
            trips.push_back(Trip(vi, vj, fijk + 1));
        }
    }
    vv2f.setFromTriplets(trips.begin(), trips.end());

    return vv2f;
}

VectorX compute_cone_angles(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT)
{
    // compute uv corner angles
    Eigen::MatrixXd corner_angles;
    igl::internal_angles(uv, FT, corner_angles);

    // sum up corner angles
    int num_vertices = V.rows();
    int num_faces = F.rows();
    VectorX cone_angles = VectorX::Zero(num_vertices);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int vi = F(fijk, i);
            cone_angles[vi] += corner_angles(fijk, i);
        }
    }

    return cone_angles;
}

} // namespace Penner
