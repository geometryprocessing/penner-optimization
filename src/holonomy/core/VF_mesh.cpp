#include "holonomy/core/VF_mesh.h"

#include "holonomy/core/vector.h"

#include <igl/boundary_facets.h>
#include <igl/unique.h>
#include <igl/per_vertex_normals.h>

namespace PennerHolonomy {

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
    for (int i = 0; i < num_bd_vertices; ++i)
    {
        int vi = bd_vertices[i];
        is_boundary_vertex[vi] = true;
    }

    return is_boundary_vertex;
}

Eigen::MatrixXd inflate_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double inflation_distance)
{
    // Get vertex normals
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);

    // Displace mesh vertices along normal
    int num_vertices = V.rows();
    Eigen::MatrixXd V_inflated(num_vertices, 3);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        // Add displacement along the vertex normal (if the normal is well defined)
        Eigen::RowVector3d ni = Eigen::RowVector3d::Zero();
        if (!(isnan(N(vi, 0)) || isnan(N(vi, 1)) || isnan(N(vi, 2)))) {
            ni = N.row(vi);
        }
        V_inflated.row(vi) = V.row(vi) + inflation_distance * ni;
    }

    return V_inflated;
}

} // namespace PennerHolonomy
