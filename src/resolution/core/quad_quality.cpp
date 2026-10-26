
#include "resolution/core/quad_quality.h"

namespace Penner {
namespace Resolution {

Vector3 normalize(const Vector3& d)
{
    return (d.squaredNorm() > 1e-24) ? d.normalized() : d;
}
    
// compute the four edge direction vectors of a quad
std::vector<std::array<Vector3, 4>> compute_quad_edges(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    int num_faces = F.rows();
    std::vector<std::array<Vector3, 4>> quad_edges(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        Vector3 x0 = V.row(F(f, 0));
        Vector3 x1 = V.row(F(f, 1));
        Vector3 x2 = V.row(F(f, 2));
        Vector3 x3 = V.row(F(f, 3));
        quad_edges[f][0] = x1 - x0;
        quad_edges[f][1] = x2 - x1;
        quad_edges[f][2] = x3 - x2;
        quad_edges[f][3] = x0 - x3;
    }

    return quad_edges;
}

std::vector<std::array<Scalar, 4>> compute_quad_lengths(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    // get the four quad edges
    std::vector<std::array<Vector3, 4>> quad_edges = compute_quad_edges(V, F);

    // get the lengths of the edges
    int num_faces = F.rows();
    std::vector<std::array<Scalar, 4>> quad_lengths(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        for (int i = 0; i < 4; ++i)
        {
            quad_lengths[f][i] = quad_edges[f][i].norm();
        }
    }

    return quad_lengths;
}

// compute area weighted normals of quad corners, i.e., cross products
// of adjacent quad edges
std::vector<std::array<Vector3, 4>> compute_quad_normals(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    std::vector<std::array<Vector3, 4>> quad_edges = compute_quad_edges(V, F);

    int num_faces = F.rows();
    std::vector<std::array<Vector3, 4>> quad_normals(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        for (int i = 0; i < 4; ++i)
        {
            int j = (i + 3) % 4;
            quad_normals[f][i] = quad_edges[f][j].cross(quad_edges[f][i]);
        }
    }

    return quad_normals;
}

// compute area of quads corresponding to corners
std::vector<std::array<Scalar, 4>> compute_quad_vertex_areas(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    std::vector<std::array<Vector3, 4>> quad_normals = compute_quad_normals(V, F);

    int num_faces = F.rows();
    std::vector<std::array<Scalar, 4>> quad_vertex_areas(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        Vector3 x0 = V.row(F(f, 0));
        Vector3 x1 = V.row(F(f, 1));
        Vector3 x2 = V.row(F(f, 2));
        Vector3 x3 = V.row(F(f, 3));

        // compute princpal normal
        Vector3 X1 = (x1 - x0) + (x2 - x3);
        Vector3 X2 = (x2 - x1) + (x3 - x0);
        Vector3 Nc = X1.cross(X2);
        Vector3 nc = normalize(Nc);
        
        // compute vertex areas
        for (int i = 0; i < 4; ++i)
        {
            quad_vertex_areas[f][i] = nc.dot(quad_normals[f][i]);
        }
    }

    return quad_vertex_areas;
}

// compute the angle between two vectors
Scalar compute_quad_corner_angle(
    const Vector3& d20,
    const Vector3& d01,
    bool is_inverted)
{
    Scalar dot_prod = -d20.dot(d01);
    Scalar norm_prod = d20.norm() * d01.norm();
    if (norm_prod < 1e-24) return 0.;

    // compute the angle between [0, pi]
    Scalar angle = acos(max(min(dot_prod / norm_prod, 1.), -1.));

    // if the angle is known to be obtuse, subtract complementary acute angle from 2 pi
    return (is_inverted) ? ((2. * PI) - angle) : angle;
}

std::vector<std::array<Scalar, 4>> compute_quad_angles(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    std::vector<std::array<Vector3, 4>> quad_edges = compute_quad_edges(V, F);
    std::vector<std::array<Scalar, 4>> quad_vertex_areas = compute_quad_vertex_areas(V, F);

    int num_faces = F.rows();
    std::vector<std::array<Scalar, 4>> quad_angles(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        for (int i = 0; i < 4; ++i)
        {
            int j = (i + 3) % 4;

            // determine if inverted angle from quad areas
            bool is_inverted = (quad_vertex_areas[f][i] < 0.);

            // compute angle of quad corner
            quad_angles[f][i] = compute_quad_corner_angle(
                quad_edges[f][j],
                quad_edges[f][i],
                is_inverted);
        }
    }

    return quad_angles;
}

std::vector<Scalar> compute_scaled_jacobian(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    std::vector<std::array<Scalar, 4>> quad_vertex_areas = compute_quad_vertex_areas(V, F);
    std::vector<std::array<Scalar, 4>> quad_lengths = compute_quad_lengths(V, F);

    int num_faces = F.rows();
    std::vector<Scalar> scaled_jacobian(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        Eigen::Vector4<Scalar> vals;
        for (int i = 0; i < 4; ++i)
        {
            int j = (i + 3) % 4;
            vals[i] = quad_vertex_areas[f][i] / (quad_lengths[f][i] * quad_lengths[f][j]);
        }
        scaled_jacobian[f] = vals.minCoeff();
    }

    return scaled_jacobian;
}

}
}
