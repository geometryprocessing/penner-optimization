#include "feature/core/common.h"

#include "holonomy/field/intrinsic_field.h"

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>

namespace Penner {
namespace Feature {

bool is_manifold(const Eigen::MatrixXi& F)
{
    Eigen::VectorXi B;
    if (!igl::is_edge_manifold(F)) return false;
    if (!igl::is_vertex_manifold(F, B)) return false;

    return true;
}

std::vector<int> generate_vertex_one_ring(const Mesh<Scalar>& m, int vertex_index)
{
    // circulate over one ring
    int h_start = m.out[vertex_index];
    int hij = h_start;
    std::vector<int> one_ring = {};
    do 
    {
        one_ring.push_back(hij);
        hij = m.n[m.opp[hij]];
    }
    while (hij != h_start);

    return one_ring;
}


// difference of principal curvatures relative to their total magnitude
Scalar compute_relative_anisotropy(Scalar max_val, Scalar min_val)
{
    return abs(max_val - min_val) / (abs(max_val) + abs(min_val));
}


// absolute difference of principal curvatures
Scalar compute_absolute_anisotropy(Scalar max_val, Scalar min_val)
{
    return abs(max_val - min_val);
}


// mean of two principal curvatures
Scalar compute_mean_anisotropy(Scalar max_val, Scalar min_val)
{
    return (max_val + min_val) / 2.;
}


// this measurement is near 0 for parabolic regions and near 1 for highly anisotropic regions
Scalar compute_parabolic_anisotropy(Scalar max_val, Scalar min_val)
{
    return abs(abs(max_val) - abs(min_val)) / max(abs(max_val), abs(min_val));
}


std::tuple<Eigen::MatrixXd, std::vector<bool>> compute_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius,
    Scalar abs_threshold,
    Scalar rel_threshold)
{
    //auto[max_direction, min_direction, _max_curvature, _min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, radius);
    //auto[_max_direction, _min_direction, max_curvature, min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, 3);
    auto[max_direction, min_direction, max_curvature, min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, radius);
    int num_faces = F.rows();
    std::vector<bool> is_fixed_direction(num_faces, false);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        Scalar kmax = max_curvature[fijk];
        Scalar kmin = min_curvature[fijk];
        if (compute_mean_anisotropy(kmax, kmin) < abs_threshold) continue;
        is_fixed_direction[fijk] = (compute_parabolic_anisotropy(kmax, kmin) > rel_threshold);
    }

    return std::make_tuple(max_direction, is_fixed_direction);
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> reindex_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& vtx_reindex)
{
    int num_orig_vertices = vtx_reindex.size();

    // reindex vertices
    Eigen::MatrixXd V_reindex = V;
    for (int vi = 0; vi < num_orig_vertices; ++vi) {
        V_reindex.row(vtx_reindex[vi]) = V.row(vi);
    }

    // reindex faces
    int num_faces = F.rows();
    Eigen::MatrixXi F_reindex = F;
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            if (F(fijk, i) >= num_orig_vertices) continue;
            F_reindex(fijk, i) = vtx_reindex[F(fijk, i)];
        }
    }

    return std::make_tuple(V_reindex, F_reindex);
}

std::vector<std::pair<int, int>> reindex_endpoints(
    const std::vector<std::pair<int, int>>& endpoints,
    const std::vector<int>& vtx_reindex)
{
    std::vector<int> vtx_reindex_inverse = invert_map(vtx_reindex);

    int num_orig_vertices = vtx_reindex.size();
    std::vector<std::pair<int, int>> endpoints_reindex = endpoints;
    for (int vi = 0; vi < num_orig_vertices; ++vi) {
        endpoints_reindex[vtx_reindex[vi]] = endpoints[vi];

        // check if need to reindex endpoints
        int e0 = endpoints[vi].first;
        int e1 = endpoints[vi].second;
        if ((e0 >= 0) || (e1 >= 0)) {
            spdlog::error("{} should be original vertex", vi);
        }
    }

    int num_vertices = endpoints.size();
    for (int vi = num_orig_vertices; vi < num_vertices; ++vi) {
        endpoints_reindex[vi] = endpoints[vi];

        // check if need to reindex endpoints
        int e0 = endpoints[vi].first;
        int e1 = endpoints[vi].second;
        if (e0 >= 0) {
            endpoints_reindex[vi].first = vtx_reindex[e0];
        }
        if (e1 >= 0) {
            endpoints_reindex[vi].second = vtx_reindex[e1];
        }
    }

    return endpoints_reindex;
}


} // namespace Feature
} // namespace Penner
