
#include "feature/feature/gluing.h"

#include "feature/core/vf_corners.h"
#include "holonomy/holonomy/cones.h"

namespace Penner {
namespace Feature {



std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
generate_boundary_pairs(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    // build map from glued vertex edge pairs to primal cut halfedges
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(m, vtx_reindex, V_map);

    // find boundary halfedges
    int n_he = m.n_halfedges();
    std::vector<int> he2bd(n_he, -1);
    std::vector<int> boundary_he = {};
    for (int hij = 0; hij < n_he; ++hij) {
        int hji = m.opp[hij];
        if ((m.type[hij] == 1) && (m.type[hji] == 2)) {
            he2bd[hij] = boundary_he.size();
            boundary_he.push_back(hij);
        }
    }
    spdlog::debug("{} boundary he found", boundary_he.size());

    // Create length identification pairs
    std::vector<std::pair<int, int>> bd_pairs;
    bd_pairs.reserve(boundary_he.size());
    for (int hij : boundary_he) {
        // get paired halfedge from edge endpoints map
        int hji = m.opp[hij];
        int vi = V_map[vtx_reindex[m.v_rep[m.to[hji]]]];
        int vj = V_map[vtx_reindex[m.v_rep[m.to[hij]]]];
        assert(hij == vv2he.coeffRef(vi, vj) - 1);
        int he_pair = vv2he.coeffRef(vj, vi) - 1;
        if (hij < he_pair) continue; // only add one to avoid redundancy

        // add pair to list
        bd_pairs.push_back(std::make_pair(hij, he_pair));
    }
    spdlog::debug("{} boundary pairs found", bd_pairs.size());

    return std::make_tuple(bd_pairs, boundary_he, he2bd);
}

std::vector<Scalar> compute_glued_angles(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    int num_glued_vertices = V_map.maxCoeff() + 1;
    std::vector<Scalar> Th_hat(num_glued_vertices, 0.);

    // sum up angles, dividing by 2 to account for doubling
    int num_ind_vertices = m.n_ind_vertices();
    for (int vi = 0; vi < num_ind_vertices; ++vi)
    {
        Th_hat[V_map[vtx_reindex[vi]]] += (m.Th_hat[vi] / 2.);
    }

    return Th_hat;
}

std::vector<Scalar> compute_glued_angle_defects(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    // get target angles
    std::vector<Scalar> Th_hat = compute_glued_angles(m, vtx_reindex, V_map);

    // subtract target from flat angle
    int num_ind_vertices = Th_hat.size();
    for (int vi = 0; vi < num_ind_vertices; ++vi)
    {
        Th_hat[vi] = (2. * M_PI) - Th_hat[vi];
    }

    return Th_hat;
}


std::pair<int, int> count_glued_cones(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map
) {
    // if mesh is not cut, use regular cone counting method
    bool is_symmetric = (m.type[0] != 0);
    if (!is_symmetric) return Holonomy::count_cones(m);

    // get glued cone angles
    std::vector<Scalar> Th_hat = compute_glued_angles(m, vtx_reindex, V_map);

    // Check for cones
    int num_ind_vertices = Th_hat.size();
    int num_neg_cones = 0;
    int num_pos_cones = 0;
    for (int vi = 0; vi < num_ind_vertices; ++vi) {
        Scalar flat_angle = 2. * M_PI;
        if (Th_hat[vi] > flat_angle + 1e-3) {
            spdlog::trace("{} cone found", Th_hat[vi]);
            num_neg_cones++;
        }
        if (Th_hat[vi] < flat_angle - 1e-3) {
            spdlog::trace("{} cone found", Th_hat[vi]);
            num_pos_cones++;
        }
    }

    return std::make_pair(num_neg_cones, num_pos_cones);
}

} // namespace Feature
} // namespace Penner