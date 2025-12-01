
#include "feature/experimental/transfer.h"

#include "feature/core/union_meshes.h"
#include "holonomy/holonomy/cones.h"

#include "util/vector.h"

namespace Penner {
namespace Feature {


// compute map from vertex-vertex edges to halfedges for cut mesh
Eigen::SparseMatrix<int> generate_cut_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    int n_v = V_map.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);

    // build sparse matrix
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int n_cut_he = m.n_halfedges();
    for (int hij = 0; hij < n_cut_he; ++hij) {
        if (m.type[hij] > 1) continue;

        int hji = m.opp[hij];
        int vi = V_map[vtx_reindex[m.v_rep[m.to[hji]]]];
        int vj = V_map[vtx_reindex[m.v_rep[m.to[hij]]]];
        trips.push_back(Trip(vi, vj, hij + 1));
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

VectorX transfer_edge_data(
    const Mesh<Scalar>& m_0,
    const std::vector<int>& vtx_reindex_0,
    const Eigen::VectorXi& V_map_0,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const VectorX& halfedge_data,
    bool is_signed)
{
    Eigen::SparseMatrix<int> vv2he = generate_cut_VV_to_halfedge_map(m, vtx_reindex, V_map);

    // iterate over halfedges
    int num_halfedges = m_0.n_halfedges();
    VectorX halfedge_data_0(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m_0.opp[hij];
        if (hij < hji) continue; // only process each edge once
        if ((m_0.type[hij] == 2) && (m_0.type[hji] == 2))
            continue; // don't process copied interior edges

        // get corrresponding (potentially cut) halfedge in domain mesh from VV values
        int vi = V_map_0[vtx_reindex_0[m_0.v_rep[m_0.to[hji]]]];
        int vj = V_map_0[vtx_reindex_0[m_0.v_rep[m_0.to[hij]]]];
        int domain_hij = vv2he.coeffRef(vi, vj) - 1;
        assert(domain_hij >= 0);

        // if target mesh edge is cut, just copy edge value directly
        if (m_0.type[hij] != m_0.type[hji])
        {
            halfedge_data_0[hij] = halfedge_data[domain_hij];
            halfedge_data_0[hji] = halfedge_data[m.opp[domain_hij]];
        }

        // get (type 1) domain halfedge corresponding to the opposite halfedge in the original mesh before cuts
        // WARNING: this is generally different than the (type 2) opposite halfedge in the cut mesh
        int domain_hji = vv2he.coeffRef(vj, vi) - 1;
        assert(domain_hji >= 0);

        // for signed values, take half of difference between values
        if (is_signed)
        {
            halfedge_data_0[hij] = 0.5 * (halfedge_data[domain_hij] - halfedge_data[domain_hji]);
            halfedge_data_0[hji] = -halfedge_data_0[hij];

            // copy to reflection with correct sign
            if (m_0.type[hji] == 1) {
                halfedge_data_0[m_0.R[hij]] = -halfedge_data_0[hij];
                halfedge_data_0[m_0.R[hji]] = halfedge_data_0[hij];
            }
        }
        // for unsigned values, take standard average
        else
        {
            halfedge_data_0[hij] = 0.5 * (halfedge_data[domain_hij] + halfedge_data[domain_hji]);
            halfedge_data_0[hji] = halfedge_data_0[hij];

            // copy to reflection
            if (m_0.type[hji] == 1) {
                halfedge_data_0[m_0.R[hij]] = halfedge_data_0[hij];
                halfedge_data_0[m_0.R[hji]] = halfedge_data_0[hij];
            }
        }
    }

    return halfedge_data_0;
}

void transfer_metric(
    MarkedPennerConeMetric& marked_metric_0,
    const std::vector<int>& vtx_reindex_0,
    const Eigen::VectorXi& V_map_0,
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    // transfer unsigned metric values
    VectorX metric_coords = marked_metric.get_metric_coordinates();
    bool is_signed = false;
    VectorX metric_coords_0 = transfer_edge_data(
        marked_metric_0,
        vtx_reindex_0,
        V_map_0,
        marked_metric,
        vtx_reindex,
        V_map,
        metric_coords,
        is_signed);

    // change metric with new coordinates
    marked_metric_0.change_metric(marked_metric_0, metric_coords_0);
}

VectorX transfer_rotation_form(
    const Mesh<Scalar>& m_0,
    const std::vector<int>& vtx_reindex_0,
    const Eigen::VectorXi& V_map_0,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const VectorX& rotation_form)
{
    // transfer signed rotation form
    bool is_signed = true;
    return transfer_edge_data(
        m_0,
        vtx_reindex_0,
        V_map_0,
        m,
        vtx_reindex,
        V_map,
        rotation_form,
        is_signed);
}

/*
DEPRECATED
std::tuple<DirichletPennerConeMetric, std::vector<int>, VectorX, std::vector<Scalar>>
generate_transferred_metric(
    const Eigen::MatrixXd& V_cut,
    const Eigen::MatrixXi& F_cut,
    const Eigen::VectorXi& V_map,
    const MarkedPennerConeMetric& domain_marked_metric,
    const std::vector<int>& domain_vtx_reindex,
    const Eigen::VectorXi& domain_V_map,
    const VectorX& domain_rotation_form,
    Holonomy::MarkedMetricParameters marked_metric_params)
{
    // generate mesh for the new cut topology
    auto [union_mesh, union_vtx_reindex] = generate_union_mesh(V_cut, F_cut);

    // transfer rotation form and cone constraints to new union mesh
    VectorX transferred_rotation_form = transfer_rotation_form(
        union_mesh,
        union_vtx_reindex,
        V_map,
        domain_marked_metric,
        domain_vtx_reindex,
        domain_V_map,
        domain_rotation_form);
    std::vector<Scalar> transferred_Th_hat = Holonomy::generate_cones_from_rotation_form(
        union_mesh,
        union_vtx_reindex,
        transferred_rotation_form,
        true);
    union_mesh.Th_hat = Holonomy::generate_cones_from_rotation_form(union_mesh, transferred_rotation_form);

    // add transferred metric to union mesh
    auto union_metric = generate_marked_metric_components(
        union_mesh,
        transferred_rotation_form,
        marked_metric_params);
    transfer_metric(
        union_metric,
        union_vtx_reindex,
        V_map,
        domain_marked_metric,
        domain_vtx_reindex,
        domain_V_map);

    // add Dirichlet conditions to union metric
    DirichletPennerConeMetric transfered_metric =
        generate_dirichlet_metric_from_mesh(union_metric, union_vtx_reindex, V_map);

    return std::make_tuple(
        transfered_metric,
        union_vtx_reindex,
        transferred_rotation_form,
        transferred_Th_hat);
}
*/


} // namespace Feature
} // namespace Penner