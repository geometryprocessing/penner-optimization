#include "util/vf_corners.h"

#include <igl/triangle_triangle_adjacency.h>
#include <igl/remove_unreferenced.h>

namespace Penner {

Eigen::MatrixXi mask_difference(
    const Eigen::MatrixXi& F_is_in,
    const Eigen::MatrixXi& F_is_out)
{
    int num_faces = F_is_in.rows();
    Eigen::MatrixXi F_mask(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            F_mask(fijk, i) = ((F_is_in(fijk, i)) && (!F_is_out(fijk, i))) ? 1 : 0;
        }
    }

	return F_mask;
}


std::vector<std::pair<int, int>> compute_mask_corners(
    const Eigen::MatrixXi& F_mask
) {
    int num_faces = F_mask.rows();

    // scan matrix for nonzero entries
    std::vector<std::pair<int, int>> mask_corners;
    mask_corners.reserve(3 * num_faces);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (!F_mask(fijk, i)) continue;
            mask_corners.push_back({fijk, i});
        }
    }

	return mask_corners;
}


Eigen::MatrixXi compute_mask_from_corners(
    int num_faces,
    const std::vector<std::pair<int, int>>& mask_corners
) {
    // mark matrix entries in the corner list with a 1
    Eigen::MatrixXi F_mask = Eigen::MatrixXi::Zero(num_faces, 3);
    for (const auto& mask_corner : mask_corners)
    {
        auto [f, i] = mask_corner;
        F_mask(f, i) = 1;
    }

	return F_mask;
}


std::vector<VertexEdge> compute_corner_edges(
    const std::vector<std::pair<int, int>>& corners,
    const Eigen::MatrixXi& F)
{
    int num_corners = corners.size();
    std::vector<VertexEdge> corner_edges(num_corners);
    for (int ci = 0; ci < num_corners; ++ci)
    {
        // get face and local corner index
        auto [fijk, k] = corners[ci];

        // get local indices of opposite edge
        int i = (k + 1) % 3;
        int j = (i + 1) % 3;

        // get global indices of opposite edge
        int vi = F(fijk, i);
        int vj = F(fijk, j);

        // write edge vertices
        corner_edges[ci] = {vi, vj};
    }

    return corner_edges;
}


// build a mask marking oriented halfedge endpoints (vi, vj) 
Eigen::SparseMatrix<int> generate_edge_matrix(
    int num_vertices,
    const std::vector<VertexEdge>& edges)
{
    Eigen::SparseMatrix<int> vv_is_edge(num_vertices, num_vertices);
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    for (auto [vi, vj] : edges)
    {
        trips.push_back(Trip(vi, vj, 1));
    }
    vv_is_edge.setFromTriplets(trips.begin(), trips.end());
    return vv_is_edge;
}

std::vector<std::pair<int, int>> compute_edge_corners(
    const std::vector<VertexEdge>& edges,
    const Eigen::MatrixXi& F)
{
    // get mask for edges
    int num_vertices = F.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv_is_edge = generate_edge_matrix(num_vertices, edges);

    // iterate over all faces and list face corners corresponding to an oriented edge
    int num_edges = edges.size();
    std::vector<std::pair<int, int>> edge_corners;
    edge_corners.reserve(num_edges);
    int num_faces = F.rows();
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int k = 0; k < 3; ++k)
        {
            // get local indices of opposite edge
            int i = (k + 1) % 3;
            int j = (i + 1) % 3;

            // get global indices of opposite edge
            int vi = F(fijk, i);
            int vj = F(fijk, j);

            // check if edge and add to list if so
            if (!vv_is_edge.coeff(vi, vj)) continue;
            edge_corners.push_back({fijk, k});
        }
    }

    return edge_corners;
}

std::vector<FaceEdge> compute_face_edges_from_corners(
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& corners
) {
    // scan matrix for nonzero entries
    std::vector<FaceEdge> edges;
    int num_corners = corners.size();
    edges.reserve(num_corners);

    // initialize a matrix to track seen edges
    int n_v = F.maxCoeff() + 1;
    Eigen::SparseMatrix<int> corner_map(n_v, n_v);

    // find unique corner per edge
    for (int ci = 0; ci < num_corners; ++ci)
    {

        // get global edge endpoint indices
        auto [fijk, k] = corners[ci];
        int i = (k + 1) % 3;
        int j = (k + 2) % 3;
        int vi = F(fijk, i);
        int vj = F(fijk, j);

        // skip seen corners
        if (corner_map.coeff(vi, vj) >= 1) continue;

        // check if edge is seen and add face edge if so
        int opp_corner = corner_map.coeff(vj, vi) - 1;
        if (opp_corner >= 0)
        {
            auto [fjil, l] = corners[opp_corner];
            edges.push_back(FaceEdge(fijk, k, fjil, l));
        }

        // add edge and mark as seen
        corner_map.coeffRef(vi, vj) = ci + 1;
    }
    spdlog::info("Extracted {} edges from {} corners", edges.size(), corners.size());

	return edges;
}


std::vector<VertexEdge> compute_face_edge_endpoints(
    const std::vector<FaceEdge>& face_edges,
    const Eigen::MatrixXi& F)
{
    int num_edges = face_edges.size();
    std::vector<VertexEdge> edge_endpoints(num_edges);
    for (int eij = 0; eij < num_edges; ++eij)
    {
        // get face and local corner index
        auto [fijk, k] = face_edges[eij].right_corner();

        // get local indices of opposite edge
        int i = (k + 1) % 3;
        int j = (i + 1) % 3;

        // get global indices of opposite edge
        int vi = F(fijk, i);
        int vj = F(fijk, j);

        // write edge vertices
        edge_endpoints[eij] = {vi, vj};
    }

    return edge_endpoints;
}


Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex)
{

    // generate IJV representation of the matrix
    int n_he = m.n_halfedges();
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    trips.reserve(n_he);
    for (int hij = 0; hij < n_he; ++hij) {
        // only process primal mesh
        if (m.type[hij] == 2) continue;

        // get edge endpoints in VF mesh indexing
        int hji = m.opp[hij];
        int vi = vtx_reindex[m.v_rep[m.to[hji]]];
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];

        // add entry mapping (vi, vj) to the one-indexed halfedge index
        trips.push_back(Trip(vi, vj, hij + 1));
    }

    // convert IJV to sparse matrix
    int n_v = m.n_ind_vertices();
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{

    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int n_he = m.n_halfedges();
    for (int hij = 0; hij < n_he; ++hij) {
        // only process primal mesh
        if (m.type[hij] == 2) continue;

        // get edge endpoints in glued VF mesh indexing
        int hji = m.opp[hij];
        int vi = V_map[vtx_reindex[m.v_rep[m.to[hji]]]];
        int vj = V_map[vtx_reindex[m.v_rep[m.to[hij]]]];

        // add entry mapping (vi, vj) to the one-indexed halfedge index
        trips.push_back(Trip(vi, vj, hij + 1));
    }

    // convert IJV to sparse matrix
    int n_v = V_map.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

std::vector<std::pair<int, int>> compute_relaxed_edges(
    const std::vector<std::pair<int, int>>& relaxed_corners,
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXi& F_cut
) {
    // get map from oriented vertex pairs to cut halfedges
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(m, vtx_reindex, V_map);

    // construct relaxed edges
    std::vector<std::pair<int, int>> relaxed_edges;
    relaxed_edges.reserve(relaxed_corners.size());
    for (const auto& ci : relaxed_corners)
    {
        // get local edge indices
        auto [fijk, k] = ci;
        int i = (k + 1) % 3;
        int j = (k + 2) % 3;

        // get global glued edge indices
        int vi = V_map[F_cut(fijk, i)];
        int vj = V_map[F_cut(fijk, j)];

        // only process each edge once
        if (vi < vj) continue;

        // get primal cut mesh halfedges corresponding to relaxed edge
        int hij = vv2he.coeff(vi, vj) - 1;
        int hji = vv2he.coeff(vj, vi) - 1;

        // add edge halfedges to list
        relaxed_edges.push_back({hij, hji});
        spdlog::debug("Adding edge ({}, {})", vi, vj);
    }

    return relaxed_edges;
}

Eigen::MatrixXi generate_overlay_cut_mask(
    const Eigen::MatrixXi& F_overlay,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::MatrixXi& F_cut,
    const Eigen::MatrixXi& F_is_cut
    )
{
    // build sparse matrix marking cut edges
    int num_vertices = F_cut.maxCoeff() + 1;
    Eigen::SparseMatrix<int> vv_is_cut(num_vertices, num_vertices);
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int num_faces = F_cut.rows();
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;

            // skip edges that are not cut
            if (!F_is_cut(fijk, k)) continue;

            // add mask for cut edge
            int vi = F_cut(fijk, i);
            int vj = F_cut(fijk, j);
            trips.push_back(Trip(vi, vj, 1));
        }
    }
    vv_is_cut.setFromTriplets(trips.begin(), trips.end());

    // mark overlay edges corresponding to cut original edges
    int num_overlay_faces = F_overlay.rows();
    Eigen::MatrixXi F_overlay_is_cut = Eigen::MatrixXi::Zero(num_overlay_faces, 3);
    for (int fijk = 0; fijk < num_overlay_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            // get overlay vertices of edge eij
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            int vi = F_overlay(fijk, i);
            int vj = F_overlay(fijk, j);

            // determine endpoints of the segment containing edge eij
            int ei0 = endpoints[vi].first;
            int ei1 = endpoints[vi].second;
            int ej0 = endpoints[vj].first;
            int ej1 = endpoints[vj].second;
            int e0, e1;

            // case: both vi and vj are original; overlay edge is uncut
            if ((ei0 == -1) && (ei1 == -1) && (ej0 == -1) && (ej1 == -1))
            {
                e0 = vj;
                e1 = vi;
            }
            // case: both vi and vj are splits on the same segment; use either endpoints
            else if ((ei0 == ej0) && (ei1 == ej1))
            {
                e0 = ei0;
                e1 = ej1;
            }
            // case: vi is original and vj is a split; use endpoints of vj
            else if ((ei0 == -1) && (ei1 == -1) && ((ej0 == vi) || (ej1 == vi)))
            {
                e0 = ej0;
                e1 = ej1;
            }
            // case: vj is original and vi is a split; use endpoints of vi
            else if ((ej0 == -1) && (ej1 == -1) && ((ei0 == vj) || (ei1 == vj)))
            {
                e0 = ei0;
                e1 = ei1;
            }
            // case: segment not on an original edge; skip
            else
            {
                continue;
            }

            // mark overlay edge as cut if original edge is cut
            if ((vv_is_cut.coeff(e0, e1)) || (vv_is_cut.coeff(e1, e0)))
            {
                F_overlay_is_cut(fijk, k) = 1;
            }
        }
    }

    return F_overlay_is_cut;
}

std::vector<std::pair<int, int>> prune_redundant_edge_corners(
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int, int>>& corners)
{
    std::vector<std::pair<int, int>> pruned_corners = {};
    pruned_corners.reserve(corners.size());

    // initialize a matrix to track seen edges
    int n_v = F.maxCoeff() + 1;
    Eigen::SparseMatrix<bool> is_edge_seen(n_v, n_v);

    // find unique corner per edge
    for (const auto& corner : corners)
    {
        // get global edge endpoint indices
        auto [fijk, k] = corner;
        int i = (k + 1) % 3;
        int j = (k + 2) % 3;
        int vi = F(fijk, i);
        int vj = F(fijk, j);

        // check if edge is seen
        if (is_edge_seen.coeff(vi, vj)) continue;

        // add edge and mark as seen
        pruned_corners.push_back(corner);
        is_edge_seen.coeffRef(vi, vj) = true;
        is_edge_seen.coeffRef(vj, vi) = true;
    }
    spdlog::info("Pruned {} to {} corners", corners.size(), pruned_corners.size());

    return pruned_corners;
}


VectorX transfer_corner_function_to_halfedge(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& corner_func)
{
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(m, vtx_reindex, V_map);
    int num_faces = F.rows();
    VectorX halfedge_func(m.n_halfedges());
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;
            int vj = F(fijk, j);
            int vk = F(fijk, k);
            int hjk = vv2he.coeff(vj, vk) - 1;
            halfedge_func[hjk] = corner_func(fijk, i);
            if (m.type[hjk] > 0)
            {
                halfedge_func[m.R[hjk]] = halfedge_func[hjk];
            }
        }
    }

    return halfedge_func;
}


Eigen::MatrixXd transfer_halfedge_function_to_corner(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    const Eigen::MatrixXi& F,
    const VectorX& halfedge_func)
{
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(m, vtx_reindex, V_map);
    int num_faces = F.rows();
    Eigen::MatrixXd corner_func(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;
            int vj = F(fijk, j);
            int vk = F(fijk, k);
            int hjk = vv2he.coeff(vj, vk) - 1;
            corner_func(fijk, i) = (double)(halfedge_func[hjk]);
        }
    }

    return corner_func;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
generate_edges(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_is_edge)
{
    // get edges from mask
    std::vector<std::pair<int, int>> edge_corners = compute_mask_corners(F_is_edge);
    edge_corners = prune_redundant_edge_corners(F, edge_corners);
    std::vector<VertexEdge> refined_edges = compute_corner_edges(edge_corners, F);

    // build edge list
    int num_edges = refined_edges.size();
    Eigen::MatrixXi E(num_edges, 2);
    for (int eij = 0; eij < num_edges; ++eij)
    {
        E(eij, 0) = refined_edges[eij][0];
        E(eij, 1) = refined_edges[eij][1];
    }

    // get edge vertices from mesh vertices
    Eigen::MatrixXd VN;
    Eigen::MatrixXi EN;
    Eigen::VectorXi I;
    igl::remove_unreferenced(V, E, VN, EN, I);

    return std::make_tuple(VN, EN);
}


} // namespace Penner