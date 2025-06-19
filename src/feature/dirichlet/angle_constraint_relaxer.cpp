#include "feature/dirichlet/angle_constraint_relaxer.h"
#include "feature/core/component_mesh.h"
#include "holonomy/holonomy/constraint.h"
#include "holonomy/holonomy/holonomy.h"
#include "optimization/core/constraint.h"
#include "optimization/core/viewer.h"
#include "util/io.h"
#include "util/vector.h"

namespace Penner {
namespace Feature {

AngleConstraintMatrixRelaxer::AngleConstraintMatrixRelaxer()
    : joined_corners({})
    , joined_corner_map({})
{}

MatrixX AngleConstraintMatrixRelaxer::run(const Mesh<Scalar>& m, const std::vector<std::pair<int, int>>& relaxed_edges)
{
    int num_relaxed_edges = relaxed_edges.size();
    int num_ind_vertices = m.n_ind_vertices();
    joined_corners.reserve(num_relaxed_edges);
    joined_corner_map = std::vector<int>(num_ind_vertices, -1);

    // for all relaxed edges, join the tip and tail vertices
    for (const auto& eij : relaxed_edges) {
        join_tip(m, eij);
        join_tail(m, eij);
    }

    // build constraint entries
    int count = 0;
    std::vector<T> tripletList;
    tripletList.reserve(num_ind_vertices);
    add_joined_vertex_constraints(m, tripletList, count);
    add_edge_holonomy_constraints(m, relaxed_edges, tripletList, count);

    // Create the constraint matrix from the triplets
    spdlog::info("relaxation to {} constraints for {} vertices", count, num_ind_vertices);
    MatrixX relaxation_matrix;
    relaxation_matrix.resize(count, num_ind_vertices);
    relaxation_matrix.reserve(tripletList.size());
    relaxation_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return relaxation_matrix;
}

void AngleConstraintMatrixRelaxer::view(const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V) const
{
    auto [V_double, F_mesh, F_halfedge] = Optimization::generate_doubled_mesh(V, m, vtx_reindex);

    // propagate joined corner labels to vertices and edges
    int num_ind_vertices = m.n_ind_vertices();
    int num_halfedges = m.n_halfedges();
    VectorX corner_label = VectorX::Zero(num_ind_vertices);
    VectorX edge_label = VectorX::Zero(num_halfedges);
    int num_corners = joined_corners.size();
    for (int ci = 0; ci < num_corners; ++ci)
    {
        // label corner vertex with joined corner index
        for (const auto& cIjk : joined_corners[ci].get_corners())
        {
            corner_label[cIjk] = ci;
        }

        // label halfedges with the corner at the base
        for (const auto& eij : joined_corners[ci].get_edges())
        {
            auto [hij, hji] = eij;
            edge_label[hij] = ci;
        }
    }
    VectorX edge_label_VF = Optimization::generate_FV_halfedge_data(F_halfedge, edge_label);
    VectorX corner_label_VF = vector_compose(corner_label, m.v_rep);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    std::string mesh_handle = "angle constraint matrix relaxer";
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addVertexScalarQuantity(
            "corner",
            convert_scalar_to_double_vector(corner_label_VF))
        ->setColorMap("spectral")
        ->setEnabled(true);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "edge",
            convert_scalar_to_double_vector(edge_label_VF))
        ->setColorMap("spectral")
        ->setEnabled(true);
    polyscope::show();
#endif
}

// join the corners across the edge at the tip of the given edge
void AngleConstraintMatrixRelaxer::join_tip(const Mesh<Scalar>& m, const std::pair<int, int>& eij)
{
    // get two corners at tip of edge
    auto [hij, hji] = eij;
    int cJki = m.v_rep[m.to[hij]];
    int cJil = m.v_rep[m.to[m.opp[hji]]];

    // case: both corners already part of a joined corner collection
    if ((is_joined_corner(cJki)) && (is_joined_corner(cJil))) {
        // unify joined corner index for two collections
        int cj0 = joined_corner_map[cJki];
        int cj1 = joined_corner_map[cJil];
        for (int corner : joined_corners[cj1].get_corners())
        {
            joined_corner_map[corner] = cj0;
        }

        // merge second corner into first
        joined_corners[cj0].merge(m, eij, joined_corners[cj1]);

        // remove data for merged second corner
        joined_corners[cj1] = {};
        spdlog::trace("Both vertices joined");
    }
    // case: only left corner part of a joined corner collection
    else if (is_joined_corner(cJki)) {
        // mark right corner as part of left collection
        int cj = joined_corner_map[cJki];
        joined_corner_map[cJil] = cj;

        // join new corner
        joined_corners[cj].join_adjacent_corner(m, eij);
    }
    // case: only right corner part of a joined corner collection
    else if (is_joined_corner(cJil)) {
        // mark left corner as part of right collection
        int cj = joined_corner_map[cJil];
        joined_corner_map[cJki] = cj;

        // join new corner
        joined_corners[cj].join_adjacent_corner(m, eij);
    }
    // case: neither corner part of an existing joined corner collection
    else {
        // build new joined corner index
        int cj = joined_corners.size();
        joined_corner_map[cJki] = cj;
        joined_corner_map[cJil] = cj;

        // make new joined corner for the given edge
        joined_corners.push_back(JoinedCorners(m, eij));
    }
}

// join the corners across the edge at the base of the given edge
void AngleConstraintMatrixRelaxer::join_tail(const Mesh<Scalar>& m, const std::pair<int, int>& eij)
{
    // equivalent to joining the tip of the reversed edge
    join_tip(m, swap_edge_vertex(eij));
}

// add constraints to the IJV matrix for joined vertices
void AngleConstraintMatrixRelaxer::add_joined_vertex_constraints(const Mesh<Scalar>& m, std::vector<T>& tripletList, int& count)
{
    // add unjoined vertex constraints: use identity rows
    int num_ind_vertices = m.n_ind_vertices();
    for (int vi = 0; vi < num_ind_vertices; ++vi) {
        // error: cannot join fixed vertices
        if ((is_joined_corner(vi)) && (m.fixed_dof[vi])) {
            spdlog::error("Fixed vertex in joined map");
            return;
        }

        // skip joined vertices (to process later)
        if (is_joined_corner(vi)) continue;

        // skip marked redundant vertex
        if (m.fixed_dof[vi]) continue;

        // add constraint for vertix
        tripletList.push_back(T(count, vi, 1.0));
        ++count;
    }

    // add joined vertex constraints: sum corner angles in collection
    int num_joined_corners = joined_corners.size();
    for (int i = 0; i < num_joined_corners; ++i) {
        const auto& c = joined_corners[i];
        if (c.n_corners() == 0) continue;
        for (int vi : c.get_corners()) {
            tripletList.push_back(T(count, vi, 1.0));
        }
        ++count;
    }
}

// add edge holonomy constraints to the IJV matrix
void AngleConstraintMatrixRelaxer::add_edge_holonomy_constraints(
    const Mesh<Scalar>& m,
    const std::vector<std::pair<int, int>>& relaxed_edges,
    std::vector<T>& tripletList,
    int& count)
{
    for (const auto& eij : relaxed_edges) {
        // get corners adjacent to edges
        auto [hij, hji] = eij;
        int ci = joined_corner_map[m.v_rep[m.to[hji]]];
        int cj = joined_corner_map[m.v_rep[m.to[hij]]];

        spdlog::debug(
            "Adding joined corner {}, {}, of size {}, {}",
            ci,
            cj,
            joined_corners[ci].n_corners(),
            joined_corners[cj].n_corners());
        
        // get alternate joined corners split by given edge
        auto ci_prefix = joined_corners[ci].get_prefix(swap_edge_vertex(eij));
        auto cj_suffix = joined_corners[cj].get_suffix(eij);
        spdlog::debug("First corner is {}", formatted_vector(ci_prefix));
        spdlog::debug("Second corner is {}", formatted_vector(cj_suffix));

        // add all angles in alternate joined corners
        for (int vi : ci_prefix) {
            tripletList.push_back(T(count, vi, 1.0));
        }
        for (int vj : cj_suffix) {
            tripletList.push_back(T(count, vj, 1.0));
        }
        ++count;
    }
}

// helper function to swap edge eij to eji (i.e., swap the vertex)
std::pair<int, int> AngleConstraintMatrixRelaxer::swap_edge_vertex(const std::pair<int, int>& eij) const
{
    return {eij.second, eij.first};
}

// determine if a corner is part of a joined collection
bool AngleConstraintMatrixRelaxer::is_joined_corner(int ci) const
{
    return (joined_corner_map[ci] >= 0);

}

std::vector<int> AngleConstraintMatrixRelaxer::JoinedCorners::get_prefix(const std::pair<int, int>& eij)
{
    std::vector<int> prefix;
    prefix.reserve(n_corners());

    // iterate over corners until the edge is found
    int num_edges = n_edges();
    for (int ij = 0; ij < num_edges; ++ij) {
        prefix.push_back(m_corners[ij]);

        // check for splitting edge
        if (eij == m_edges[ij]) {
            return prefix;
        }
    }

    // case: edge not found
    spdlog::error("Edge not in joined corner");
    return {};
}

std::vector<int> AngleConstraintMatrixRelaxer::JoinedCorners::get_suffix(const std::pair<int, int>& eij)
{
    std::vector<int> suffix;
    suffix.reserve(n_corners());

    // iterate over corners in reverse until the edge is found
    int num_edges = n_edges();
    for (int ij = num_edges - 1; ij >= 0; --ij) {
        suffix.push_back(m_corners[ij + 1]);

        // check for splitting edge
        if (eij == m_edges[ij]) {
            return suffix;
        }
    }

    // case: edge not found
    spdlog::error("Edge not in joined corner");
    return {};
}


void AngleConstraintMatrixRelaxer::JoinedCorners::join_adjacent_corner(const Mesh<Scalar>& m, const std::pair<int, int>& eij)
{
    int hij = eij.first;
    int hji = eij.second;

    // get corners adjacent to j
    int cJki = m.v_rep[m.to[hij]];
    int cJil = m.v_rep[m.to[m.opp[hji]]];

    // case: first edge 
    if (m_edges.empty()) {
        m_corners = {cJki, cJil};
        m_edges = {eij};
        return;
    }

    // case: corner at end of existing joined corners
    if (cJki == m_corners.back()) {
        m_corners.push_back(cJil);
        m_edges.push_back(eij);
    }
    // case: corner at front of existing joined corners
    else if (cJil == m_corners.front()) {
        m_corners.push_front(cJki);
        m_edges.push_front(eij);
    }
    // error: cannot append edge
    else {
        spdlog::error("Joining corner to non-adjacent collection");
        return;
    }

#if CHECK_VALIDITY
    // check validity
    if (!is_valid_joined_corners(m)) {
        spdlog::error("Invalid joined corners");
    }
#endif
}

void AngleConstraintMatrixRelaxer::JoinedCorners::merge(const Mesh<Scalar>& m, const std::pair<int, int>& eij, const JoinedCorners& adjacent_corners)
{
    // case: trivial corner collection
    if (m_edges.empty()) {
        m_corners = adjacent_corners.m_corners;
        m_edges = adjacent_corners.m_edges;
        return;
    }

    // get edge halfedges
    int hij = eij.first;
    int hji = eij.second;

    // get corners adjacent to j
    int cJki = m.v_rep[m.to[hij]];
    int cJil = m.v_rep[m.to[m.opp[hji]]];

    // case: append new corners to end of current joined corners
    if (cJki == m_corners.back()) {
        // add separating edge
        m_edges.push_back(eij);

        // append new joined corners and edges to the back
        for (int corner : adjacent_corners.m_corners)
        {
            m_corners.push_back(corner);
        }
        for (const auto& edge : adjacent_corners.m_edges)
        {
            m_edges.push_back(edge);
        }
    }
    // case: append new corners to front of currnet joined corners
    else if (cJil == m_corners.front()) {
        // add separating edge
        m_edges.push_front(eij);

        // append new joined corners and edges to the front in reverse
        for (auto itr = adjacent_corners.m_corners.rbegin(); itr != adjacent_corners.m_corners.rend(); ++itr)
        {
            m_corners.push_front(*itr);
        }
        for (auto itr = adjacent_corners.m_edges.rbegin(); itr != adjacent_corners.m_edges.rend(); ++itr)
        {
            m_edges.push_front(*itr);
        }
    }
    // case: cannot append edge
    else {
        spdlog::error("No match found");
        return;
    }

#if CHECK_VALIDITY
    // check validity
    if (!is_valid_joined_corners(m)) {
        spdlog::error("Invalid joined corners");
    }
#endif
}

// check validity of the joined corners
bool AngleConstraintMatrixRelaxer::JoinedCorners::is_valid_joined_corners(const Mesh<Scalar>& m) const
{
    int num_edges = m_edges.size();
    int num_corners = m_corners.size();

    // check for trivial joined corners
    if (num_edges == 0) {
        return (num_corners == 0);
    }

    // check one more corner than edges
    if (num_corners != (num_edges + 1))
    {
        spdlog::error("Inconsistent number of edges and corners");
        return false;
    }

    // check consistency of edges and corners
    for (int ij = 0; ij < num_edges; ++ij) {
        auto [hij, hji] = m_edges[ij];
        int cJki = m.v_rep[m.to[hij]];
        int cJil = m.v_rep[m.to[m.opp[hji]]];
        if (m_corners[ij] != cJki)
        {
            spdlog::error("Inconsistent right corner");
            return false;
        }
        if (m_corners[ij + 1] != cJil)
        {
            spdlog::error("Inconsistent left corner");
            return false;
        }
    }

    // no issues found
    return true;
}

MatrixX compute_relaxed_angle_constraint_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::pair<int, int>>& relaxed_edges)
{
    AngleConstraintMatrixRelaxer relaxer;
    return relaxer.run(m, relaxed_edges);
}


} // namespace Feature
} // namespace Penner