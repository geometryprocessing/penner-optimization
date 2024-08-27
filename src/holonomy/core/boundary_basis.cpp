#include "holonomy/core/boundary_basis.h"

#include "holonomy/core/dual_lengths.h"
#include "util/boundary.h"

namespace Penner {
namespace Holonomy {

BoundaryBasisGenerator::BoundaryBasisGenerator(const Mesh<Scalar>& m)
    : m_mesh(m)
{
    // Build halfedge to edge maps
    build_edge_maps(m, m_he2e, m_e2he);

    // Get boundary components
    m_basis_boundary_handles = find_boundary_components(m);
    int root = 0;
    if (!m_basis_boundary_handles.empty()) {
        m_root_boundary_handle = m_basis_boundary_handles.back();
        root = m.f[m_root_boundary_handle];
        assert(m.type[m.h[root]] == 1);
        m_basis_boundary_handles.pop_back();
    }

    // Build spanning dual tree from the root boundary
    std::vector<Scalar> dual_edge_lengths = compute_dual_edge_lengths(m);
    m_dual_tree = DualTree(m, dual_edge_lengths, root, true);
};

std::vector<int> BoundaryBasisGenerator::construct_boundary_basis_loop(int index) const
{
    // Start from face adjacent to handle
    int start_h = m_basis_boundary_handles[index];

    // Circulate around boundary to build basis
    std::vector<int> basis_loop = {};
    int h = start_h;
    do
    {
        // Iterate once to prevent duplication
        h = m_mesh.opp[m_mesh.n[h]];

        // Circulate to next boundary edge, adding faces to basis
        while (m_mesh.type[h] != 2) {
            basis_loop.push_back(m_mesh.f[h]);
            h = m_mesh.opp[m_mesh.n[h]];
        }
        h = m_mesh.opp[h];

    } while (h != start_h);

    return basis_loop;
}

std::vector<int> BoundaryBasisGenerator::construct_boundary_path_basis_loop(int index) const
{
    int start_face = m_mesh.f[m_basis_boundary_handles[index]];
    std::vector<int> dual_path = {start_face};
    std::vector<int> dual_edges = {};

    // Trace up the dual tree until a root is reached
    int curr_face_index = start_face;
    while (!m_dual_tree.is_root(curr_face_index)) {
        // Get parent face of current face
        int edge_index = m_dual_tree.out(curr_face_index);
        curr_face_index = m_dual_tree.to(edge_index);
        assert(m_dual_tree.from(edge_index) == dual_path.back());

        // Add face and edge to the path
        dual_path.push_back(curr_face_index);
    }
    assert(curr_face_index == m_mesh.f[m_root_boundary_handle]);
    
    // Build dual loop from the path and its double copy
    std::vector<int> basis_loop = dual_path;
    basis_loop.reserve(2 * dual_path.size());
    for (auto itr = dual_path.rbegin(); itr != dual_path.rend(); ++itr)
    {
        int primal_face_index = *itr;
        int copy_face_index = m_mesh.f[m_mesh.R[m_mesh.h[primal_face_index]]];
        assert(m_mesh.type[m_mesh.h[copy_face_index]] == 2);
        basis_loop.push_back(copy_face_index);
    }

    return basis_loop;
}

} // namespace Holonomy
} // namespace Penner