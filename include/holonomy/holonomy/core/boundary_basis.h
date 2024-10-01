#pragma once

#include "holonomy/core/common.h"
#include "util/spanning_tree.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Class to generate boundary loops and boundary path basis loops
 */
class BoundaryBasisGenerator
{
public:
   /**
     * @brief Construct a new Boundary Basis Generator object on a mesh
     * 
     * @param m: mesh
     */
    BoundaryBasisGenerator(const Mesh<Scalar>& m);

    /**
     * @brief Get number of boundaries corresponding to basis loops.
     * 
     * This value is one less than the total number of boundary loops, and each
     * basis boundary corresponds to two loops
     * 
     * @return number of boundary basis loops
     */
    int n_basis_boundaries() const { return (m_basis_boundary_handles.size()); }

    /**
     * @brief Construct a basis loop corresponding to the basis boundary with given index
     * 
     * @param index: index of the basis boundary
     * @return sequence of faces defining the dual loop
     */
    std::vector<int> construct_boundary_basis_loop(int index) const;

    /**
     * @brief Construct a basis loop corresponding to the path from the basis boundary 
     * with the given index to a designated base boundary.
     * 
     * @param index: index of the basis boundary
     * @return sequence of faces defining the dual loop
     */
    std::vector<int> construct_boundary_path_basis_loop(int index) const;

    /**
     * @brief Modify basis loops to avoid marked halfedges on the boundary
     * 
     * @param is_marked_halfedge: list of marked halfedges
     * @return true if the basis loops were perturbed to avoid the marked edges
     * @return false otherwise
     */
    bool avoid_marked_halfedges(const std::vector<int>& marked_halfedges);

private:
    Mesh<Scalar> m_mesh;
    std::vector<int> m_he2e;
    std::vector<int> m_e2he;

    int m_root_boundary_handle;
    std::vector<int> m_basis_boundary_handles;
    DualTree m_dual_tree;
};

} // namespace Holonomy
} // namespace Penner
