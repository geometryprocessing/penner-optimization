// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "holonomy/core/common.h"
#include "util/spanning_tree.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Methods to generate basis dual loops for the double of meshes with boundary.
 * 
 * These loops correspond to alignment constraints between different boundary components
 * of a mesh with boundary. For a mesh with b boundary components, b - 1 constraints are needed.
 * The constraint measures the holonomy between a base boundary component and another component.
 * 
 * In the doubled mesh for a mesh with boundary, the paths between components become dual loops.
 * 
 * Note that the double mesh has b - 1 handles corresponding to the boundary loops, which results
 * in an increase in genus of 2b - 2. However, the holonomy of b - 1 loops are determined by cone
 * angle constraints and symmetry.
 * 
 */

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