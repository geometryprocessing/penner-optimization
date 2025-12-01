#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/dual_loop.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Determine if a one form is valid.
 *
 * @param m: mesh
 * @param one_form: per-halfedge one-form
 * @return true iff the one form is valid
 */
bool is_valid_one_form(const Mesh<Scalar>& m, const VectorX& one_form);

/**
 * @brief Determine if a one form is closed
 *
 * @param m: mesh
 * @param one_form: per-halfedge one-form
 * @return true iff the one form is closed
 */
bool is_closed_one_form(const Mesh<Scalar>& m, const VectorX& one_form);

/**
 * @brief Given a list of dual loops, compute a matrix with columns given by the corresponding
 * closed one forms for the dual loops.
 *
 * @param m: mesh
 * @param dual_loops: loops defining the one forms
 * @return matrix of dual loop one forms
 */
MatrixX build_dual_loop_basis_one_form_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops);

/**
 * @brief Given a mesh and a homology basis, compute a matrix for the space of closed one forms
 * with the first |V| (or |V| - 1) columns corresponding to vertex hat function derivatives and the
 * last 2g corresponding to the homology basis loops.
 *
 * @param m: mesh
 * @param homology_basis_loops: homology basis loops for the mesh
 * @param eliminate_vertex: remove the last vertex of the mesh so that the matrix is full rank
 * @return matrix with basis forms as columns
 */
MatrixX build_closed_one_form_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    bool eliminate_vertex = false);

/**
 * @brief Compute the matrix that integrates a closed one-form over a cut of the mesh to a disk.
 *
 * @param m: mesh
 * @param cut_h: per-halfedge list of halfedges to cut (or an empty vector for no pregiven cuts)
 * @param is_cut_h: per-halfedge list of halfedges that are cut
 * @param start_h: (optional) halfedge to start the integration at
 * @return matrix representing the linear integration operation
 */
MatrixX build_one_form_integral_matrix(
    const Mesh<Scalar>& m,
    const std::vector<bool>& cut_h,
    std::vector<bool>& is_cut_h,
    int start_h=0);

/**
 * @brief Given a closed one form, compute its integral over a cut of the mesh to a disk.
 *
 * The integrated form is represented as a per-halfedge attribute, where the value corresponds
 * to the integrated value at the tip of the halfedge. The cut is implicitly defined by the
 * discontinuities in the resulting per-corner values.
 *
 * @param m: mesh
 * @param one_form: per-halfedge closed one-form
 * @param cut_h: per-halfedge list of halfedges to cut (or an empty vector for no pregiven cuts)
 * @param is_cut_h: per-halfedge list of halfedges that are cut
 * @param start_h: (optional) halfedge to start the integration at
 * @return: per-halfedge integrated one-form
 */
VectorX integrate_one_form(
    const Mesh<Scalar>& m,
    const VectorX& one_form,
    const std::vector<bool>& cut_h,
    std::vector<bool>& is_cut_h,
    int start_h = 0);

/**
 * @brief Generate matrix to scale the halfedges of a mesh with metric by scale factors associated
 * to the tips of halfedges of an integrated one form.
 *
 * @param m: mesh
 */
MatrixX build_integrated_one_form_scaling_matrix(const Mesh<Scalar>& m);

/**
 * @brief Scale the halfedges of a mesh with metric by scale factors associated to the tips of
 * halfedges of an integrated one form.
 *
 * Since we are using log coordinates, the scaling corresponds to addition of the values at
 * corners adjacent to the halfedge. Note that this operation may not preserve discrete metrics
 * as values for paired halfedges can be different.
 *
 * @param m: mesh
 * @param metric_coords: initial per-halfedge metric coordinates
 * @param integrated_one_form: per-halfedge integrated one-form
 * @return scaled metric: metric after scaling
 */
VectorX scale_halfedges_by_integrated_one_form(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const VectorX& integrated_one_form);

/**
 * @brief Scale the edges of a mesh with metric by scale factors associated to vertices.
 *
 * Since we are using log coordinates, the scaling corresponds to addition of the values at
 * corners adjacent to the halfedge. Note that this operation will preserve discrete metrics.
 *
 * @param m: mesh
 * @param metric_coords: initial per-halfedge metric coordinates
 * @param zero_form: per-vertex zero-form
 * @return scaled metric: metric after scaling
 */
VectorX scale_edges_by_zero_form(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const VectorX& zero_form);

} // namespace Holonomy
} // namespace Penner