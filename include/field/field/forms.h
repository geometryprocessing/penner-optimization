// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/common.h"
#include "metric/cone_metric.h"

/**
 * @brief Methods for creating and manipulating one forms on a surface, represented as
 * anti-symmetric halfedge values.
 * 
 * These halfedge values are interpretable as the integrated values of the form along the
 * oriented halfedges.
 * 
 * Includes methods to check whether forms are valid and closed, to integrate closed one
 * forms over a disk, and to scale vertices by integrated one forms. 
 * 
 */

namespace Penner {
namespace Field {

/**
 * @brief Determine if a one form is valid.
 *
 * @param m: mesh
 * @param one_form: per-halfedge one-form
 * @return true iff the one form is valid
 */
bool is_valid_one_form(const Mesh<Scalar>& m, const VectorX& one_form);

/**
 * @brief Determine if a one form is closed, meaning the sum of halfedge values in a face is zero
 *
 * @param m: mesh
 * @param one_form: per-halfedge one-form
 * @return true iff the one form is closed
 */
bool is_closed_one_form(const Mesh<Scalar>& m, const VectorX& one_form);

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

} // namespace Field
} // namespace Penner