// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/common.h"

/**
 * @brief Data structure to iteratively compute the change of coordinate matrix for intrinsic flips, e.g., in Penner coordinates.
 * 
 */

namespace Penner {

/// Class for iteratively building the flip change of coordinate matrix
class FlipMatrixGenerator
{
public:
    /// Initialize an identity matrix
    ///
    /// @param[in] size: dimension of the flip matrix
    FlipMatrixGenerator(int size);

    /// Reset the matrix to the identity
    void reset();

    /// Multiply the current matrix by a flip matrix Di for a given edge on the left.
    ///
    /// The flip matrix is the identity except the row ed, which has the values indicated
    /// by the (ed, j, v) specified by matrix indices and scalars
    ///
    /// @param[in] column_indices: column indices j for row ed of the matrix Di
    /// @param[in] values: values v for row ed of the matrix Di
    /// @param[in] ed: edge index for the nontrivial row of the matrix Di
    void multiply_by_matrix(
        const std::array<int, 5>& column_indices,
        const std::array<Scalar, 5>& values,
        int ed);

    /// Generate a standard sparse representation of the current matrix
    ///
    /// @return flip matrix
    MatrixX build_matrix() const;

private:
    int m_size;
    std::vector<std::vector<std::pair<int, Scalar>>> m_list_of_lists;
};

/// Class for incrementally building the flip change of coordinate matrix using a standard template
/// library map representation
///
/// NOTE: Potential alternative representation of the flip matrix generator.
class FlipMapMatrixGenerator
{
public:
    /// Initialize an identity matrix
    ///
    /// @param[in] size: dimension of the flip matrix
    FlipMapMatrixGenerator(int size);

    /// Reset the matrix to the identity
    void reset();

    /// Multiply the current matrix by a flip matrix Di for a given edge on the left.
    ///
    /// The flip matrix is the identity except the row ed, which has the values indicated
    /// by the (ed, j, v) specified by matrix indices and scalars
    ///
    /// @param[in] column_indices: column indices j for row ed of the matrix Di
    /// @param[in] values: values v for row ed of the matrix Di
    /// @param[in] ed: edge index for the nontrivial row of the matrix Di
    void multiply_by_matrix(
        const std::array<int, 5>& column_indices,
        const std::array<Scalar, 5>& values,
        int ed);

    /// Generate a standard sparse representation of the current matrix
    ///
    /// @return flip matrix
    MatrixX build_matrix() const;

private:
    int m_size;
    std::vector<std::map<int, Scalar>> m_list_of_lists;
};

} // namespace Penner