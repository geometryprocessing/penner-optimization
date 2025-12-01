/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#pragma once

#include "optimization/core/common.h"

namespace Penner {
namespace Optimization {

/// Class for incrementally building the flip change of coordinate matrix
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

} // namespace Optimization
} // namespace Penner