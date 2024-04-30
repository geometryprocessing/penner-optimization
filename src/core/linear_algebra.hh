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

#include "common.hh"

namespace CurvatureMetric {

/// Compute the Kronecker product of two vectors.
///
/// @param[in] vec_1: first vector to product
/// @param[in] vec_2: second vector to product
/// @return product of vec_1 and vec_2
VectorX kronecker_product(const VectorX& vec_1, const VectorX& vec_2);

/// Create an identity sparse matrix of dimension nxn
///
/// @param[in] n: matrix dimension
/// @return nxn identity matrix
MatrixX id_matrix(int n);

/// Create an empty IJV representation of a matrix with allocated space.
///
/// @param[in] capacity: space to reserve in the IJV arrays
/// @return IJV representation of a matrix with reserved space
std::tuple<std::vector<int>, std::vector<int>, std::vector<Scalar>> allocate_triplet_matrix(
    int capacity);

/// Compute the condition number of a matrix.
///
/// Note that this is a slow and unstable operation for large matrices
///
/// @param[in] matrix: input matrix
/// @return condition number of the matrix
Scalar compute_condition_number(const Eigen::MatrixXd matrix);

/// Solve a positive symmetric definite system Ax = b
///
/// The method used depends on the chosen compiler options.
///
/// @param[in] A: PSD matrix of the system Ax = b
/// @param[in] b: right hand side of the system Ax = b
/// @return solution x to Ax = b
VectorX solve_psd_system(const MatrixX& A, const VectorX& b);

/// Solve a general linear system Ax = b
///
/// The method used depends on the chosen compiler options.
///
/// @param[in] A: matrix of the system Ax = b
/// @param[in] b: right hand side of the system Ax = b
/// @return solution x to Ax = b
VectorX solve_linear_system(const MatrixX& A, const VectorX& b);

/// Given a matrix and lists of row and column indices, compute the
/// corresponding submatrix
///
/// @param[in] matrix: full matrix
/// @param[in] row_indices: indices of the rows to keep
/// @param[in] col_indices: indices of the cols to keep
/// @param[out] submatrix: corresponding submatrix
void compute_submatrix(
    const MatrixX& matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices,
    MatrixX& submatrix);

} // namespace CurvatureMetric
