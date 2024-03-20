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
