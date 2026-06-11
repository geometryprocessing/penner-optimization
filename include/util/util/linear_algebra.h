// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "util/common.h"
#include <Eigen/SparseLU>

namespace Penner {

/**
 * @brief Compute the dot product of two 3D vectors.
 * 
 * @tparam vector representation of R3 supporting indexing
 * @param v1: first vector
 * @param v2: second vector
 * @return dot product of the two vectors
 */
template <typename VectorType>
double dot_prod(const VectorType& v1, const VectorType& v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/**
 * @brief Compute the cross product of two 3D vectors.
 * 
 * @tparam vector representation of R3 supporting indexing
 * @param v1: first vector
 * @param v2: second vector
 * @return cross product of the two vectors
 */
template <typename VectorType>
VectorType cross_prod(const VectorType& v1, const VectorType& v2)
{
    return VectorType(
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]);
}

/**
 * @brief Compute the signed angle between two 3D vectors around a normal axis.
 * 
 * @tparam vector representation of R3 supporting indexing
 * @param v1: first vector
 * @param v2: second vector
 * @param normal: normal vector representing the axis of rotation
 * @return signed angle between the two vectors around the axis of rotation
 */
template <typename VectorType>
double signed_angle(const VectorType& v1, const VectorType& v2, const VectorType& normal)
{
    double s = dot_prod(normal, cross_prod(v1, v2));
    double c = dot_prod(v1, v2);
    const double angle = (s == 0 && c == 0) ? 0.0 : atan2(s, c);
    return angle;
}


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

template <typename PreciseScalar, int Cols>
Eigen::Matrix<PreciseScalar, Eigen::Dynamic, Cols> solve_high_precision(
    const MatrixX& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Cols>& b
)
{
    typedef Eigen::SparseMatrix<PreciseScalar> MatrixXp;
    typedef Eigen::Matrix<PreciseScalar, Eigen::Dynamic, Cols> VectorXp;

    MatrixXp Ap = A.cast<PreciseScalar>();
    VectorXp bp = b.template cast<PreciseScalar>();
    Eigen::SparseLU<MatrixXp> solver;
    solver.compute(Ap);
    if (solver.info() != Eigen::Success) spdlog::error("matrix factorization failed");
    VectorXp xp = solver.solve(bp);
    Scalar solver_error = (Scalar)((Ap * xp - bp).norm());
    spdlog::trace("precise tutte solver error is {}", solver_error);

    return xp;
}

/**
 * @brief Solve a linear system with matrix valued right hand side
 * 
 * @param A: matrix to invert
 * @param B: right hand side matrix
 * @return solution to AX = B
 */
MatrixX solve_linear_system(const MatrixX& A, const MatrixX& B);

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

/**
 * @brief Compute a rotation matrix corresponding to a given angle
 * 
 * @param theta: rotation angle
 * @return rotation angle
 */
Matrix2x2 compute_rotation(Scalar theta);

/**
 * @brief Generate equally spaced values in an interval [a, b]
 * 
 * @param a: starting value
 * @param b: ending value
 * @param num_steps: number of steps to use in the linspace
 * @return a vector of equally spaced values between a and b
 */
std::vector<Scalar> generate_linspace(Scalar a, Scalar b, int num_steps);


} // namespace Penner