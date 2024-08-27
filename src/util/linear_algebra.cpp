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
#include "util/linear_algebra.h"

#ifdef USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#endif

#include "util/vector.h"

namespace Penner {

VectorX kronecker_product(const VectorX& vec_1, const VectorX& vec_2)
{
    if (vec_1.size() != vec_2.size()) {
        spdlog::error("Cannot multiply two vectors of different sizes");
        return VectorX();
    }

    // Build product component-wise
    size_t vec_size = vec_1.size();
    VectorX product(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        product[i] = vec_1[i] * vec_2[i];
    }

    return product;
}

MatrixX id_matrix(int n)
{
    // Build triplet lists for the identity
    std::vector<T> tripletList;
    tripletList.reserve(n);
    for (int i = 0; i < n; ++i) {
        tripletList.push_back(T(i, i, 1.0));
    }

    // Create the matrix from the triplets
    MatrixX id(n, n);
    id.reserve(tripletList.size());
    id.setFromTriplets(tripletList.begin(), tripletList.end());

    return id;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<Scalar>> allocate_triplet_matrix(
    int capacity)
{
    std::vector<int> I;
    std::vector<int> J;
    std::vector<Scalar> V;
    I.reserve(capacity);
    J.reserve(capacity);
    V.reserve(capacity);

    return std::make_tuple(I, J, V);
}

Scalar compute_condition_number(const Eigen::MatrixXd matrix)
{
    // Check for square matrix
    if (matrix.rows() != matrix.cols()) return 0.0;

    // Compute condition number with singular values
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
    Scalar sigma_1 = svd.singularValues()(0);
    Scalar sigma_n = svd.singularValues()(svd.singularValues().size() - 1);
    spdlog::trace("Min and max singular values are {} and {}", sigma_n, sigma_1);

    return sigma_1 / sigma_n;
}

VectorX solve_psd_system(const MatrixX& A, const VectorX& b)
{
#ifdef USE_SUITESPARSE
    Eigen::CholmodSupernodalLLT<MatrixX> solver;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
#endif

    solver.compute(A);
    return solver.solve(b);
}

VectorX solve_linear_system(const MatrixX& A, const VectorX& b)
{
#ifdef WITH_MPFR
    Eigen::SparseQR<MatrixX, Eigen::COLAMDOrdering<int>> solver;
#else
    Eigen::SparseLU<MatrixX> solver;
#endif

    solver.compute(A);
    return solver.solve(b);
}

void compute_submatrix(
    const MatrixX& matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices,
    MatrixX& submatrix)
{
    size_t num_rows = matrix.rows();
    size_t num_cols = matrix.cols();
    size_t num_subrows = row_indices.size();
    size_t num_subcols = col_indices.size();
    assert(num_rows >= num_subrows);
    assert(num_cols >= num_subcols);

    // Get mappings from rows and columns to submatrix rows and columns
    std::vector<int> row_indexing_map, col_indexing_map;
    compute_set_to_subset_mapping(num_rows, row_indices, row_indexing_map);
    compute_set_to_subset_mapping(num_cols, col_indices, col_indexing_map);

    // To compute the sparse matrix subset, we get iterate over nonzero entries
    // and prune those that are not in the given indexing set while remapping kept
    // indices to their index in the subset
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplet_list;
    triplet_list.reserve(matrix.nonZeros());
    for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
            // Check if entry is in the submatrix
            int submatrix_row = row_indexing_map[it.row()];
            int submatrix_col = col_indexing_map[it.col()];
            if (submatrix_row < 0) continue;
            if (submatrix_col < 0) continue;

            // Add reindexed entry to the triplet list
            triplet_list.push_back(T(submatrix_row, submatrix_col, it.value()));
        }
    }
    submatrix.resize(num_subrows, num_subcols);
    submatrix.reserve(triplet_list.size());
    submatrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

Matrix2x2 compute_rotation(Scalar theta)
{
    Matrix2x2 R;
    R.row(0) << cos(theta), -sin(theta);
    R.row(1) << sin(theta), cos(theta);
    
    return R;
}

MatrixX solve_linear_system(const MatrixX& A, const MatrixX&B)
{
#ifdef WITH_MPFR
    Eigen::SparseQR<MatrixX, Eigen::COLAMDOrdering<int>> solver;
#else
    Eigen::SparseLU<MatrixX> solver;
#endif

    solver.compute(A);
    return solver.solve(B);
}

std::vector<Scalar> generate_linspace(Scalar a, Scalar b, int num_steps)
{
    std::vector<Scalar> linspace(num_steps + 1);

    // iteratively compute linspace
    Scalar delta = (b - a) / static_cast<Scalar>(num_steps);
    linspace[0] = a;
    for (int i = 0; i < num_steps; ++i)
    {
        linspace[i + 1] = linspace[i] + delta;
    }

    // clamp last value exactly to b
    linspace.back() = b;

    return linspace;
}

} // namespace Penner
