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

#include "util/common.h"

namespace Penner {

/**
 * @brief Convert an Eigen vector to a std vector
 *
 * @tparam Type of the vector data
 * @param v: Eigen vector
 * @return std vector
 */
template <typename Type>
std::vector<Type> convert_vector(const Eigen::Matrix<Type, Eigen::Dynamic, 1>& v)
{
    return std::vector<Type>(v.data(), v.data() + v.size());
}

/**
 * @brief Reverse the entries in a vector
 *
 * @tparam Type of the vector data
 * @param v: vector
 * @return reversed vector
 */
template <typename Type>
std::vector<Type> reverse_vector(const std::vector<Type>& v)
{
    std::vector<Type> w = v;
    std::reverse(w.begin(), w.end());
    return w;
}

/**
 * @brief Concatenate two vectors together
 *
 * @tparam Type of the vector data
 * @param v: first vector
 * @param w: second vector
 * @return std vector
 */
template <typename Type>
std::vector<Type> concatenate_vector(const std::vector<Type>& v, const std::vector<Type>& w)
{
    std::vector<Type> concat = v;
    concat.reserve(v.size() + w.size());
    concat.insert(concat.end(), w.begin(), w.end());
    return concat;
}

/**
 * @brief Check if two vectors are equal, up to numerical tolerance.
 * 
 * @tparam VectorType 
 * @param v: first vector
 * @param w: second vector
 * @return true if equal
 * @return false otherwise
 */
template <typename VectorType>
bool vector_equal(const VectorType& v, const VectorType& w)
{
    int n = v.size();
    int m = w.size();

    // check for consistent sizes
    if (n != m) return false;

    // check all entries
    for (int i = 0; i < n; ++i)
    {
        if (!float_equal(v[i], w[i])) return false;
    }

    return true;
}

/**
 * @brief Check if a vector contains a value
 * 
 * @tparam VectorType 
 * @tparam ElemType
 * @param v: vector
 * @param x: value to check for
 * @return true iff v contains x
 */
template <typename VectorType, typename ElemType>
bool vector_contains(const VectorType& v, const ElemType& x)
{
    int n = v.size();

    // check all entries
    for (int i = 0; i < n; ++i)
    {
        if (v[i] == x) return true;
    }

    return false;
}

template <typename VectorType>
Scalar compute_total_sum(const VectorType& v)
{
    Scalar total_sum = 0.;
    int n = v.size();
    for (int i = 0; i < n; ++i)
    {
        total_sum += v[i];
    }
    return total_sum;
}

/// Convert standard template library vector to an Eigen vector.
///
/// @param[in] vector_std: standard template library vector to copy
/// @param[out] vector_eigen: copied vector
template <typename VectorScalar, typename MatrixScalar>
void convert_std_to_eigen_vector(
    const std::vector<VectorScalar>& vector_std,
    Eigen::Matrix<MatrixScalar, Eigen::Dynamic, 1>& vector_eigen)
{
    size_t vector_size = vector_std.size();
    vector_eigen.resize(vector_size);
    for (size_t i = 0; i < vector_size; ++i) {
        vector_eigen[i] = MatrixScalar(vector_std[i]);
    }
}

/// Convert standard template library vector of vectors to an Eigen matrix.
///
/// @param[in] matrix_vec: vector matrix to copy
/// @param[out] matrix: copied matrix
template <typename VectorScalar, typename MatrixScalar>
void convert_std_to_eigen_matrix(
    const std::vector<std::vector<VectorScalar>>& matrix_vec,
    Eigen::Matrix<MatrixScalar, Eigen::Dynamic, Eigen::Dynamic>& matrix)
{
    matrix.setZero(0, 0);
    if (matrix_vec.empty()) return;

    // Get dimensions of matrix
    int rows = matrix_vec.size();
    int cols = matrix_vec[0].size();
    matrix.resize(rows, cols);

    // Copy matrix by row
    for (int i = 0; i < rows; ++i) {
        // Check size validity
        if (static_cast<int>(matrix_vec[i].size()) != cols) {
            spdlog::error("Cannot copy vector of vectors of inconsistent sizes to a matrix");
            matrix.setZero(0, 0);
            return;
        }

        // Copy row
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = MatrixScalar(matrix_vec[i][j]);
        }
    }
}

template <typename VectorScalar, typename MatrixScalar, std::size_t Dimension>
void convert_std_to_eigen_matrix(
    const std::vector<std::array<VectorScalar, Dimension>>& matrix_vec,
    Eigen::Matrix<MatrixScalar, Eigen::Dynamic, Eigen::Dynamic>& matrix)
{
    matrix.setZero(0, 0);
    if (matrix_vec.empty()) return;

    // Get dimensions of matrix
    int rows = matrix_vec.size();
    matrix.resize(rows, Dimension);

    // Copy matrix by row
    for (int i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < Dimension; ++j) {
            matrix(i, j) = MatrixScalar(matrix_vec[i][j]);
        }
    }
}

template <typename Scalar, std::size_t Dimension>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
convert_std_to_eigen_matrix(const std::vector<std::array<Scalar, Dimension>>& lol_matrix)
{
    int num_rows = lol_matrix.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(num_rows, Dimension);
    for (int i = 0; i < num_rows; ++i)
    {
        for (std::size_t j = 0; j < Dimension; ++j)
        {
            eigen_matrix(i, j) = lol_matrix[i][j];
        }
    }

    return eigen_matrix;
}

/// Convert Eigen dense vector to sparse.
///
/// @param[in] vector_dense: input dense vector
/// @param[out] vector_sparse: output sparse vector
void convert_dense_vector_to_sparse(const VectorX& vector_dense, MatrixX& vector_sparse);

/// Convert Eigen vector to standard template library vector.
///
/// @param[in] vector_eigen: Eigen vector to copy
/// @param[out] vector_std: copied vector
void convert_eigen_to_std_vector(const VectorX& vector_eigen, std::vector<Scalar>& vector_std);

/// Convert vector of scalars to doubles (with potential precision loss)
///
/// @param[in] vector_scalar: vector of scalars to copy
/// @return vector of scalars cast to doubles
Eigen::Matrix<double, Eigen::Dynamic, 1> convert_scalar_to_double_vector(
    const VectorX& vector_scalar);

/// Convert vector of vector of scalars to doubles
///
/// @param[in] vector_scalar: vector of vector of scalars to copy
/// @return vector of vector of scalars cast to doubles
std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> convert_scalar_to_double_vector(
    const std::vector<VectorX>& vector_scalar);

template <typename OldScalar, typename NewScalar>
Eigen::Matrix<NewScalar, Eigen::Dynamic, 1> convert_vector_type(
    const Eigen::Matrix<OldScalar, Eigen::Dynamic, 1>& vector)
{
    int num_entries = vector.size();
    Eigen::Matrix<NewScalar, Eigen::Dynamic, 1> vector_converted(num_entries);
    for (int i = 0; i < num_entries; ++i) {
        vector_converted[i] = (NewScalar)(vector[i]);
    }

    return vector_converted;
}

template <typename OldScalar, typename NewScalar>
std::vector<NewScalar> convert_vector_type(const std::vector<OldScalar>& vector)
{
    int num_entries = vector.size();
    std::vector<NewScalar> vector_converted(num_entries);
    for (int i = 0; i < num_entries; ++i) {
        vector_converted[i] = (NewScalar)(vector[i]);
    }

    return vector_converted;
}

/// Fill a vector with some value.
///
/// @param[in] size: size of the output vector
/// @param[in] value: fill value
/// @param[out] vec: vector to fill
template <typename T>
void fill_vector(size_t size, T value, std::vector<T>& vec)
{
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = value;
    }
}

/// Given a universe set and a list of subset indices, mask the corresponding
/// subset entries with 0.
///
/// @param[in] mask: list of indices to mask
/// @param[in, out] set: object to mask
void mask_subset(const std::vector<int>& mask, VectorX& set);

/// Given a universe set and a list of subset indices, compute the corresponding
/// subset.
///
/// @param[in] universe: full set
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[out] subset: subset corresponding to the indices
template <typename T>
void compute_subset(
    const std::vector<T>& universe,
    const std::vector<int>& subset_indices,
    std::vector<T>& subset)
{
    size_t subset_size = subset_indices.size();
    subset.resize(subset_size);
    for (size_t i = 0; i < subset_size; ++i) {
        subset[i] = universe[subset_indices[i]];
    }
}

/// Given a universe set and a list of subset indices, compute the corresponding
/// subset.
///
/// @param[in] universe: full set
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[out] subset: subset corresponding to the indices
void compute_subset(
    const VectorX& universe,
    const std::vector<int>& subset_indices,
    VectorX& subset);

/// Given a universe set size and a list of subset indices, compute the inverse
/// mapping from original subset indices to their location in the new subset or
/// -1 if the item has been removed
///
/// @param[in] set_size: size of the full set
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[out] set_to_subset_mapping: subset corresponding to the indices
void compute_set_to_subset_mapping(
    size_t set_size,
    const std::vector<int>& subset_indices,
    std::vector<int>& set_to_subset_mapping);

/// Write the subset with given indices into the universe.
///
/// @param[in] subset: subset corresponding to the indices
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[in, out] set: full set to overwrite
void write_subset(const VectorX& subset, const std::vector<int>& subset_indices, VectorX& set);

/// @brief From a vector of the indices, build a boolean array marking these indices as true
///
/// @param[in] index_vector: list of indices
/// @param[in] num_indices: total number of indices
/// @param[out] boolean_array: array of boolean values marking indices
template <typename Index>
void convert_index_vector_to_boolean_array(
    const std::vector<Index>& index_vector,
    Index num_indices,
    std::vector<bool>& boolean_array)
{
    boolean_array.resize(num_indices, false);
    for (size_t i = 0; i < index_vector.size(); ++i) {
        boolean_array[index_vector[i]] = true;
    }
}

/// @brief From a boolean array, build a vector of the indices that are true.
///
/// @param[in] boolean_array: array of boolean values
/// @param[out] index_vector: indices where the array is true
template <typename Index>
void convert_boolean_array_to_index_vector(
    const std::vector<bool>& boolean_array,
    std::vector<Index>& index_vector)
{
    size_t num_indices = boolean_array.size();
    index_vector.clear();
    index_vector.reserve(num_indices);
    for (size_t i = 0; i < num_indices; ++i) {
        if (boolean_array[i]) {
            index_vector.push_back(i);
        }
    }
}

/// @brief From a vector of the indices, build the complement of indices
///
/// @param[in] index_vector: list of indices
/// @param[in] num_indices: total number of indices
/// @param[in] complement_vector: complement of indices
template <typename Index>
void index_vector_complement(
    const std::vector<Index>& index_vector,
    Index num_indices,
    std::vector<Index>& complement_vector)
{
    // Build index boolean array
    std::vector<bool> boolean_array;
    convert_index_vector_to_boolean_array(index_vector, num_indices, boolean_array);

    // Build complement
    complement_vector.clear();
    complement_vector.reserve(num_indices - index_vector.size());
    for (Index i = 0; i < num_indices; ++i) {
        if (!boolean_array[i]) {
            complement_vector.push_back(i);
        }
    }
}


/// Given a boolean array, enumerate the true and false entries.
///
/// @param[in] boolean_array: array of boolean values
/// @param[out] true_entry_list: indices where the array is true
/// @param[out] false_entry_list: indices where the array is false
/// @param[out] array_to_list_map: map from indices to their position in the true or false lists
void enumerate_boolean_array(
    const std::vector<bool>& boolean_array,
    std::vector<int>& true_entry_list,
    std::vector<int>& false_entry_list,
    std::vector<int>& array_to_list_map);
std::vector<int> enumerate_boolean_array(const std::vector<bool>& boolean_array);

} // namespace Penner
