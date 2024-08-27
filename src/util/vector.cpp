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
#include "util/vector.h"

namespace Penner {

void convert_dense_vector_to_sparse(const VectorX& vector_dense, MatrixX& vector_sparse)
{
    // Copy the vector to a triplet list
    int vec_size = vector_dense.size();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplet_list;
    triplet_list.reserve(vec_size);
    for (int i = 0; i < vec_size; ++i) {
        triplet_list.push_back(T(i, 0, vector_dense[i]));
    }

    // Build the sparse vector from the triplet list
    vector_sparse.resize(vec_size, 1);
    vector_sparse.reserve(triplet_list.size());
    vector_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

void convert_eigen_to_std_vector(const VectorX& vector_eigen, std::vector<Scalar>& vector_std)
{
    size_t vector_size = vector_eigen.size();
    vector_std.resize(vector_size);
    for (size_t i = 0; i < vector_size; ++i) {
        vector_std[i] = vector_eigen[i];
    }
}

Eigen::Matrix<double, Eigen::Dynamic, 1> convert_scalar_to_double_vector(
    const VectorX& vector_scalar)
{
    int num_entries = vector_scalar.size();
    Eigen::Matrix<double, Eigen::Dynamic, 1> vector_double(num_entries);
    for (int i = 0; i < num_entries; ++i) {
        vector_double[i] = (double)(vector_scalar[i]);
    }

    return vector_double;
}

std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> convert_scalar_to_double_vector(
    const std::vector<VectorX>& vector_scalar)
{
    int num_entries = vector_scalar.size();
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> vector_double(num_entries);
    for (int i = 0; i < num_entries; ++i) {
        vector_double[i] = convert_scalar_to_double_vector(vector_scalar[i]);
    }

    return vector_double;
}

void mask_subset(const std::vector<int>& mask, VectorX& set)
{
    for (size_t i = 0; i < mask.size(); ++i) {
        set[mask[i]] = 0;
    }
}

void compute_subset(
    const VectorX& universe,
    const std::vector<int>& subset_indices,
    VectorX& subset)
{
    subset.resize(subset_indices.size());
    for (size_t i = 0; i < subset_indices.size(); ++i) {
        subset[i] = universe[subset_indices[i]];
    }
}

void compute_set_to_subset_mapping(
    size_t set_size,
    const std::vector<int>& subset_indices,
    std::vector<int>& set_to_subset_mapping)
{
    size_t subset_size = subset_indices.size();
    fill_vector(set_size, -1, set_to_subset_mapping);
    for (size_t i = 0; i < subset_size; ++i) {
        set_to_subset_mapping[subset_indices[i]] = i;
    }
}

void write_subset(const VectorX& subset, const std::vector<int>& subset_indices, VectorX& set)
{
    size_t subset_size = subset.size();
    assert(subset_indices.size() == subset_size);
    for (size_t i = 0; i < subset_size; ++i) {
        set[subset_indices[i]] = subset[i];
    }
}

void enumerate_boolean_array(
    const std::vector<bool>& boolean_array,
    std::vector<int>& true_entry_list,
    std::vector<int>& false_entry_list,
    std::vector<int>& array_to_list_map)
{
    int num_entries = boolean_array.size();
    true_entry_list.clear();
    true_entry_list.reserve(num_entries);
    false_entry_list.clear();
    false_entry_list.reserve(num_entries);

    // Iterate over the boolean array to enumerate the true and false entries
    for (int i = 0; i < num_entries; ++i) {
        if (boolean_array[i]) {
            true_entry_list.push_back(i);
        } else {
            false_entry_list.push_back(i);
        }
    }

    // Generate the reverse map from array entries to list indices
    int num_true_entries = true_entry_list.size();
    int num_false_entries = false_entry_list.size();
    array_to_list_map.clear();
    array_to_list_map.resize(num_entries, -1);
    for (int i = 0; i < num_true_entries; ++i) {
        array_to_list_map[true_entry_list[i]] = i;
    }
    for (int i = 0; i < num_false_entries; ++i) {
        array_to_list_map[false_entry_list[i]] = i;
    }
}

} // namespace Penner
