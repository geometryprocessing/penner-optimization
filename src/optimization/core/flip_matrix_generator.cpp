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
#include "optimization/core/flip_matrix_generator.h"

namespace CurvatureMetric {

FlipMatrixGenerator::FlipMatrixGenerator(int size)
    : m_size(size)
{
    reset();
}

void FlipMatrixGenerator::reset()
{
    m_list_of_lists = std::vector<std::vector<std::pair<int, Scalar>>>(
        m_size,
        std::vector<std::pair<int, Scalar>>());
    for (int i = 0; i < m_size; ++i) {
        m_list_of_lists[i].push_back(std::make_pair(i, 1.0));
    }
}

void FlipMatrixGenerator::multiply_by_matrix(
    const std::vector<int>& column_indices,
    const std::vector<Scalar>& values,
    int ed)
{
    int num_entries = 0;
    for (int i = 0; i < 5; ++i) {
        int ei = column_indices[i];
        num_entries += m_list_of_lists[ei].size();
    }

    // Compute the new row of J_del corresponding to edge ed, which is the only
    // edge that changes
    std::vector<std::pair<int, Scalar>> J_del_d_new;
    J_del_d_new.reserve(num_entries);
    for (int i = 0; i < 5; ++i) {
        int ei = column_indices[i];
        Scalar Di = values[i];
        for (auto it : m_list_of_lists[ei]) {
            J_del_d_new.push_back(std::make_pair(it.first, Di * it.second));
        }
    }
    std::sort(J_del_d_new.begin(), J_del_d_new.end());

    // Compress vector
    m_list_of_lists[ed] = std::vector<std::pair<int, Scalar>>();
    m_list_of_lists[ed].reserve(num_entries);
    int index = -1;
    Scalar value = 0.0;
    for (const auto& entry : J_del_d_new) {
        if (index != entry.first) {
            if (index >= 0) {
                m_list_of_lists[ed].push_back(std::make_pair(index, value));
            }
            index = entry.first;
            value = entry.second;
        } else {
            value += entry.second;
        }
    }
    m_list_of_lists[ed].push_back(std::make_pair(index, value));
}

MatrixX FlipMatrixGenerator::build_matrix() const
{
    // Build triplets from list of lists
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(5 * m_size);
    for (int i = 0; i < m_size; ++i) {
        for (auto it : m_list_of_lists[i]) {
            tripletList.push_back(T(i, it.first, it.second));
        }
    }

    // Create the matrix from the triplets
    MatrixX matrix;
    matrix.resize(m_size, m_size);
    matrix.reserve(tripletList.size());
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return matrix;
}


FlipMapMatrixGenerator::FlipMapMatrixGenerator(int size)
    : m_size(size)
{
    reset();
}

void FlipMapMatrixGenerator::reset()
{
    m_list_of_lists = std::vector<std::map<int, Scalar>>(m_size, std::map<int, Scalar>());
    for (int i = 0; i < m_size; ++i) {
        m_list_of_lists[i][i] = 1.0;
    }
}

void FlipMapMatrixGenerator::multiply_by_matrix(
    const std::vector<int>& column_indices,
    const std::vector<Scalar>& values,
    int ed)
{
    // Compute the new row of J_del corresponding to edge ed, which is the only
    // edge that changes
    std::map<int, Scalar> J_del_d_new;
    for (int i = 0; i < 5; ++i) {
        int ei = column_indices[i];
        Scalar Di = values[i];
        for (auto it : m_list_of_lists[ei]) {
            J_del_d_new[it.first] += Di * it.second;
        }
    }
    m_list_of_lists[ed] = J_del_d_new;
}

MatrixX FlipMapMatrixGenerator::build_matrix() const
{
    // Build triplets from list of lists
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(5 * m_size);
    for (int i = 0; i < m_size; ++i) {
        for (auto it : m_list_of_lists[i]) {
            tripletList.push_back(T(i, it.first, it.second));
        }
    }

    // Create the matrix from the triplets
    MatrixX matrix;
    matrix.resize(m_size, m_size);
    matrix.reserve(tripletList.size());
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return matrix;
}

} // namespace CurvatureMetric
