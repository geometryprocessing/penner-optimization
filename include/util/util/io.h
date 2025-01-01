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

#include "util/embedding.h"

#include <filesystem>
#include <fstream>

namespace Penner {

/// Join two filepaths.
///
/// @param[in] first_path: first path to join
/// @param[in] second_path: second path to join
/// @return combined path
inline std::filesystem::path join_path(
    const std::filesystem::path& first_path,
    const std::filesystem::path& second_path)
{
    return first_path / second_path;
}

/// Format a vector as a human readable string
///
/// @param[in] vec: vector to format
/// @param[in] delim: deliminator between vector entries
/// @return formatted vector
template <typename VectorType>
std::string formatted_vector(const VectorType& vec, std::string delim = " ", int precision=8)
{
    std::stringstream vector_string;
    int n = vec.size();
    for (int i = 0; i < n; ++i) {
        vector_string << std::fixed << std::setprecision(precision) << vec[i] << delim;
    }

    return vector_string.str();
}

/// Read a vector from a file.
///
/// @param[in] filename: file with vector to read
/// @param[out] vec: vector from file
template <typename T>
void read_vector_from_file(const std::string& filename, std::vector<T>& vec)
{
    vec.clear();

    // Open file
    std::ifstream input_file(filename);
    if (!input_file) return;

    // Read file
    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        T value;
        iss >> value;
        vec.push_back(value);
    }

    // Close file
    input_file.close();
}

/// Read a vector of pairs from a file.
///
/// @param[in] filename: file with vector to read
/// @param[out] vec: vector from file
template <typename T>
void read_vector_of_pairs_from_file(const std::string& filename, std::vector<std::pair<T, T>>& vec)
{
    vec.clear();

    // Open file
    std::ifstream input_file(filename);
    if (!input_file) return;

    // Read file
    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        T first_value, second_value;
        iss >> first_value >> second_value;
        vec.push_back(std::make_pair(first_value, second_value));
    }

    // Close file
    input_file.close();
}

/// @brief Create an output log with a given name with output to a directory or to
/// standard output if an empty directory is specified.
///
/// @param[in] log_dir: directory to write output (or an empty path for standard output)
/// @param[in] log_name: name of the log for global access
void create_log(const std::filesystem::path& log_dir, const std::string& log_name);

/// @brief Log information about the given mesh to a logger
///
/// @param[in] m: mesh
/// @param[in] log_name: name of the log for global access
void log_mesh_information(const Mesh<Scalar>& m, const std::string& log_name);

/**
 * @brief Write a vector to file.
 *
 * @tparam Random access vector with size method
 * @param v: vector to write
 * @param output_filename: filename for writing
 */
template <typename VectorType>
void write_vector(const VectorType& v, const std::string& output_filename, int precision = 17)
{
    std::ofstream output_file(output_filename, std::ios::out | std::ios::trunc);
    int n = v.size();
    for (int i = 0; i < n; ++i) {
        output_file << std::setprecision(precision) << v[i] << std::endl;
    }
    output_file.close();
}

/// Write a matrix to file.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
/// @param[in] separator: (optional) separator for columns
void write_matrix(const Eigen::MatrixXd& matrix, const std::string& filename, std::string separator=",");

Eigen::MatrixXd read_matrix(const std::string& filename);

/// Write a sparse matrix to file in i,j,v format.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
/// @param[in] format: format (csv or matlab) to write the matrix in
void write_sparse_matrix(
    const MatrixX& matrix,
    const std::string& filename,
    std::string format = "csv");

} // namespace Penner
