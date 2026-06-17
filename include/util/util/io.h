// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

/**
 * @brief Methods to read and write vectors, matrices, and meshes to and from file.
 * 
 */

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
void write_vector(
    const VectorType& v,
    const std::string& output_filename,
    int precision = 17,
    std::string sep = "\n",
    bool append = false)
{
    std::ofstream output_file;
    if (append)
    {
        output_file = std::ofstream(output_filename, std::ios::out | std::ios::app);
    }
    else
    {
        output_file = std::ofstream(output_filename, std::ios::out | std::ios::trunc);
    }
    int n = v.size();
    for (int i = 0; i < n; ++i) {
        output_file << std::setprecision(precision) << v[i] << sep;
    }
    output_file.close();
}

/// Write a matrix to file.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
/// @param[in] separator: (optional) separator for columns
void write_matrix(const Eigen::MatrixXd& matrix, const std::string& filename, std::string separator=",");

/// Write matrix of integers to file.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
/// @param[in] separator: (optional) separator for columns
void write_integer_matrix(const Eigen::MatrixXi& matrix, const std::string& filename, std::string separator=",");

Eigen::MatrixXd read_matrix(const std::string& filename);

/// Write an obj file with uv coordinates.
///
/// @param[in] filename: obj output file location
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv corner coordinates
/// @param[in] F_uv: mesh uv faces
void write_obj_with_uv(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv);

void write_hdf5_mesh(const std::string& path,
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXi& F);

void write_hdf5_mesh_with_uv(const std::string& path,
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXi& F,
                const Eigen::MatrixXd& uv,
                const Eigen::MatrixXi& FT);

void read_hdf5_mesh(const std::string& path,
                Eigen::MatrixXd& V,
                Eigen::MatrixXi& F);

void read_hdf5_mesh_with_uv(const std::string& path,
                Eigen::MatrixXd& V,
                Eigen::MatrixXi& F,
                Eigen::MatrixXd& uv,
                Eigen::MatrixXi& FT);

void write_hdf5_vector_field(const std::string& path,
                        const std::string& name,
                        const Eigen::MatrixXd& vector_field);
void write_hdf5_integer_matrix(const std::string& path,
                        const std::string& name,
                        const Eigen::MatrixXi& vector_field);

Eigen::MatrixXd read_hdf5_vector_field(const std::string& path,
                                   const std::string& name);
Eigen::MatrixXi read_hdf5_integer_matrix(const std::string& path,
                                   const std::string& name);


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