#pragma once

#include "common.hh"

namespace CurvatureMetric {

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
template <typename T>
std::string formatted_vector(const std::vector<T>& vec, std::string delim = " ")
{
    std::stringstream vector_string;
    for (size_t i = 0; i < vec.size(); ++i) {
        vector_string << vec[i] << delim;
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

/// Write vector to file
///
/// @param[in] vec: vector to write to file
/// @param[in] filename: file to write to
/// @param[in] precision: precision for output
void write_vector(const VectorX& vec, const std::string& filename, int precision = 17);

/// Write a matrix to file.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
void write_matrix(const Eigen::MatrixXd& matrix, const std::string& filename);

/// Write a sparse matrix to file in i,j,v format.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
/// @param[in] format: format (csv or matlab) to write the matrix in
void write_sparse_matrix(
    const MatrixX& matrix,
    const std::string& filename,
    std::string format = "csv");

} // namespace CurvatureMetric
