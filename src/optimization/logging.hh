#pragma once
#include "common.hh"

/// @file Methods to log the results of optimization, including diagnostic information and per
/// iteration values.


namespace CurvatureMetric {

/// @brief Create an output log with a given name with output to a directory or to
/// standard output if an empty directory is specified.
///
/// @param[in] log_dir: directory to write output (or an empty path for standard output)
/// @param[in] log_name: name of the log for global access
void
create_log(const std::filesystem::path& log_dir, const std::string& log_name);

/// @brief Log information about the given mesh to a logger
///
/// @param[in] m: mesh
/// @param[in] log_name: name of the log for global access
void
log_mesh_information(const Mesh<Scalar>& m, const std::string& log_name);

/// Write vector to file
///
/// @param[in] vec: vector to write to file
/// @param[in] filename: file to write to
/// @param[in] precision: precision for output
void write_vector(
	const VectorX &vec,
	const std::string &filename,
	int precision=17
);

}