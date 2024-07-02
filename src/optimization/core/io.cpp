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
#include "optimization/core/io.h"

#include "optimization/core/embedding.h"
#include "optimization/core/vector.h"

namespace CurvatureMetric {

void create_log(const std::filesystem::path& log_dir, const std::string& log_name)
{
    // If the log directory is trivial, use standard output
    if (log_dir.empty()) {
        auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(std::cout);
        auto logger = std::make_shared<spdlog::logger>(log_name, ostream_sink);
        spdlog::register_logger(logger);
        // Make a file logger if an output path is specified
    } else {
        std::filesystem::path log_dir_path(log_dir);
        std::filesystem::create_directories(log_dir_path);
        std::filesystem::path log_path = join_path(log_dir_path, log_name + ".log");
        spdlog::basic_logger_mt(log_name, log_path);
    }
}

void log_mesh_information(const Mesh<Scalar>& m, const std::string& log_name)
{
    ReductionMaps reduction_maps(m);
    std::vector<int> cone_vertices;
    // compute_cone_vertices(m, reduction_maps, cone_vertices);
    spdlog::get(log_name)->trace("Fixed vertices are {}", formatted_vector(reduction_maps.fixed_v));
    // spdlog::get(log_name)->trace(
    //   "Cone vertices are {}",
    //   formatted_vector(cone_vertices)
    //);
    spdlog::get(log_name)->trace("Mesh next map: {}", formatted_vector(m.n));
    spdlog::get(log_name)->trace("Mesh face to halfedge map: {}", formatted_vector(m.h));
    spdlog::get(log_name)->trace("Mesh halfedge to face map: {}", formatted_vector(m.f));
}

void write_vector(const VectorX& vec, const std::string& filename, int precision)
{
    std::ofstream output_file(filename, std::ios::out | std::ios::trunc);
    for (Eigen::Index i = 0; i < vec.size(); ++i) {
        output_file << std::setprecision(precision) << vec[i] << std::endl;
    }
    output_file.close();
}

void write_matrix(const Eigen::MatrixXd& matrix, const std::string& filename)
{
    if (matrix.cols() == 0) {
        return;
    }

    // Open file
    std::ofstream output_file;
    output_file.open(filename);

    // Iterate over rows
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        // Iterate over columns of row i
        Scalar v = matrix(i, 0);
        output_file << std::fixed << std::setprecision(17) << v;
        for (Eigen::Index j = 1; j < matrix.cols(); ++j) {
            Scalar v = matrix(i, j);
            output_file << std::fixed << std::setprecision(17) << "," << v;
        }

        // Add newline to end of row
        output_file << std::endl;
    }

    // Close file
    output_file.close();
}

void write_sparse_matrix(const MatrixX& matrix, const std::string& filename, std::string format)
{
    /// Open file
    std::ofstream output_file;
    output_file.open(filename);

    // Iterate over nonzero entries
    for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            Scalar v = it.value();

            // CSV format has comma separated 0-indexed values
            if (format == "csv") {
                output_file << std::fixed << std::setprecision(17) << i << "," << j << "," << v
                            << std::endl;
            }
            // MATLAB uses space separated 1-indexed values
            else if (format == "matlab") {
                output_file << std::fixed << std::setprecision(17) << (i + 1) << "  " << (j + 1)
                            << "  " << v << std::endl;
            }
        }
    }

    // Close file
    output_file.close();
}

} // namespace CurvatureMetric
