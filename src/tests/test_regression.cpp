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

#include <catch2/catch_test_macros.hpp>
#include <igl/readOBJ.h>

#include "common.hh"
#include "penner_optimization_interface.hh"
#include "io.hh"
#include "vector.hh"

using namespace CurvatureMetric;

namespace
{
	bool is_optimized_metric_reproduced(
			const Eigen::MatrixXd &V,
			const Eigen::MatrixXi &F,
			const std::vector<Scalar> &Th_hat,
			const VectorX &ground_truth_metric_coords)
	{
		Mesh<Scalar> m;
		std::vector<int> vtx_reindex;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = generate_initial_mesh(V, F, V, F, Th_hat, vtx_reindex);
		LogLengthEnergy opt_energy(*cone_metric);

		// Make default parameters with 5 iterations
		auto proj_params = std::make_shared<ProjectionParameters>();
		auto opt_params = std::make_shared<OptimizationParameters>();
		opt_params->num_iter = 5;

		// Optimize the metric
    std::unique_ptr<DifferentiableConeMetric> optimized_cone_metric = optimize_metric(
        *cone_metric,
        opt_energy,
        proj_params,
        opt_params);
    VectorX optimized_reduced_metric_coords = optimized_cone_metric->get_reduced_metric_coordinates();

		// Check if the metrics are equal
		spdlog::info("Error is {}", sup_norm(optimized_reduced_metric_coords - ground_truth_metric_coords));
		return (vector_equal(optimized_reduced_metric_coords, ground_truth_metric_coords));
	}
} // namespace

TEST_CASE("Optimization produces consistent results", "[regression]")
{
  spdlog::set_level(spdlog::level::off);
	std::vector<std::string> meshes = {"knot1", "eight"};
	for (const auto &mesh : meshes)
	{
		std::string regression_dir = TEST_DATA_DIR;
		std::string input_filename = join_path(regression_dir, mesh + ".obj");
		std::string Th_hat_filename = join_path(regression_dir, mesh + "_Th_hat");
		std::string metric_filename = join_path(regression_dir, mesh + "_metric_coords");

		// Get input mesh
		Eigen::MatrixXd V, uv, N;
		Eigen::MatrixXi F, FT, FN;
		spdlog::info("Optimizing mesh at {}", input_filename);
		igl::readOBJ(input_filename, V, uv, N, F, FT, FN);

		// Get input angles
		std::vector<Scalar> Th_hat;
		spdlog::info("Using cone angles at {}", Th_hat_filename);
		read_vector_from_file(Th_hat_filename, Th_hat);

		// Get ground truth metric from file or target
		VectorX ground_truth_metric_coords;
		std::vector<Scalar> ground_truth_metric_coords_vec;
		read_vector_from_file(metric_filename, ground_truth_metric_coords_vec);
		convert_std_to_eigen_vector(ground_truth_metric_coords_vec, ground_truth_metric_coords);

		// Run regression test
		CHECK(is_optimized_metric_reproduced(V, F, Th_hat, ground_truth_metric_coords));
	}
}
