
#include <catch2/catch_test_macros.hpp>
#include <igl/readOBJ.h>

#include "common.hh"
#include "optimization_interface.hh"

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
		VectorX reduced_metric_target;
		generate_initial_mesh(V, F, Th_hat, m, vtx_reindex, reduced_metric_target);
		VectorX reduced_metric_init = reduced_metric_target;

		// Make default parameters with 5 iterations
		auto proj_params = std::make_shared<ProjectionParameters>();
		auto opt_params = std::make_shared<OptimizationParameters>();
		opt_params->num_iter = 5;

		// Optimize the metric
		PennerConeMetric cone_metric(m, reduced_metric_init);
		VectorX optimized_reduced_metric_coords;
		optimize_metric(cone_metric,
										reduced_metric_target,
										reduced_metric_init,
										optimized_reduced_metric_coords,
										proj_params,
										opt_params);

		// Check if the metrics are equal
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
