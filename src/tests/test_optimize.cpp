#include <catch2/catch_test_macros.hpp>

#include "implicit_optimization.hh"
#include "shapes.hh"
#include "common.hh"

using namespace CurvatureMetric;

TEST_CASE("The optimium edge length for a triangle can be computed", "[optimize]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<double> Th_hat;
  generate_triangle(V, F, Th_hat);
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  bool fix_boundary = false;
  Mesh<double> m = FV_to_double<double>(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, std::vector<int>(), fix_boundary);

  // Use default optimization without edge flips
  auto proj_params = std::make_shared<ProjectionParameters>();
  auto opt_params = std::make_shared<OptimizationParameters>();
  opt_params->use_edge_lengths = true;
  opt_params->num_iter = 5;

  // Use right triangle as the initial metric
  VectorX metric_coords(3);
  metric_coords << 0.0, 0.0, 2.0 * std::log(std::sqrt(2));
  VectorX metric_target(3);
  metric_target << 0.0, 0.0, 0.0;
	PennerConeMetric cone_metric(m, metric_coords);

  SECTION("p-norm")
  {
    // For a triangle with the p norm, the optimal solution should be the mean of the target
    opt_params->energy_choice = "p_norm";
    metric_target << -1.0, 0.0, 4.0;

    VectorX optimized_metric_coords;
    optimize_metric(
        cone_metric,
        metric_target,
        metric_coords,
        optimized_metric_coords,
        proj_params,
        opt_params);

    REQUIRE(optimized_metric_coords.size() == 3);
    CHECK(float_equal(optimized_metric_coords[0], 1.0, 1e-6));
    CHECK(float_equal(optimized_metric_coords[1], 1.0, 1e-6));
    CHECK(float_equal(optimized_metric_coords[2], 1.0, 1e-6));
  }
}

TEST_CASE("The optimium edge length for a square can be computed", "[optimize]")
{
  Eigen::MatrixXd V(4, 3);
  V << 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, -1.0, 0.0;
  Eigen::MatrixXi F(2, 3);
  F << 0, 1, 2,
      0, 3, 1;
  std::vector<double> Th_hat = {
      0.5 * M_PI,
      0.5 * M_PI,
      0.5 * M_PI,
      0.5 * M_PI};
  std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
  bool fix_boundary = false;
  Mesh<double> m = FV_to_double<double>(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, std::vector<int>(), fix_boundary);

  // Use default optimization without edge flips
  auto proj_params = std::make_shared<ProjectionParameters>();
  auto opt_params = std::make_shared<OptimizationParameters>();
  opt_params->use_edge_lengths = true;
  opt_params->num_iter = 5;

  // Use right triangle as the initial metric
  VectorX metric_coords(5);
  metric_coords << 0.0, 0.0, 0.0, 0.0, 0.0;
  VectorX metric_target(5);
  metric_target << 0.0, 0.0, 0.0, 0.0, 0.0;
	PennerConeMetric cone_metric(m, metric_coords);

  SECTION("p-norm")
  {
    // For a split square with the p norm, the optimal solution should be the mean of the target
    // TODO Make debug mesh where can control individual edges
    opt_params->energy_choice = "p_norm";
    metric_target << -1.0, 0.0, 4.0, 0.0, 0.0;

    VectorX optimized_metric_coords;
    optimize_metric(
        cone_metric,
        metric_target,
        metric_coords,
        optimized_metric_coords,
        proj_params,
        opt_params);

    REQUIRE(optimized_metric_coords.size() == 5);
  }
}
