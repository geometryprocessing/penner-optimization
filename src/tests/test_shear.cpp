#include <catch2/catch_test_macros.hpp>

#include "shear.hh"
#include "shapes.hh"
#include "embedding.hh"
#include "common.hh"
#include "cone_metric.hh"
#include "optimization_interface.hh"
#include "projection.hh"

using namespace CurvatureMetric;

TEST_CASE( "A dual shear coordinate basis can be computed", "[shear]" )
{
  spdlog::set_level(spdlog::level::debug);
	std::vector<int> vtx_reindex;
  VectorX reduced_metric_target;
  
  SECTION ( "Triangle" )
  {
    Eigen::MatrixXd V(3, 3);
    V <<
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0;

    Eigen::MatrixXi F(1, 3);
    F <<
    0, 1, 2;
    
    std::vector<double> Th_hat(3, M_PI / 3.0);

    // Generate mesh
    std::unique_ptr<DifferentiableConeMetric> cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

    // Compute shear basis
    MatrixX shear_basis_matrix;
    std::vector<int> independent_edges;
    compute_shear_dual_basis(*cone_metric, shear_basis_matrix, independent_edges);

    REQUIRE( shear_basis_matrix.rows() == 6 );
    REQUIRE( shear_basis_matrix.cols() == 0 );

		// Compute corresponding scaling matrix
    MatrixX scaling_matrix = conformal_scaling_matrix(*cone_metric);
    MatrixX inner_product_matrix = scaling_matrix.transpose() * shear_basis_matrix;
    
    // Check that the basis is orthogonal to the scaling
    REQUIRE( float_equal(matrix_sup_norm(inner_product_matrix), 0.0) );
  }

  SECTION ( "Double triangle" )
  {
    Eigen::MatrixXd V(3, 3);
    V <<
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0;

    Eigen::MatrixXi F(2, 3);
    F <<
    0, 1, 2,
    0, 2, 1;
    
    std::vector<double> Th_hat(3, 2.0 * M_PI / 3.0);

    // Generate mesh
    std::unique_ptr<DifferentiableConeMetric> cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

    // Compute shear basis
    MatrixX shear_basis_matrix;
    std::vector<int> independent_edges;
    compute_shear_dual_basis(*cone_metric, shear_basis_matrix, independent_edges);

    REQUIRE( shear_basis_matrix.rows() == 6 );
    REQUIRE( shear_basis_matrix.cols() == 0 );

		// Compute corresponding scaling matrix
    MatrixX scaling_matrix = conformal_scaling_matrix(*cone_metric);
    MatrixX inner_product_matrix = scaling_matrix.transpose() * shear_basis_matrix;
    
    // Check that the basis is orthogonal to the scaling
    REQUIRE( float_equal(matrix_sup_norm(inner_product_matrix), 0.0) );
  }

  SECTION ( "Tetrahedron" )
  {
    Eigen::MatrixXd V(4, 3);
    V <<
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

    Eigen::MatrixXi F(4, 3);
    F <<
    0, 1, 3,
    1, 2, 3,
    2, 0, 3,
    0, 2, 1;
    
    std::vector<double> Th_hat(4, M_PI);

    // Generate mesh
    std::unique_ptr<DifferentiableConeMetric> cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

    // Compute shear basis and ensure it is the correct size
    MatrixX shear_basis_matrix;
    std::vector<int> independent_edges;
    compute_shear_dual_basis(*cone_metric, shear_basis_matrix, independent_edges);
    spdlog::info("Shear basis matrix is {}", shear_basis_matrix);

    REQUIRE( shear_basis_matrix.rows() == 12 );
    REQUIRE( shear_basis_matrix.cols() == 2 );

		// Compute corresponding scaling matrix
    MatrixX scaling_matrix = conformal_scaling_matrix(*cone_metric);
    MatrixX inner_product_matrix = scaling_matrix.transpose() * shear_basis_matrix;
    spdlog::info("Conformal scaling matrix is {}", scaling_matrix);
    
    // Check that the basis is orthogonal to the scaling
    REQUIRE( float_equal(matrix_sup_norm(inner_product_matrix), 0.0) );
  }

  SECTION ( "Pyramid" )
  {
    Eigen::MatrixXd V(5, 3);
    V <<
    0.0, -1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0;

    Eigen::MatrixXi F(6, 3);
    F <<
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
    3, 0, 4,
    0, 2, 1,
    0, 3, 2;
    
    std::vector<double> Th_hat(5, (10.0 - 4.0) * M_PI / 5.0);

    // Generate mesh
    std::unique_ptr<DifferentiableConeMetric> cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

    // Compute shear basis and ensure it is the correct size
    MatrixX shear_basis_matrix;
    std::vector<int> independent_edges;
    compute_shear_dual_basis(*cone_metric, shear_basis_matrix, independent_edges);
    spdlog::info("Shear basis matrix is {}", shear_basis_matrix);

    REQUIRE( shear_basis_matrix.rows() == 18 );
    REQUIRE( shear_basis_matrix.cols() == 4 );

		// Compute corresponding scaling matrix
    MatrixX scaling_matrix = conformal_scaling_matrix(*cone_metric);
    MatrixX inner_product_matrix = scaling_matrix.transpose() * shear_basis_matrix;
    spdlog::info("Conformal scaling matrix is {}", scaling_matrix);
    
    // Check that the basis is orthogonal to the scaling
    REQUIRE( float_equal(matrix_sup_norm(inner_product_matrix), 0.0) );
  }
}

TEST_CASE( "Shear dual basis coordinates for a tetrahedron can be computed", "[shear]" )
{
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vtx_reindex;
  VectorX reduced_metric_target;
  
  Eigen::MatrixXd V(4, 3);
  V <<
  0.0, 0.0, 0.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0;

  Eigen::MatrixXi F(4, 3);
  F <<
  0, 1, 3,
  1, 2, 3,
  2, 0, 3,
  0, 2, 1;
  
  std::vector<double> Th_hat(4, M_PI);

  // Generate mesh
  std::unique_ptr<DifferentiableConeMetric> initial_cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

  // Compute shear and conformal basis
  MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
  compute_shear_dual_basis(*initial_cone_metric, shear_basis_matrix, independent_edges);
  MatrixX scaling_matrix = conformal_scaling_matrix(*initial_cone_metric);

  SECTION ( "Zero" )
  {
    // Compute metric from arbitrary initial coordinates
    VectorX exact_scale_factors(4);
    exact_scale_factors << 0, 0, 0, 0;
    VectorX exact_shear_coords(2);
    exact_shear_coords << 0, 0;
    VectorX metric_coords = shear_basis_matrix * exact_shear_coords + scaling_matrix * exact_scale_factors;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = initial_cone_metric->set_metric_coordinates(metric_coords);

    // Compute the shear and scale factors for the metric
    VectorX shear_coords, scale_factors;
    compute_shear_basis_coordinates(*cone_metric, shear_basis_matrix, shear_coords, scale_factors);

    REQUIRE( vector_equal(shear_coords, exact_shear_coords) );
    REQUIRE( vector_equal(scale_factors, exact_scale_factors) );
  }

  SECTION ( "Conformal" )
  {
    // Compute metric from arbitrary initial coordinates
    VectorX exact_scale_factors(4);
    exact_scale_factors << 1, 2, 0, -1;
    VectorX exact_shear_coords(2);
    exact_shear_coords << 0, 0;
    VectorX metric_coords = shear_basis_matrix * exact_shear_coords + scaling_matrix * exact_scale_factors;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = initial_cone_metric->set_metric_coordinates(metric_coords);

    // Compute the shear and scale factors for the metric
    VectorX shear_coords, scale_factors;
    compute_shear_basis_coordinates(*cone_metric, shear_basis_matrix, shear_coords, scale_factors);

    REQUIRE( vector_equal(shear_coords, exact_shear_coords) );
    REQUIRE( vector_equal(scale_factors, exact_scale_factors) );
  }

  SECTION ( "Shear" )
  {
    // Compute metric from arbitrary initial coordinates
    VectorX exact_scale_factors(4);
    exact_scale_factors << 0, 0, 0, 0;
    VectorX exact_shear_coords(2);
    exact_shear_coords << 2, -1;
    VectorX metric_coords = shear_basis_matrix * exact_shear_coords + scaling_matrix * exact_scale_factors;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = initial_cone_metric->set_metric_coordinates(metric_coords);

    // Compute the shear and scale factors for the metric
    VectorX shear_coords, scale_factors;
    compute_shear_basis_coordinates(*cone_metric, shear_basis_matrix, shear_coords, scale_factors);

    REQUIRE( vector_equal(shear_coords, exact_shear_coords) );
    REQUIRE( vector_equal(scale_factors, exact_scale_factors) );
  }

  SECTION ( "General" )
  {
    // Compute metric from arbitrary initial coordinates
    VectorX exact_scale_factors(4);
    exact_scale_factors << 1, 2, 0, -1;
    VectorX exact_shear_coords(2);
    exact_shear_coords << 2, -1;
    VectorX metric_coords = shear_basis_matrix * exact_shear_coords + scaling_matrix * exact_scale_factors;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = initial_cone_metric->set_metric_coordinates(metric_coords);

    // Compute the shear and scale factors for the metric
    VectorX shear_coords, scale_factors;
    compute_shear_basis_coordinates(*cone_metric, shear_basis_matrix, shear_coords, scale_factors);

    REQUIRE( vector_equal(shear_coords, exact_shear_coords) );
    REQUIRE( vector_equal(scale_factors, exact_scale_factors) );
  }
}

TEST_CASE( "Shear dual basis coordinates for a pyramid can be computed", "[shear]" )
{
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vtx_reindex;
  VectorX reduced_metric_target;

  Eigen::MatrixXd V(5, 3);
  V <<
  0.0, -1.0, 0.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  -1.0, 0.0, 0.0,
  0.0, 0.0, 1.0;

  Eigen::MatrixXi F(6, 3);
  F <<
  0, 1, 4,
  1, 2, 4,
  2, 3, 4,
  3, 0, 4,
  0, 2, 1,
  0, 3, 2;
  
  std::vector<double> Th_hat(5, (10.0 - 4.0) * M_PI / 5.0);

  // Generate mesh
  std::unique_ptr<DifferentiableConeMetric> initial_cone_metric = generate_initial_mesh(V, F, Th_hat, vtx_reindex);

  // Compute shear and conformal basis
  MatrixX shear_basis_matrix;
  std::vector<int> independent_edges;
  compute_shear_dual_basis(*initial_cone_metric, shear_basis_matrix, independent_edges);
  MatrixX scaling_matrix = conformal_scaling_matrix(*initial_cone_metric);
  
  SECTION ( "Zero" )
  {
    // Compute metric from arbitrary initial coordinates
    VectorX exact_scale_factors(5);
    exact_scale_factors << 0, 0, 0, 0, 0;
    VectorX exact_shear_coords(4);
    exact_shear_coords << 0, 0, 0, 0;
    VectorX metric_coords = shear_basis_matrix * exact_shear_coords + scaling_matrix * exact_scale_factors;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = initial_cone_metric->set_metric_coordinates(metric_coords);

    // Compute the shear and scale factors for the metric
    VectorX shear_coords, scale_factors;
    compute_shear_basis_coordinates(*cone_metric, shear_basis_matrix, shear_coords, scale_factors);

    REQUIRE( vector_equal(shear_coords, exact_shear_coords) );
    REQUIRE( vector_equal(scale_factors, exact_scale_factors) );
  }

  SECTION ( "General" )
  {
    // Compute metric from arbitrary initial coordinates
    VectorX exact_scale_factors(5);
    exact_scale_factors << 1, 2, 0, -1, -2;
    VectorX exact_shear_coords(4);
    exact_shear_coords << 1, 2, 0, -1;
    VectorX metric_coords = shear_basis_matrix * exact_shear_coords + scaling_matrix * exact_scale_factors;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = initial_cone_metric->set_metric_coordinates(metric_coords);

    // Compute the shear and scale factors for the metric
    VectorX shear_coords, scale_factors;
    compute_shear_basis_coordinates(*cone_metric, shear_basis_matrix, shear_coords, scale_factors);

    REQUIRE( vector_equal(shear_coords, exact_shear_coords) );
    REQUIRE( vector_equal(scale_factors, exact_scale_factors) );
  }
}
