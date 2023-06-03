#include <catch2/catch_test_macros.hpp>

#include "area.hh"
#include "shapes.hh"
#include "common.hh"

using namespace CurvatureMetric;

TEST_CASE( "The squared area of a triangle can be computed", "[area]" )
{
  SECTION ( "Equilateral triangle" )
  {
	double result = area_squared(1.0, 1.0, 1.0);
	double answer = 3.0 / 16.0;

    REQUIRE( float_equal(result, answer) );
  }

  SECTION ( "Right triangle" )
  {
	double result = area_squared(3.0, 4.0, 5.0);
	double answer = 36.0;

    REQUIRE( float_equal(result, answer) );
  }

  SECTION ( "Line segment" )
  {
	double result = area_squared(2.0, 1.0, 1.0);
	double answer = 0.0;

    REQUIRE( float_equal(result, answer) );
  }

  SECTION ( "Point" )
  {
	double result = area_squared(0.0, 0.0, 0.0);
	double answer = 0.0;

    REQUIRE( float_equal(result, answer) );
  }
}

TEST_CASE( "The derivative of the squared area of a triangle can be computed", "[area]" )
{
  SECTION ( "Equilateral triangle" )
  {
	double result = area_squared_derivative(1.0, 1.0, 1.0);
	double answer = 0.25;

    REQUIRE( float_equal(result, answer) );
  }

  SECTION ( "Right triangle" )
  {
	double result = area_squared_derivative(3.0, 4.0, 5.0);
	double answer = 24.0;

    REQUIRE( float_equal(result, answer) );
  }

  SECTION ( "Line segment" )
  {
	double result = area_squared_derivative(2.0, 1.0, 1.0);
	double answer = -1.0;

    REQUIRE( float_equal(result, answer) );
  }

  SECTION ( "Point" )
  {
	double result = area_squared_derivative(0.0, 0.0, 0.0);
	double answer = 0.0;

    REQUIRE( float_equal(result, answer) );
  }

}

TEST_CASE( "The squared area of mesh faces can be computed", "[area]" ) {

  SECTION ( "Regular tetrahedron" )
  {
    // Build a tetrahedron with unit length edges
    Mesh<double> m;
    std::vector<int> vtx_reindex;
    VectorX log_length_coords = VectorX::Constant(6, 0.0);
    generate_tetrahedron_mesh(m, vtx_reindex);

    // Compute the areas
    VectorX he2areasq;
    areas_squared_from_log_lengths(m, log_length_coords, he2areasq);

    // Check the areas
    REQUIRE( he2areasq.size() == 12 );
    for (size_t i = 0; i < 12; ++i)
    {
      REQUIRE( float_equal(he2areasq[i], 3.0 / 16.0) );
    }
  }
}


TEST_CASE( "The squared area derivatives of mesh faces can be computed", "[area]" ) {

  SECTION ( "Regular tetrahedron" )
  {
    // Build a tetrahedron with unit length edges
    Mesh<double> m;
    std::vector<int> vtx_reindex;
    VectorX log_length_coords = VectorX::Constant(6, 0.0);
    generate_tetrahedron_mesh(m, vtx_reindex);

    // Compute the area derivatives
    VectorX he2areasqderiv;
    area_squared_derivatives_from_log_lengths(m, log_length_coords, he2areasqderiv);

    // Check the area derivatives
    REQUIRE( he2areasqderiv.size() == 12 );
    for (size_t i = 0; i < 12; ++i)
    {
      REQUIRE( float_equal(he2areasqderiv[i], 0.125) );
    }
  }
}