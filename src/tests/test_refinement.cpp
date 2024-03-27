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

#include "common.hh"
#include "refinement.hh"
#include "triangulation.hh"

using namespace CurvatureMetric;

TEST_CASE( "A self overlapping polygon can be identified", "[refinement]" )
{
	std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
	std::vector<std::vector<int>> splitting_vertices;
	std::vector<std::vector<Scalar>> min_face_areas;
  std::vector<VectorX> uv_vertices, vertices;

  SECTION ( "Vertex" )
  {
    uv_vertices = { Eigen::Vector2d(0, 0) };
    vertices = { Eigen::Vector3d(0, 0, 0) };
    bool is_self_overlapping = is_self_overlapping_polygon(
      uv_vertices,
      vertices,
      is_self_overlapping_subpolygon,
      splitting_vertices,
      min_face_areas
    );

    // Check is self overlapping
    REQUIRE( is_self_overlapping );

    // Construct 1x1 tables, all true
    REQUIRE( is_self_overlapping_subpolygon.size() == 1 );
    REQUIRE( is_self_overlapping_subpolygon[0].size() == 1 );
    REQUIRE( splitting_vertices.size() == 1 );
    REQUIRE( splitting_vertices[0].size() == 1 );
    REQUIRE( min_face_areas.size() == 1 );
    REQUIRE( min_face_areas[0].size() == 1 );
    REQUIRE( is_self_overlapping_subpolygon[0][0] );
  }

  SECTION ( "Edge" )
  {
    uv_vertices = {
      Eigen::Vector2d(0, 0),
      Eigen::Vector2d(1, 0)
    };
    vertices = {
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0)
    };
    bool is_self_overlapping = is_self_overlapping_polygon(
      uv_vertices,
      vertices,
      is_self_overlapping_subpolygon,
      splitting_vertices,
      min_face_areas
    );

    // Check is self overlapping
    REQUIRE( is_self_overlapping );

    // Construct 2x2 tables, all true
    int num_vertices = vertices.size();
    REQUIRE( is_self_overlapping_subpolygon.size() == num_vertices );
    REQUIRE( splitting_vertices.size() == num_vertices );
    REQUIRE( min_face_areas.size() == num_vertices );
    for (int i = 0; i < num_vertices; ++i)
    {
      REQUIRE( is_self_overlapping_subpolygon[i].size() == num_vertices );
      REQUIRE( splitting_vertices[i].size() == num_vertices );
      REQUIRE( min_face_areas[i].size() == num_vertices );
      for (int j = 0; j < num_vertices; ++j)
      {
        REQUIRE( is_self_overlapping_subpolygon[i][j] );
      }
    }
  }

  SECTION ( "Triangle" )
  {
    uv_vertices = {
      Eigen::Vector2d(0, 0),
      Eigen::Vector2d(1, 0),
      Eigen::Vector2d(0, 1)
    };
    vertices = {
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(0, 1, 0)
    };
    bool is_self_overlapping = is_self_overlapping_polygon(
      uv_vertices,
      vertices,
      is_self_overlapping_subpolygon,
      splitting_vertices,
      min_face_areas
    );

    // Check is self overlapping
    REQUIRE( is_self_overlapping );

    // Construct 3x3 tables, all true
    int num_vertices = vertices.size();
    REQUIRE( is_self_overlapping_subpolygon.size() == num_vertices );
    REQUIRE( splitting_vertices.size() == num_vertices );
    REQUIRE( min_face_areas.size() == num_vertices );
    for (int i = 0; i < num_vertices; ++i)
    {
      REQUIRE( is_self_overlapping_subpolygon[i].size() == num_vertices );
      REQUIRE( splitting_vertices[i].size() == num_vertices );
      REQUIRE( min_face_areas[i].size() == num_vertices );
      for (int j = 0; j < num_vertices; ++j)
      {
        REQUIRE( is_self_overlapping_subpolygon[i][j] );
      }
    }
  }

  SECTION ( "Inverted triangle" )
  {
    uv_vertices = {
      Eigen::Vector2d(0, 0),
      Eigen::Vector2d(0, 1),
      Eigen::Vector2d(1, 0)
    };
    vertices = {
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(1, 0, 0)
    };
    bool is_self_overlapping = is_self_overlapping_polygon(
      uv_vertices,
      vertices,
      is_self_overlapping_subpolygon,
      splitting_vertices,
      min_face_areas
    );

    // Check is not self overlapping
    REQUIRE( !is_self_overlapping );

    // Construct 3x3 tables
    int num_vertices = vertices.size();
    REQUIRE( is_self_overlapping_subpolygon.size() == num_vertices );
    REQUIRE( splitting_vertices.size() == num_vertices );
    REQUIRE( min_face_areas.size() == num_vertices );
    for (int i = 0; i < num_vertices; ++i)
    {
      REQUIRE( is_self_overlapping_subpolygon[i].size() == num_vertices );
      REQUIRE( splitting_vertices[i].size() == num_vertices );
      REQUIRE( min_face_areas[i].size() == num_vertices );
    }
    
    // Vertex subpolygons are true
    REQUIRE( is_self_overlapping_subpolygon[0][0] );
    REQUIRE( is_self_overlapping_subpolygon[1][1] );
    REQUIRE( is_self_overlapping_subpolygon[2][2] );

    // Edge subpolygons are true
    REQUIRE( is_self_overlapping_subpolygon[0][1] );
    REQUIRE( is_self_overlapping_subpolygon[1][2] );
    REQUIRE( is_self_overlapping_subpolygon[2][0] );

    // Triangle subpolygons are false
    REQUIRE( !is_self_overlapping_subpolygon[0][2] );
    REQUIRE( !is_self_overlapping_subpolygon[1][0] );
    REQUIRE( !is_self_overlapping_subpolygon[2][1] );
  }

  SECTION ( "Square" )
  {
    uv_vertices = {
      Eigen::Vector2d(0, 0),
      Eigen::Vector2d(1, 0),
      Eigen::Vector2d(1, 1),
      Eigen::Vector2d(0, 1)
    };
    vertices = {
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(1, 1, 0),
      Eigen::Vector3d(0, 1, 0)
    };
    bool is_self_overlapping = is_self_overlapping_polygon(
      uv_vertices,
      vertices,
      is_self_overlapping_subpolygon,
      splitting_vertices,
      min_face_areas
    );

    // Check is self overlapping
    REQUIRE( is_self_overlapping );
  }

  SECTION ( "Not self-overlapping quadrilateral" )
  {
    uv_vertices = {
      Eigen::Vector2d(0, 0),
      Eigen::Vector2d(1, 1),
      Eigen::Vector2d(0, 1),
      Eigen::Vector2d(1, 0)
    };
    vertices = {
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 1, 0),
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(1, 0, 0)
    };
    bool is_self_overlapping = is_self_overlapping_polygon(
      uv_vertices,
      vertices,
      is_self_overlapping_subpolygon,
      splitting_vertices,
      min_face_areas
    );

    // Check is self overlapping
    REQUIRE( !is_self_overlapping );
  }
}

TEST_CASE( "A valid refinement mesh can be built", "[refinement]" )
{

}
