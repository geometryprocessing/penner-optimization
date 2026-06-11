// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <catch2/catch_test_macros.hpp>

#include "metric/common.h"
#include "parametrization/refinement.h"
#include "parametrization/triangulation.h"

using namespace Penner;
using namespace Penner::Optimization;

TEST_CASE( "A self overlapping polygon can be identified", "[refinement]" )
{
	std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
	std::vector<std::vector<int>> splitting_vertices;
	std::vector<std::vector<Scalar>> min_face_areas;
  std::vector<Eigen::Vector2d> uv_vertices;
  std::vector<Eigen::Vector3d> vertices;

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