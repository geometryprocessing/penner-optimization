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

#include "energies.hh"
#include "shapes.hh"
#include "common.hh"
#include "cone_metric.hh"

using namespace CurvatureMetric;

TEST_CASE( "The energies of a triangle can be computed", "[energies]" )
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<double> Th_hat;
  generate_triangle(V, F, Th_hat);
	std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
	bool fix_boundary = false;
	Mesh<double> m = FV_to_double<double>(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops, std::vector<int>(), fix_boundary);

  // Use right triangle as the metric and uniform unit lengths as the target
  VectorX metric_coords(3);
  VectorX metric_target(3);
  metric_coords << 0.0, 0.0, 2.0 * std::log(std::sqrt(2));
  metric_target << 0.0, 0.0, 0.0;
  PennerConeMetric cone_metric(m, metric_target);
  
  SECTION ( "First invariant" )
  {
    VectorX f2J1;
    MatrixX J_f2J1;
    bool need_jacobian = true;
    first_invariant(
      cone_metric,
      metric_coords,
      f2J1,
      J_f2J1,
      need_jacobian
    );

    double A0 = std::sqrt(3.0) / 4.0;
    double cot0 = 1.0 / std::sqrt(3.0);
    REQUIRE( f2J1.size() == 2 );
    REQUIRE( float_equal(f2J1[0], (4.0 * cot0) / (2.0 * A0)) );
    REQUIRE( float_equal(f2J1[1], (4.0 * cot0) / (2.0 * A0)) );
    REQUIRE( J_f2J1.rows() == 2 );
    REQUIRE( J_f2J1.cols() == 3 );
    REQUIRE( float_equal(J_f2J1.coeff(0, 0), (1.0 * cot0) / (2.0 * A0)) );
    REQUIRE( float_equal(J_f2J1.coeff(0, 1), (1.0 * cot0) / (2.0 * A0)) );
    REQUIRE( float_equal(J_f2J1.coeff(0, 2), (2.0 * cot0) / (2.0 * A0)) );
  }

  SECTION ( "Second invariant" )
  {
    VectorX f2J2;
    MatrixX J_f2J2;
    bool need_jacobian = true;
    second_invariant_squared(
      cone_metric,
      metric_coords,
      f2J2,
      J_f2J2,
      need_jacobian
    );

    double A0_sq = 3.0 / 16.0;
    double A_sq = 0.25;
    REQUIRE( f2J2.size() == 2 );
    REQUIRE( float_equal(f2J2[0], A_sq / A0_sq) );
    REQUIRE( float_equal(f2J2[1], A_sq / A0_sq) );
    REQUIRE( J_f2J2.rows() == 2 );
    REQUIRE( J_f2J2.cols() == 3 );
    REQUIRE( float_equal(J_f2J2.coeff(0, 0), A_sq / A0_sq) );
    REQUIRE( float_equal(J_f2J2.coeff(0, 1), A_sq / A0_sq) );
    REQUIRE( float_equal(J_f2J2.coeff(0, 2), 0.0) );
  }
}

TEST_CASE( "The errors of a triangle can be computed", "[energies]" ) {
  SECTION ( "Errors" )
  {
    VectorX x(3);
    VectorX x0(3);
    x << 1.0, 2.0, 0.5;
    x0 << 2.0, 2.0, 1.0;
    // err is 1.0, 0.0, 0.5
    // Sq err is 1.0, 0.0, 0.25
    // Rel err is 0.5, 0.0, 0.5
    // Rel sq err is 0.25, 0.0, 0.25
    // total Sq err is 1.25
    // rel factor is 9
    CHECK( float_equal(root_mean_square_error(x, x0), std::sqrt(1.25 / 3.0)) );
    CHECK( float_equal(relative_root_mean_square_error(x, x0), std::sqrt(1.25 / (3.0 * 9.0))) );
    CHECK( float_equal(root_mean_square_relative_error(x, x0), std::sqrt(0.5 / 3.0)) );
  }
}
