#pragma once

#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/globals.hh"
#include <Eigen/Sparse>
#include <string>
#include "mpreal.h"

namespace CurvatureMetric {
using namespace OverlayProblem;

#ifdef MULTIPRECISION
#include <unsupported/Eigen/MPRealSupport>
typedef mpfr::mpreal Scalar;
#else
typedef double Scalar;
#endif

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> MatrixX;
typedef Eigen::Triplet<Scalar> T;
const Scalar INF = 1e10;

// Parameters to pass to the conformal method for projecting to the constraint.
// More detail on these parameters can be found in the documentation for that
// method.
//
// max_itr: maximum number of iterations
// bound_norm_thres: line search threshold for dropping the gradient norm bound
// error_eps: minimum error termination condition
// do_reduction: reduce the initial line step if the range of coordinate values
// is large initial_ptolemy: use ptolemy flips for the initial make_delaunay
// use_edge_flips: use intrinsic edge flips
struct ProjectionParameters
{
  int max_itr = 100;
  Scalar bound_norm_thres = 1e-10;
#ifdef MULTIPRECISION
  Scalar error_eps = 1e-24;
#else
  Scalar error_eps = 1e-8;
#endif
  bool do_reduction = true;
  bool initial_ptolemy = true;
  bool use_edge_flips = true;
  std::string output_dir = "";
};

struct OptimizationParameters
{
  // Logging
  std::string output_dir = "";
  bool use_checkpoints = false;

  // Convergence parameters
  Scalar min_ratio = 0.0;
  int num_iter = 200;

  // Line step choice parameters
  bool require_energy_decr = true;
  bool require_gradient_proj_negative = true;
  Scalar max_angle_incr = INF;
  Scalar max_energy_incr = 1e-8;

  // Optimization method choices
  std::string direction_choice = "gradient";
  bool use_optimal_projection = false;
  bool use_edge_lengths = false;

  // Energy parameters
  std::string energy_choice = "p_norm";
  int p = 2;
  bool use_log = true;
  bool fix_bd_lengths = false;

  // Quadratic energy parameters
  Scalar cone_weight = 1.0;
  Scalar bd_weight = 1.0;
  Scalar reg_factor = 0.0;

  // Numerical stability parameters
  Scalar beta_0 = 1.0;
  Scalar max_beta = 1e16;
  Scalar max_grad_range = 10;
  Scalar max_angle = INF;
};
}
