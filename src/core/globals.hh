#pragma once

#include <Eigen/Sparse>
#include <string>
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/globals.hh"

namespace CurvatureMetric {
using namespace OverlayProblem;

#ifdef MULTIPRECISION
#include <unsupported/Eigen/MPRealSupport>
#include "mpreal.h"
typedef mpfr::mpreal Scalar;
#else
typedef double Scalar;
#endif

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<Scalar> MatrixX;
typedef Eigen::Triplet<Scalar> T;
const Scalar INF = 1e10;

/// Energies available for optimization
enum class EnergyChoice {
    log_length, // 2-norm of the metric coordinates
    log_scale, // scaling energy of best fit scale factors
    quadratic_sym_dirichlet, // quadratic approximation to symmetric Dirichlet
    sym_dirichlet, // symmetric Dirichlet
    p_norm // 4-norm of the metric coordinates
};

// Parameters to pass to the conformal method for projecting to the constraint.
// More detail on these parameters can be found in the documentation for that
// method.
struct ProjectionParameters
{
    int max_itr = 100; // maximum number of iterations
    Scalar bound_norm_thres = 1e-10; // line search threshold for dropping the gradient norm bound
#ifdef MULTIPRECISION
    Scalar error_eps = 1e-24; // minimum error termination condition
#else
    Scalar error_eps = 1e-8; // minimum error termination condition
#endif
    bool do_reduction =
        true; // reduce the initial line step if the range of coordinate values is large
    bool initial_ptolemy = true; // initial_ptolemy: use ptolemy flips for the initial make_delaunay
    bool use_edge_flips = true; // use intrinsic edge flips
    std::string output_dir = "";
};

// Parameters for the optimization method
struct OptimizationParameters
{
    // Logging
    std::string output_dir = ""; // output directory for file logs
    bool use_checkpoints = false; // if true, checkpoint state to output directory

    // Convergence parameters
    Scalar min_ratio =
        0.0; // minimum ratio of projected to ambient descent direction for convergence
    int num_iter = 200; // maximum number of iterations

    // Line step choice parameters
    bool require_energy_decr = true; // if true, require energy to decrease in each iteration
    bool require_gradient_proj_negative = true; // if true, require projection of the gradient onto
    // the descent direction to remain negative
    Scalar max_angle_incr = INF; // maximum allowed angle error increase in line step
    Scalar max_energy_incr = 1e-8; // maximum allowed energy increase in iteration

    // Optimization method choices
    std::string direction_choice = "projected_gradient"; // choice of direction

    // Numerical stability parameters
    Scalar beta_0 = 1.0; // initial line step size to try
    Scalar max_beta = 1e16; // maximum allowed line step size
    Scalar max_grad_range = 10; // maximum allowed gradient range (reduce if larger)
    Scalar max_angle = INF; // maximum allowed cone angle error (reduce if larger)
};
} // namespace CurvatureMetric
