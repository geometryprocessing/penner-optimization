#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include <igl/Timer.h>

namespace PennerHolonomy {

/**
 * @brief Parameters for Newton holonomy optimization
 *
 */
struct NewtonParameters
{
    std::string output_dir = ""; // directory name for genearting all stats
    bool error_log = false; // when true: write out per-newton iterations stats
    int checkpoint_frequency = 0; // when true: write out checkpoint state per n iterations
    bool reset_lambda = true; // when true: start with lambda = lambda0 for each newton iteration;
                              // if false, start with lambda from the previous
    bool do_reduction =
        false; // when true: reduce step, if the components of descent direction vary too much
    Scalar lambda0 = 1.0; // starting lambda value for the line search, normally 1
    Scalar error_eps = 0; // max angle error tolerance, terminate if below
    int max_itr = 500; // upper bound for newton iterations
    double max_time = 1e10; // upper bound for runtime (in seconds) before termination
    Scalar min_lambda = 1e-16; // terminate if lambda drops below this threshold
    int log_level = -1; // controlling detail of console logging
    Scalar bound_norm_thres = 1e-8; // line step threshold to stop bounding the error norm

#ifdef USE_SUITESPARSE
    std::string solver = "cholmod"; // solver to use for pseudoinverse computation
#else
    std::string solver = "ldlt"; // solver to use for pseudoinverse computation
#endif
};

/**
 * @brief Per iteration data log for Newton optimization.
 *
 * This data is used for writing per iteration data to file, but can also be used for extracting
 * final optimization status.
 *
 */
struct NewtonLog
{
    int num_iter = 0; // iterations of Newton descent performed
    Scalar max_error = 0.0; // maximum holonomy error (sup norm)
    Scalar step_size = 0.0; // step size taken along the Newton descent direction
    int num_flips = 0; // number of flips to make delaunay from initial connectivity

    Scalar l2_energy = 0.0; // l2 deviation from original metric coordinates
    Scalar rmse = 0.0; // root-mean-square-error of metric coordinates
    Scalar rrmse = 0.0; // relative-root-mean-square-error of metric coordinates
    Scalar rmsre = 0.0; // root-mean-square-relative-error of metric coordinates

    double time = 0.0; // time since start of Newton optimization
    double solve_time =
        0.0; // iteration time spent solving the linear system for the descent direction
    double constraint_time = 0.0; // iteration time spent constructing the constraint and jacobian
    double direction_time =
        0.0; // iteration time spent finding the descent direction (includes solve time)
    double line_search_time = 0.0; // time spent in the line search along the Newton direction

    Scalar min_corner_angle = 0.0; // minimum angle at a corner
    Scalar max_corner_angle = 0.0; // maximum angle at a corner

    Scalar direction_angle_change = 0.0; // angle between current and previous iteration descent direction

    Scalar direction_norm = 0.0; // norm of the Newton descent direction
    Scalar direction_residual = 0.0; // residual ||Ax - b|| of the linear solve

    Scalar error_norm_sq; // TODO
    Scalar proj_grad; // TODO
};

/**
 * @brief Optime holonomy constraints at vertices and along dual loop markings on the marked metric.
 *
 * This optimization minimizes deviation of the computed holonomy from constraints, and is expected
 * to produce solutions that satisfy the constraints up to near numerical precision. Constraints at
 * vertices (satisfying Gauss-Bonnet) and along a full system of loops on the surface are sufficient
 * to completely constrain the holonomy of any loop on the surface.
 * 
 * @param initial_marked_metric: mesh with metric, dual loop markings, and holonomy constraint values
 * @param alg_params: parameters for the optimization
 * @return mesh with metric optimized to satisfy holonomy constraints
 */
MarkedPennerConeMetric optimize_metric_angles(
    const MarkedPennerConeMetric& initial_marked_metric,
    const NewtonParameters& alg_params);

/**
 * @brief Optime holonomy constraints at vertices and along dual loop markings on the marked metric using
 * a subspace of the metric coordinate space.
 *
 * @param initial_marked_metric: mesh with metric, dual loop markings, and holonomy constraint values
 * @param metric_basis_matrix: matrix with basis vectors for the metric coordinate space as columns
 * @param alg_params: parameters for the optimization
 * @return mesh with metric optimized to satisfy holonomy constraints
 */
MarkedPennerConeMetric optimize_subspace_metric_angles(
    const MarkedPennerConeMetric& initial_marked_metric,
    const MatrixX& metric_basis_matrix,
    const NewtonParameters& alg_params);

/**
 * @brief Optime holonomy constraints at vertices and along dual loop markings on the marked metric using
 * a subspace of the metric coordinate space with exposed log for analysis of the final optimization state.
 *
 * @param initial_marked_metric: mesh with metric, dual loop markings, and holonomy constraint values
 * @param metric_basis_matrix: matrix with basis vectors for the metric coordinate space as columns
 * @param alg_params: parameters for the optimization
 * @param log: Newton iteration log
 * @return mesh with metric optimized to satisfy holonomy constraints
 */
MarkedPennerConeMetric optimize_subspace_metric_angles_log(
    const MarkedPennerConeMetric& initial_marked_metric,
    const MatrixX& metric_basis_matrix,
    const NewtonParameters& alg_params,
    NewtonLog& log);

/**
 * @brief Add the state of the optimized metric to the viewer.
 * 
 * @param marked_metric: mesh with initial metric
 * @param marked_metric: mesh with metric after optimization
 * @param vtx_reindex: map from halfedge to VF vertex indices
 * @param V: input mesh vertices
 * @param mesh_handle: (optional) handle for mesh in viewer
 * @param show: (optional) show viewer if true
 */
void view_optimization_state(
    const MarkedPennerConeMetric& init_marked_metric,
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    std::string mesh_handle="",
    bool show=true);

class OptimizeHolonomyNewton
{
public:
    MarkedPennerConeMetric run(
        const MarkedPennerConeMetric& initial_marked_metric,
        const MatrixX& metric_basis_matrix,
        const NewtonParameters& input_alg_params);

    OptimizeHolonomyNewton() {}

    NewtonLog get_log() { return log; }

protected:
    // Metric data
    VectorX reduced_metric_init;
    VectorX reduced_metric_coords;
    VectorX alpha;
    VectorX cot_alpha;

    // Constraint and descent direction data
    VectorX constraint;
    MatrixX J;
    VectorX descent_direction;

    // Previous descent direction data (for logging)
    VectorX prev_descent_direction;

    // Algorithm data
    Scalar lambda;
    NewtonParameters alg_params;

    // Logging data
    std::string checkpoint_dir;
    std::ofstream log_file;
    std::ofstream timing_file;
    std::ofstream energy_file;
    std::ofstream stability_file;
    std::ofstream metric_status_file;
    igl::Timer timer;
    NewtonLog log;
    std::unique_ptr<CurvatureMetric::EnergyFunctor> l2_energy;

    void initialize_logging();
    void initialize_metric_status_log(MarkedPennerConeMetric& marked_metric);

    void initialize_data_log();
    void write_data_log_entry();

    void initialize_timing_log();
    void write_timing_log_entry();

    void initialize_energy_log();
    void write_energy_log_entry();

    void initialize_stability_log();
    void write_stability_log_entry();

    void initialize_logs();
    void write_log_entries();
    void close_logs();

    void initialize_checkpoints();
    void checkpoint_direction();
    void checkpoint_metric(const MarkedPennerConeMetric& marked_metric);

    void update_log_error(const MarkedPennerConeMetric& marked_metric);

    void solve_linear_system(const MatrixX& metric_basis_matrix);

    void update_lambda();
    void update_holonomy_constraint(MarkedPennerConeMetric& marked_metric);
    void update_descent_direction(
        MarkedPennerConeMetric& marked_metric,
        const MatrixX& metric_basis_matrix);

    void perform_line_search(
        const MarkedPennerConeMetric& initial_marked_metric,
        MarkedPennerConeMetric& marked_metric);

    bool is_converged();
};

} // namespace PennerHolonomy