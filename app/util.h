#pragma once 

#include "feature/core/common.h"
#include "holonomy/interface.h"
#include "holonomy/holonomy/newton.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <CLI/CLI.hpp>

namespace Penner {

const std::map<std::string, spdlog::level::level_enum> log_level_map {
    {"trace",    spdlog::level::trace},
    {"debug",    spdlog::level::debug},
    {"info",     spdlog::level::info},
    {"warn",     spdlog::level::warn},
    {"critical", spdlog::level::critical},
    {"off",      spdlog::level::off},
};

inline void add_marked_metric_parameters(
    CLI::App& app,
    Holonomy::MarkedMetricParameters& marked_metric_params
) {
    app.add_flag(
        "--use_initial_zero",
        marked_metric_params.use_initial_zero,
        "Use initial zero coordinates");
    app.add_flag(
        "--remove_loop_constraints",
        marked_metric_params.remove_loop_constraints,
        "Remove homology basis loop holonomy constraints");
    app.add_flag(
        "--remove_symmetry",
        marked_metric_params.remove_symmetry,
        "Remove symmetry structure from mesh");
}

inline void add_newton_parameters(
    CLI::App& app,
    Holonomy::NewtonParameters& alg_params 
) {
    app.add_flag(
        "--do_reduction",
        alg_params.do_reduction,
        "Reduce line step size to bounded range for stability");
    app.add_option(
        "--solver",
        alg_params.solver,
        "Solver to use for linear systems");
    app.add_flag(
        "--reset_lambda",
        alg_params.reset_lambda,
        "Start with lambda = lambda0 for each newton iteration");
    app.add_option("--max_itr", alg_params.max_itr, "Upper bound for newton iterations")
        ->check(CLI::NonNegativeNumber);
    app.add_option(
           "--bound_norm_thres",
           alg_params.bound_norm_thres,
           "Line step threshold to stop bounding the error norm")
        ->check(CLI::Number);
    app.add_option(
           "--checkpoint_frequency",
           alg_params.checkpoint_frequency,
           "Number of iterations to wait between checkpoints (nonpositive for never)")
        ->check(CLI::Number);
}

} // namespace Penner
