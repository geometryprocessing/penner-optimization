// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "optimization/metric_optimization/uv_optimization.h"

#if USE_UV_OPTIMIZATION 


namespace Penner {
namespace Optimization {

SymDir::Parameters load_parameters(const std::string& filepath)
{
    std::ifstream js_in(filepath);
    nlohmann::json config = nlohmann::json::parse(js_in);
    return read_parameters(config);
}

SymDir::Parameters read_parameters(const nlohmann::json& config)
{
    SymDir::Parameters param;
    param.max_iters = config["max_iters"]; // iterations
    param.max_time = config["max_time"]; // time in seconds
    param.smooth_only_iters = config["smooth_only_iters"];
    param.E_target = config["E_target"]; // Energy target
    param.ls_iters = config["ls_iters"]; // param for linesearch in smoothing operation
    param.do_newton = config["do_newton"]; // do newton/gd steps for smoothing operation
    param.local_smooth = config["local_smooth"];
    param.global_smooth = config["global_smooth"];
    param.elen_alpha = config["elen_alpha"];
    param.do_projection = config["do_projection"];
    param.with_cons = config["with_cons"];
    param.Lp = config["Lp"];
    param.save_meshes = config["save_meshes"];
    param.do_feature_alignment = config["do_feature_alignment"]; // align feature edges
    param.symdir_weight = config["symdir_weight"];
    param.alignment_weight = config["alignment_weight"];
    param.degenerate_weight = config["degenerate_weight"];
    param.fix_misaligned = config["fix_misaligned"];
    param.use_rref = config["use_rref"];
    // param.solver_type = config["solver_type"];
    param.cg_rel_err = config["cg_rel_err"];
    param.cg_iters = config["cg_iters"];
    
    param.percentages = config["percentages"].get<std::vector<double>>();
    param.percentage_target = config["percentage_target"];
    param.percentage_target_value = config["percentage_target_value"];
    param.save_percentages_meshes = config["save_percentages_meshes"];

    param.E_abs_err = config["E_abs_err"];
    param.E_rel_err = config["E_rel_err"];
    param.diff_err = config["diff_err"];
    param.grad_abs_err = config["grad_abs_err"];
    param.grad_rel_err = config["grad_rel_err"];
    
    param.precompute_seamless = config["precompute_seamless"];
    param.projected_newton = config["projected_newton"];
    param.soft_max = config["soft_max"];
    param.t = config["t"];
    param.precompute_seamless = config["precompute_seamless"];
    
    param.percentage_target_converge = config["percentage_target_converge"];
    param.max_grad_abs_converge = config["max_grad_abs_converge"];
    param.max_grad_rel_converge = config["max_grad_rel_converge"];
    param.energy_diff_converge = config["energy_diff_converge"];
    param.use_worst_n_energy_in_ls = config["use_worst_n_energy_in_ls"];
    param.E_abs_converge = config["E_abs_converge"];
    param.E_rel_converge = config["E_rel_converge"];

    param.last_screenshot_after_optimization = config["last_screenshot_after_optimization"];
    param.screenshot_interval = config["screenshot_interval"];
    param.output_dir_for_screenshots = config["output_dir_for_screenshots"];
    param.uv_scale_for_screenshots = config["uv_scale_for_screenshots"];
    param.angle_to_rotate_model_for_screenshots = config["angle_to_rotate_model_for_screenshots"];
    param.screenshot_during_optimization = config["screenshot_during_optimization"];

    param.degenerate_vertices_preconditioner = config["degenerate_vertices_preconditioner"];
    param.precond_dim = config["precond_dim"];
    param.triangle_threshold = config["triangle_threshold"];

    return param;
}   

Eigen::MatrixXd optimize_seamless_parameterization(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXi& F_init,
    const Eigen::MatrixXd& uv_init,
    const Eigen::MatrixXi& FT_init,
    const Eigen::MatrixXd& Du,
    const Eigen::MatrixXd& Dv,
    SymDir::Parameters param)
{
    // set trivial feature edges
    Eigen::MatrixXi FE(0, 0);
    Eigen::MatrixXi ME(0, 0);
    return optimize_aligned_parameterization(
      V_init,
      F_init,
      uv_init,
      FT_init,
      FE,
      ME,
      Du,
      Dv,
      param);
}

Eigen::MatrixXd optimize_aligned_parameterization(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXi& F_init,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& FE_init,
    const Eigen::MatrixXi& ME,
    const Eigen::MatrixXd& Du,
    const Eigen::MatrixXd& Dv,
    SymDir::Parameters param)
{
    // cut mesh along seams
    MeshCutter meshcutter(V_init, uv, F_init, F);
    auto [V, EE] = meshcutter.cut_mesh();
    Eigen::MatrixXi FE(0, 0);
    if (param.do_feature_alignment)
    {
        // Loading the feature edge constraints
        FE = meshcutter.reindex_feature_edges(FE_init);
    }

    // initialize data structures
    SymDir::ExtremeOpt extremeopt(V, F);
    extremeopt.m_params = param;
    extremeopt.create_mesh(V, F, uv);
    extremeopt.set_v_map(F_init, F);

    // set constraints
    if (extremeopt.m_params.with_cons)
    {
        std::vector<std::vector<int>> EE_e = transform_EE(F, EE);
        std::vector<std::vector<int>> FE_e;
        if (extremeopt.m_params.do_feature_alignment) {
            FE_e = transform_FE(F, FE);
        }
        extremeopt.init_constraints(EE_e);
        extremeopt.EE = EE;
        extremeopt.FE = FE;
        extremeopt.ME = ME;
    }

    // set field directions
    extremeopt.PD1 = Du;
    extremeopt.PD2 = Dv;

    // optimize parametrization
    Eigen::MatrixXi F_opt = F;
    Eigen::MatrixXd uv_opt;
    extremeopt.do_optimization_without_log();
    extremeopt.export_mesh(V, F_opt, uv_opt);

    return uv_opt;
}

} // namespace Optimization
} // namespace Penner

#endif