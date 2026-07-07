/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH 2024 Paper          *
*  `Seamless Parametrization in Penner Coordinates`                              *
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
#include "field/frame_field.h"
#include "field/intrinsic_field.h"
#include "holonomy/interface.h"
#include "holonomy/holonomy/newton.h"
#include "holonomy/holonomy/cones.h"
#include "holonomy/core/viewer.h"
#include "parametrization/refinement.h"
#include "parametrization/parametrize.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <CLI/CLI.hpp>
#include <filesystem>

#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "main_helper.h"

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Holonomy;



Eigen::MatrixXd optimize_seamless_parameterization(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXi& F_init,
    const Eigen::MatrixXd& uv_init,
    const Eigen::MatrixXi& FT_init,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& thetas,
    const Eigen::MatrixXi& period_jumps,
    const nlohmann::json& config
)
{
    Eigen::MatrixXd uv = uv_init;
    Eigen::MatrixXi F = FT_init;
    
    SymDir::Parameters param;
    param.max_iters = config["max_iters"]; // iterations
    param.smooth_only_iters = config["smooth_only_iters"];
    param.E_target = config["E_target"]; // Energy target
    param.ls_iters = config["ls_iters"]; // param for linesearch in smoothing operation
    param.do_newton = config["do_newton"]; // do newton/gd steps for smoothing operation
    // do global/local smooth (local smooth does not optimize boundary vertices)
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
    param.fix_misaligned = config["fix_misaligned"];
    param.use_rref = config["use_rref"];
    param.model_name = config["model"];
    param.percentage_target_converge = false;
    param.use_worst_n_energy_in_ls = false;

	MeshCutter meshcutter(V_init, uv, F_init, F);

	auto [V, EE] = meshcutter.cut_mesh();

    Eigen::MatrixXi FE_init;
    Eigen::MatrixXi FE(0, 0);
    Eigen::MatrixXi ME(0, 0);
    if (param.do_feature_alignment)
    {
        // TODO
        // Loading the feature edge constraints
        //FE_init = meshcutter.load_feature_edges(input_file);
        //FE = meshcutter.reindex_feature_edges(FE_init);
        //if (param.fix_misaligned)
        //{
        //    std::string misaligned_file = input_dir + "/" + model + "_misaligned_edges";
        //    ME = meshcutter.load_misaligned_edges(misaligned_file);
        //}
    }
    
    double cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Initial constraints error {}", cons_residual);

    Eigen::MatrixXi new_F;
    Eigen::MatrixXd new_V, new_uv;
    SymDir::ExtremeOpt extremeopt(V, F);
    extremeopt.m_params = param;
    
    extremeopt.create_mesh(V, F, uv);

    nlohmann::json opt_log;
    opt_log["model_name"] = config["model"];
    opt_log["args"] = config;

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
    extremeopt.comb_matchings(reference_field, thetas, period_jumps);
    extremeopt.do_optimization(opt_log);

    extremeopt.export_mesh(V, F, uv);
    cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Final constraints error {}", cons_residual);

    if (extremeopt.m_params.with_cons) extremeopt.export_EE(EE);

    return uv;
}


int main(int argc, char* argv[])
{
    std::map<std::string, spdlog::level::level_enum> log_level_map {
        {"trace",    spdlog::level::trace},
        {"debug",    spdlog::level::debug},
        {"info",     spdlog::level::info},
        {"warn",     spdlog::level::warn},
        {"critical", spdlog::level::critical},
        {"off",      spdlog::level::off},
    };

    // Get command line arguments
    CLI::App app{"Generate a constrained seamless parametrization."};
    std::filesystem::path mesh_filename = "";
    std::filesystem::path Th_hat_filename = "";
    std::filesystem::path field_filename = "";
    std::filesystem::path output_dir = "./";
    std::filesystem::path input_json = "../scripts/symdir.json";

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->check(CLI::ExistingFile)->required();
    app.add_option("--cones", Th_hat_filename, "Cone angle filepath")
        ->check(CLI::ExistingFile);
    app.add_option("--field", field_filename, "Rotation field one form")
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output", output_dir, "Output directory");
    app.add_option("--json", input_json, "JSON specification for uv optimization");

    // Marked Metric Parameters
    // NOTE: Only several parameters are exposed to the CLI
    MarkedMetricParameters marked_metric_params;
    NewtonParameters alg_params;
    //bool use_free_cones = false;
    bool optimize = false;
    app.add_flag(
        "--remove_loop_constraints",
        marked_metric_params.remove_loop_constraints,
        "Remove homology basis loop holonomy constraints");
    app.add_option("--max_itr", alg_params.max_itr, "Upper bound for newton iterations")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--error_eps", alg_params.error_eps, "Error convergence threshold")
        ->check(CLI::NonNegativeNumber);
    app.add_flag("--use_initial_zero", marked_metric_params.use_initial_zero, "Use zero coordinates");
    app.add_flag("--use_free_cones", marked_metric_params.use_free_cones, "Let cones have free angles");
    app.add_flag("--optimize", optimize, "Optimize uv map for distortion and field alignment");
    alg_params.output_dir = output_dir;
    alg_params.error_log = true;

    // Miscellaneous
    double max_triangle_quality = 0.;
    bool use_delaunay = false;
    bool fit_field = false;
    bool show_parameterization = false;
    spdlog::level::level_enum log_level = spdlog::level::info;
    app.add_option(
           "--max_triangle_quality",
           max_triangle_quality,
           "Maximum allowed triangle quality (0 for unbounded)")
        ->check(CLI::NonNegativeNumber);
    app.add_flag("--use_delaunay", use_delaunay, "Use Delaunay connectivity");
    app.add_flag("--fit_field", fit_field, "Fit new cross field");
    app.add_flag("--show_parameterization", show_parameterization, "Show final paramaterization");
    app.add_option("--log_level", log_level, "Level of logging")
        ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));

    CLI11_PARSE(app, argc, argv);
    spdlog::set_level(log_level);
    std::string mesh = mesh_filename.stem();
    std::filesystem::create_directory(output_dir);

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("Using mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    // Get input angles from cross field or file
    std::string field_format = field_filename.extension();
    MarkedPennerConeMetric marked_metric;
    std::vector<int> vtx_reindex;
    std::vector<Scalar> Th_hat;
    VectorX rotation_form(F.rows() * 3);
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    if ((fit_field) || (field_filename == "")) {
        FieldParameters field_params;
        field_params.use_principal_directions = true;
        std::tie(reference_field, theta, kappa, period_jump) = generate_frame_field(V, F, field_params);
        std::tie(marked_metric, vtx_reindex, rotation_form, Th_hat) = generate_metric_from_field(V, F, theta, kappa, period_jump, marked_metric_params);

        //std::tie(rotation_form, Th_hat) = generate_intrinsic_rotation_form(V, F, field_params);
        //std::string mesh_name = std::filesystem::path(mesh_filename).filename().replace_extension();
        //write_vector(Th_hat, join_path(output_dir, mesh_name + "_Th_hat"));
    }
    else if (field_format == ".ffield")
    {
        auto [m, vtx_reindex] = generate_mesh(V, F, V, F, Th_hat);
        std::tie(reference_field, theta, kappa, period_jump) = load_frame_field(field_filename);
        std::tie(marked_metric, vtx_reindex, rotation_form, Th_hat) = generate_metric_from_field(V, F, theta, kappa, period_jump, marked_metric_params);
    }
    else if (field_format == ".rosy")
    {
        auto [m, vtx_reindex] = generate_mesh(V, F, V, F, Th_hat);
        auto rosy_field = load_rosy_field(field_filename);

        // initialize feild generator with the given theta 
        int num_faces = F.rows();
        Eigen::VectorXi reference_corner(num_faces);
        theta.resize(num_faces);
        kappa.resize(num_faces, 3);
        period_jump.resize(num_faces, 3);
        IntrinsicNRosyField field_generator;
        field_generator.min_angle = M_PI / 2.;
        field_generator.use_trivial_boundary = true;
        field_generator.initialize(m);
        field_generator.get_field(m, vtx_reindex, F, reference_corner, theta, kappa, period_jump);
        theta = infer_theta(V, F, reference_corner, rosy_field);
        field_generator.set_field(m, vtx_reindex, F, theta, kappa, period_jump);
        field_generator.compute_principal_matchings(m);

        // get field
        field_generator.get_field(m, vtx_reindex, F, reference_corner, theta, kappa, period_jump);
        reference_field = generate_reference_field(V, F, reference_corner);
        rotation_form = field_generator.compute_rotation_form(m);
        Th_hat = generate_cones_from_rotation_form(m, vtx_reindex, rotation_form);
        std::vector<int> free_cones = {};
        std::tie(marked_metric, vtx_reindex) =
            generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params);
    }
    else {
        // Get input rotation
        std::vector<Scalar> rotation_form_vec;
        spdlog::info("Using rotation_form at {}", field_filename);
        read_vector_from_file(field_filename, rotation_form_vec);
        convert_std_to_eigen_vector(rotation_form_vec, rotation_form);

        if (Th_hat_filename != "")
        {
            spdlog::info("Using cone angles at {}", Th_hat_filename);
            read_vector_from_file(Th_hat_filename, Th_hat);
        }
        else
        {
            auto [m, vtx_reindex] = generate_mesh(V, F, V, F, Th_hat);
            Th_hat = generate_cones_from_rotation_form(m, vtx_reindex, rotation_form);
        }

        // Generate initial marked mesh for optimization
        std::vector<int> free_cones = {};
        std::tie(marked_metric, vtx_reindex) =
            generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params);
    }

    // Check for invalid cones and fix any issues
    if (!validate_cones(marked_metric)) {
        spdlog::info("Fixing invalid cones");
        fix_cones(marked_metric);
    }

    // add constraints to viewer
    view_rotation_form(marked_metric, vtx_reindex, V, rotation_form, Th_hat, "rotation", false);

    // Make initial mesh Delaunay if desired
    std::vector<int> flip_seq = {};
    if (use_delaunay) {
        marked_metric.make_discrete_metric();
        flip_seq = marked_metric.get_flip_sequence();
        marked_metric.reset();
    }

    // Regularize
    if (max_triangle_quality > 0.) {
        regularize_metric(marked_metric, max_triangle_quality);
    }

    // Optimize metric
    spdlog::info("Beginning optimization");
    auto opt_marked_metric = optimize_metric_angles(marked_metric, alg_params);

    // Undo any initial flips to make Delaunay
    for (auto iter = flip_seq.rbegin(); iter != flip_seq.rend(); ++iter) {
        int h = *iter;
        spdlog::trace("Flipping {} cw", h);
        opt_marked_metric.flip_ccw(h, true);
        opt_marked_metric.flip_ccw(h, true);
        opt_marked_metric.flip_ccw(h, true);
    }

    // Write the output metric coordinates
    std::string output_filename;
    output_filename = join_path(output_dir, "optimized_metric_coords");
    write_vector(opt_marked_metric.get_reduced_metric_coordinates(), output_filename);

    // Generate full overlay
    std::vector<bool> is_cut = {};
    auto vf_res = generate_VF_mesh_from_metric(
        V,
        F,
        Th_hat,
        marked_metric,
        opt_marked_metric.get_metric_coordinates(),
        is_cut,
        false);
    Eigen::MatrixXd V_o = std::get<1>(vf_res);
    Eigen::MatrixXi F_o = std::get<2>(vf_res);
    Eigen::MatrixXd uv_o = std::get<3>(vf_res);
    Eigen::MatrixXi FT_o = std::get<4>(vf_res);
    std::vector<int> fn_to_f_o = std::get<7>(vf_res);
    std::vector<std::pair<int, int>> endpoints_o = std::get<8>(vf_res);

    // Generate minimal refinement
    RefinementMesh refinement_mesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o);
    refinement_mesh.refine_mesh();
    refinement_mesh.simplify_mesh();
    auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r] = refinement_mesh.get_VF_mesh();
    auto [reference_field_r, theta_r, kappa_r, period_jump_r] = refine_frame_field(
        F_r,
        FT_r,
        fn_to_f_r,
        endpoints_r,
        F,
        reference_field,
        theta,
        kappa,
        period_jump);

    bool write_field = true;
    std::string ffield_file = join_path(output_dir, mesh+".ffield");
    if (write_field)
    {
        write_frame_field(ffield_file,  reference_field_r, theta_r, kappa_r, period_jump_r);
    }

    // Optionally optimize parameterization 
    if (optimize)
    {
        std::ifstream js_in(input_json);
        nlohmann::json config = nlohmann::json::parse(js_in);
        config["model"] = mesh;
        uv_r = optimize_seamless_parameterization(
            V_r,
            F_r, 
            uv_r,
            FT_r,
            reference_field_r,
            theta_r,
            period_jump_r,
            config);
    }

    if (show_parameterization) view_triangulation(V_o, F_o, fn_to_f_o, endpoints_o, "refinement", false);
    if (show_parameterization) view_seamless_parameterization(V_o, F_o, uv_o, FT_o, "overlay", false);
    if (show_parameterization) view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "simplified");

    // Write the output mesh
    output_filename = join_path(output_dir, mesh + "_param.obj");
    write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);

}
