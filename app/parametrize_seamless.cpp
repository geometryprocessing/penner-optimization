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
#include "field/cross_field.h"
#include "field/intrinsic_field.h"
#include "holonomy/interface.h"
#include "holonomy/holonomy/newton.h"
#include "field/cones.h"
#include "holonomy/core/viewer.h"
#include "parametrization/refinement.h"
#include "parametrization/parametrize.h"

// post process uv optimization
#include "optimization/metric_optimization/uv_optimization.h"

// cone validation
#include "holonomy/holonomy/constraint.h"

// infer theta from direction field
#include "field/vector_field.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <CLI/CLI.hpp>
#include <filesystem>

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Optimization;
using namespace Penner::Holonomy;


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
    std::filesystem::path current_dir = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path input_json = current_dir / "symdir.json";

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
    bool show_field = false;
    spdlog::level::level_enum log_level = spdlog::level::info;
    app.add_option(
           "--max_triangle_quality",
           max_triangle_quality,
           "Maximum allowed triangle quality (0 for unbounded)")
        ->check(CLI::NonNegativeNumber);
    app.add_flag("--use_delaunay", use_delaunay, "Use Delaunay connectivity");
    app.add_flag("--fit_field", fit_field, "Fit new cross field");
    app.add_flag("--show_parameterization", show_parameterization, "Show final paramaterization");
    app.add_flag("--show_field", show_field, "Show guiding cross field");
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
    if (show_field) view_rotation_form(marked_metric, vtx_reindex, V, rotation_form, Th_hat, "rotation", false);

    // Make initial mesh Delaunay if desired
    std::vector<int> flip_seq = {};
    if (use_delaunay) {
        marked_metric.make_discrete_metric();
        flip_seq = marked_metric.get_flip_sequence();
        marked_metric.reset_flip_sequence();
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
#if USE_UV_OPTIMIZATION
        std::ifstream js_in(input_json);
        nlohmann::json config = nlohmann::json::parse(js_in);
        config["model"] = mesh;
        SymDir::Parameters uv_param = read_parameters(config);
        auto [Du, Dv] = comb_frame_field(V_r, F_r, uv_r, FT_r, reference_field_r, theta_r, period_jump_r);
        uv_r = optimize_seamless_parameterization(
            V_r,
            F_r, 
            uv_r,
            FT_r,
            Du,
            Dv,
            uv_param);
#else
    spdlog::warn("uv optimization disabled");
#endif
    }

    //if (show_parameterization) view_triangulation(V_o, F_o, fn_to_f_o, endpoints_o, "refinement", false);
    //if (show_parameterization) view_seamless_parameterization(V_o, F_o, uv_o, FT_o, "overlay", false);
    if (show_parameterization) view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "simplified");

    // Write the output mesh
    output_filename = join_path(output_dir, mesh + "_param.obj");
    write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);

}
