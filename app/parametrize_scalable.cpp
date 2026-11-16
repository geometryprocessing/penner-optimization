#include "holonomy/core/common.h"
#include "holonomy/core/viewer.h"
#include "holonomy/interface.h"
#include "holonomy/holonomy/constraint.h"
#include "field/frame_field.h"
#include "parametrization/layout.h"
#include "metric/area.h"
#include "holonomy/core/dual_lengths.h"
#include "feature/feature/error.h"
#include "feature/feature/error.h"

#include "resolution/optimization/field.h"
#include "holonomy/core/viewer.h"
#include "field/cones.h"
#include "resolution/interface.h"
#include "resolution/hierarchy/overlay.h"
#include "resolution/hierarchy/voronoi.h"
#include "resolution/core/geodesics.h"
#include "resolution/core/geodesic_flipper.h"
#include "feature/core/component_mesh.h"
#include "feature/core/io.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"
#endif

#include <igl/readOBJ.h>
#include <igl/readMSH.h>
#include <igl/doublearea.h>
#include <igl/writeMSH.h>
#include <igl/local_basis.h>
#include <igl/rotate_vectors.h>
#include <igl/Timer.h>
#include <igl/write_triangle_mesh.h>
#include <igl/internal_angles.h>

#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Holonomy;
using namespace Penner::Feature;
using namespace Penner::Resolution;

int main(int argc, char *argv[])
{

    // Get command line arguments
    CLI::App app{"Generate a parametrization with a scalable hierarchical method."};
    std::filesystem::path mesh_filename = "";
    std::string output_dir = "./";
    std::string input_json = "";
    std::string symdir_json = "";
    bool show_parametrization = false;
    bool write_mesh = true;
    bool silent = false;
    bool optimize = false;

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->required();
    app.add_option("--output", output_dir, "Output directory");
    app.add_option("-j,--json", input_json, "JSON of default parameters");
    app.add_option("--symdir_json", symdir_json, "JSON of symdir optimization parameters");

    app.add_option("--ratio", "ratio of vertices to simplify");
    app.add_flag("--show_parametrization", show_parametrization, "show parametrized mesh");
    app.add_flag("--write_mesh", write_mesh, "save mesh to file");
    app.add_flag("--silent", silent, "disable logging");
    app.add_flag("--optimize", optimize, "apply uv optimization");

    // Miscellaneous
    spdlog::level::level_enum log_level = spdlog::level::info;
    CLI11_PARSE(app, argc, argv);
    if (silent) log_level = spdlog::level::off;
    spdlog::set_level(log_level);

    // separate file paths and types
    std::string file_format = mesh_filename.extension();
    std::string name = mesh_filename.stem();
    std::string input_dir = mesh_filename.parent_path();

    // get input mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    read_mesh(mesh_filename, V, F);

    // get input field if needed
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    if (file_format == ".obj")
    {
        std::string field_filename = join_path(input_dir, name + ".ffield");
        std::tie(reference_field, theta, kappa, period_jump) = load_frame_field(field_filename);
    }
    else if (file_format == ".h5")
    {
        std::string mesh_filename = join_path(input_dir, name + ".h5");
        spdlog::info("Using mesh at {}", mesh_filename);
        read_hdf5_mesh(mesh_filename, V, F);
        reference_field = read_hdf5_vector_field(mesh_filename, "reference_field");
        theta = read_hdf5_vector_field(mesh_filename, "theta");
        kappa = read_hdf5_vector_field(mesh_filename, "kappa");
        period_jump = read_hdf5_integer_matrix(mesh_filename, "period_jump");
    }

    // get method parameters
    json results;
    json config = ScalableConfig{};                                  // 1. defaults
    if (!input_json.empty()) {
        std::ifstream in(input_json);
        config.merge_patch(json::parse(in));                         // 2. file
    }
    overwrite_json_with_cli(config, app);                            // 3. command line
    ScalableConfig cfg = config_from_json(config);
    results["config"] = config;

    NewtonParameters alg_params;
    ResolutionMetricGenerator resolution_metric(V, F);
    resolution_metric.set_field(reference_field, theta, kappa, period_jump);
    if (symdir_json != "") resolution_metric.symdir_json = symdir_json;
    resolution_metric.run(cfg, alg_params);
    if (optimize) resolution_metric.optimize();

    auto [V_o, F_o, uv_o, FT_o] = resolution_metric.get_parametrized_mesh();

    // optionally visualize mesh
    if (show_parametrization) Holonomy::view_seamless_parameterization(V_o, F_o, uv_o, FT_o);

    // optionally write mesh to file
    if (write_mesh)
    {
        // write refined mesh with uvs to file
        std::string output_path = join_path(output_dir, name + "_param.h5");
        spdlog::info("writing mesh to {}", output_path);
        write_hdf5_mesh_with_uv(output_path, V_o, F_o, uv_o, FT_o);

        spdlog::info("writing field to {}", output_path);
        auto [PD1, PD2] = resolution_metric.get_combed_field();
        write_hdf5_vector_field(output_path, "u_target_field", PD1);
        write_hdf5_vector_field(output_path, "v_target_field", PD2);

        // write field data
        bool write_full_field = true;
        if (write_full_field)
        {
            auto [reference_field_r, theta_r, kappa_r, period_jump_r] = resolution_metric.get_refined_field();
            write_hdf5_vector_field(output_path, "reference_field", reference_field_r);
            write_hdf5_vector_field(output_path, "theta", theta_r);
            write_hdf5_vector_field(output_path, "kappa", kappa_r);
            write_hdf5_integer_matrix(output_path, "period_jump", period_jump_r);
        }
    }
}
