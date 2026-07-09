#include "field/frame_field.h"
#include "field/intrinsic_field.h"
#include "holonomy/core/viewer.h"
#include "feature/interface.h"
#include "feature/core/io.h"
#include "feature/core/viewer.h"
#include "feature/surgery/cut_metric_generator.h"
#include "util/vector.h"

#include <igl/writeOBJ.h>

#include <igl/readOBJ.h>
#include <CLI/CLI.hpp>

#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Feature;
using namespace Penner::Holonomy;

int main(int argc, char* argv[])
{
    // Get command line arguments
    CLI::App app{"Modify frame field in interactive viewer"};
    std::string mesh_filename = "./";
    std::string output_dir = "./";
    bool show_field = false;
    bool collapse_cones = false;

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Input mesh")->check(CLI::ExistingFile)->required();
    app.add_option("--output", output_dir, "Output directory");
    app.add_flag("--view", show_field, "open viewer to show field");
    app.add_flag("--collapse_cones", collapse_cones, "collapse adjacent cones");
    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(spdlog::level::info);
    std::filesystem::create_directory(output_dir);
    std::string mesh = std::filesystem::path(mesh_filename).stem().string();

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("optimizing mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    // refine input mesh
    std::vector<VertexEdge> feature_edges, hard_feature_edges;
    std::tie(V, F, feature_edges, hard_feature_edges) = generate_refined_feature_mesh(V, F, false);

    // get cut feature mesh
    FeatureFinder feature_finder(V, F);
    feature_finder.mark_features(feature_edges);
    Eigen::MatrixXd V_cut;
    Eigen::MatrixXi F_cut, F_is_feature;
    Eigen::VectorXi V_map;
    std::tie(V_cut, F_cut, V_map, F_is_feature) = feature_finder.generate_feature_cut_mesh();

    // generate initial field
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    std::tie(reference_field, theta, kappa, period_jump) = generate_refined_feature_field(V_cut, F_cut, V_map, collapse_cones);

    if (show_field) view_cross_field(V, F, reference_field, theta, kappa, period_jump);

    // write output
    std::string output_filename;
    output_filename = join_path(output_dir, mesh + ".obj");
    igl::writeOBJ(output_filename, V, F);
    output_filename = join_path(output_dir, mesh + "_features");
    write_feature_edges(output_filename, feature_edges);
    output_filename = join_path(output_dir, mesh + "_hard_features");
    write_feature_edges(output_filename, hard_feature_edges);
    output_filename = join_path(output_dir, mesh+".ffield");
    Penner::Field::write_frame_field(output_filename, reference_field, theta, kappa, period_jump);

}
