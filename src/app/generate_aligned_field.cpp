#include "holonomy/core/common.h"
#include "feature/core/viewer.h"
#include "feature/interface.h"
#include "feature/feature/features.h"
#include "feature/surgery/cut_metric_generator.h"
#include "feature/core/io.h"
#include "util/io.h"
#include "field/frame_field.h"
#include "holonomy/core/viewer.h"
#include <igl/bounding_box_diagonal.h>
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
    Scalar displacement = 0.;
    bool show_field = false;
    bool collapse_cones = false;

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Input mesh")->check(CLI::ExistingFile)->required();
    app.add_option("--output", output_dir, "Output directory");
    app.add_option("--displacement", displacement, "displacement for cross field viewer");
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
    FeatureFinder feature_finder(V, F);
    feature_finder.mark_features(feature_edges);
    Eigen::MatrixXd V_cut;
    Eigen::MatrixXi F_cut, F_is_feature;
    Eigen::VectorXi V_map;
    std::tie(V_cut, F_cut, V_map, F_is_feature) = feature_finder.generate_feature_cut_mesh();

    // fit field directions
    int radius = 5;
    Scalar rel_anisotropy=0.9;
    Scalar abs_anisotropy=0.2;
    Scalar bb_diag = igl::bounding_box_diagonal(V);
    auto [direction, is_fixed_direction] = Penner::Field::compute_field_direction(
        V_cut,
        F_cut,
        radius,
        abs_anisotropy / bb_diag,
        rel_anisotropy);

    // generate initial field and halfedge mesh
    MarkedMetricParameters marked_metric_params;
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    marked_metric_params.use_log_length = true;
    CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, {});
    cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction);
    std::tie(reference_field, theta, kappa, period_jump) = cut_metric_generator.get_field();
    Eigen::VectorXi reference_corner(reference_field.rows());

    MarkedPennerConeMetric marked_metric;
    std::vector<int> vtx_reindex;
    std::vector<int> face_reindex;
    VectorX rotation_form;
    std::vector<Scalar> Th_hat;
    std::tie(marked_metric, vtx_reindex, face_reindex, rotation_form, Th_hat) = cut_metric_generator.get_union_metric( marked_metric_params);

    // modify field
    Penner::Field::IntrinsicNRosyField field_generator;
    field_generator.initialize(marked_metric);
    field_generator.set_field(marked_metric, vtx_reindex, F_cut, face_reindex, theta, kappa, period_jump);
    Eigen::MatrixXd V_disp = displace_cut_faces(V_cut, F_cut, displacement);
            
    if (collapse_cones)
    {
        field_generator.collapse_nearby_cones(marked_metric);
        field_generator.get_field(marked_metric, vtx_reindex, F_cut, face_reindex, reference_corner, theta, kappa, period_jump);
    }

    if (show_field)
    {
        polyscope::init();
        field_generator.update_viewer(marked_metric, vtx_reindex, V_disp);
        auto callback = [&] () {
            if (ImGui::Button("collapse cones")) {
                field_generator.collapse_adjacent_cones(marked_metric);
                field_generator.update_viewer(marked_metric, vtx_reindex, V_disp);
            }
            if (ImGui::Button("parameterize")) {
                field_generator.get_field(marked_metric, vtx_reindex, F_cut, face_reindex, reference_corner, theta, kappa, period_jump);

                NewtonParameters alg_params;
                alg_params.output_dir = "./";
                alg_params.error_eps = 1e-10;
                alg_params.solver = "ldlt";
                MarkedMetricParameters marked_metric_params;
                AlignedMetricGenerator aligned_metric_generator(
                    V,
                    F,
                    feature_edges,
                    hard_feature_edges,
                    reference_field,
                    theta,
                    kappa,
                    period_jump,
                    marked_metric_params);
                aligned_metric_generator.optimize_full(alg_params);
                aligned_metric_generator.optimize_relaxed(alg_params);
                aligned_metric_generator.parameterize(false);
                auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r] = aligned_metric_generator.get_parameterization();
                view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "refined mesh", false);

                std::string output_filename = join_path(output_dir, mesh+"_param.obj");
                write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);
            }
            if (ImGui::Button("write field")) {
                field_generator.get_field(marked_metric, vtx_reindex, F_cut, face_reindex, reference_corner, theta, kappa, period_jump);
                std::string field_filename = join_path(output_dir, mesh+".ffield");
                Penner::Field::write_frame_field(field_filename, reference_field, theta, kappa, period_jump);
            }
        };
        polyscope::state::userCallback = callback;
        polyscope::show();
        field_generator.get_field(marked_metric, vtx_reindex, F_cut, face_reindex, reference_corner, theta, kappa, period_jump);
    }

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
