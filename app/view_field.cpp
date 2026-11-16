#include "field/frame_field.h"
#include "field/vector_field.h"
#include "field/viewer.h"
#include "util/io.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <CLI/CLI.hpp>

#include <fstream>
#include <iomanip>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"
#endif

using namespace Penner;
using namespace Penner::Field;

// write a per-face direction field to file, one "<dx> <dy> <dz>" line per face
void write_direction_field(const std::string& output_filename, const Eigen::MatrixXd& direction_field)
{
    std::ofstream output_file(output_filename, std::ios::out | std::ios::trunc);
    if (!output_file) {
        spdlog::error("Could not open {} for writing", output_filename);
        return;
    }
    for (int f = 0; f < direction_field.rows(); ++f) {
        for (int i : {0, 1, 2}) {
            output_file << std::fixed << std::setprecision(17) << direction_field(f, i);
            output_file << ((i < 2) ? " " : "\n");
        }
    }
    output_file.close();
    spdlog::info("Wrote {} face directions to {}", direction_field.rows(), output_filename);
}

int main(int argc, char* argv[])
{
    // Get command line arguments
    CLI::App app{"View a frame field"};
    std::filesystem::path mesh_filename = "";

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->required();
    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(spdlog::level::debug);

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;

    std::string file_format = mesh_filename.extension();
    std::string name = mesh_filename.stem();
    std::string input_dir = mesh_filename.parent_path();
    if (file_format == ".obj")
    {
        spdlog::info("Using mesh at {}", mesh_filename);
        igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

        // get input field field
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

    // add buttons to save one representative direction of the frame field per face and the
    // field cone positions (as a vertex-only obj)
    std::string direction_filename = join_path(input_dir, name + ".field");
    std::string cone_filename = join_path(input_dir, name + "_cones.obj");
    std::string cone_index_filename = join_path(input_dir, name + "_cone_indices.obj");
#ifdef ENABLE_VISUALIZATION
    polyscope::state::userCallback = [&]() {
        if (ImGui::Button("Save face directions")) {
            Eigen::MatrixXd direction_field = generate_vector_field(V, F, reference_field, theta);
            write_direction_field(direction_filename, direction_field);
        }
        if (ImGui::Button("Save field cones")) {
            std::vector<Scalar> Th_hat = compute_frame_field_cones(V, F, kappa, period_jump);
            auto [cone_positions, cone_values] = generate_cone_vertices(V, Th_hat);
            igl::writeOBJ(cone_filename, cone_positions, Eigen::MatrixXi(0, 3));
            spdlog::info("Wrote {} cones to {}", cone_positions.rows(), cone_filename);

            // write cone angle defects as the first coordinate of padded vertices
            Eigen::MatrixXd cone_values_padded = Eigen::MatrixXd::Zero(cone_values.size(), 3);
            cone_values_padded.col(0) = cone_values;
            igl::writeOBJ(cone_index_filename, cone_values_padded, Eigen::MatrixXi(0, 3));
            spdlog::info("Wrote {} cone indices to {}", cone_values.size(), cone_index_filename);
        }
    };
#endif

    view_cross_field(V, F, reference_field, theta, kappa, period_jump);
}
