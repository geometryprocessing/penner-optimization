#include "holonomy/core/common.h"
#include "feature/core/viewer.h"
#include "feature/core/quads.h"

#include <igl/readOBJ.h>
#include <CLI/CLI.hpp>

using namespace Penner;
using namespace Penner::Feature;

int main(int argc, char* argv[])
{
    // Get command line arguments
    CLI::App app{"View a quad mesh"};
    std::string mesh_filename = "";

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->check(CLI::ExistingFile)->required();
    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(spdlog::level::debug);

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("Using mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    view_quad_mesh(V, F);
}
