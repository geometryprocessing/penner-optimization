#include "holonomy/core/viewer.h"
#include "feature/feature/error.h"
#include "util/io.h"
#include "feature/core/io.h"
#include "util/vector.h"
#include "util/vf_mesh.h"

#include <igl/readOBJ.h>
#include <CLI/CLI.hpp>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#endif

using namespace Penner;
using namespace Penner::Holonomy;
using namespace Penner::Feature;

int main(int argc, char* argv[])
{
    // Get command line arguments
    CLI::App app{"View a quad mesh"};
    std::string mesh_filename = "";
    bool show_layout = false;
    bool show_seams = false;

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->check(CLI::ExistingFile)->required();
    app.add_flag("--show_layout", show_layout, "Show layout of parametrization");
    app.add_flag("--show_seams", show_seams, "Show seam edges of parametrization");
    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(spdlog::level::debug);

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("Using mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    // get tagged feature edges
    std::vector<VertexEdge> feature_edges = load_mesh_edges(mesh_filename);

    // infer seame edges from parametrization connectivity
    Eigen::MatrixXi F_is_seam = find_seams(F, FT);
    auto [V_seams, E_seams] = generate_edges(V, F, F_is_seam);

    // view seamless parametrization
    view_seamless_parameterization(V, F, uv, FT, "seamless mesh", false);

    // show feature edges if any are tagged
    if (feature_edges.size() > 0)
    {
        // compute feature alignment error
        std::vector<double> feature_alignment = compute_feature_alignment(F, uv, FT, feature_edges);
        spdlog::info("Maximum feature alignment: {}", vector_max(feature_alignment));

        // plot edges
        Eigen::MatrixXi E = compute_edge_matrix(feature_edges);
        auto [VN, EN] = remove_unreferenced(V, E);
        polyscope::registerCurveNetwork("features", VN, EN);
    }

    // optional viewers
    if (show_layout) polyscope::registerSurfaceMesh2D("layout", uv, FT);
    if (show_seams) polyscope::registerCurveNetwork("seams", V_seams, E_seams);

    polyscope::show();

}
