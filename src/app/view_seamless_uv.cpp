#include "holonomy/core/common.h"
#include "holonomy/core/viewer.h"
#include "feature/feature/error.h"
#include "feature/core/io.h"
#include "feature/surgery/cut_mesh_layout.h"
#include "feature/interface.h"

#include "util/vf_mesh.h"

#include <igl/readOBJ.h>
#include <igl/remove_unreferenced.h>
#include <CLI/CLI.hpp>
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

using namespace Penner;
using namespace Penner::Holonomy;
using namespace Penner::Feature;



std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_cone_vertices(
    const Eigen::MatrixXd& V,
    const std::vector<Scalar>& Th_hat)
{
    int num_vertices = Th_hat.size();
    std::vector<int> cone_indices;
    cone_indices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (float_equal(Th_hat[vi], 2 * M_PI, 1e-8)) continue;
        cone_indices.push_back(vi);
    }

    int num_cones = cone_indices.size();
    Eigen::MatrixXd cone_positions(num_cones, 3);
    Eigen::VectorXd cone_values(num_cones);
    for (int i = 0; i < num_cones; ++i) {
        int vi = cone_indices[i];
        cone_positions.row(i) = V.row(vi);
        cone_values[i] = (double)(Th_hat[vi]) - (2 * M_PI);
    }

    return std::make_tuple(cone_positions, cone_values);
}

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

    std::vector<VertexEdge> E = load_mesh_edges(mesh_filename);

    Eigen::MatrixXi F_is_seam = find_seams(F, FT);
    auto [V_seams, E_seams] = generate_edges(V, F, F_is_seam);

    std::vector<double> feature_alignment = compute_feature_alignment(F, uv, FT, E);
    spdlog::info("Maximum feature alignment: {}", vector_max(feature_alignment));
    view_seamless_parameterization(V, F, uv, FT, "seamless mesh", false);
    //VectorX cone_angles = compute_cone_angles(V, F, uv, FT);
    //std::vector<Scalar> cone_angles_vec;
    //convert_eigen_to_std_vector(cone_angles, cone_angles_vec);

    //auto [uv_n, FT_n] = generate_connected_parameterization<Scalar>(V, F, uv, FT);
    //view_seamless_parameterization(V, F, uv_n, FT_n, "connected seamless mesh", false);

    //std::vector<double> feature_alignment = compute_feature_alignment(F, uv_n, FT_n, E);
    //spdlog::info("Maximum feature alignment: {}", vector_max(feature_alignment));

    //auto [cone_positions, cone_values] = generate_cone_vertices(V, cone_angles_vec);
    bool show_layout = false;
    if (show_layout) polyscope::registerSurfaceMesh2D("layout", uv, FT);
    //polyscope::getSurfaceMesh("base mesh")
    //    ->addVertexScalarQuantity("cone angles", cone_angles);
    //polyscope::registerPointCloud("cones", cone_positions);
    //polyscope::getPointCloud("cones")
    //    ->addScalarQuantity("index", cone_values)
    //    ->setColorMap("coolwarm")
    //    ->setMapRange({-M_PI, M_PI})
    //    ->setEnabled(true);
    Eigen::MatrixXd NV;
    Eigen::MatrixXi NE;
    Eigen::VectorXi I;
    //igl::remove_unreferenced(V, E, NV, NV, I);
    polyscope::registerCurveNetwork("features", V, E);
    bool show_seams = false;
    if (show_seams) polyscope::registerCurveNetwork("seams", V_seams, E_seams);

    polyscope::show();

}
