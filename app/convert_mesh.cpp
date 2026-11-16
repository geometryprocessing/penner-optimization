
#include "util/io.h"

#include <igl/doublearea.h>

#include <CLI/CLI.hpp>

using namespace Penner;

int main(int argc, char *argv[])
{
#ifdef WITH_MPFR
    spdlog::info("Using multiprecision");
//    mpfr::mpreal::set_default_prec(100);
//    mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
//    mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

    // Get command line arguments
    CLI::App app{"Generate a regularized parametrization with the modified Newton's method."};
    std::filesystem::path mesh_filename = "";
    std::filesystem::path output_dir = "./";
    Scalar ratio = 0.;

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->required();
    app.add_option("--output", output_dir, "Output directory");
    app.add_option("--scale", ratio, "Larger corresponds to finer mesh");
    CLI11_PARSE(app, argc, argv);

    // Miscellaneous
    spdlog::level::level_enum log_level = spdlog::level::info;
    spdlog::set_level(log_level);

    // separate file paths and types
    std::string file_format = mesh_filename.extension();
    std::string name = mesh_filename.stem();

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("Using mesh at {}", mesh_filename);
    read_mesh_with_uv(mesh_filename, V, F, uv, FT);

    // compute target scale
    if (ratio > 0.)
    {
        Eigen::VectorXd double_area;
        igl::doublearea(V, F, double_area);
        Scalar mesh_area = double_area.sum() / 2.;
        int num_vertices = V.rows();
        Scalar target_length = sqrt((ratio * num_vertices) / mesh_area);
        uv *= target_length;
    }

    // Write the output mesh
    std::string output_filename = join_path(output_dir, name + "_scaled.obj");
    write_obj_with_uv(output_filename, V, F, uv, FT);
}
