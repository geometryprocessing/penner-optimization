/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
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
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include "common.hh"
#include "explicit_optimization.hh"
#include "io.hh"
#include "penner_optimization_interface.hh"
#include "shear.hh"
#include "viewers.hh"
#include "refinement.hh"
#include <CLI/CLI.hpp>

using namespace CurvatureMetric;

int main(int argc, char* argv[])
{
#ifdef MULTIPRECISION
    spdlog::info("Using multiprecision");
    mpfr::mpreal::set_default_prec(60);
    mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
    mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
#endif

    // Build maps from strings to enums
    std::map<std::string, EnergyChoice> energy_choice_map{
        {"log_length", EnergyChoice::log_length},
        {"log_scale", EnergyChoice::log_scale},
        {"quadratic_sym_dirichlet", EnergyChoice::quadratic_sym_dirichlet},
        {"sym_dirichlet", EnergyChoice::sym_dirichlet},
        {"p_norm", EnergyChoice::p_norm},
    };

    // Get command line arguments
    CLI::App app{"Generate approximately isometric parameterization for a mesh."};
    std::string mesh_filename = "";
    std::string Th_hat_filename = "";
    std::string output_dir = "./";
		EnergyChoice energy_choice = EnergyChoice::log_length;
    bool show_parameterization = false;
    auto proj_params = std::make_shared<ProjectionParameters>();
    auto opt_params = std::make_shared<OptimizationParameters>();
    opt_params->direction_choice = "gradient";
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->check(CLI::ExistingFile)->required();
    app.add_option("--cones", Th_hat_filename, "Cone angle filepath")
        ->check(CLI::ExistingFile)
        ->required();
    app.add_option("--energy", energy_choice, "Energy to minimize")
        ->transform(CLI::CheckedTransformer(energy_choice_map, CLI::ignore_case));
    app.add_option("--direction", opt_params->direction_choice, "Descent direction: gradient, conjugate_gradient, lbfgs");
    app.add_option(
           "--num_iter",
           opt_params->num_iter,
           "Maximum number of iterations to perform")
        ->check(CLI::NonNegativeNumber);
    app.add_flag("--show_parameterization", show_parameterization, "Show final parameterization");
    app.add_option("-o,--output", output_dir, "Output directory");
    CLI11_PARSE(app, argc, argv);

    // Make output directory
    spdlog::set_level(spdlog::level::info);
    std::filesystem::create_directories(output_dir);
    opt_params->output_dir = output_dir;

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("Optimizing mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    // Get input angles
    std::vector<Scalar> Th_hat_init;
    spdlog::info("Using cone angles at {}", Th_hat_filename);
    read_vector_from_file(Th_hat_filename, Th_hat_init);
    std::vector<Scalar> Th_hat = correct_cone_angles(Th_hat_init);

    // Get initial mesh for optimization
    std::vector<int> vtx_reindex;
    std::vector<int> free_cones = {};
    bool fix_boundary = false;
    bool use_discrete_metric = false;
    std::unique_ptr<DifferentiableConeMetric> cone_metric =
        generate_initial_mesh(V, F, V, F, Th_hat, vtx_reindex, free_cones, fix_boundary, use_discrete_metric);

    // Get energy
    std::unique_ptr<EnergyFunctor> opt_energy = generate_energy(V, F, Th_hat, *cone_metric, energy_choice);

    // Compute shear dual basis and the corresponding inner product matrix
    MatrixX shear_basis_matrix;
    std::vector<int> independent_edges;
    compute_shear_dual_basis(*cone_metric, shear_basis_matrix, independent_edges);

    // Compute the shear dual coordinates for this basis
    VectorX shear_basis_coords_init;
    VectorX scale_factors_init;
    compute_shear_basis_coordinates(
        *cone_metric,
        shear_basis_matrix,
        shear_basis_coords_init,
        scale_factors_init);

    // Optimize the metric
    VectorX optimized_metric_coords = optimize_shear_basis_coordinates(
        *cone_metric,
        *opt_energy,
        shear_basis_matrix,
        proj_params,
        opt_params);

    // Write the metric coordinate output
    std::string output_filename = join_path(output_dir, "reduced_metric_coords");
    write_vector(optimized_metric_coords, output_filename);

    // Generate overlay mesh
    std::vector<bool> is_cut = {};
    bool do_best_fit_scaling = false;
    auto vf_res = generate_VF_mesh_from_metric(
            V,
            F,
            Th_hat,
            *cone_metric,
            optimized_metric_coords,
            is_cut,
            do_best_fit_scaling);
    OverlayMesh<Scalar> m_o = std::get<0>(vf_res);
    Eigen::MatrixXd V_o = std::get<1>(vf_res);
    Eigen::MatrixXi F_o = std::get<2>(vf_res);
    Eigen::MatrixXd uv_o = std::get<3>(vf_res);
    Eigen::MatrixXi FT_o = std::get<4>(vf_res);
    std::vector<int> fn_to_f_o = std::get<7>(vf_res);
    std::vector<std::pair<int, int>> endpoints_o = std::get<8>(vf_res);

    // Write the overlay output
    output_filename = join_path(output_dir, "overlay_mesh_with_uv.obj");
    write_obj_with_uv(output_filename, V_o, F_o, uv_o, FT_o);

    // Get refinement mesh
    Eigen::MatrixXd V_r;
    Eigen::MatrixXi F_r;
    Eigen::MatrixXd uv_r;
    Eigen::MatrixXi FT_r;
    std::vector<int> fn_to_f_r;
    std::vector<std::pair<int, int>> endpoints_r;
    RefinementMesh refinement_mesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o);
    refinement_mesh.get_VF_mesh(V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r);

    // Write the refined output
    output_filename = join_path(output_dir, "refined_mesh_with_uv.obj");
    write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);

    // Optionally show final parameterization
    if (show_parameterization) view_parameterization(V_r, F_r, uv_r, FT_r);
}
