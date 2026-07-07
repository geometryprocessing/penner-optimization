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
#include "field/field.h"
#include "holonomy/interface.h"
#include "holonomy/holonomy/cones.h"
#include "util/io.h"

#include <igl/boundary_facets.h>
#include <igl/facet_components.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>

#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/writeOBJ.h>

#include <CLI/CLI.hpp>
#include <filesystem>

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Holonomy;

void write_error_record(const std::filesystem::path& filepath, const std::string& error_message)
{
    std::ofstream output_file(filepath, std::ios::out | std::ios::app);
    output_file << error_message << std::endl;
    output_file.close();
}

int main(int argc, char* argv[])
{
    // get command line arguments
    CLI::App app{"Generate smooth cross field data for a mesh."};
    std::string mesh_file = "";
    std::string input_dir = "./";
    std::string output_dir = "./";
    bool is_stl = false;
    app.add_option("--mesh", mesh_file, "Mesh file")->required();
    app.add_option("-i,--input", input_dir, "Input filepath")
        ->check(CLI::ExistingDirectory)
        ->required();
    app.add_option("-o,--output", output_dir, "Output directory")->check(CLI::ExistingDirectory);
    app.add_flag("--is_stl", is_stl, "Input is an stl mesh");
    CLI11_PARSE(app, argc, argv);
    spdlog::set_level(spdlog::level::debug);

    // get mesh name from file
    std::string mesh_name = std::filesystem::path(mesh_file).replace_extension();

    // get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    std::string mesh_filename = join_path(input_dir, mesh_file);
    spdlog::info("Using mesh at {}", mesh_filename);
    if (is_stl) {
        std::ifstream input_stream(mesh_filename);
        Eigen::MatrixXd V_stl, N_stl;
        Eigen::MatrixXi F_stl;
        Eigen::VectorXi I, J;
        igl::readSTL(input_stream, V_stl, F_stl, N_stl);
        igl::remove_duplicate_vertices(V_stl, F_stl, 1e-14, V, I, J, F);
        input_stream.close();
    } else {
        igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);
    }

    // get mesh topology information
    Eigen::MatrixXd bd;
    Eigen::VectorXi components;
    igl::boundary_facets(F, bd);
    igl::facet_components(F, components);

    // check validity
    std::string error_filename = join_path(output_dir, "error_mesh_list");
    if (bd.size() > 1) {
        spdlog::warn("Mesh has boundary");
        write_error_record(error_filename, mesh_name + " has boundary");
    }
    if (components.maxCoeff() != 0) {
        spdlog::error("Mesh has multiple components");
        write_error_record(error_filename, mesh_name + " has multiple components");
        return 1;
    }
    if (!igl::is_edge_manifold(F)) {
        spdlog::error("Mesh is not edge manifold");
        write_error_record(error_filename, mesh_name + " is not edge manifold");
        return 1;
    }
    Eigen::VectorXi B;
    if (!igl::is_vertex_manifold(F, B)) {
        spdlog::error("Mesh is not vertex manifold");
        write_error_record(error_filename, mesh_name + " is not vertex manifold");
        return 1;
    }

    // generate cross field
    auto [frame_field, field_Th_hat] = generate_cross_field(V, F);

    // build halfedge mesh with angles
    std::vector<int> vtx_reindex;
    std::vector<int> free_cones(0);
    bool fix_boundary = false;
    bool use_discrete_metric = true;
    std::unique_ptr<DifferentiableConeMetric> cone_metric =
        Optimization::generate_initial_mesh(
            V,
            F,
            V,
            F,
            field_Th_hat,
            vtx_reindex,
            free_cones,
            fix_boundary,
            use_discrete_metric);

    // check for zero edge lengths and degenerate angles
    VectorX he2angle, he2cot;
    cone_metric->get_corner_angles(he2angle, he2cot);
    int num_halfedges = cone_metric->n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (float_equal(cone_metric->l[hij], 0.)) {
            spdlog::error("Mesh has {} length edge", cone_metric->l[hij]);
            write_error_record(error_filename, mesh_name + " has 0 length edge");
            return 1;
        }
        if (float_equal(he2angle[hij], 0.)) {
            spdlog::error("Mesh has {} angle", he2angle[hij]);
            write_error_record(error_filename, mesh_name + " has 0 corner angle");
            return 1;
        }
        if (float_equal(he2angle[hij], M_PI)) {
            spdlog::error("Mesh has {} angle", cone_metric->l[hij]);
            write_error_record(error_filename, mesh_name + " has pi corner angle");
            return 1;
        }
    }

    // generate rotation form and cones from cross field
    VectorX rotation_form = generate_rotation_form_from_cross_field(*cone_metric, vtx_reindex, V, F, frame_field);
    std::vector<Scalar> form_Th_hat = generate_cones_from_rotation_form(
            *cone_metric,
            vtx_reindex,
            rotation_form,
            bd.size() > 1);

    // check cones are consistent
    int num_vertices = V.rows();
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (!float_equal(form_Th_hat[vi], field_Th_hat[vi], 1e-6)) {
            spdlog::warn("Inconsistent cones {} and {}", form_Th_hat[vi], field_Th_hat[vi]);
            write_error_record(error_filename, mesh_name + " has inconsistent cones");
        }
    }

    // write output
    igl::writeOBJ(join_path(output_dir, mesh_name + ".obj"), V, F);
    write_vector(rotation_form, join_path(output_dir, mesh_name + "_kappa_hat"));
    write_vector(form_Th_hat, join_path(output_dir, mesh_name + "_Th_hat"));

    return 0;
}
