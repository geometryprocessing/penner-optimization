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
#include "field/frame_field.h"
#include "field/cross_field.h"
#include "field/intrinsic_field.h"
#include "holonomy/interface.h"
#include "feature/interface.h"
#include "holonomy/holonomy/newton.h"
#include "field/cones.h"
#include "holonomy/core/viewer.h"
#include "parametrization/refinement.h"
#include "parametrization/parametrize.h"

// cone validation
#include "holonomy/holonomy/constraint.h"

// infer theta from direction field
#include "field/vector_field.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <CLI/CLI.hpp>
#include <filesystem>

#if USE_UV_OPTIMIZATION 
#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "main_helper.h"
#endif

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Holonomy;
using namespace Feature;


int main(int argc, char* argv[])
{

    // Get command line arguments
    CLI::App app{"Generate a constrained seamless parametrization."};
    std::filesystem::path mesh_filename = "";

    // IO Parameters
    app.add_option("--mesh", mesh_filename, "Mesh filepath")->check(CLI::ExistingFile)->required();
    FieldParameters field_params;
    MarkedMetricParameters marked_metric_params;
    NewtonParameters alg_params;

    CLI11_PARSE(app, argc, argv);
    std::string mesh = mesh_filename.stem();

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("Using mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    //parametrize_seamless(V, F, field_params, alg_params);
    auto field_data = generate_feature_aligned_frame_field(V, F, field_params);
    auto [V_ref, F_ref, feature_edges, hard_feature_edges, reference_field, theta, kappa, period_jump] = field_data;
    generate_feature_aligned_parameterization(
        V_ref,
        F_ref,
        feature_edges,
        hard_feature_edges,
        reference_field,
        theta,
        kappa,
        period_jump,
        alg_params);
}
