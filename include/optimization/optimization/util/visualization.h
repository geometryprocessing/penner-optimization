// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <string>

#ifdef RENDER_TEXTURE
//#include <igl/png/writePNG.h> deprecated
#endif


namespace Penner {
namespace Optimization {

typedef igl::opengl::glfw::Viewer Viewer;

Viewer generate_mesh_viewer(const Eigen::MatrixXd &v,
                            const Eigen::MatrixXi &f,
                            bool show_lines);

void view_mesh(Viewer viewer);

void add_normals_to_mesh(Viewer &viewer,
                         const Eigen::MatrixXd &n);

void add_shading_to_mesh(Viewer &viewer,
                         const Eigen::MatrixXd &color,
                         const Eigen::VectorXd &AO,
                         double lighting_factor);

[[deprecated]]
void save_mesh_screen_capture(Viewer &viewer,
                              std::string image_path,
                              int width,
                              int height);


} // namespace Optimization
} // namespace Penner