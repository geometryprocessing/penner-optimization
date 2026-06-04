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
