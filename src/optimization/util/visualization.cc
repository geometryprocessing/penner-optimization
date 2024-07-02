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
#include "visualization.h"

namespace CurvatureMetric
{


Viewer generate_mesh_viewer(const Eigen::MatrixXd &v,
                            const Eigen::MatrixXi &f,
                            bool show_lines)
{
    // Initialize viewer with mesh
    Viewer viewer;
    viewer.data().set_mesh(v, f);

    // Optionally show lines
    viewer.data().show_lines = show_lines;

    return viewer;
}

void view_mesh(Viewer viewer)
{
    // Launch widget
    viewer.launch();
}


void add_normals_to_mesh(Viewer &viewer,
                         const Eigen::MatrixXd &n)
{
    viewer.data().set_normals(n);
}

void add_shading_to_mesh(Viewer &viewer,
                         const Eigen::MatrixXd &color,
                         const Eigen::VectorXd &AO,
                         double lighting_factor)
{
    // Generate shading color map
    Eigen::MatrixXd C = color;
    for (unsigned i = 0; i < AO.rows(); ++i)
    {
        C.row(i) *= AO(i);
    }

    // Set viewer colors
    viewer.data().set_colors(C);

    // Set lighting factor
    viewer.core().lighting_factor = lighting_factor;
}

void save_mesh_screen_capture(Viewer &viewer,
                              std::string image_path,
                              int width,
                              int height)
{
    // Initialize windowless viewer
    glfwInit();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    viewer.launch_init(false, false, "", width, height);
    viewer.launch_rendering(false);

    // Allocate temporary buffers for 1280x800 image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

    // Draw the scene in the buffers
    viewer.core().draw_buffer(viewer.data(), true, R, G, B, A);

    // Save it to specified output path
    igl::png::writePNG(R, G, B, A, image_path);
}

}
