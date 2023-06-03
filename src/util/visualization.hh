#ifndef VISUALIZATION_HH
#define VISUALIZATION_HH
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>
#include <string>

namespace CurvatureMetric
{

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

void save_mesh_screen_capture(Viewer &viewer,
                              std::string image_path,
                              int width,
                              int height);

}
#endif
