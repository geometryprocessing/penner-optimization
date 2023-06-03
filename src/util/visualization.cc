#include "visualization.hh"

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
