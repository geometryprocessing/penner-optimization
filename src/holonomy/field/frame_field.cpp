#include "holonomy/field/frame_field.h"
#include "holonomy/holonomy/constraint.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/holonomy/holonomy.h"
#include "optimization/parameterization/refinement.h"
#include "holonomy/core/viewer.h"

#include <igl/per_face_normals.h>
#include <igl/local_basis.h>
#include <igl/rotate_vectors.h>
#include <igl/internal_angles.h>
#include <igl/comb_cross_field.h>
#include <igl/cross_field_mismatch.h>
#include <igl/find_cross_field_singularities.h>

// TODO: Cleaning pass

namespace Penner {
namespace Holonomy {

Eigen::Vector3d generate_reference_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int fijk,
    int local_index)
{
    int i = local_index;
    int j = (i + 1) % 3;
    int k = (j + 1) % 3;
    int vj = F(fijk, j);
    int vk = F(fijk, k);
    return (V.row(vk) - V.row(vj)).normalized();
}

Eigen::MatrixXd generate_reference_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    int num_faces = F.rows();
    int local_index = 1;
    Eigen::MatrixXd reference_field(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        reference_field.row(fijk) = generate_reference_direction(V, F, fijk, local_index);
    }

    return reference_field;
}

Eigen::MatrixXd generate_reference_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& reference_corner)
{
    int num_faces = F.rows();
    Eigen::MatrixXd reference_field(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        int local_index = reference_corner[fijk];
        reference_field.row(fijk) = generate_reference_direction(V, F, fijk, local_index);
    }

    return reference_field;
}

Eigen::MatrixXd generate_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta)
{
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V, F, B1, B2, B3);
    return igl::rotate_vectors(reference_field, theta, B1, B2);
}

void write_frame_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump)
{
    std::ofstream field_file(output_filename, std::ios::out | std::ios::trunc);

    // write all feature edge vertices
    int num_faces = theta.size();
    field_file << num_faces << std::endl;
    for (int f = 0; f < reference_field.rows(); ++f)
    {
        // write reference direction
        for (int i : {0 , 1, 2})
        {
            field_file << std::fixed << std::setprecision(17) << reference_field(f, i) << " ";
        }

        // write face rotation angle
        field_file << std::fixed << std::setprecision(17) << theta[f] << " ";

        // write corner kappas
        for (int i : {0 , 1, 2})
        {
            field_file << std::fixed << std::setprecision(17) << kappa(f, i) << " ";
        }

        // write corner period jumps
        for (int i : {0 , 1, 2})
        {
            field_file << period_jump(f, i) << " ";
        }
        field_file << std::endl;
    }

    // close output file
    field_file.close();
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi>
load_frame_field(const std::string& filename)
{
    // Open file
    spdlog::debug("opening field at {}", filename);
    std::ifstream input_file(filename);
    if (!input_file) return {{}, {}, {}, {}};

    // get number of faces
    std::string line;
    std::getline(input_file, line);
    std::istringstream iss(line);
    int num_faces;
    iss >> num_faces;
    spdlog::debug("{} faces", num_faces);

    // initialize vectors
    Eigen::MatrixXd reference_field(num_faces, 3);
    Eigen::VectorXd theta(num_faces);
    Eigen::MatrixXd kappa(num_faces, 3);
    Eigen::MatrixXi period_jump(num_faces, 3);

    // Read file one face at a time
    int f = 0;
    while ((f < num_faces) && (std::getline(input_file, line))) {
        std::istringstream iss(line);
        for (int i : {0 , 1, 2})
        {
            iss >> reference_field(f, i);
        }
        iss >> theta(f);
        for (int i : {0 , 1, 2})
        {
            iss >> kappa(f, i);
        }
        for (int i : {0 , 1, 2})
        {
            iss >> period_jump(f, i);
        }

        ++f;
    }

    if (num_faces != f)
    {
        spdlog::error("Number of faces inconsistent with number of lines");
    }

    // Close file
    input_file.close();
    return std::make_tuple(reference_field, theta, kappa, period_jump);
}

std::vector<Scalar> compute_cone_angle( 
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump)
{
    // Compute the corner angles
    Eigen::MatrixXd angles;
    igl::internal_angles(V, F, angles);

    int num_vertices = V.rows();
    int num_faces = F.rows();
    std::vector<Scalar> Th_hat(num_vertices, 0);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int vi = F(fijk, i);
            Th_hat[vi] += angles(fijk, i);
            Th_hat[vi] += kappa(fijk, j);
            Th_hat[vi] += (M_PI / 2.) * period_jump(fijk, j);
        }
    }

    return Th_hat;
}

Eigen::VectorXd infer_theta(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& reference_corner,
    const Eigen::MatrixXd& direction_field)
{
    Eigen::MatrixXd N;
    igl::per_face_normals(V, F, N);

    Eigen::MatrixXd reference_field = generate_reference_field(V, F, reference_corner);
    int num_faces = F.rows();
    Eigen::VectorXd theta(num_faces);
    for (int f = 0; f < num_faces; ++f)
    {
        // TODO: Check if can replace sign by reversing order, i.e., the signed angle is anticommutative
        theta[f] = -signed_angle<Eigen::Vector3d>(direction_field.row(f), reference_field.row(f), N.row(f));
    }

    return theta;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi> refine_frame_field(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::MatrixXi& F_base,
    const Eigen::MatrixXd& reference_vector,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump)
{
    // build subface structure
    // TODO: make version that does not require uv map
    Eigen::MatrixXi F_orig, F_uv_orig, halfedge_map;
    std::vector<std::array<std::vector<int>, 3>> corner_v_points, corner_uv_points;
    std::vector<std::vector<int>> F_to_Fn = Optimization::build_F_to_Fn(Fn_to_F);
    Optimization::build_faces(
        F,
        F_uv,
        F_to_Fn,
        endpoints,
        F_orig,
        corner_v_points,
        F_uv_orig,
        corner_uv_points,
        halfedge_map);

    // build refined field
    int num_faces = F.rows(); 
    Eigen::MatrixXd reference_vector_r(num_faces, 3);
    Eigen::VectorXd theta_r(num_faces);
    Eigen::MatrixXd kappa_r(num_faces, 3);
    Eigen::MatrixXi period_jump_r(num_faces, 3);
    for (int fi = 0; fi < num_faces; ++fi)
    {
        // get direction and offset from original face
        int f_orig = Fn_to_F[fi];
        reference_vector_r.row(fi) = reference_vector.row(f_orig);
        theta_r[fi] = theta[f_orig];

        // find offset to match reconstructed base mesh indexing with input base mesh faces
        int offset = 0;
        while (F_base(f_orig, offset) != F_orig(f_orig, 0))
        {
            offset++;
            if (offset >= 3)
            {
                spdlog::error("could not match offset");
                break;
            }
        }

        for (int j = 0; j < 3; ++j)
        {
            // set period jump and field rotation to zero for interior edges
            if (halfedge_map(fi, j) < 0)
            {
                kappa_r(fi, j) = 0.;
                period_jump_r(fi, j) = 0;
                continue;
            }

            // copy base mesh period jump and field rotation for refined edges of base mesh triangles
            int j_orig = (halfedge_map(fi, j) + offset) % 3; // update halfedge corner data with offset
            kappa_r(fi, j) = kappa(f_orig, j_orig);
            period_jump_r(fi, j) = period_jump(f_orig, j_orig);
        }
    }

    return std::make_tuple(reference_vector_r, theta_r, kappa_r, period_jump_r);
}

} // namespace Holonomy
} // namespace Penner