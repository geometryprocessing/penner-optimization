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
#include <igl/bounding_box_diagonal.h>

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

Eigen::MatrixXd generate_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta)
{
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V, F, B1, B2, B3);
    return igl::rotate_vectors(reference_field, theta, B1, B2);
}

// difference of principal curvatures relative to their total magnitude
Scalar compute_relative_anisotropy(Scalar max_val, Scalar min_val)
{
    return abs(max_val - min_val) / (abs(max_val) + abs(min_val));
}


// absolute difference of principal curvatures
Scalar compute_absolute_anisotropy(Scalar max_val, Scalar min_val)
{
    return abs(max_val - min_val);
}


// mean of two principal curvatures
Scalar compute_mean_anisotropy(Scalar max_val, Scalar min_val)
{
    return (max_val + min_val) / 2.;
}


// this measurement is near 0 for parabolic regions and near 1 for highly anisotropic regions
Scalar compute_parabolic_anisotropy(Scalar max_val, Scalar min_val)
{
    return abs(abs(max_val) - abs(min_val)) / max(abs(max_val), abs(min_val));
}

std::tuple<Eigen::MatrixXd, std::vector<bool>> compute_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius,
    Scalar abs_threshold,
    Scalar rel_threshold)
{
    //auto[max_direction, min_direction, _max_curvature, _min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, radius);
    //auto[_max_direction, _min_direction, max_curvature, min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, 3);
    auto[max_direction, min_direction, max_curvature, min_curvature] = Holonomy::compute_facet_principal_curvature(V, F, radius);
    int num_faces = F.rows();
    std::vector<bool> is_fixed_direction(num_faces, false);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        Scalar kmax = max_curvature[fijk];
        Scalar kmin = min_curvature[fijk];
        if (compute_mean_anisotropy(kmax, kmin) < abs_threshold) continue;
        is_fixed_direction[fijk] = (compute_parabolic_anisotropy(kmax, kmin) > rel_threshold);
    }

    return std::make_tuple(max_direction, is_fixed_direction);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi>
generate_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const FieldParameters& field_params)
{
    // Convert VF mesh to halfedge
    std::vector<Scalar> flat_Th_hat(V.rows(), 2. * M_PI);
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    std::vector<int> free_cones(0);
    bool fix_boundary = false;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        V,
        F,
        flat_Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);
    std::vector<int> face_map = generate_face_map(m);

    // initalize field generator
    Holonomy::IntrinsicNRosyField field_generator;
    field_generator.min_cone = field_params.min_cone;
    field_generator.use_roundings= field_params.use_roundings;
    field_generator.initialize(m);

    // set initial field
    int num_faces = F.rows();
    Eigen::VectorXi reference_corner(num_faces);
    Eigen::VectorXd theta(num_faces);
    Eigen::MatrixXd kappa(num_faces, 3);
    Eigen::MatrixXi period_jump(num_faces, 3);

    // (optionally) fit principal directions
    if (field_params.use_principal_directions)
    {
        field_generator.get_field(m, vtx_reindex, F, face_map, reference_corner, theta, kappa, period_jump);

        // fit field directions
        int radius = 5;
        Scalar bb_diag = igl::bounding_box_diagonal(V);
        auto [direction, is_fixed_direction] = compute_field_direction(
            V,
            F,
            radius,
            field_params.abs_anisotropy / bb_diag,
            field_params.rel_anisotropy);

        // compute normals for angle computations
        Eigen::MatrixXd N;
        igl::per_face_normals(V, F, N);

        // get fixed angles from input direction field
        std::vector<Scalar> target_theta(num_faces);
        std::vector<bool> is_fixed(num_faces);
        for (int i = 0; i < num_faces; ++i)
        {
            int fijk = face_map[i];
            is_fixed[i] = is_fixed_direction[fijk];

            // convert extrinsic direction to intrinsic angle
            Eigen::Vector3d reference_direction = Holonomy::generate_reference_direction(V, F, fijk, reference_corner[fijk]);
            target_theta[i] = -Holonomy::signed_angle<Eigen::Vector3d>(direction.row(fijk), reference_direction, N.row(fijk));
        }

        // set fixed directions 
        field_generator.set_fixed_directions(m, target_theta, is_fixed);
    }

    // find initial field
    field_generator.solve(m);

    // change period jups to fix invalid configuration
    if (field_params.fix_cone_pair) field_generator.fix_cone_pair(m);
    if (field_params.fix_cone_pair) field_generator.fix_zero_cones(m);

    // optionally remove nearby cones
    if (field_params.min_cone_pair_distance > 0.) field_generator.remove_close_cone_pairs(m, field_params.min_cone_pair_distance);

    // optionally collapse all curvature
    if (field_params.collapse_cones) field_generator.remove_greedy_cone_pairs(m);
    if (field_params.collapse_cones) field_generator.concentrate_curvature(m);

    // get field from generator
    field_generator.get_field(m, vtx_reindex, F, face_map, reference_corner, theta, kappa, period_jump);

    // convert face corner to tangent direction
    Eigen::MatrixXd reference_field = Holonomy::generate_reference_field(V, F, reference_corner);

    return std::make_tuple(reference_field, theta, kappa, period_jump);
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