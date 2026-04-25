#include "holonomy/field/frame_field.h"
#include "holonomy/holonomy/constraint.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/holonomy/holonomy.h"
#include "optimization/parameterization/refinement.h"
#include "holonomy/core/viewer.h"
#include "util/vf_mesh.h"

#include <igl/per_face_normals.h>
#include <igl/local_basis.h>
#include <igl/rotate_vectors.h>
#include <igl/internal_angles.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/grad.h>
#include <igl/doublearea.h>

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

Eigen::VectorXd rotate_vector(
                    const Eigen::VectorXd& vec,
                    double angle,
                    const Eigen::VectorXd& B1,
                    const Eigen::VectorXd& B2)
{
    double norm = vec.norm();

    // project onto the tangent plane and convert to angle
    double a = atan2(B2.dot(vec), B1.dot(vec));

    // rotate
    a += angle;

    // move it back to global coordinates
    return norm*cos(a) * B1 + norm*sin(a) * B2;
}

std::tuple<int, int>
find_u_aligned_edges(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT)
{
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(FT, TT, TTi);
    Eigen::ArrayXX<bool> is_boundary_edge = (TT.array() < 0);

    double min_v_diff = 1e10;
    int min_face = -1;
    int min_edge = -1;
    for (int fijk = 0; fijk < FT.rows(); ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (!is_boundary_edge(fijk, i)) continue;

            int vi = FT(fijk, (i + 0) % 3);
            int vj = FT(fijk, (i + 1) % 3);
            double signed_u_diff = uv(vj, 0) - uv(vi, 0);
            if (signed_u_diff < 0.) continue;

            double v_diff = abs(uv(vj, 1) - uv(vi, 1));
            if (v_diff < min_v_diff) {
                min_v_diff = v_diff;
                min_face = fijk;
                min_edge = i;
            }
        }
    }

    return std::make_tuple(min_face, min_edge);
}

std::tuple<Eigen::VectorXi, std::deque<int>, Eigen::VectorXi> initialize_matchings(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& frame_field, 
    const Eigen::MatrixXd& B1,
    const Eigen::MatrixXd& B2)
{   
    Eigen::VectorXi matchings = Eigen::VectorXi::Constant(FT.rows(), 0);
    std::deque<int> d;
    Eigen::VectorXi mark = Eigen::VectorXi::Constant(FT.rows(), 0);

    auto [fijk, i] = find_u_aligned_edges(uv, FT);
    int vi = F(fijk, (i + 0) % 3);
    int vj = F(fijk, (i + 1) % 3);
    Eigen::VectorXd dij = V.row(vj) - V.row(vi);
    dij.normalize();
    double min = (frame_field.row(fijk) - dij.transpose()).norm();
    for (int i = 1; i < 4; ++i)
    {
        Eigen::VectorXd rot_dfijk = rotate_vector(frame_field.row(fijk), i * (PI / 2.0), B1.row(fijk), B2.row(fijk));
        double curr_diff = (rot_dfijk - dij).norm();
        if (curr_diff < min)
        {
            min = curr_diff;
            matchings[fijk] = i;
        }
    }
    d.push_back(fijk);
    mark[fijk] = 1;

    return { matchings, d, mark };
}

Eigen::SparseMatrix<double> compute_area_weight_matrix(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
)
{
    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);
    int num_faces = dblarea.size();
    Eigen::SparseMatrix<double> weights(3 * num_faces, 3 * num_faces);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * num_faces);
    double total_area = dblarea.sum();

    for (int f = 0; f < num_faces; ++f)
    {
        double area_f = dblarea[f] / total_area;
        for (int i = 0; i < 3; ++i)
        {
            int j = f + (i * num_faces);
            triplets.emplace_back(j, j, area_f);
        }
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());
    return weights;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
maximize_combed_frame_alignment(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2)
{
    // get local basis for rotations
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V, F, B1, B2, B3);

    // get gradient of uv coordinates
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, FT, V_cut);
    Eigen::SparseMatrix<double> Grad;
    igl::grad(V_cut, FT, Grad);
    Eigen::MatrixXd Guv = Grad * uv;

    // minimize misalignment
    Eigen::SparseMatrix<double> weights = compute_area_weight_matrix(V, F);
    Eigen::VectorXd right_angles = Eigen::VectorXd::Constant(F.rows(), PI / 2.);
    double min_energy = 1e10;
    int j = 0;
    for (int i = 0; i < 4; ++i)
    {
        // rotate field
        Eigen::MatrixXd RPD1 = igl::rotate_vectors(PD1, i * right_angles, B1, B2);
        Eigen::MatrixXd RPD2 = igl::rotate_vectors(PD2, i * right_angles, B1, B2);

        // compute energy for current rotation
        Eigen::MatrixXd uT_vT(3*F.rows(), 2);
        uT_vT.col(0) = Eigen::Map<const Eigen::VectorXd>(RPD1.data(), 3 * FT.rows());
        uT_vT.col(1) = Eigen::Map<const Eigen::VectorXd>(RPD2.data(), 3 * FT.rows());
        Eigen::MatrixXd R = Guv - uT_vT;
        double energy = (R.transpose() * (weights * R)).trace();
        if (energy < min_energy)
        {
            min_energy = energy;
            j = i;
        }
        spdlog::info("energy for rotation {} is {}", i * PI / 2., energy);
    }

    // rotate field with minimizing angle
    Eigen::MatrixXd RPD1 = igl::rotate_vectors(PD1, j * right_angles, B1, B2);
    Eigen::MatrixXd RPD2 = igl::rotate_vectors(PD2, j * right_angles, B1, B2);
    return { RPD1, RPD2 };
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
comb_frame_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& thetas,
    const Eigen::MatrixXi& period_jumps)
{
    // get local basis for rotations
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V, F, B1, B2, B3);

    // rotate initial reference field to a field direction
    Eigen::MatrixXd frame_field = igl::rotate_vectors(reference_field, thetas, B1, B2);

    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(FT, TT, TTi);

    auto [ matchings, d, mark ] = initialize_matchings(V, F, uv, FT, frame_field, B1, B2);
    while (!d.empty())
    {
        int fijk = d.at(0);
        d.pop_front();
        for (int local_hij = 0; local_hij < 3; ++local_hij)
        {
            int fjil = TT(fijk, (local_hij + 1) % 3);
            if ((fjil >= 0) && (mark[fjil] == 0))
            {
                matchings[fjil] = matchings[fijk] - period_jumps(fijk, local_hij);
                while (matchings[fjil] < 0)
                {
                    matchings[fjil] += 4;
                }
                matchings[fjil] = matchings[fjil] % 4;
                mark[fjil] = 1;
                d.push_back(fjil);
            }
        }
    }
    Eigen::VectorXd u_angles = (matchings.cast<double>().array()) * (PI / 2.0);
    Eigen::VectorXd v_angles = (matchings.cast<double>().array() + 1) * (PI / 2.0);
    Eigen::MatrixXd PD1 = igl::rotate_vectors(frame_field, u_angles, B1, B2);
    Eigen::MatrixXd PD2 = igl::rotate_vectors(frame_field, v_angles, B1, B2);

    return maximize_combed_frame_alignment(V, F, uv, FT, PD1, PD2);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
load_combed_field(const std::string& ffield_file)
{
    std::ifstream inf(ffield_file);
    if (!inf) {
        spdlog::error("Failed to load frame field file\n");
        exit(EXIT_FAILURE);
    }
    spdlog::info("aligning to frame field file\n");
    
    int i = 0;
    int num_vectors;
    inf >> num_vectors;
    Eigen::MatrixXd PD1(num_vectors, 3);
    Eigen::MatrixXd PD2(num_vectors, 3);
    inf.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string line{};
    while (std::getline(inf, line)) {
        std::istringstream iss(line);
        double d1x;
        double d1y;
        double d1z;
        double d2x;
        double d2y;
        double d2z;

        iss >> d1x >> d1y >> d1z >> d2x >> d2y >> d2z;
        PD1.row(i) << d1x, d1y, d1z;
        PD2.row(i) << d2x, d2y, d2z;

        ++i;
    }

    return {PD1, PD2};
}


void write_combed_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2)
{
    std::ofstream field_file(output_filename, std::ios::out | std::ios::trunc);

    // write all feature edge vertices
    int num_faces = PD1.rows();
    field_file << num_faces << std::endl;
    for (int f = 0; f < PD1.rows(); ++f)
    {
        // write first direction
        for (int i : {0 , 1, 2})
        {
            field_file << std::fixed << std::setprecision(17) << PD1(f, i) << " ";
        }

        // write second direction
        for (int i : {0 , 1, 2})
        {
            field_file << std::fixed << std::setprecision(17) << PD2(f, i) << " ";
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

Eigen::VectorXi transfer_period_jumps_to_halfedge(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F, 
    const Eigen::MatrixXi& corner_period_jump)
{
    int num_halfedges = m.n_halfedges();
    int num_faces = F.rows();
    Eigen::VectorXi period_jump = Eigen::VectorXi::Zero(num_halfedges);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        // get reference halfedge
        int hij = m.h[fijk];

        // get local vertex index opposite reference halfedge
        int vk = vtx_reindex[m.v_rep[m.to[m.n[hij]]]];
        int k = 0;
        while (F(fijk, k) != vk)
        {
            k = (k + 1) % 3;
        }


        // get period jumps and rotations across edges
        for (int i = 0; i < 3; ++i)
        {
            period_jump[hij] = corner_period_jump(fijk, k);

            // increment local index and halfedge
            k = (k + 1) % 3;
            hij = m.n[hij];
        }

        // Mirror data onto reflected boundary halfedges on doubled meshes.
        if (m.type[hij] == 0) continue;
        hij = m.h[fijk];
        for (int i = 0; i < 3; ++i)
        {
            period_jump[m.R[hij]] = -period_jump[hij];
            hij = m.n[hij];
        }
    }

    return period_jump;
}


} // namespace Holonomy
} // namespace Penner
