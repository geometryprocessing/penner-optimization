// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "field/vector_field.h"

#include "util/map.h"
#include "util/linear_algebra.h"
#include "util/vector.h"
#include "util/vf_mesh.h"
//#include "field/forms.h"
//#include "metric/constraint.h"

#include <igl/local_basis.h>
#include <igl/per_face_normals.h>
#include <igl/rotate_vectors.h>
#include <igl/boundary_facets.h>
#include <igl/principal_curvature.h>
#include <igl/average_onto_faces.h>

namespace Penner {
namespace Field {

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

Eigen::MatrixXd generate_vector_field(
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

Eigen::MatrixXd transfer_vertex_direction_to_corner(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& vertex_direction,
    int corner_index
)
{
    int num_faces = F.rows();
    int dim = vertex_direction.cols();
    Eigen::MatrixXd face_direction(num_faces, dim);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        int vi = F(fijk, corner_index);
        face_direction.row(fijk) = vertex_direction.row(vi);
    }

    return face_direction;
}


Eigen::Vector3d average_line_field(const Eigen::Vector3d& d0, const Eigen::Vector3d& d1, const Eigen::Vector3d& d2)
{
    //start with first vector (implicitly fixing sign)
    Eigen::Vector3d d = d0;

    // add second vector with sign corrected to d0
    if ((d0 - d1).norm() < (d0 + d1).norm())
    {
        d += d1;
    } else {
        d -= d1;
    }

    // add third vector with sign corrected to d0
    if ((d0 - d2).norm() < (d0 + d2).norm())
    {
        d += d2;
    } else {
        d -= d2;
    }

    return d / 3.;
}

Eigen::MatrixXd average_line_field(const std::array<Eigen::MatrixXd, 3>& corner_directions)
{
    int num_faces = corner_directions[0].rows();
    Eigen::MatrixXd face_direction(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        face_direction.row(fijk) = average_line_field(corner_directions[0].row(fijk), corner_directions[1].row(fijk), corner_directions[2].row(fijk));
    }

    return face_direction;
}


Eigen::MatrixXd average_line_field_onto_faces(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& vertex_direction)
{
    std::array<Eigen::MatrixXd, 3> corner_directions;
    for (int i = 0; i < 3; ++i)
    {
        corner_directions[i] = transfer_vertex_direction_to_corner(F, vertex_direction, i);
    }

    return average_line_field(corner_directions);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
compute_facet_principal_curvature(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius)
{
    // find principal curvature
      // Compute curvature directions via quadric fitting
    Eigen::MatrixXd PD1,PD2;
    Eigen::VectorXd PV1,PV2;
    std::vector<int> bad_vertices;
    igl::principal_curvature(V,F,PD1,PD2,PV1,PV2,bad_vertices,radius,(radius!=1));
    Eigen::VectorXd face_max_curvature, face_min_curvature;
    igl::average_onto_faces(F, PV1, face_max_curvature);
    igl::average_onto_faces(F, PV2, face_min_curvature);
    Eigen::MatrixXd face_max_direction = average_line_field_onto_faces(F, PD1);
    Eigen::MatrixXd face_min_direction = average_line_field_onto_faces(F, PD2);

    return std::make_tuple(face_max_direction, face_min_direction, face_max_curvature, face_min_curvature);
}

std::tuple<Eigen::MatrixXd, std::vector<bool>> compute_field_direction(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius,
    Scalar abs_threshold,
    Scalar rel_threshold,
    Scalar sample_rate)
{
    //auto[max_direction, min_direction, _max_curvature, _min_curvature] = compute_facet_principal_curvature(V, F, radius);
    //auto[_max_direction, _min_direction, max_curvature, min_curvature] = compute_facet_principal_curvature(V, F, 3);
    auto[max_direction, min_direction, max_curvature, min_curvature] = compute_facet_principal_curvature(V, F, radius);
    int num_faces = F.rows();
    std::vector<bool> is_fixed_direction(num_faces, false);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        Scalar kmax = max_curvature[fijk];
        Scalar kmin = min_curvature[fijk];
        if (compute_mean_anisotropy(kmax, kmin) < abs_threshold) continue;
        if (compute_parabolic_anisotropy(kmax, kmin) < rel_threshold) continue;
        is_fixed_direction[fijk] = true;
    }

    if (sample_rate < 1)
    {
        // shuffle directions
        std::vector<int> fixed_directions;
        convert_boolean_array_to_index_vector(is_fixed_direction, fixed_directions);
        std::vector<int> shuffled_directions = shuffle_map_image(fixed_directions);

        // compute number of sampled directions
        int num_fixed_directions = fixed_directions.size();
        int num_sampled_directions = sample_rate * num_fixed_directions;
        num_sampled_directions = std::min<int>(num_sampled_directions, num_fixed_directions);

        // get first n shuffled directions
        is_fixed_direction = std::vector<bool>(num_faces, false);
        for (int i = 0; i < num_sampled_directions; ++i)
        {
            is_fixed_direction[shuffled_directions[i]] = true;
        }
    }

    return std::make_tuple(max_direction, is_fixed_direction);
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



} // namespace Holonomy
} // namespace Penner