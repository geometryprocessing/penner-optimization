#pragma once

#include "holonomy/core/common.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Parameters for cross field generation
 *
 */
struct FieldParameters
{
    Scalar min_angle = 0.; // minimum allowed cone angle in the cross field
};

/**
 * @brief Given a mesh with a per-face frame field, 
 * 
 * @param m: halfedge mesh
 * @param vtx_reindex: map from halfedge to VF vertex indices
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param frame_field: |F|x3 frame field of per-face field direction vectors 
 * @return per-halfedge rotation form
 */
VectorX generate_rotation_form_from_cross_field(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field);

VectorX generate_intrinsic_rotation_form(const Mesh<Scalar>& m, const FieldParameters& field_params);

VectorX generate_intrinsic_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const FieldParameters& field_params);

} // namespace Holonomy
} // namespace Penner