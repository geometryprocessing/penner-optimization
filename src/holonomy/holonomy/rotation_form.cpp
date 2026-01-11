#include "holonomy/holonomy/rotation_form.h"

#include "holonomy/field/field.h"
#include "holonomy/field/intrinsic_field.h"
#include "holonomy/field/forms.h"

#include <igl/boundary_facets.h>
#include <igl/per_face_normals.h>

namespace Penner {
namespace Holonomy {

// Priority function for halfedges
// Assumes that vertex indices are unique
bool has_priority(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex, int h)
{
    assert(m.to[h] != m.to[m.opp[h]]);
    return h < m.opp[h];
    return (vtx_reindex[m.to[h]] < vtx_reindex[m.to[m.opp[h]]]);
}

// Get vertex in the embedded mesh
int get_projected_vertex(const Mesh<Scalar>& m, int h)
{
    return m.v_rep[m.to[h]];
    // TODO
    // if (m.type[h] < 2) return m.to[h];
    // else return m.to[m.opp[m.R[h]]];
}

// Get face in the embedded mesh
int get_projected_face(const Mesh<Scalar>& m, int h)
{
    if (m.type[h] < 2)
        return m.f[h];
    else
        return m.f[m.R[h]];
}

// Measure the intrinsic angle between frame field vectors in two faces across an edge
Scalar compute_cross_field_edge_angle(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& N,
    int h)
{
    // Use consistent halfedge for stability
    // if (m.type[h] > 1) { FIXME
    //    return -compute_cross_field_edge_angle(m, vtx_reindex, V, R, N, m.R[h]);
    //}
    if (!has_priority(m, vtx_reindex, h)) {
        return -compute_cross_field_edge_angle(m, vtx_reindex, V, R, N, m.opp[h]);
    }

    // Get halfedge direction
    int hij = h;
    int hji = m.opp[hij];
    int vi = vtx_reindex[get_projected_vertex(m, hji)];
    int vj = vtx_reindex[get_projected_vertex(m, hij)];
    Eigen::Vector3d h_direction = V.row(vj) - V.row(vi);

    // Get frame field directions
    int f0 = get_projected_face(m, hij);
    int f1 = get_projected_face(m, hji);
    Eigen::Vector3d R0 = R.row(f0);
    Eigen::Vector3d R1 = R.row(f1);

    // Get signed face normals
    // In doubled meshes, the normal is inverted
    // double s0 = (m.type[hij] < 2) ? 1.0 : -1.0;
    // double s1 = (m.type[hji] < 2) ? 1.0 : -1.0;
    double s0 = 1.0;
    double s1 = 1.0;
    Eigen::Vector3d N0 = s0 * N.row(f0);
    Eigen::Vector3d N1 = s1 * N.row(f1);

    // Get angle of rotation across the edge
    Scalar d0 = signed_angle<Eigen::Vector3d>(h_direction, R0, N0);
    Scalar d1 = signed_angle<Eigen::Vector3d>(h_direction, R1, N1);
    Scalar alpha = (2 * M_PI) + (M_PI / 4) + d0 - d1;

    return pos_fmod(double(alpha), M_PI / 2.0) - (M_PI / 4);
}

VectorX generate_rotation_form_from_cross_field(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& frame_field)
{
    // Compute face normals
    Eigen::MatrixXd N;
    igl::per_face_normals(V, F, N);

    // Compute rotation form from frame field
    int num_halfedges = m.n_halfedges();
    VectorX rotation_form(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        rotation_form[h] = compute_cross_field_edge_angle(m, vtx_reindex, V, frame_field, N, h);
    }

    assert(is_valid_one_form(m, rotation_form));
    return rotation_form;
}

VectorX generate_intrinsic_rotation_form(const Mesh<Scalar>& m, const FieldParameters& field_params)
{
    IntrinsicNRosyField field_generator;
    field_generator.min_cone = field_params.min_cone;
    field_generator.use_roundings = field_params.use_roundings;
    
    return field_generator.run(m);
}

VectorX generate_intrinsic_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V,
    const FieldParameters& field_params)
{
    IntrinsicNRosyField field_generator;
    field_generator.use_roundings = field_params.use_roundings;
    field_generator.min_cone = field_params.min_cone;

    return field_generator.run_with_viewer(m, vtx_reindex, V);
}

} // namespace Holonomy
} // namespace Penner