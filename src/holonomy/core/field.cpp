#include "holonomy/core/field.h"

#include "util/vf_mesh.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/core/forms.h"

#include "optimization/core/constraint.h"
#include "util/vector.h"

#if USE_COMISO
#include <igl/copyleft/comiso/nrosy.h>
#endif

namespace Penner {
namespace Holonomy {

std::tuple<Eigen::MatrixXd, std::vector<Scalar>> generate_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    // Compute cross field and singularities from comiso
    Eigen::MatrixXd frame_field;
    Eigen::VectorXd S;
#if USE_COMISO
    Eigen::VectorXi b(1);
    Eigen::MatrixXd bc(1, 3);
    b << 0;
    bc << 1, 1, 1;
    int degree = 4;
    igl::copyleft::comiso::nrosy(V, F, b, bc, degree, frame_field, S);
#else
    int num_vertices = V.rows();
    int num_faces = F.rows();
    spdlog::error(
        "Comiso solver not enabled for #V={}, #F={} mesh. Set USE_COMISO to use.",
        num_vertices,
        num_faces);
#endif

    // Get the boundary vertices
    std::vector<bool> is_boundary_vertex = compute_boundary_vertices(F, V.rows());

    // Turn singularities into a flat metric
    // FIXME This is only accurate for closed meshes; singularities only make sense with doubling
    int num_cone_vertices = S.size();
    std::vector<Scalar> Th_hat(num_cone_vertices);
    for (int vi = 0; vi < num_cone_vertices; ++vi) {
        if (is_boundary_vertex[vi]) {
            Th_hat[vi] = M_PI - (2 * M_PI * S[vi]);
        } else {
            Th_hat[vi] = 2 * M_PI * (1 - S[vi]);
        }
    }

    return std::make_tuple(frame_field, Th_hat);
}

} // namespace Holonomy
} // namespace Penner