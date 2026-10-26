#include "resolution/core/layout.h"

namespace Penner {
namespace Resolution {

double compute_seamless_error(const Eigen::Matrix2d& R)
{
  // goal: check fourth power is identity
  Eigen::Matrix2d R2 = R * R;
  Eigen::Matrix2d R4 = R2 * R2;

  // compute max abs error
  double error = 0.; 
  error = max(error, abs(R4(0, 0) - 1.));
  error = max(error, abs(R4(1, 0)));
  error = max(error, abs(R4(0, 1)));
  error = max(error, abs(R4(1, 1) - 1.));

  return error;
}

double compute_seamless_error(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij)
{
    return compute_seamless_error(infer_float_rotation(halfedge, uv, hij));
}

bool is_seamless_rotation(const Eigen::Matrix2d& R)
{
  return (float_equal(compute_seamless_error(R), 0., 1e-6));
}

Eigen::Matrix2d infer_float_rotation(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij
)
{
    // get relevant halfedge uv data
    int hji = halfedge.opp[hij];
    Eigen::Vector2d uvi = uv[halfedge.prev[hij]];
    Eigen::Vector2d uvj = uv[hij];
    Eigen::Vector2d uvio = uv[hji];
    Eigen::Vector2d uvjo = uv[halfedge.prev[hji]];

    // check for trivial identity case
    if ((uvi == uvio) && (uvj == uvjo))
    {
        return Eigen::Matrix2d::Identity();
    }

    // compute edge vectors
    Eigen::Vector2d dij = uvj - uvi;
    Eigen::Vector2d dijo = uvjo - uvio;
    assert(float_equal(dij.norm(), dijo.norm()));

    auto perp = [](const Eigen::Vector2d& v) -> Eigen::Vector2d {
        return Eigen::Vector2d(-v.y(), v.x());
    };

    // Set matrices with edge vectors as columns
    Eigen::Matrix2d B1, B2;
    B1.col(0) = dij;
    B1.col(1) = perp(dij);
    B2.col(0) = dijo;
    B2.col(1) = perp(dijo);

    // the rotation is the unique matrix that maps the edge vectors
    // to each other
    return B1 * B2.inverse();
}

Eigen::Matrix2d infer_rotation(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij
)
{
    // compute rotation with generic method (for metric parametrizations)
    Eigen::Matrix2d R = infer_float_rotation(halfedge, uv, hij);
    assert(is_seamless_rotation(R));

    // round to integer valued matrix for stability
    return R.array().round().matrix();
}

Eigen::Matrix2d transform_opposite_face(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int hij
) {
    int hji = halfedge.opp[hij];
    int hjk = halfedge.n[hij];
    int hki = halfedge.n[hjk];
    int hil = halfedge.n[hji];
    int hlj = halfedge.n[hil];

    // compute the rotation to carry the mate onto the halfedge
    Eigen::Matrix2d R = infer_rotation(halfedge, uv, hij);

    // compute the edge vector of the opposite vertex in the face to move
    Eigen::Vector2d uvio = uv[hji];
    Eigen::Vector2d uvlo = uv[hil];
    Eigen::Vector2d dilo = uvlo - uvio;

    // set the new opposite face vertices, laying out the opposite vertex using
    // the rotated edge vector rooted at the fixed input halfedge
    Eigen::Vector2d uvi = uv[hki];
    Eigen::Vector2d uvj = uv[hij];
    uv[hji] = uvi;
    uv[hlj] = uvj;
    uv[hil] = uvi + (R * dilo);

    assert(float_equal(dilo.norm(), (uv[hil] - uv[hji]).norm()));
    assert(float_equal((uvlo - uv[hlj]).norm(), (uv[hil] - uv[hlj]).norm()));

    // return rotation matrix for convenience
    return R;
}

Eigen::Vector3d rotate_in_face_plane(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector3d>& V,
    const Eigen::Matrix2d& R,
    const Eigen::Vector3d& d,
    int face_index)
{
    // get face vertices
    const int hij = halfedge.f2h[face_index];
    const int hjk = halfedge.n[hij];
    const int hki = halfedge.n[hjk];
    const int vi = halfedge.to[hki];
    const int vj = halfedge.to[hij];
    const int vk = halfedge.to[hjk];

    // compute local frame for the face
    const Eigen::Vector3d eij = V[vj] - V[vi];
    const Eigen::Vector3d eik = V[vk] - V[vi];
    Eigen::Vector3d b1 = eij.normalized();
    Eigen::Vector3d n = eij.cross(eik).normalized();
    Eigen::Vector3d b2 = n.cross(b1).normalized();

    // compute local frame coordinates of the tangent vector
    Eigen::Vector2d d_local(d.dot(b1), d.dot(b2));

    // apply rotation to 2D frame coordinates 
    Eigen::Vector2d d_rot = R * d_local;

    // expand back to 3D vector coefficients
    return d_rot[0] * b1 + d_rot[1] * b2;
}

Eigen::Matrix2d transform_opposite_face(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V,
    std::vector<Eigen::Vector3d>& PD1,
    std::vector<Eigen::Vector3d>& PD2,
    int hij
) {
    // transform rotation compute rotation 
    Eigen::Matrix2d R = transform_opposite_face(halfedge, uv, hij);

    // rotate the opposite face target fields so that the norm of the error is invariant
    // a clockwise rotation of the triangle results in a counterclockwise rotation of the
    // gradient, and vice versa 
    const int hji = halfedge.opp[hij];
    const int fjil = halfedge.face[hji];
    PD1[fjil] = rotate_in_face_plane(halfedge, V, R.transpose(), PD1[fjil], fjil);
    PD2[fjil] = rotate_in_face_plane(halfedge, V, R.transpose(), PD2[fjil], fjil);

    return R;
}

}
}