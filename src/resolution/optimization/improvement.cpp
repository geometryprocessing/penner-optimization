#include "resolution/optimization/improvement.h"

#include "resolution/core/layout.h"
#include "util/vf_mesh.h"
#include "metric/quality.h"
#include "parametrization/layout.h"
#include "util/union_find.h"

#include <igl/doublearea.h>
#include <igl/local_basis.h>
#include <igl/avg_edge_length.h>
#include <igl/predicates/predicates.h>

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "conformal_ideal_delaunay/Layout.hh"

#include <igl/Timer.h>


#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "energy.h"
#include "main_helper.h"


namespace Penner {
namespace Resolution {

// helper functions
namespace {

Eigen::MatrixXd rows_to_matrix(const std::vector<Eigen::Vector3d>& rows, int cols)
{
    Eigen::MatrixXd M(rows.size(), cols);
    for (int i = 0; i < rows.size(); ++i)
    {
        M.row(i) = rows[i].head(cols);
    }

    return M;
}

Eigen::MatrixXd rows_to_matrix(const std::vector<Eigen::Vector2d>& rows)
{
    Eigen::MatrixXd M(rows.size(), 2);
    for (int i = 0; i < rows.size(); ++i)
    {
        M.row(i) = rows[i];
    }

    return M;
}

std::vector<Eigen::Vector2d> matrix_to_rows2(const Eigen::MatrixXd& M)
{
    std::vector<Eigen::Vector2d> rows(M.rows());
    for (int i = 0; i < M.rows(); ++i)
    {
        rows[i] = M.row(i);
    }

    return rows;
}

std::vector<Eigen::Vector3d> matrix_to_rows3(const Eigen::MatrixXd& M)
{
    std::vector<Eigen::Vector3d> rows(M.rows());
    for (int i = 0; i < M.rows(); ++i)
    {
        rows[i] = M.row(i);
    }

    return rows;
}

} // namespace


// ----------------
// Geometry Helpers
// ----------------

// measure the error in length of paired halfedges
double compute_length_error(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int hij)
{
    int hji = halfedge.opp[hij];
    Eigen::Vector2d uvi = uv[halfedge.prev[hij]];
    Eigen::Vector2d uvj = uv[hij];
    Eigen::Vector2d uvio = uv[hji];
    Eigen::Vector2d uvjo = uv[halfedge.prev[hji]];
    return abs((uvj - uvi).norm() - (uvjo - uvio).norm());
}

// get the normal of a face in 3D space
Eigen::Vector3d compute_face_normal(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector3d>& V,
    int face_index)
{
    int hij = halfedge.f2h[face_index];
    int hjk = halfedge.n[hij];
    int hki = halfedge.n[hjk];

    int vi = halfedge.to[hki];
    int vj = halfedge.to[hij];
    int vk = halfedge.to[hjk];

    Eigen::Vector3d eij = (V[vj] - V[vi]).transpose();
    Eigen::Vector3d eik = (V[vk] - V[vi]).transpose();
    Eigen::Vector3d n = eij.cross(eik);
    double n_norm = n.norm();
    if (n_norm > 0.) n /= n_norm;

    return n.transpose();
}

// average the two face normals adjacent to an edge
Eigen::Vector3d average_edge_normal(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector3d>& V,
    int halfedge_index)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int f0 = halfedge.face[hij];
    int f1 = halfedge.face[hji];

    Eigen::Vector3d n0 = compute_face_normal(halfedge, V, f0);
    Eigen::Vector3d n1 = compute_face_normal(halfedge, V, f1);
    return 0.5 * (n0 + n1);
}

// measure the dihedral angle of two faces adjacent to an edge
double edge_dihedral_angle(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector3d>& V,
    int halfedge_index)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int f0 = halfedge.face[hij];
    int f1 = halfedge.face[hji];

    double d = compute_face_normal(halfedge, V, f0).dot(compute_face_normal(halfedge, V, f1));
    if (d < -1.) d = -1.;
    if (d > 1.) d = 1.;
    return acos(d);
}

// ----------
// Predicates
// ----------

// determine if the one ring of a vertex is seamless
bool is_one_ring_seamless(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int vertex_index)
{
    int h_start = halfedge.out[vertex_index];
    int hij = h_start;
    do
    {
        if (!is_seamless_rotation(infer_float_rotation(halfedge, uv, hij))) return false;
        hij = halfedge.n[halfedge.opp[hij]];
    } while (hij != h_start);

    return true;
}

// determine if a face is flipped in the uv plane with robust predicates
bool is_face_flipped(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int face_index)
{
    int hij = halfedge.f2h[face_index];
    int hjk = halfedge.n[hij];
    int hki = halfedge.n[hjk];
    return (igl::predicates::orient2d(
        uv[hki],
        uv[hij],
        uv[hjk]) != igl::predicates::Orientation::POSITIVE);
}

// determine if any faces in the one ring of a vertex are flipped
bool is_one_ring_flipped(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int vertex_index)
{
    int h_start = halfedge.out[vertex_index];
    int hij = h_start;
    do
    {
        if (is_face_flipped(halfedge, uv, halfedge.face[hij])) return true;
        hij = halfedge.opp[halfedge.prev[hij]];
    } while (hij != h_start);

    return false;
}

// determine if any faces adjacent to an edge are flipped
bool is_edge_flap_flipped(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int halfedge_index)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    return is_face_flipped(halfedge, uv, halfedge.face[hij]) || is_face_flipped(halfedge, uv, halfedge.face[hji]);
}

// check if link around vi contains vk
bool link_contains_vertex(
    const HalfedgeMesh& halfedge,
    int vi,
    int vk)
{
    int h_start = halfedge.out[vi];
    int hij = h_start;
    do
    {
        if (halfedge.to[hij] == vk) return true;

        hij = halfedge.opp[halfedge.prev[hij]];
    } while (hij != h_start);

    return false;
}

// check if link around vi does not contain duplicates
bool is_link_simple(
    const HalfedgeMesh& halfedge,
    int vi)
{
    int h_start = halfedge.out[vi];
    int hij = h_start;
    std::set<int> link;
    int count = 0;
    do
    {
        link.insert(halfedge.to[hij]);
        ++count;
        hij = halfedge.opp[halfedge.prev[hij]];
    } while (hij != h_start);

    return (link.size() == count);
}


// -------------
// Displacements
// -------------

// compute displacements to apply to all vertices in the same vertex cycle as a halfedge
// tip to displace the halfedge tip to a given position while maintaining seamless constraints
std::vector<Eigen::Vector2d> compute_halfedge_tip_displacements(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int halfedge_index,
    Eigen::Vector2d& position
) {
    int hij = halfedge_index;
    Eigen::Vector2d d = position - uv[hij];

    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    std::vector<Eigen::Vector2d> displacements = {};
    do
    {
        Eigen::Vector2d Rd = (R.inverse()) * d;
        int hjk = halfedge.n[hij];
        if (halfedge.opp[hjk] != halfedge_index)
        {
            R = R * infer_rotation(halfedge, uv, hjk);
        }
        displacements.push_back(Rd);
        hij = halfedge.opp[hjk];
    } while (hij != halfedge_index);

    return displacements;
}

// compute all positions of the tips of the halfedges in the same vertex cycle as
// the tip of the given halfedge
std::vector<Eigen::Vector2d> compute_halfedge_tip_positions(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int halfedge_index
) {
    int hij = halfedge_index;
    std::vector<Eigen::Vector2d> positions = {};
    do
    {
        positions.push_back(uv[hij]);
        hij = halfedge.opp[halfedge.n[hij]];
    } while (hij != halfedge_index);

    return positions;
}

// apply the given vector of displacements (assumed generated from compute_halfedge_tip_displacements)
// to the tips of the halfedges in the vertex cycle
void apply_halfedge_tip_displacements(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int halfedge_index,
    const std::vector<Eigen::Vector2d>& displacements)
{
  int hij = halfedge_index;
  for (int i = 0; i < displacements.size(); ++i)
  {
    uv[hij] += displacements[i];
    hij = halfedge.opp[halfedge.n[hij]];
  }
}


// set the given tips of the halfedges in the vertex cycle to the given positions
// e.g., obtained from compute_halfedge_tip_positions
void set_halfedge_tip_positions(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int halfedge_index,
    const std::vector<Eigen::Vector2d>& positions)
{
  int hij = halfedge_index;
  for (int i = 0; i < positions.size(); ++i)
  {
    uv[hij] = positions[i];
    hij = halfedge.opp[halfedge.n[hij]];
  }
}

// scale a vector of displacements by a given value (e.g., to shrink a line step)
void scale_displacements(
    std::vector<Eigen::Vector2d>& displacements,
    double scale)
{
  for (int i = 0; i < displacements.size(); ++i)
  {
    displacements[i] *= scale;
  }
}

// set the given tips of the halfedges in the vertex cycle to displace the halfedge
// tip to a given position while maintaining seamless constraints
bool set_halfedge_tip_position(
    const HalfedgeMesh& halfedge,
    std::vector<Eigen::Vector2d>& uv,
    int halfedge_index,
    Eigen::Vector2d& position
) {
    std::vector<Eigen::Vector2d> displacements = compute_halfedge_tip_displacements(halfedge, uv, halfedge_index, position);
    apply_halfedge_tip_displacements(halfedge, uv, halfedge_index, displacements);
    return true;
}

// ------------
// Local Charts
// ------------

// build parametric layout of the given subset of faces
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> get_local_halfedge_layout(
    const HalfedgeMesh& halfedge,
    const Eigen::MatrixXd& uv,
    const std::vector<int>& faces
) {
    int num_faces = faces.size();
    Eigen::MatrixXd uv_l(3 * num_faces, 2);
    Eigen::MatrixXi F_l(num_faces, 3);

    // build local connectivity
    int count = 0;
    for (int i = 0; i < num_faces; ++i)
    {
        int f = faces[i];
        int hij = halfedge.f2h[f];
        for (int j = 0; j < 3; ++j)
        {
            uv_l.row(count) = uv.row(hij);
            F_l(i, j) = count;

            // increment
            ++count;
            hij = halfedge.n[hij];
        }
    }

    return std::make_tuple(uv_l, F_l);
}

// viewer for parametrization around an edge
void view_edge_flap(
    const HalfedgeMesh& halfedge,
    const Eigen::MatrixXd& uv,
    int hij
) {
    int hji = halfedge.opp[hij];
    std::vector<int> flap_faces = {halfedge.face[hij], halfedge.face[hji]};
    auto [uv_l, F_l] = get_local_halfedge_layout(halfedge, uv, flap_faces);
#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh2D("edge flap layout", uv_l, F_l);
    polyscope::show();
#endif
}

void view_vertex_adjacent_faces(
    const HalfedgeMesh& halfedge,
    const Eigen::MatrixXd& uv,
    int vertex_index,
    std::string mesh_handle="",
    bool show=true)
{
    if (mesh_handle == "") {
        mesh_handle = "vertex adjacent face layout";
    }

    if ((vertex_index < 0) || (vertex_index >= halfedge.out.size()))
    {
        spdlog::warn("invalid vertex index {}", vertex_index);
        return;
    }

    // get faces adjacent to vertex
    std::vector<int> adjacent_faces = {};
    int h_start = halfedge.out[vertex_index];
    int hij = h_start;
    do
    {
        adjacent_faces.push_back(halfedge.face[hij]);
        hij = halfedge.opp[halfedge.prev[hij]];
    } while (hij != h_start);

    // build local layout
    auto [uv_l, F_l] = get_local_halfedge_layout(halfedge, uv, adjacent_faces);

    // view adjacent faces
#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh2D(mesh_handle, uv_l, F_l);
    if (show) polyscope::show();
#endif
}

// build local 3D mesh and parametrization in the one ring of a vertex
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd>
build_local_chart(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V,
    int vertex_index
) {
    int degree = halfedge.vertex_degree(vertex_index);
    int hij = halfedge.out[vertex_index];

    // build local chart
    std::vector<Eigen::Vector3d> V_l(degree + 1);
    std::vector<Eigen::Vector2d> uv_l(degree + 1);
    Eigen::MatrixXi F_l(degree, 3);
    V_l[0] = V[vertex_index];
    uv_l[0] = Eigen::Vector2d::Zero();

    // set all vertex positions in one ring using rotated displacement
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    for (int i = 0; i < degree; ++i)
    {
        // set 3D geometry and face
        int vj = halfedge.to[hij];
        int j = (i + 1) % degree;
        V_l[i + 1] = V[vj];
        F_l.row(i) << 0, i + 1, j + 1; 

        // get direction vector from i to j
        Eigen::Vector2d uvi = uv[halfedge.prev[hij]];
        Eigen::Vector2d uvj = uv[hij];
        Eigen::Vector2d d = uvj - uvi;

        // get rotated direction
        Eigen::Vector2d Rd = R * d;
        uv_l[i + 1] = Rd; 
        spdlog::trace("rotation is \n{}", R);
        spdlog::trace("displacing {} by {}", hij, Rd);

        // rotate outgoing direction clockwise
        R = R * infer_rotation(halfedge, uv, hij);
        hij = halfedge.n[halfedge.opp[hij]];
    }

    return std::make_tuple(rows_to_matrix(V_l, 3), F_l, rows_to_matrix(uv_l));
}

// build a VF mesh with parametrization with all triangles disconnected
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd> explode_halfedge(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector3d>& V,
    const std::vector<Eigen::Vector2d>& uv)
{
    int num_faces = halfedge.count_faces();
    Eigen::MatrixXd V_c(3 * num_faces, 3);
    Eigen::MatrixXi F_c(num_faces, 3);
    Eigen::MatrixXd uv_c(3 * num_faces, 2);

    auto [he_new2old, he_old2new] = halfedge.get_halfedge_reductions();
    auto [f_new2old, f_old2new] = halfedge.get_face_reductions();

    for (int Fijk = 0; Fijk < f_new2old.size(); ++Fijk)
    {
        int f = f_new2old[Fijk];
        int hij = halfedge.f2h[f];
        for (int i = 0; i < 3; ++i)
        {
            int Hij = he_old2new[hij];
            V_c.row(Hij) = V[halfedge.to[hij]];
            F_c(Fijk, i) = Hij;
            uv_c.row(Hij) = uv[hij];
            hij = halfedge.n[hij];
        }
    }

    return std::make_tuple(V_c, F_c, uv_c);
}

// -----------
// Frame Field
// -----------


// project field directions to the tangent planes of the face
void project_direction_to_tangent(
    Eigen::MatrixXd& PD1,
    Eigen::MatrixXd& PD2,
    const Eigen::MatrixXd& V,
    const HalfedgeMesh& halfedge,
    int face_index)
{
    const int hij = halfedge.f2h[face_index];
    const int hjk = halfedge.n[hij];
    const int hki = halfedge.n[hjk];
    const int vi = halfedge.to[hki];
    const int vj = halfedge.to[hij];
    const int vk = halfedge.to[hjk];

    // get face tangent
    const Eigen::Vector3d eij = (V.row(vj) - V.row(vi)).transpose();
    const Eigen::Vector3d eik = (V.row(vk) - V.row(vi)).transpose();
    Eigen::Vector3d b1 = eij.normalized();
    Eigen::Vector3d n = eij.cross(eik).normalized();
    Eigen::Vector3d b2 = n.cross(b1).normalized();

    // do projection
    Eigen::Vector3d v = PD1.row(face_index).transpose();
    Eigen::Vector2d v_local(v.dot(b1), v.dot(b2));
    PD1.row(face_index) = (v_local[0] * b1 + v_local[1] * b2).stableNormalized().transpose();
    PD2.row(face_index) = (-v_local[1] * b1 + v_local[0] * b2).stableNormalized().transpose();
}

// project field directions to the tangent planes of the face
void project_direction_to_tangent(
    std::vector<Eigen::Vector3d>& PD1,
    std::vector<Eigen::Vector3d>& PD2,
    const std::vector<Eigen::Vector3d>& V,
    const HalfedgeMesh& halfedge,
    int face_index)
{
    const int hij = halfedge.f2h[face_index];
    const int hjk = halfedge.n[hij];
    const int hki = halfedge.n[hjk];
    const int vi = halfedge.to[hki];
    const int vj = halfedge.to[hij];
    const int vk = halfedge.to[hjk];

    const Eigen::Vector3d eij = (V[vj] - V[vi]).transpose();
    const Eigen::Vector3d eik = (V[vk] - V[vi]).transpose();
    Eigen::Vector3d b1 = eij.normalized();
    Eigen::Vector3d n = eij.cross(eik).normalized();
    Eigen::Vector3d b2 = n.cross(b1).normalized();

    Eigen::Vector3d v = PD1[face_index].transpose();
    Eigen::Vector2d v_local(v.dot(b1), v.dot(b2));
    PD1[face_index] = (v_local[0] * b1 + v_local[1] * b2).stableNormalized().transpose();
    PD2[face_index] = (-v_local[1] * b1 + v_local[0] * b2).stableNormalized().transpose();
}

// void average_field_across_edge(
//     Eigen::MatrixXd& PD1,
//     Eigen::MatrixXd& PD2,
//     const Eigen::MatrixXd& V,
//     const HalfedgeMesh& halfedge,
//     int halfedge_index
// )
// {
//     int hij = halfedge_index;
//     int hji = halfedge.opp[hij];
//     int fijk = halfedge.face[hij];
//     int fjil = halfedge.face[hji];

//     // get average direction of faces
//     Eigen::Vector3d v0 = PD1.row(fijk).transpose();
//     Eigen::Vector3d v1 = PD1.row(fjil).transpose();
//     Eigen::Vector3d v = 0.5 * (v0 + v1);
    
//     // project directions to faces
//     for (int f : { fijk, fjil })
//     {
//         PD1.row(f) = v;
//         project_direction_to_tangent(PD1, PD2, V, halfedge, f);
//     }

// }

// average the field directions across the given edge and project to the tangent planes
void average_field_across_edge(
    std::vector<Eigen::Vector3d>& PD1,
    std::vector<Eigen::Vector3d>& PD2,
    const std::vector<Eigen::Vector3d>& V,
    const HalfedgeMesh& halfedge,
    int halfedge_index
)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int fijk = halfedge.face[hij];
    int fjil = halfedge.face[hji];

    Eigen::Vector3d v = 0.5 * (PD1[fijk] + PD1[fjil]).transpose();
    for (int f : { fijk, fjil })
    {
        PD1[f] = v.transpose();
        project_direction_to_tangent(PD1, PD2, V, halfedge, f);
    }
}

// ----------------
// Triangle Quality
// ----------------

template <typename Vector>
Eigen::Vector3d compute_triangle_lengths(
    const HalfedgeMesh& halfedge,
    const std::vector<Vector>& positions,
    int face_index,
    bool is_tip_index)
{
    int hij = halfedge.f2h[face_index];
    std::array<Vector, 3> face_uv;
    for (int i = 0; i < 3; ++i)
    {
        face_uv[i] = (is_tip_index) ? positions[hij] : positions[halfedge.to[hij]];
        hij = halfedge.n[hij];
    }

    Eigen::Vector3d lengths;
    for (int i = 0; i < 3; ++i)
    {
        int j = (i + 1) % 3;
        lengths[i] = (face_uv[j] - face_uv[i]).norm();
    }

    return lengths;
}

template <typename Vector>
double compute_triangle_quality(
    const HalfedgeMesh& halfedge,
    const std::vector<Vector>& positions,
    int face_index,
    bool is_tip_index)
{
    Eigen::Vector3d l = compute_triangle_lengths(halfedge, positions, face_index, is_tip_index);
    return Holonomy::compute_triangle_quality(l[0], l[1], l[2]);
}

template <typename Vector>
double compute_edge_quality(
    const HalfedgeMesh& halfedge,
    const std::vector<Vector>& positions,
    int halfedge_index,
    bool is_tip_index)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    return compute_triangle_quality(halfedge, positions, halfedge.face[hij], is_tip_index)
         + compute_triangle_quality(halfedge, positions, halfedge.face[hji], is_tip_index);
}


template <typename Vector>
double compute_vertex_quality(
    const HalfedgeMesh& halfedge,
    const std::vector<Vector>& positions,
    int vertex_index,
    bool is_tip_index)
{
    double quality = 0.;
    int h_start = halfedge.out[vertex_index];
    int hij = h_start;
    do
    {
        quality += compute_triangle_quality(halfedge, positions, halfedge.face[hij], is_tip_index);
        hij = halfedge.opp[halfedge.prev[hij]];
    } while (hij != h_start);

    return quality;
}

template <typename Vector>
double compute_total_quality(
    const HalfedgeMesh& halfedge,
    const std::vector<Vector>& positions,
    bool is_tip_index)
{
    double quality = 0.;
    int num_faces = halfedge.f2h.size();
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        if (halfedge.is_face_deleted(fijk)) continue;
        quality += compute_triangle_quality(halfedge, positions, fijk, is_tip_index);
    }

    return quality;
}

// -------------------
// Symmetric Dirichlet
// -------------------

// compute the symmetric dirichlet energy for a parametrization
double symmetric_dirichlet_energy(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V,
    int vertex_index,
    int norm_p=1)
{
    // build local patch
    auto [V_l, F_l, uv_l] = build_local_chart(halfedge, uv, V, vertex_index);

    // get gradient from local patch
    Eigen::SparseMatrix<double> G;
    SymDir::get_grad_op(V_l, F_l, G);

    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, uv_l, Ji);

    Eigen::VectorXd area;
    igl::doublearea(V_l, F_l, area);
    area = 0.5 * area;
    double energy = SymDir::compute_energy_from_jacobian(Ji, area, norm_p);
    spdlog::info("symmetric dirichlet energy is {}", energy);
    spdlog::info("area is {}", area.sum());

    return energy * area.sum();
}

double symmetric_dirichlet_energy(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V,
    int norm_p=1)
{
    auto [V_c, F_c, uv_c] = explode_halfedge(halfedge, V, uv);

    // get gradient from local patch
    Eigen::SparseMatrix<double> G;
    SymDir::get_grad_op(V_c, F_c, G);

    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, uv_c, Ji);

    Eigen::VectorXd area;
    igl::doublearea(V_c, F_c, area);
    double energy = SymDir::compute_energy_from_jacobian(Ji, area, norm_p);

    return energy;
}

Eigen::VectorXd face_symmetric_dirichlet_values(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V)
{
    auto [V_c, F_c, uv_c] = explode_halfedge(halfedge, V, uv);

    Eigen::VectorXd face_energy = SymDir::symmetric_dirichlet_energy(V_c, F_c, uv_c, 1);

    // map from reduced to full face list
    auto [f_new2old, f_old2new] = halfedge.get_face_reductions();
    Eigen::VectorXd full_face_energy = Eigen::VectorXd::Zero(f_old2new.size());
    for (int f = 0; f < f_new2old.size(); ++f)
    {
        full_face_energy[f_new2old[f]] = face_energy[f];
    }

    return full_face_energy;
}

Eigen::VectorXd vertex_symmetric_dirichlet_values(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V)
{
    Eigen::VectorXd face_sym_dir = face_symmetric_dirichlet_values(halfedge, uv, V);

    int num_vertices = halfedge.out.size();
    int num_halfedges = halfedge.n.size();
    Eigen::VectorXd vertex_sym_dir = Eigen::VectorXd::Zero(num_vertices);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (halfedge.is_halfedge_deleted(hij)) continue;
        int vi = halfedge.to[hij];
        int f = halfedge.face[hij];
        vertex_sym_dir(vi) = max(vertex_sym_dir(vi), face_sym_dir(f));
    }

    return vertex_sym_dir;
}


void view_halfedge_parameterization(
    const HalfedgeMesh& halfedge,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& uv
) {
    auto V_rows = matrix_to_rows3(V);
    auto uv_rows = matrix_to_rows2(uv);
    auto [V_c, F_c, uv_c] = explode_halfedge(halfedge, V_rows, uv_rows);

    auto [f_new2old, f_old2new] = halfedge.get_face_reductions();
    Eigen::VectorXd sym_dir = vector_compose(face_symmetric_dirichlet_values(halfedge, uv_rows, V_rows), f_new2old);
    Eigen::VectorXd area;
    igl::doublearea(V_c, F_c, area);

    // view adjacent faces
#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh("mesh", V_c, F_c)
        ->addVertexParameterizationQuantity("uv", uv_c);
    polyscope::getSurfaceMesh("mesh")
        ->addFaceScalarQuantity("sym dirichlet", sym_dir);
    polyscope::getSurfaceMesh("mesh")
        ->addFaceScalarQuantity("area", area);
    polyscope::show();
#endif
}

Eigen::VectorXd symmetric_dirichlet(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT)
{
    Eigen::MatrixXd V_cut;
    cut_mesh_along_parametrization_seams(V, F, uv, FT, V_cut);
    return SymDir::symmetric_dirichlet_energy(V_cut, FT, uv, 1);
}


MeshImprover::MeshImprover(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT
) {
    avg_edge_length = igl::avg_edge_length(V, F);
    spdlog::info("average edge length: {}", avg_edge_length);
    abs_collapse_edge_thres = 0.01 * avg_edge_length;

    Eigen::VectorXd area;
    igl::doublearea(V, F, area);
    avg_area = 0.5 * area.mean();
    
    // Convert to halfedge
    std::tie(halfedge, vtx_reindex, he_to_corner) = generate_halfedge(F);

    // TODO: replace with final mesh connectivity
    VectorX cone_angles = compute_cone_angles(V, F, uv, FT);
    int num_vertices = V.rows();
    is_cone = std::vector<bool>(num_vertices, false);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (!float_equal(cone_angles[vtx_reindex[vi]], 2 * PI))
        {
            is_cone[vi] = true;
        }
    }

    // transfer uvs to halfedge
    halfedge_uv.resize(he_to_corner.size(), Eigen::Vector2d::Zero());
    for (int hij = 0; hij < he_to_corner.size(); ++hij)
    {
        auto [f, i] = he_to_corner[hij];
        int hjk = halfedge.n[hij];
        Eigen::Vector2d uvk = uv.row(FT(f, i));
        halfedge_uv[hjk] = uvk;
    }

    // reindex vertices
    V_r.resize(V.rows());
    for (int vi = 0; vi < vtx_reindex.size(); ++vi)
    {
        V_r[vi] = V.row(vtx_reindex[vi]);
    }
}


void MeshImprover::set_combed_field(
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2
)
{
    auto [f_new2old, f_old2new] = halfedge.get_face_reductions();
    const int num_faces = halfedge.count_faces();
    assert(PD1.rows() == f_old2new.size());
    assert(PD2.rows() == f_old2new.size());
    assert(PD1.cols() == PD2.cols());

    PD1_r.resize(num_faces, Eigen::Vector3d::Zero());
    PD2_r.resize(num_faces, Eigen::Vector3d::Zero());

    for (int fi = 0; fi < num_faces; ++fi)
    {
        const int f = f_new2old[fi];
        PD1_r[fi] = PD1.row(f);
        PD2_r[fi] = PD2.row(f);
    }
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
MeshImprover::get_combed_field() const
{
    auto [f_new2old, f_old2new] = halfedge.get_face_reductions();
    Eigen::MatrixXd PD1 = Eigen::MatrixXd::Zero(f_new2old.size(), 3);
    Eigen::MatrixXd PD2 = Eigen::MatrixXd::Zero(f_new2old.size(), 3);

    if (PD1_r.empty() || PD2_r.empty())
    {
        return std::make_tuple(PD1, PD2);
    }

    for (int fi = 0; fi < f_new2old.size(); ++fi)
    {
        const int f = f_new2old[fi];
        PD1.row(fi) = PD1_r[f];
        PD2.row(fi) = PD2_r[f];
    }

    return std::make_tuple(PD1, PD2);
}

void MeshImprover::improve(int num_iter)
{
    igl::Timer timer, method_timer;

    // get initial mesh quality
    results["initial-quality"] = compute_total_quality(halfedge, halfedge_uv, true);
    //results["initial-sym-dir"] = symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r);
    results["initial-energy"] = compute_energy();
    spdlog::info("initial quality: {}", results["initial-quality"]);
    if (show_smooth) view_halfedge_parameterization(halfedge, rows_to_matrix(V_r, 3), rows_to_matrix(halfedge_uv));

    timer.start();
    std::vector<json> iter_results(num_iter);
    for (int i = 0; i < num_iter; ++i)
    {
        // smooth vertices
        if (!skip_smooth)
        {
            // get pre-smooth statistics
            method_timer.start();
            spdlog::info("Optimizing symmetric Dirichlet energy with seamless constraints");
            iter_results[i]["symdir-energy"] = compute_energy();
            iter_results[i]["symdir-time"] = method_timer.getElapsedTime();
            spdlog::info("post-symdir energy: {}", iter_results[i]["symdir-energy"]);

            // smooth all vertices
            method_timer.start();
            for (int vi = 0; vi < halfedge.out.size(); ++vi)
            {
                if (halfedge.is_vertex_deleted(vi)) continue;

                // cannot move cone vertex positions
                if (is_cone[vi])
                {
                    spdlog::debug("skipping nonplanar vertex");
                    continue; // skip cone vertices with seam
                }
                assert(!is_one_ring_flipped(halfedge, halfedge_uv, vi));
                
                // locally optimize vertex position
                local_energy_opt(vi);
            }

            // get post-smooth statistics
            iter_results[i]["smooth-quality"] = compute_total_quality(halfedge, V_r, false);
            iter_results[i]["smooth-uv-quality"] = compute_total_quality(halfedge, halfedge_uv, true);
            //iter_results[i]["smooth-sym-dir"] = symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r);
            iter_results[i]["smooth-energy"] = compute_energy();
            iter_results[i]["smooth-time"] = method_timer.getElapsedTime();
            spdlog::info("post-smoothing energy: {}", iter_results[i]["smooth-energy"]);
            spdlog::info("post-smoothing quality: {}", iter_results[i]["smooth-quality"]);
            spdlog::info("post-smoothing uv quality: {}\n", iter_results[i]["smooth-uv-quality"]);
        }

        // try flipping edges
        if (!skip_flips)
        {
            method_timer.start();
            int flip_count = 0;
            int num_edges = halfedge.e2h.size();
            for (int eij = 0; eij < num_edges; ++eij)
            {
                if (halfedge.is_edge_deleted(eij)) continue;
                int hij = halfedge.e2h[eij];

                double orig_quality = compute_edge_quality(halfedge, V_r, hij, false);
                double orig_uv_quality = compute_edge_quality(halfedge, halfedge_uv, hij, true);
                double orig_total_quality = orig_quality + orig_uv_quality;
                spdlog::debug("original quality is {}, {}", orig_quality, orig_uv_quality);

                // don't flip to self adjacent edge
                int hji = halfedge.opp[hij];
                int hjk = halfedge.n[hij];
                int hil = halfedge.n[hji];
                if (halfedge.to[hjk] == halfedge.to[hil])
                {
                    spdlog::debug("tried to flip to self adjacent edge");
                    continue;
                }
                int vk = halfedge.to[hjk];
                int vl = halfedge.to[hil];
                if ((link_contains_vertex(halfedge, vk, vl)) || (link_contains_vertex(halfedge, vl, vk)))
                {
                    spdlog::debug("flipping edge would create double edge");
                    continue;
                }

                // dihedral angle pre-check
                double flip_angle_thres = PI / 90.;
                if (abs(edge_dihedral_angle(halfedge, V_r, hij)) > flip_angle_thres)
                {
                    spdlog::debug("edge too sharp to flip");
                    continue;
                }
                Eigen::Vector3d nij = average_edge_normal(halfedge, V_r, hij);

                // get initial field (ater fixing local patch rotation)
                transform_opposite_face(hij);
                int fijk = halfedge.face[hij];
                int fjil = halfedge.face[hji];
                Eigen::Vector3d d11 = PD1_r[fijk];
                Eigen::Vector3d d12 = PD1_r[fjil];
                Eigen::Vector3d d21 = PD2_r[fijk];
                Eigen::Vector3d d22 = PD2_r[fjil];

                // perform flip
                flip_edge_ccw(hij);

                // get post-flip statistics
                double quality = compute_edge_quality(halfedge, V_r, hij, false);
                double uv_quality = compute_edge_quality(halfedge, halfedge_uv, hij, true);
                double total_quality = quality + uv_quality;
                Eigen::Vector3d nkl = average_edge_normal(halfedge, V_r, hij);
                spdlog::debug("new quality is {}, {}", quality, uv_quality);

                // check if edge flip improves quality and doesn't flip sharp edges
                if ((is_edge_flap_flipped(halfedge, halfedge_uv, hij))
                    || (quality > orig_quality)
                    || (uv_quality > orig_uv_quality)
                    || (nij.dot(nkl) < 0.5))
                {
                    flip_edge_clockwise(hij);
                    PD1_r[fijk] = d11;
                    PD1_r[fjil] = d12;
                    PD2_r[fijk] = d21;
                    PD2_r[fjil] = d22;

                    // check if flip undone successfully
                    quality = compute_edge_quality(halfedge, V_r, hij, false);
                    uv_quality = compute_edge_quality(halfedge, halfedge_uv, hij, true);
                    assert(float_equal(uv_quality, orig_uv_quality, 1e-8));
                }
                else
                {
                    flip_count++;
                }
            }

            // get post-flip statistics
            iter_results[i]["flip-count"] = flip_count;
            iter_results[i]["flip-quality"] = compute_total_quality(halfedge, V_r, false);
            iter_results[i]["flip-uv-quality"] = compute_total_quality(halfedge, halfedge_uv, true);
            //iter_results[i]["flip-sym-dir"] = symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r);
            iter_results[i]["flip-energy"] = compute_energy();
            iter_results[i]["flip-time"] = method_timer.getElapsedTime();
            spdlog::info("number of edge flips: {}", flip_count);
            spdlog::info("post-flip energy: {}", iter_results[i]["flip-energy"]);
            spdlog::info("post-flip quality: {}", iter_results[i]["flip-quality"]);
            spdlog::info("post-flip uv quality: {}\n", iter_results[i]["flip-uv-quality"]);
        }

        // try collapsing edges
        if (!skip_collapse)
        {
            method_timer.start();
            int collapse_count = 0;
            int num_halfedges = halfedge.n.size();
            for (int hij = 0; hij < num_halfedges; ++hij)
            {
                if (halfedge.is_halfedge_deleted(hij)) continue;

                // make sure edge can be collapsed
                if (!halfedge.is_collapse_edge_valid(hij)) continue;
                if (!halfedge.is_collapse_link_valid(hij)) continue;
                int vi = halfedge.to[halfedge.opp[hij]];
                int vj = halfedge.to[hij];
                assert(is_link_simple(halfedge, vi));
                assert(is_link_simple(halfedge, vj));
                if (!is_link_simple(halfedge, vi)) continue;
                if (!is_link_simple(halfedge, vj)) continue;


                // check if collapsed vertex vj is a cone and prohibit if so
                if (is_cone[vj]) continue;

                // only collapse small edges
                if ((V_r[vi] - V_r[vj]).norm() > abs_collapse_edge_thres)
                {
                    continue;
                }

                // collapse edge
                bool success = collapse_edge(hij);
                if (!success) continue;
                else collapse_count++;
            }

            // check post-collapse data
            iter_results[i]["collape-count"] = collapse_count;
            iter_results[i]["collapse-quality"] = compute_total_quality(halfedge, V_r, false);
            iter_results[i]["collapse-uv-quality"] = compute_total_quality(halfedge, halfedge_uv, true);
            //iter_results[i]["collapse-sym-dir"] = symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r);
            iter_results[i]["collapse-energy"] = compute_energy();
            iter_results[i]["collapse-time"] = method_timer.getElapsedTime();
            spdlog::info("number of edge collapses: {}", collapse_count);
            spdlog::info("post-collapse energy: {}", iter_results[i]["collapse-energy"]);
            spdlog::info("post-collapse quality: {}", iter_results[i]["collapse-quality"]);
            spdlog::info("post-collapse uv quality: {}\n", iter_results[i]["collapse-uv-quality"]);
        }

        // try refining edges
        double abs_split_edge_thres = 10 * avg_edge_length;
        if (!skip_split)
        {
            method_timer.start();
            int split_count = 0;
            int num_edges = halfedge.e2h.size();
            for (int eij = 0; eij < num_edges; ++eij)
            {
                if (halfedge.is_edge_deleted(eij)) continue;
                int hij = halfedge.e2h[eij];

                // check if edge is too long to split
                if (halfedge.is_halfedge_deleted(hij)) continue;
                int vi = halfedge.to[halfedge.opp[hij]];
                int vj = halfedge.to[hij];
                double lij = (V_r[vi] - V_r[vj]).norm();
                if (lij < abs_split_edge_thres) continue;

                // split the edge
                spdlog::debug("splitting edge {} of length {}", hij, lij);
                split_edge(hij);
                split_count++;
            }

            // get post-split statistic
            iter_results[i]["split-count"] = split_count;
            iter_results[i]["split-quality"] = compute_total_quality(halfedge, V_r, false);
            iter_results[i]["split-uv-quality"] = compute_total_quality(halfedge, halfedge_uv, true);
            //iter_results[i]["split-sym-dir"] = symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r);
            iter_results[i]["split-energy"] = compute_energy();
            iter_results[i]["split-time"] = method_timer.getElapsedTime();
            spdlog::info("number of splits: {}", split_count);
            spdlog::info("post-split energy: {}", iter_results[i]["split-energy"]);
            spdlog::info("post-split quality: {}", iter_results[i]["split-quality"]);
            spdlog::info("post-split uv quality: {}\n", iter_results[i]["split-uv-quality"]);
        }
    }

    // get final statistics
    results["iter-results"] = iter_results;
    results["final-sym-dir"] = symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r);
    results["final-energy"] = compute_energy();
    results["total-time"] = timer.getElapsedTime();

    // check final validity
    if (!halfedge.is_valid()) spdlog::error("halfedge is not valid");
    if (!halfedge.is_simplicial()) spdlog::error("mesh is not simplicial");
    assert(halfedge.is_valid());
    assert(halfedge.is_simplicial());

    // check final seamless error
    int num_halfedges = halfedge.n.size();
    double length_error = 0.;
    double seamless_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (halfedge.is_halfedge_deleted(hij)) continue;
        length_error = max(length_error, compute_length_error(halfedge, halfedge_uv, hij));
        seamless_error = max(seamless_error, compute_seamless_error(halfedge, halfedge_uv, hij));
    }
    spdlog::info("length error: {}", length_error); 
    spdlog::info("seamless error: {}", seamless_error); 
    spdlog::info("final energy: {}", results["final-energy"]); 
    spdlog::info("total time: {}", results["total-time"]); 

    if (show_smooth) view_halfedge_parameterization(halfedge, rows_to_matrix(V_r, 3), rows_to_matrix(halfedge_uv));
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi>
MeshImprover::get_improved_mesh()
{
    int num_faces = halfedge.f2h.size();
    int num_halfedges = halfedge.n.size();
    
    // build new layout
    std::vector<bool> is_processed_face(num_faces, false);
    std::vector<bool> is_cut_halfedge(num_halfedges, true);

    // Initialize queue and record of faces to process
    std::queue<int> Q;
    int f_start = 0;
    while (halfedge.is_face_deleted(f_start)) ++f_start;
    Q.push(halfedge.f2h[f_start]);
    is_processed_face[f_start] = true;

    // get halfedge cut and make sure layout faces are adjacent in interior
    while (!Q.empty()) {
        // Get next halfedge to process
        int h = Q.front();
        Q.pop();

        // Get other triangle edges
        int hn = halfedge.n[h];
        int hp = halfedge.n[hn];
        int hno = halfedge.opp[hn];
        int hpo = halfedge.opp[hp];
        int ho = halfedge.opp[h];

        // Check if next edge triangle should be laid out
        if (!is_processed_face[halfedge.face[hno]]) {
            transform_opposite_face(hn);
            is_cut_halfedge[hn] = false;
            is_cut_halfedge[hno] = false;

            is_processed_face[halfedge.face[hno]] = true;
            Q.push(hno);
        }

        // Check if previous edge triangle should be laid out
        if (!is_processed_face[halfedge.face[hpo]]) {
            transform_opposite_face(hp);
            is_cut_halfedge[hp] = false;
            is_cut_halfedge[hpo] = false;

            is_processed_face[halfedge.face[hpo]] = true;
            Q.push(hpo);
        }

        // Check if current edge triangle should be laid out
        // NOTE: Should only be used once for original edge
        if (!is_processed_face[halfedge.face[ho]]) {
            transform_opposite_face(h);
            is_cut_halfedge[h] = false;
            is_cut_halfedge[ho] = false;

            is_processed_face[halfedge.face[ho]] = true;
            Q.push(ho);
        }
    }

    // trim cut
    bool any_trimmed = true;
    while (any_trimmed)
    {
        any_trimmed = false;
        for (int hij = 0; hij < num_halfedges; hij++)
        {
            if (halfedge.is_halfedge_deleted(hij)) continue;
            if (!is_cut_halfedge[hij]) continue;
            int vj = halfedge.to[hij];
            if ((count_valence(halfedge.n, halfedge.opp, hij, is_cut_halfedge) == 1) && (!is_cone[vj]))
            {
                int hji = halfedge.opp[hij];
                is_cut_halfedge[hij] = false;
                is_cut_halfedge[hji] = false;
                any_trimmed = true;
                continue;
            }
        }
    }

    Scalar length_error = 0.;
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (halfedge.is_halfedge_deleted(hij)) continue;
        length_error = max(length_error, compute_length_error(halfedge, halfedge_uv, hij));
    }
    spdlog::info("length error after layout: {}", length_error); 

    // get uv vertices
    UnionFind uv_vertices(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (halfedge.is_halfedge_deleted(hij)) continue;

        // check if cut halfedge
        int hkj = halfedge.opp[halfedge.n[hij]];
        if (is_cut_halfedge[hkj]) continue;

        // union tips otherwise
        uv_vertices.union_sets(hij, hkj);
    }

    // build uv vertices
    auto [he_new2old, he_old2new] = halfedge.get_halfedge_reductions();
    auto [f_new2old, f_old2new] = halfedge.get_face_reductions();
    int num_uv_vertices = uv_vertices.count_sets();
    std::vector<int> h2fv = uv_vertices.index_sets();
    Eigen::MatrixXd uv_r(num_uv_vertices, 2);
    Eigen::MatrixXi F_r(halfedge.count_faces(), 3);
    Eigen::MatrixXi FT_r(halfedge.count_faces(), 3);
    for (int f = 0; f < num_faces; ++f)
    {
        if (halfedge.is_face_deleted(f)) continue;

        int hij = halfedge.f2h[f];
        for (int i = 0; i < 3; ++i)
        {
            int Fijk = f_old2new[f];
            int vi = halfedge.to[hij];
            int uvi = h2fv[hij];
            uv_r.row(uvi) = halfedge_uv[hij];
            F_r(Fijk, i) = vi;
            FT_r(Fijk, i) = uvi;
            hij = halfedge.n[hij];
        }
    }

    Eigen::MatrixXd V_out = rows_to_matrix(V_r, 3);
    remove_unreferenced_inplace(V_out, F_r);
    remove_unreferenced_inplace(uv_r, FT_r);

    return std::make_tuple(V_out, F_r, uv_r, FT_r);
}

void MeshImprover::get_uv(
    Eigen::MatrixXd& uv,
    Eigen::MatrixXi& FT
) const
{
    // transfer uvs to halfedge
    for (int hij = 0; hij < he_to_corner.size(); ++hij)
    {
        auto [f, i] = he_to_corner[hij];
        int hjk = halfedge.n[hij];
        uv.row(FT(f, i)) = halfedge_uv[hjk];
    }
}

void MeshImprover::set_uv(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT
)
{
    // transfer uvs to halfedge
    for (int hij = 0; hij < he_to_corner.size(); ++hij)
    {
        auto [f, i] = he_to_corner[hij];
        int hjk = halfedge.n[hij];
        halfedge_uv[hjk] = uv.row(FT(f, i));
    }
}

void MeshImprover::transform_opposite_face(int hij)
{
    if (PD1_r.empty() || PD2_r.empty())
    {
        Penner::Resolution::transform_opposite_face(halfedge, halfedge_uv, hij);
        return;
    }
    Penner::Resolution::transform_opposite_face(halfedge, halfedge_uv, V_r, PD1_r, PD2_r, hij);
}

void MeshImprover::transform_vertex_adjacent_faces(
    int vertex_index
) {
    int h_start = halfedge.out[vertex_index];
    int hij = h_start;
    do
    {
        // transform opposite face (clockwise from current face)
        transform_opposite_face(hij);

        // need to rotate halfedge clockwise into newly moved face
        hij = halfedge.n[halfedge.opp[hij]];
    } while (hij != h_start);
}

Eigen::Matrix<double, 2, 3> MeshImprover::compute_face_tangent_gradient(
    int face_index) const
{
    // renaming indices of vertices of triangles for convenience
    int f = face_index;
    int hki = halfedge.f2h[f];
    int hij = halfedge.n[hki];
    int hjk = halfedge.n[hij];
    int vi = halfedge.to[hki];
    int vj = halfedge.to[hij];
    int vk = halfedge.to[hjk];

    // #F x 3 matrices of triangle edge vectors, named after opposite vertices
    Eigen::Vector3d vkj = V_r[vk] - V_r[vj];
    Eigen::Vector3d vik = V_r[vi] - V_r[vk];
    Eigen::Vector3d vji = V_r[vj] - V_r[vi];
    Eigen::Vector3d n = vkj.cross(vik);
    
    // area of parallelogram is twice area of triangle
    // area of parallelogram is || v1 x v2 ||
    // This does correct l2 norm of rows, so that it contains #F list of twice
    // triangle areas
    double dblA = std::sqrt(n.dot(n));
    Eigen::Vector3d u = n / dblA;

    // if area of face is too small, use scaled standard simplex
    if (dblA < 2. * min_rel_area * avg_area)
    {
        spdlog::debug("using standard simplex for face {} with area {}", f, dblA);
        double target_edge_length = std::sqrt(4 * avg_area / std::sqrt(3));
        Eigen::Vector3d Vi, Vj, Vk;
        Vi.setZero();
        Vj.setZero();
        Vk.setZero();
        Vi(0) = target_edge_length * std::sqrt(min_rel_area);
        Vj(1) = target_edge_length * std::sqrt(min_rel_area);
        Vk(2) = target_edge_length * std::sqrt(min_rel_area);
        vkj = Vk - Vj;
        vik = Vi - Vk;
        vji = Vj - Vi;
        n = vkj.cross(vik);
        dblA = std::sqrt(n.dot(n));
        u = n / dblA;
    }
    
    // rotate each vector 90 degrees around normal
    double normji = std::sqrt(vji.dot(vji));
    double normik = std::sqrt(vik.dot(vik));
    Eigen::Vector3d eperp_ji = u.cross(vji);
    eperp_ji = eperp_ji / std::sqrt(eperp_ji.dot(eperp_ji));
    eperp_ji *= normji / dblA;
    Eigen::Vector3d eperp_ik = u.cross(vik);
    eperp_ik = eperp_ik / std::sqrt(eperp_ik.dot(eperp_ik));
    eperp_ik *= normik / dblA;
    Eigen::Vector3d eperp_kj = -eperp_ji - eperp_ik;

    // get surface frame
    Eigen::Vector3d b1 = vji.stableNormalized();
    Eigen::Vector3d b2 = (n.cross(b1)).stableNormalized();

    // compute map from vertex values to x, y derivatives
    Eigen::Matrix<double, 2, 3> G;
    G(0, 0) = eperp_kj.dot(b1);
    G(0, 1) = eperp_ik.dot(b1);
    G(0, 2) = eperp_ji.dot(b1);
    G(1, 0) = eperp_kj.dot(b2);
    G(1, 1) = eperp_ik.dot(b2);
    G(1, 2) = eperp_ji.dot(b2);

    return G;
}

Eigen::Matrix<double, 3, 3> MeshImprover::compute_face_gradient(
    int face_index) const
{
    // renaming indices of vertices of triangles for convenience
    int f = face_index;
    int hki = halfedge.f2h[f];
    int hij = halfedge.n[hki];
    int hjk = halfedge.n[hij];
    int vi = halfedge.to[hki];
    int vj = halfedge.to[hij];
    int vk = halfedge.to[hjk];

    // #F x 3 matrices of triangle edge vectors, named after opposite vertices
    Eigen::Vector3d vkj = V_r[vk] - V_r[vj];
    Eigen::Vector3d vik = V_r[vi] - V_r[vk];
    Eigen::Vector3d vji = V_r[vj] - V_r[vi];
    Eigen::Vector3d n = vkj.cross(vik);
    // area of parallelogram is twice area of triangle
    // area of parallelogram is || v1 x v2 ||
    // This does correct l2 norm of rows, so that it contains #F list of twice
    // triangle areas
    double dblA = std::sqrt(n.dot(n));
    Eigen::Vector3d u = n / dblA;

    // rotate each vector 90 degrees around normal
    double normji = std::sqrt(vji.dot(vji));
    double normik = std::sqrt(vik.dot(vik));
    Eigen::Vector3d eperp_ji = u.cross(vji);
    eperp_ji = eperp_ji / std::sqrt(eperp_ji.dot(eperp_ji));
    eperp_ji *= normji / dblA;
    Eigen::Vector3d eperp_ik = u.cross(vik);
    eperp_ik = eperp_ik / std::sqrt(eperp_ik.dot(eperp_ik));
    eperp_ik *= normik / dblA;
    Eigen::Vector3d eperp_kj = -eperp_ji - eperp_ik;

    // compute map from vertex values to x, y derivatives
    Eigen::Matrix<double, 3, 3> G;
    G.col(0) = eperp_kj;
    G.col(1) = eperp_ik;
    G.col(2) = eperp_ji;

    return G;
}

double MeshImprover::compute_face_area(int face_index) const
{
    int hij = halfedge.f2h[face_index];
    int hjk = halfedge.n[hij];
    int hki = halfedge.n[hjk];
    int vi = halfedge.to[hki];
    int vj = halfedge.to[hij];
    int vk = halfedge.to[hjk];

    return compute_triangle_area(V_r[vi], V_r[vj], V_r[vk]);
}

double MeshImprover::compute_total_area() const
{
    double area = 0.;
    int num_faces = halfedge.f2h.size();
    for (int f = 0; f < num_faces; ++f)
    {
        if (halfedge.is_face_deleted(f)) continue;
        area += compute_face_area(f);
    }

    return area;
}

double MeshImprover::compute_face_energy(int face_index) const
{
    int f = face_index;
    double energy = 0.;

    // get face uv vertices
    Eigen::Matrix<double, 3, 2> face_uv = compute_face_uv(f);

    // compute face energy
    double area = compute_face_area(f);
    Eigen::Matrix<double, 2, 3> G = compute_face_tangent_gradient(f);
    Eigen::Matrix<double, 2, 2> J_sq = G * face_uv;
    energy += area * symdir_weight * SymDir::symmetric_dirichlet_energy_t(J_sq(0, 0), J_sq(1, 0), J_sq(0, 1), J_sq(1, 1), norm_p);

    if ((alignment_weight > 0) && (area > min_rel_area * avg_area))
    {
        Eigen::Matrix<double, 3, 3> grad = compute_face_gradient(f);
        Eigen::Matrix<double, 3, 2> Guv = grad * face_uv;
        Eigen::Matrix<double, 3, 2> uT_vT;
        uT_vT.col(0) = PD1_r[f];
        uT_vT.col(1) = PD2_r[f];
        Eigen::MatrixXd R = Guv - uT_vT;
        energy += alignment_weight * area * (R.transpose() * R).trace();
    }

    return energy;
}

double MeshImprover::compute_vertex_energy(int vertex_index) const
{
    int vi = vertex_index;
    int hij = halfedge.out[vi];
    int hik = hij;
    double energy = 0.;
    do
    {
        int f = halfedge.face[hik];
        energy += compute_face_energy(f);

        // iterate around vertex
        hik = halfedge.n[halfedge.opp[hik]];
    } while (hik != hij);

    return energy;
}

double MeshImprover::compute_energy() const
{
    double energy = 0.;
    int num_faces = halfedge.f2h.size();
    for (int f = 0; f < num_faces; ++f)
    {
        if (halfedge.is_face_deleted(f)) continue;
        energy += compute_face_energy(f);
    }

    // normalize energy by area 
    energy = energy / compute_total_area();

    return energy;
}

std::tuple<double, Eigen::Vector2d> MeshImprover::vertex_energy_direction(int vertex_index) const
{
    int vi = vertex_index;
    int hij = halfedge.out[vi];
    int hik = hij;
    double energy = 0.;
    double total_area = 0.;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    do
    {
        // get face uv vertices
        Eigen::Matrix<double, 3, 2> face_uv;
        int f = halfedge.face[hik];

        // need to use this halfedge for consistency
        int h = halfedge.f2h[f];
        int local_vertex_index = -1;
        for (int i = 0; i < 3; ++i)
        {
            face_uv.row(i) = halfedge_uv[h];

            // look for index correponding to the central vertex
            if (halfedge.to[h] == vi)
            {
                local_vertex_index = i;
            }

            h = halfedge.n[h];
        }

        // compute face symdir energy
        double area = compute_face_area(f);
        Eigen::Matrix<double, 2, 3> G = compute_face_tangent_gradient(f);
        Eigen::Vector2d g = G.col(local_vertex_index);
        Eigen::Matrix<double, 2, 2> J_sq = G * face_uv;
        Eigen::Matrix<Scalar, 1, 4> J = Eigen::Map<Eigen::Matrix<Scalar, 1, 4>>(J_sq.data(), 4);
        Eigen::Matrix<Scalar, 4, 4> local_hessian;
        Eigen::Matrix<Scalar, 1, 4> local_grad;
        energy += area * symdir_weight * jakob::gradient_and_hessian_from_J(J, local_grad, local_hessian, norm_p);
        total_area += area;

        // extrat gradient and hessian for face
        for (int i = 0; i < 2; ++i)
        {
            gradient[i] += area * symdir_weight * g.dot(local_grad.segment(2 * i, 2));
            for (int j = 0; j < 2; ++j)
            {
                H(i, j) += area * symdir_weight * g.dot(local_hessian.block(2 * i, 2 * j, 2, 2) * g);
            }
        }

        // get alignment energy
        if ((alignment_weight > 0) && (area > min_rel_area * avg_area))
        {
            Eigen::Matrix<double, 3, 3> grad = compute_face_gradient(f);
            Eigen::Matrix<double, 3, 2> Guv = grad * face_uv;
            Eigen::Matrix<double, 3, 2> uT_vT;
            Eigen::Vector3d gi = grad.col(local_vertex_index);
            uT_vT.col(0) = PD1_r[f];
            uT_vT.col(1) = PD2_r[f];
            Eigen::Matrix<double, 3, 2> R = Guv - uT_vT;
            energy += alignment_weight * area * (R.transpose() * R).trace();
            gradient += 2. * alignment_weight * area * (gi.transpose() * R);
            for (int i = 0; i < 2; ++i)
            {
                H(i, i) += 2. * alignment_weight * area * gi.squaredNorm();
            }
        }

        // iterate around vertex
        hik = halfedge.n[halfedge.opp[hik]];
    } while (hik != hij);

    spdlog::debug("energy is {}", energy / total_area);
    spdlog::debug("gradient is {}", gradient / total_area);
    spdlog::debug("hessian is \n{}", H / total_area);

    Eigen::Vector2d d = -H.inverse() * gradient;
    spdlog::debug("direction is {}", d);

    return std::make_tuple(energy, d);
}

void MeshImprover::local_energy_opt(int vertex_index)
{
    // WARNING: assumes hji is in same face as local layout used for the direction computation
    int vi = vertex_index; 
    int hji = halfedge.prev[halfedge.out[vi]];

    // get search direction for mesh improvement
    double orig_energy ;
    Eigen::Vector2d d;
    std::tie(orig_energy, d) = vertex_energy_direction(vi);
    spdlog::debug("original energy for vertex {} is {}", vi, orig_energy);
    spdlog::debug("displacing by {}", d);

    // try to move the full line step
    Eigen::Vector2d p = halfedge_uv[hji];
    Eigen::Vector2d pn = p + d;
    std::vector<Eigen::Vector2d> positions = compute_halfedge_tip_positions(halfedge, halfedge_uv, hji);
    std::vector<Eigen::Vector2d> displacements = compute_halfedge_tip_displacements(halfedge, halfedge_uv, hji, pn);
    apply_halfedge_tip_displacements(halfedge, halfedge_uv, hji, displacements);
    double energy = compute_vertex_energy(vi);
    spdlog::debug("new energy is {}", energy);
    //assert(float_equal(energy, symmetric_dirichlet_energy(halfedge, halfedge_uv, V_r, vi)));

    // do backtracking line search to make sure no triangles are flipped
    int count = 1;
    scale_displacements(displacements, -1.);
    while ((is_one_ring_flipped(halfedge, halfedge_uv, vi)) || (energy > orig_energy))
    {
        if (count >= 5)
        {
            set_halfedge_tip_positions(halfedge, halfedge_uv, hji, positions);
            break;
        }

        spdlog::debug("backtracking attempt {}", count);
        scale_displacements(displacements, 0.5);
        apply_halfedge_tip_displacements(halfedge, halfedge_uv, hji, displacements);

        energy = compute_vertex_energy(vi);
        spdlog::debug("energy is {}", energy);

        ++count;
    }

    energy = compute_vertex_energy(vi);
    spdlog::debug("final energy is {}\n", energy);
}

void MeshImprover::flip_edge_ccw(
    int halfedge_index
)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int hjk = halfedge.n[hij];
    int hil = halfedge.n[hji];

    // make triangle flap connected before flipping
    transform_opposite_face(halfedge_index);

    // update uv data
    halfedge_uv[hij] = halfedge_uv[hjk];
    halfedge_uv[hji] = halfedge_uv[hil];

    // do flip
    // WARNING: must update connectivity after geometry
    halfedge.flip_edge(halfedge_index);

    // set field to average
    average_field_across_edge(PD1_r, PD2_r, V_r, halfedge, hij);
}

void MeshImprover::flip_edge_clockwise(
    int halfedge_index
)
{
    // make triangle flap connected before flipping
    transform_opposite_face(halfedge_index);

    // update uv data
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int hjk = halfedge.n[hij];
    int hil = halfedge.n[hji];
    halfedge_uv[hij] = halfedge_uv[hil];
    halfedge_uv[hji] = halfedge_uv[hjk];

    // do flip
    // WARNING: must update connectivity after geometry
    for (int i = 0; i < 3; ++i)
    {
        halfedge.flip_edge(halfedge_index);
    }

    // set field to average
    average_field_across_edge(PD1_r, PD2_r, V_r, halfedge, hij);
}

// compute quality in collapse region
template <typename Vector>
double compute_collapse_quality(
    const HalfedgeMesh& halfedge,
    const std::vector<Vector>& positions,
    int halfedge_index,
    bool is_tip_index=true)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int fijk = halfedge.face[hij];
    int fjil = halfedge.face[hji];

    double quality = 0.;
    for (int h_start : {hij, hji})
    {
      int h_curr = h_start;
      do
      {
          int f = halfedge.face[h_curr];
          if ((f != fijk) && (f != fjil))
          {
              quality += compute_triangle_quality(halfedge, positions, f, is_tip_index);
          }
          h_curr = halfedge.opp[halfedge.prev[h_curr]];
      } while (h_curr != h_start);
    }

    return quality;
}

// check if face is flipped in collapsed region
bool is_collapse_region_flipped(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int halfedge_index)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int fijk = halfedge.face[hij];
    int fjil = halfedge.face[hji];

    for (int h_start : {hij, hji})
    {
      int h_curr = h_start;
      do
      {
          int f = halfedge.face[h_curr];
          if ((f != fijk) && (f != fjil))
          {
              if (is_face_flipped(halfedge, uv, f)) return true;
          }
          h_curr = halfedge.opp[halfedge.prev[h_curr]];
      } while (h_curr != h_start);
    }

    return false;
}

bool MeshImprover::collapse_edge(
    int halfedge_index
)
{
    int hij = halfedge_index;
    int hki = halfedge.prev[hij];
    int vi = halfedge.to[hki];
    int vj = halfedge.to[hij];

    // quality of region is sum of vertex regions, minus the overcounted edge flap
    double orig_quality = compute_collapse_quality(halfedge, V_r, hij, false);
    double orig_uv_quality = compute_collapse_quality(halfedge, halfedge_uv, hij, true);
    double orig_total_quality = orig_quality + orig_uv_quality;
    spdlog::debug("quality before collapse: {}", orig_quality);
    spdlog::debug("uv quality before collapse: {}", orig_uv_quality);

    if (orig_quality > 1e10)
    {
        //view_vertex_adjacent_faces(halfedge, uv, vi, "vi", false);
        //view_vertex_adjacent_faces(halfedge, uv, vj, "vj", true);
    }

    // collapse location of edge endpoints
    Eigen::Vector2d p = halfedge_uv[hki];
    std::vector<Eigen::Vector2d> positions = compute_halfedge_tip_positions(halfedge, halfedge_uv, hij);
    std::vector<Eigen::Vector2d> displacements = compute_halfedge_tip_displacements(halfedge, halfedge_uv, hij, p);
    apply_halfedge_tip_displacements(halfedge, halfedge_uv, halfedge_index, displacements);

    // collapse location of 3D position
    Eigen::Vector3d pi = V_r[vi];
    Eigen::Vector3d pj = V_r[vj];
    //Eigen::Vector3d pn = (pi + pj) / 2.;
    Eigen::Vector3d pn = pi; // collapse to pi
    V_r[vi] = pn;
    V_r[vj] = pn;

    // TODO: need to change vertex position
    // get new quality in collapse region
    double quality = compute_collapse_quality(halfedge, V_r, hij, false);
    double uv_quality = compute_collapse_quality(halfedge, halfedge_uv, hij);
    double total_quality = quality + uv_quality;
    spdlog::debug("quality after setting position: {}", quality);
    spdlog::debug("uv quality after setting position: {}", uv_quality);

    // check if element degenerates
    bool success = true;
    if (is_collapse_region_flipped(halfedge, halfedge_uv, hij))
    {
        spdlog::debug("flipped element after collapse");
        success = false;
    }
    // TODO: replace uv quality with symmetric dirichlet
    if (uv_quality > orig_uv_quality)
    {
        spdlog::debug("collapse reduces uv quality");
        success = false;
    }
    if (quality > orig_quality)
    {
        spdlog::debug("collapse reduces quality");
        success = false;
    }
    if (total_quality > orig_total_quality)
    {
        spdlog::debug("collapse reduces quality");
        success = false;
    }
    
    // reset positions if degenerate and return without changing connectivity
    if (!success)
    {
        spdlog::debug("aborting collapse");
        set_halfedge_tip_positions(halfedge, halfedge_uv, halfedge_index, positions);
        V_r[vi] = pi;
        V_r[vj] = pj;

        double final_uv_quality = compute_collapse_quality(halfedge, halfedge_uv, hij);
        assert(float_equal(orig_uv_quality, final_uv_quality));

        return false;
    }

    // do collapse
    // WARNING: must update connectivity after geometry
    int hli = halfedge.collapse_edge(halfedge_index);

    // project directions in faces of collapsed vertex one ring
    int h = hli;
    do
    {
        int f = halfedge.face[h];
        project_direction_to_tangent(PD1_r, PD2_r, V_r, halfedge, f);
        h = halfedge.opp[halfedge.n[h]];
    } while (h != hli);

    double final_uv_quality = compute_vertex_quality(halfedge, halfedge_uv, vi, true);
    spdlog::debug("quality after collapse: {}", final_uv_quality);

    //assert(halfedge.is_valid());
    assert(float_equal(uv_quality, final_uv_quality, 1e-10 * quality));
    assert(!is_one_ring_flipped(halfedge, halfedge_uv, vi));
    assert(is_one_ring_seamless(halfedge, halfedge_uv, vi));

    return true;
}

void MeshImprover::split_edge(
    int halfedge_index
)
{
    int hij = halfedge_index;
    int hji = halfedge.opp[hij];
    int hjk = halfedge.n[hij];
    int hil = halfedge.n[hji];
    int vi = halfedge.to[hji];
    int vj = halfedge.to[hij];
    int num_init_halfedges = halfedge.n.size();
    int num_init_vertices = halfedge.out.size();
    int num_init_faces = halfedge.f2h.size();

    // get initial geometry
    Eigen::Vector2d uvi0 = halfedge_uv[halfedge.prev[hij]];
    Eigen::Vector2d uvj0 = halfedge_uv[hij];
    Eigen::Vector2d uvi1 = halfedge_uv[hji];
    Eigen::Vector2d uvj1 = halfedge_uv[halfedge.prev[hji]];
    Eigen::Vector2d uvk = halfedge_uv[hjk];
    Eigen::Vector2d uvl = halfedge_uv[hil];
    Eigen::Vector3d pd10 = PD1_r[halfedge.face[hij]];
    Eigen::Vector3d pd20 = PD2_r[halfedge.face[hij]];
    Eigen::Vector3d pd11 = PD1_r[halfedge.face[hji]];
    Eigen::Vector3d pd21 = PD2_r[halfedge.face[hji]];

    // do flip and get refined quad connectivity
    int hin = halfedge.split_triangle_mesh_edge(halfedge_index);
    int hnk = halfedge.n[hin];
    int hkn = halfedge.opp[hnk];
    int hnj = halfedge.n[hkn];
    int hjn = halfedge.opp[hnj];
    int hnl = halfedge.n[hjn];
    int hln = halfedge.opp[hnl];
    int hni = halfedge.n[hln];

    // add data for six halfedges, two new faces, and one new vertex
    // WARNING: count depends on implementation, so using generic expansion method
    for (int i = num_init_halfedges; i < halfedge.n.size(); ++i)
    {
        halfedge_uv.push_back({});
    }
    for (int i = num_init_vertices; i < halfedge.out.size(); ++i)
    {
        V_r.push_back({});
    }
    for (int i = num_init_faces; i < halfedge.f2h.size(); ++i)
    {
        PD1_r.push_back({});
        PD2_r.push_back({});
    }

    // update 3D geometry
    int vn = halfedge.to[hin];
    V_r[vn] = 0.5 * (V_r[vi] + V_r[vj]);

    // halfedges and field in face 0
    Eigen::Vector2d uvn0 = 0.5 * (uvi0 + uvj0);
    halfedge_uv[hin] = uvn0;
    halfedge_uv[hnk] = uvk;
    halfedge_uv[hkn] = uvn0;
    halfedge_uv[hnj] = uvj0;
    PD1_r[halfedge.face[hin]] = pd10;
    PD2_r[halfedge.face[hin]] = pd20;
    PD1_r[halfedge.face[hnj]] = pd10;
    PD2_r[halfedge.face[hnj]] = pd20;

    // halfedges and field in face 1
    Eigen::Vector2d uvn1 = 0.5 * (uvi1 + uvj1);
    halfedge_uv[hjn] = uvn1;
    halfedge_uv[hnl] = uvl;
    halfedge_uv[hln] = uvn1;
    halfedge_uv[hni] = uvi1;
    PD1_r[halfedge.face[hjn]] = pd11;
    PD2_r[halfedge.face[hjn]] = pd21;
    PD1_r[halfedge.face[hni]] = pd11;
    PD2_r[halfedge.face[hni]] = pd21;
}

Eigen::Matrix<double, 3, 2> MeshImprover::compute_face_uv(int face_index) const
{
    Eigen::Matrix<double, 3, 2> face_uv;
    int h = halfedge.f2h[face_index];
    for (int i = 0; i < 3; ++i)
    {
        face_uv.row(i) = halfedge_uv[h];
        h = halfedge.n[h];
    }

    return face_uv;
}

// experimental
Eigen::Vector2d laplacian_direction(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    int halfedge_index)
{
    Eigen::Vector2d total_diff = Eigen::Vector2d::Zero();
    int count = 0;
    int hij = halfedge_index;
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    do
    {
        int hji = halfedge.prev[hij];
        Eigen::Vector2d vj = uv[hij];
        Eigen::Vector2d vi = uv[hji];
        Eigen::Vector2d Rd = R * (vi - vj);
        total_diff += Rd;
        ++count;

        int hjk = halfedge.n[hij];
        R = R * infer_rotation(halfedge, uv, hjk);
        hij = halfedge.opp[hjk];
    } while (hij != halfedge_index);

    return total_diff / count;
}

void MeshImprover::smooth_vertex(
    int vertex_index
)
{
    int vi = vertex_index; 

    double orig_quality= compute_vertex_quality(halfedge, halfedge_uv, vi, true);
    spdlog::debug("original quality is {}", orig_quality);

    // get search direction for mesh improvement
    int hji = halfedge.opp[halfedge.out[vi]];
    Eigen::Vector2d d = laplacian_direction(halfedge, halfedge_uv, hji);
    spdlog::debug("displacing by {}", d);

    // try to move the full line step
    Eigen::Vector2d p = halfedge_uv[hji];
    Eigen::Vector2d pn = p + d;
    set_halfedge_tip_position(halfedge, halfedge_uv, hji, pn);
    double quality = compute_vertex_quality(halfedge, halfedge_uv, vi, true);
    spdlog::debug("new quality is {}", compute_vertex_quality(halfedge, halfedge_uv, vi, true));

    // do backtracking line search to make sure no triangles are flipped
    int count = 1;
    while ((is_one_ring_flipped(halfedge, halfedge_uv, vi)) || (quality > orig_quality))
    {
        if (count >= 5)
        {
            set_halfedge_tip_position(halfedge, halfedge_uv, hji, p);
            break;
        }

        spdlog::debug("backtracking attempt {}", count);
        d /= 2.;
        pn = p + d;
        set_halfedge_tip_position(halfedge, halfedge_uv, hji, pn);

        quality = compute_vertex_quality(halfedge, halfedge_uv, vi, true);
        spdlog::debug("new quality is {}", compute_vertex_quality(halfedge, halfedge_uv, vi, true));


        if (count >= 5)
        {
            view_vertex_adjacent_faces(halfedge, rows_to_matrix(halfedge_uv), vi);
        }

        ++count;
    }

    quality = compute_vertex_quality(halfedge, halfedge_uv, vi, true);
    spdlog::debug("final quality is {}\n", compute_vertex_quality(halfedge, halfedge_uv, vi, true));

    if ((quality > 1e10) || (quality > orig_quality + 1e-8))
    {
        transform_vertex_adjacent_faces(vi);
        view_vertex_adjacent_faces(halfedge, rows_to_matrix(halfedge_uv), vi);
    }
    assert(quality <= orig_quality + 1e-8);
}

// experimental
std::tuple<double, Eigen::Vector2d> symmetric_dirichlet_direction(
    const HalfedgeMesh& halfedge,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<Eigen::Vector3d>& V,
    int vertex_index,
    int norm_p=1)
{
    // build local patch
    auto [V_l, F_l, uv_l] = build_local_chart(halfedge, uv, V, vertex_index);

    // get gradient from local patch
    Eigen::SparseMatrix<double> G;
    SymDir::get_grad_op(V_l, F_l, G);

    Eigen::VectorXd area;
    igl::doublearea(V_l, F_l, area);

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> hessian;
    bool get_hessian = true;
    double energy = SymDir::get_grad_and_hessian(G, area, uv_l, grad, hessian, get_hessian, norm_p, false);

    int N = uv_l.rows();
    Eigen::Matrix2d H;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            H(i, j) = hessian.coeff(N * i, N * j);
        }
    }
    Eigen::Vector2d gradient = { grad(0), grad(N) };
    spdlog::debug("energy is {}", energy);
    spdlog::debug("gradient is {}", gradient);
    spdlog::debug("hessian is \n{}", H);

    Eigen::Vector2d d = -H.inverse() * gradient;
    spdlog::debug("direction is {}", d);

    return std::make_tuple(energy, d);
}

}
}

#ifdef USE_UV_OPTIMIZATION
#endif
