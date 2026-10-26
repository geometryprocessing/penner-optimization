#include "resolution/core/common.h"
#include "feature/core/common.h"
#include "field/frame_field.h"
#include "field/cones.h"
#include "holonomy/core/viewer.h"

#include <igl/bounding_box_diagonal.h>
#include <igl/per_face_normals.h>

namespace Penner {
namespace Resolution {



Eigen::SparseMatrix<int> generate_VV_to_halfedge_map(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex)
{

    // generate IJV representation of the matrix
    int n_he = m.n_halfedges();
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    trips.reserve(n_he);
    for (int hij = 0; hij < n_he; ++hij) {
        // only process primal mesh
        if (m.type[hij] == 2) continue;

        // get edge endpoints in VF mesh indexing
        int hji = m.opp[hij];
        int vi = vtx_reindex[m.v_rep[m.to[hji]]];
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];

        // add entry mapping (vi, vj) to the one-indexed halfedge index
        trips.push_back(Trip(vi, vj, hij + 1));
    }

    // convert IJV to sparse matrix
    int n_v = m.n_ind_vertices();
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> transfer_corner_function_to_halfedge(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& corner_func)
{
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(m, vtx_reindex);
    int num_faces = F.rows();
    Eigen::Matrix<T, Eigen::Dynamic, 1> halfedge_func(m.n_halfedges());
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;
            int vj = F(fijk, j);
            int vk = F(fijk, k);
            int hjk = vv2he.coeff(vj, vk) - 1;
            halfedge_func[hjk] = corner_func(fijk, i);
            if (m.type[hjk] > 0)
            {
                halfedge_func[m.R[hjk]] = halfedge_func[hjk];
            }
        }
    }

    return halfedge_func;
}

std::tuple<VectorX, VectorX> compute_seamless_error(
    const Mesh<Scalar> &m,
    const Eigen::MatrixXd &uv)
{
    // Iterate over edges to check the length inconsistencies
    int num_halfedges = m.n_halfedges();
    VectorX uv_length_error = VectorX::Zero(num_halfedges);
    VectorX uv_angle_error = VectorX::Zero(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        int hki = m.n[m.n[hij]];
        int hji = m.opp[hij];
        int hlj = m.n[m.n[hji]];

        // Compute the length of each halfedge corresponding to the corner in the cut mesh
        Eigen::Vector2d uv_00 = uv.row(hij);
        Eigen::Vector2d uv_01 = uv.row(hki);
        Eigen::Vector2d uv_10 = uv.row(hlj);
        Eigen::Vector2d uv_11 = uv.row(hji);
        Scalar l0 = Holonomy::compute_uv_length(uv_00, uv_01);
        Scalar l1 = Holonomy::compute_uv_length(uv_10, uv_11);
        Scalar angle = Holonomy::uv_angle(uv_01 - uv_00, uv_11 - uv_10);
        Scalar length_error = abs(l0 - l1);
        Scalar angle_error = pos_fmod(angle + (4 * M_PI), M_PI / 2.);
        angle_error = min(angle_error, (M_PI / 2.) - angle_error);
        spdlog::trace("angle of {}: {}", hij, angle);
        spdlog::trace("angle error: {}", angle_error);
        spdlog::trace("length error: {}", length_error);

        // set length error for the given edge
        uv_length_error(hij) = length_error;
        uv_angle_error(hij) = angle_error;
    }

    return std::make_tuple(uv_length_error, uv_angle_error);
}


int next_orig_segment(const OverlayMesh<Scalar> &m_o, int _h) 
{
    // handle degenerate cases
    // FIXME: could be unnecessary
    if (m_o.n[m_o.opp[m_o.n[_h]]] == m_o.opp[_h])
    {
        //return m_o.n[_h];
    }

    // iterate around valence 4 vertex
    //int cur = m_o.n[m_o.opp[m_o.n[_h]]];
    int cur = m_o.n[_h];

    // if original edge, keep iterating
    while (m_o.edge_type[m_o.e(cur)] == CURRENT_EDGE)
    {
        //cur = m_o.n[m_o.opp[m_o.n[cur]]];
        cur = m_o.n[m_o.opp[cur]];
    }
    return cur;
}

void remove_unreferenced_inplace(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    std::vector<int> old2new(V.rows(), -1);
    Eigen::MatrixXd V_compact(V.rows(), V.cols());
    int num_used = 0;

    for (int f = 0; f < F.rows(); ++f)
    {
        for (int i = 0; i < F.cols(); ++i)
        {
            int v_old = F(f, i);
            int& v_new = old2new[v_old];
            if (v_new == -1)
            {
                v_new = num_used++;
                V_compact.row(v_new) = V.row(v_old);
            }
            F(f, i) = v_new;
        }
    }

    V_compact.conservativeResize(num_used, V.cols());
    V = V_compact;
}

Eigen::Vector3d project_orthogonal_to(
    const Eigen::Vector3d& v,
    const Eigen::Vector3d& u)
{
    double denom = u.squaredNorm();
    if (denom == 0.0) {
        // If u is zero, the whole space is orthogonal to it
        return v;
    }

    return v - (v.dot(u) / denom) * u;
}

double compute_triangle_area(
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3)
{
    Eigen::Vector3d d12 = v2 - v1;
    Eigen::Vector3d d13 = v3 - v1;
    Eigen::Vector3d n = d12.cross(d13);
    return 0.5 * std::sqrt(n.dot(n));
}

Eigen::Matrix2d generate_ccw_rotation()
{
    Eigen::Matrix2d R;
    R.row(0) << 0, -1;
    R.row(1) << 1,  0;

    return R;
}

} // Resolution
} // Penner
