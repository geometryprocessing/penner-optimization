#include "holonomy/core/common.h"

namespace PennerHolonomy {

int compute_euler_characteristic(const Mesh<Scalar>& m)
{
    int n_v = m.n_vertices();
    int n_e = m.n_edges();
    int n_f = m.n_faces();
    return n_v - n_e + n_f;
}

int compute_genus(const Mesh<Scalar>& m)
{
    int euler_characteristic = compute_euler_characteristic(m);
    return (2 - euler_characteristic) / 2;
}

Matrix2x2 compute_rotation(Scalar theta)
{
    Matrix2x2 R;
    R.row(0) << cos(theta), -sin(theta);
    R.row(1) << sin(theta), cos(theta);
    
    return R;
}

MatrixX solve_linear_system(const MatrixX& A, const MatrixX&B)
{
#ifdef WITH_MPFR
    Eigen::SparseQR<MatrixX, Eigen::COLAMDOrdering<int>> solver;
#else
    Eigen::SparseLU<MatrixX> solver;
#endif

    solver.compute(A);
    return solver.solve(B);
}

Eigen::SparseMatrix<int> compute_vv_to_halfedge_matrix(const Mesh<Scalar>& m)
{
    // Create the adjacency matrix (tail, head) -> halfedge index+1
    int n_v = m.n_ind_vertices();
    int n_he = m.n_halfedges();
    Eigen::SparseMatrix<int> vv2he(n_v, n_v);
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    trips.reserve(n_he);
    for (int hij = 0; hij < n_he; ++hij)
    {
        if (m.type[hij] > 1) continue; // only use primal halfedges
        int vi = m.v_rep[m.to[m.opp[hij]]];
        int vj = m.v_rep[m.to[hij]];
        trips.push_back(Trip(vi, vj, hij + 1));
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}

} // namespace PennerHolonomy
