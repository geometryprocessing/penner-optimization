#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#include "spdlog/spdlog.h"

#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include "conformal_ideal_delaunay/globals.hh"

namespace Penner {
using namespace OverlayProblem;

#ifdef MULTIPRECISION
#include <unsupported/Eigen/MPRealSupport>
#include "mpreal.h"
typedef mpfr::mpreal Scalar;
#else
typedef double Scalar;
#endif

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<Scalar> MatrixX;

typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 2, 2> Matrix2x2;

typedef Eigen::Triplet<Scalar> T;


using std::max;
using std::min;
using std::isnan;

const Scalar INF = 1e10;

/// Swap two doubles.
///
/// @param[in, out] a: first double to swap
/// @param[in, out] b: second double to swap
inline void swap(double& a, double& b)
{
    std::swap<double>(a, b);
}

/// Get the max of two doubles.
///
/// @param[in] a: first double to max
/// @param[in] b: second double to max
/// @return max of a and b
//inline double max(const double& a, const double& b)
//{
//    return std::max(a, b);
//}

/// Check if two values are equal, up to a tolerance.
///
/// @param[in] a: first value to compare
/// @param[in] b: second value to compare
/// @param[in] eps: tolerance for equality
/// @return true iff |a - b| < eps
template <typename FloatScalar>
bool float_equal(FloatScalar a, FloatScalar b, FloatScalar eps = 1e-10)
{
    return (abs(a - b) < eps);
}

/// Create a vector with values 0,1,...,n-1
///
/// @param[in] n: size of the output vector
/// @param[out] vec: output arangement vector
inline void arange(size_t n, std::vector<int>& vec)
{
    vec.resize(n);
    std::iota(vec.begin(), vec.end(), 0);
}

template <typename OldScalar, typename NewScalar>
Mesh<NewScalar> change_mesh_type(const Mesh<OldScalar>& m)
{
    Mesh<NewScalar> _m;
    _m.n = m.n;
    _m.to = m.to;
    _m.f = m.f;
    _m.h = m.h;
    _m.out = m.out;
    _m.opp = m.opp;
    _m.type = m.type;
    _m.type_input = m.type_input;
    _m.R = m.R;
    _m.v_rep = m.v_rep;
    _m.fixed_dof = m.fixed_dof;
    _m.pt_in_f = m.pt_in_f;

    int num_pts = m.pts.size();
    _m.pts.resize(num_pts);
    for (int i = 0; i < num_pts; ++i)
    {
        _m.pts[i].f_id = m.pts[i].f_id;
        for (int j = 0; j < 3; ++j)
        {
            _m.pts[i].bc[j] = (NewScalar)(m.pts[i].bc[j]);
        }
    }

    int num_halfedges = m.l.size();
    _m.l.resize(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        _m.l[hij] = (NewScalar)(m.l[hij]);
    }

    int num_vertices = m.Th_hat.size();
    _m.Th_hat.resize(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        _m.Th_hat[vi] = (NewScalar)(m.Th_hat[vi]);
    }

    return _m;
}


} // namespace Penner