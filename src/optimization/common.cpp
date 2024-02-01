#include "common.hh"

#ifdef USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#endif

namespace CurvatureMetric {


VectorX solve_psd_system(const MatrixX& A, const VectorX&b)
{
#ifdef USE_SUITESPARSE
    Eigen::CholmodSupernodalLLT<MatrixX> solver;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
#endif

    solver.compute(A);
    return solver.solve(b);
}

VectorX solve_linear_system(const MatrixX& A, const VectorX&b)
{
#ifdef WITH_MPFR
    Eigen::SparseQR<MatrixX, Eigen::COLAMDOrdering<int>> solver;
#else
    Eigen::SparseLU<MatrixX> solver;
#endif

    solver.compute(A);
    return solver.solve(b);
}

}
