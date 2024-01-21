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

}
