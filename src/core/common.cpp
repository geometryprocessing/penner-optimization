#include "common.hh"

namespace CurvatureMetric {

bool vector_equal(VectorX v, VectorX w, Scalar eps)
{
    // Check if the sizes are the same
    if (v.size() != w.size()) return false;

    // Check per element equality
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (!float_equal(v[i], w[i], eps)) return false;
    }

    // Equal otherwise
    return true;
}

bool matrix_contains_nan(const Eigen::MatrixXd& mat)
{
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            if (std::isnan(mat(i, j))) return true;
        }
    }

    return false;
}

Scalar sup_norm(const VectorX& v)
{
    Scalar norm_value = 0.0;
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        norm_value = max(norm_value, abs(v[i]));
    }

    return norm_value;
}

Scalar matrix_sup_norm(const MatrixX& matrix)
{
    // Check for trivial matrices
    if (matrix.size() == 0) return 0;

    // Iterate to determine maximum abs value
    Scalar max_value = 0.0;
    for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
            max_value = std::max(max_value, abs(it.value()));
        }
    }

    return max_value;
}

} // namespace CurvatureMetric
