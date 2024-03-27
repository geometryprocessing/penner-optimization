/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
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
