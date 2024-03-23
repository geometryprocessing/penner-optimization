#pragma once

#include "common.hh"

namespace CurvatureMetric {

/// Class for incrementally building the flip change of coordinate matrix
class FlipMatrixGenerator
{
public:
    /// Initialize an identity matrix
    ///
    /// @param[in] size: dimension of the flip matrix
    FlipMatrixGenerator(int size);

    /// Reset the matrix to the identity
    void reset();

    /// Multiply the current matrix by a flip matrix Di for a given edge on the left.
    ///
    /// The flip matrix is the identity except the row ed, which has the values indicated
    /// by the (ed, j, v) specified by matrix indices and scalars
    ///
    /// @param[in] column_indices: column indices j for row ed of the matrix Di
    /// @param[in] values: values v for row ed of the matrix Di
    /// @param[in] ed: edge index for the nontrivial row of the matrix Di
    void multiply_by_matrix(
        const std::vector<int>& column_indices,
        const std::vector<Scalar>& values,
        int ed);

    /// Generate a standard sparse representation of the current matrix
    ///
    /// @return flip matrix
    MatrixX build_matrix() const;

private:
    int m_size;
    std::vector<std::vector<std::pair<int, Scalar>>> m_list_of_lists;
};

/// Class for incrementally building the flip change of coordinate matrix using a standard template
/// library map representation
///
/// NOTE: Potential alternative representation of the flip matrix generator.
class FlipMapMatrixGenerator
{
public:
    /// Initialize an identity matrix
    ///
    /// @param[in] size: dimension of the flip matrix
    FlipMapMatrixGenerator(int size);

    /// Reset the matrix to the identity
    void reset();

    /// Multiply the current matrix by a flip matrix Di for a given edge on the left.
    ///
    /// The flip matrix is the identity except the row ed, which has the values indicated
    /// by the (ed, j, v) specified by matrix indices and scalars
    ///
    /// @param[in] column_indices: column indices j for row ed of the matrix Di
    /// @param[in] values: values v for row ed of the matrix Di
    /// @param[in] ed: edge index for the nontrivial row of the matrix Di
    void multiply_by_matrix(
        const std::vector<int>& column_indices,
        const std::vector<Scalar>& values,
        int ed);

    /// Generate a standard sparse representation of the current matrix
    ///
    /// @return flip matrix
    MatrixX build_matrix() const;

private:
    int m_size;
    std::vector<std::map<int, Scalar>> m_list_of_lists;
};

} // namespace CurvatureMetric
