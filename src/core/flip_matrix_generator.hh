#pragma once

#include "common.hh"

namespace CurvatureMetric {

class FlipMapMatrixGenerator {
public:
FlipMapMatrixGenerator(int size);

void reset();

void multiply_by_matrix(
    const std::vector<int>& matrix_indices,
    const std::vector<Scalar>& matrix_scalars,
    int ed
);

MatrixX build_matrix() const;

private:
    int m_size;
    std::vector<std::map<int, Scalar>> m_list_of_lists;
};

class FlipMatrixGenerator
{
public:
FlipMatrixGenerator(int size);

void reset();

void multiply_by_matrix(
    const std::vector<int>& matrix_indices,
    const std::vector<Scalar>& matrix_scalars,
    int ed
);
 
MatrixX build_matrix() const;

private:
    int m_size;
    std::vector<std::vector<std::pair<int, Scalar>>> m_list_of_lists;
};

} // namespace CurvatureMetric
