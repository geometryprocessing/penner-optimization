#pragma once

#include <memory>

#include "common.hh"
#include "embedding.hh"

namespace CurvatureMetric {


class FlipMapMatrixGenerator {
public:
FlipMapMatrixGenerator(int size)
: m_size(size) {
    reset();

}

void reset() {
    m_list_of_lists =
        std::vector<std::map<int, Scalar>>(m_size, std::map<int, Scalar>());
    for (int i = 0; i < m_size; ++i) {
        m_list_of_lists[i][i] = 1.0;
    }
}

void multiply_by_matrix(
    const std::vector<int>& matrix_indices,
    const std::vector<Scalar>& matrix_scalars,
    int ed
) {
  // Compute the new row of J_del corresponding to edge ed, which is the only
  // edge that changes
  std::map<int, Scalar> J_del_d_new;
  for (int i = 0; i < 5; ++i) {
    int ei = matrix_indices[i];
    Scalar Di = matrix_scalars[i];
    for (auto it : m_list_of_lists[ei]) {
      J_del_d_new[it.first] += Di * it.second;
    }
  }
  m_list_of_lists[ed] = J_del_d_new;
}

MatrixX build_matrix() const {
    // Build triplets from list of lists
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(5 * m_size);
    for (int i = 0; i < m_size; ++i) {
      for (auto it : m_list_of_lists[i]) {
        tripletList.push_back(T(i, it.first, it.second));
      }
    }

    // Create the matrix from the triplets
    MatrixX matrix;
    matrix.resize(m_size, m_size);
    matrix.reserve(tripletList.size());
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return matrix;
}

private:
    int m_size;
    std::vector<std::map<int, Scalar>> m_list_of_lists;
};

class FlipMatrixGenerator
{
public:
FlipMatrixGenerator(int size)
: m_size(size) {
    reset();

}

void reset() {
    m_list_of_lists =
        std::vector<std::vector<std::pair<int, Scalar>>>(m_size, std::vector<std::pair<int, Scalar>>());
    for (int i = 0; i < m_size; ++i) {
        m_list_of_lists[i].push_back(std::make_pair(i, 1.0));
    }
}

void multiply_by_matrix(
    const std::vector<int>& matrix_indices,
    const std::vector<Scalar>& matrix_scalars,
    int ed
) {
    int num_entries = 0;
    for (int i = 0; i < 5; ++i) {
        int ei = matrix_indices[i];
        num_entries += m_list_of_lists[ei].size();
    }

    // Compute the new row of J_del corresponding to edge ed, which is the only
    // edge that changes
    std::vector<std::pair<int, Scalar>> J_del_d_new;
    J_del_d_new.reserve(num_entries);
    for (int i = 0; i < 5; ++i) {
        int ei = matrix_indices[i];
        Scalar Di = matrix_scalars[i];
        for (auto it : m_list_of_lists[ei]) {
            J_del_d_new.push_back(std::make_pair(it.first, Di * it.second));
        }
    }
    std::sort(J_del_d_new.begin(), J_del_d_new.end());

    // Compress vector
    m_list_of_lists[ed] = std::vector<std::pair<int, Scalar>>();
    m_list_of_lists[ed].reserve(num_entries);
    int index = -1;
    Scalar value = 0.0;
    for (const auto& entry : J_del_d_new)
    {
        if (index != entry.first)
        {
            if (index >= 0)
            {
                m_list_of_lists[ed].push_back(std::make_pair(index, value));
            }
            index = entry.first;
            value = entry.second;
        }
        else
        {
            value += entry.second;
        }
    }
    m_list_of_lists[ed].push_back(std::make_pair(index, value));
}

MatrixX build_matrix() const {
    // Build triplets from list of lists
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(5 * m_size);
    for (int i = 0; i < m_size; ++i) {
        for (auto it : m_list_of_lists[i]) {
            tripletList.push_back(T(i, it.first, it.second));
        }
    }

    // Create the matrix from the triplets
    MatrixX matrix;
    matrix.resize(m_size, m_size);
    matrix.reserve(tripletList.size());
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return matrix;
}

private:
    int m_size;
    std::vector<std::vector<std::pair<int, Scalar>>> m_list_of_lists;
};

class DifferentiableConeMetric : public Mesh<Scalar>
{
public:
    DifferentiableConeMetric(const Mesh<Scalar>& m);

    std::vector<int> he2e; // map from halfedge to edge
    std::vector<int> e2he; // map from edge to halfedge

    bool is_discrete_metric() const { return m_is_discrete_metric; };
    int num_flips() { return m_flip_seq.size(); };
    const std::vector<int>& get_flip_sequence() const { return m_flip_seq; }

    // Metric access methods: need to access metric coordinates and corner angles
    virtual VectorX get_metric_coordinates() const;
    virtual VectorX get_reduced_metric_coordinates() const = 0;
    virtual void get_corner_angles(VectorX& he2angle, VectorX& he2cot) const;

    // Flip method: need to be able to flip metric
    virtual bool flip_ccw(int _h, bool Ptolemy = true);
    void undo_flips();

    // Metric change methods: need to be able to clone metric and change metric coordinates
    virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const = 0;
    virtual std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const = 0;
    virtual std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const = 0;

    // Constraint methods: need to have a differentiable constraint and method to project to it
    virtual bool constraint(
        VectorX& constraint,
        MatrixX& J_constraint,
        bool need_jacobian = true,
        bool only_free_vertices = true) const;
    virtual std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const = 0;
    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const;

    // Discrete metric methods: need differentiable method to flip to a discrete metric
    virtual void make_discrete_metric() = 0;
    virtual MatrixX get_transition_jacobian() const = 0;

    int n_reduced_coordinates() const;

    MatrixX change_metric_to_reduced_coordinates(
        const std::vector<int>& I,
        const std::vector<int>& J,
        const std::vector<Scalar>& V,
        int num_rows) const;
    MatrixX change_metric_to_reduced_coordinates(
        const std::vector<Eigen::Triplet<Scalar>>& tripletList,
        int num_rows) const;
    MatrixX change_metric_to_reduced_coordinates(const MatrixX& halfedge_jacobian) const;

    virtual ~DifferentiableConeMetric() = default;

protected:
    bool m_is_discrete_metric;
    std::vector<int> m_flip_seq;
    MatrixX m_identification;
};

class PennerConeMetric : public DifferentiableConeMetric
{
public:
    PennerConeMetric(const Mesh<Scalar>& m, const VectorX& metric_coords);

    std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override;

    std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const override;
    VectorX get_reduced_metric_coordinates() const override;

    std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override;

    MatrixX get_transition_jacobian() const override;

    void make_discrete_metric() override;

    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const override;

    bool flip_ccw(int _h, bool Ptolemy = true) override;

    MatrixX get_flip_jacobian() const
    {
        return m_transition_jacobian_lol.build_matrix();
    }

    // Reset Jacobian
    void reset()
    {
        // Initialize jacobian to the identity
        m_transition_jacobian_lol.reset();
    }

protected:
    std::vector<int> m_embed;
    std::vector<int> m_proj;
    MatrixX m_projection;
    bool m_need_jacobian;
    FlipMatrixGenerator m_transition_jacobian_lol;

    VectorX reduce_metric_coordinates(const VectorX& metric_coords) const;
    void expand_metric_coordinates(const VectorX& metric_coords);
};

class DiscreteMetric : public DifferentiableConeMetric
{
public:
    DiscreteMetric(const Mesh<Scalar>& m, const VectorX& log_length_coords);


    std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override;

    std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const override;
    VectorX get_reduced_metric_coordinates() const override;

    std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override;

    MatrixX get_transition_jacobian() const override;

    void make_discrete_metric() override;

    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const override;

    bool flip_ccw(int _h, bool Ptolemy = true) override;

protected:
    std::vector<int> m_embed;
    std::vector<int> m_proj;
    MatrixX m_projection;

    VectorX reduce_metric_coordinates(const VectorX& metric_coords) const;
    void expand_metric_coordinates(const VectorX& metric_coords);
};

} // namespace CurvatureMetric
