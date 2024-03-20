#pragma once

#include <memory>

#include "common.hh"
#include "embedding.hh"
#include "flip_matrix_generator.hh"

namespace CurvatureMetric {

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

    MatrixX get_flip_jacobian() const { return m_transition_jacobian_lol.build_matrix(); }

    // Reset Jacobian
    void reset()
    {
        // Initialize jacobian to the identity
        m_transition_jacobian_lol.reset();
        m_flip_seq.clear();
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
