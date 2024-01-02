#pragma once

#include <memory>

#include "common.hh"
#include "embedding.hh"

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

    virtual VectorX get_metric_coordinates() const;
    virtual VectorX get_reduced_metric_coordinates() const = 0;
    virtual void get_corner_angles(VectorX& he2angle, VectorX& he2cot) const;

    MatrixX change_metric_to_reduced_coordinates(
        const std::vector<int>& I,
        const std::vector<int>& J,
        const std::vector<Scalar>& V,
        int num_rows) const;

    MatrixX change_metric_to_reduced_coordinates(const MatrixX& halfedge_jacobian) const;

    virtual ~DifferentiableConeMetric() = default;

    virtual bool flip_ccw(int _h, bool Ptolemy = true);

    virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const = 0;
    virtual std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const = 0;
    virtual std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const = 0;

    virtual void make_discrete_metric() = 0;
    virtual std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const = 0;
    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const;

    virtual MatrixX get_transition_jacobian() const = 0;

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

private:
    std::vector<int> m_embed;
    std::vector<int> m_proj;
    MatrixX m_projection;
    std::vector<std::map<int, Scalar>> m_transition_jacobian_lol;

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

private:
    std::vector<int> m_embed;
    std::vector<int> m_proj;
    MatrixX m_projection;

    VectorX reduce_metric_coordinates(const VectorX& metric_coords) const;
    void expand_metric_coordinates(const VectorX& metric_coords);
};

} // namespace CurvatureMetric
