#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/forms.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace Penner {
namespace Holonomy {

// TODO Refactor this and mesh class

/**
 * @brief Class to represent a mesh with a Penner similarity structure
 */
class SimilarityPennerConeMetric : public MarkedPennerConeMetric 
{
public:
    SimilarityPennerConeMetric(
        const Mesh<Scalar>& m,
        const VectorX& metric_coords,
        const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
        const std::vector<Scalar>& kappa,
        const VectorX& harmonic_form_coords);

    SimilarityPennerConeMetric(
        const Mesh<Scalar>& m,
        const VectorX& reduced_metric_coords,
        const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
        const std::vector<Scalar>& kappa);

    // Metric access methods
    VectorX get_reduced_metric_coordinates() const override;
    void get_corner_angles(VectorX& he2angle, VectorX& he2cot) const override;

    // Flip method
    bool flip_ccw(int _h, bool Ptolemy = true) override;

    // Metric change methods
    std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override
    {
        return std::make_unique<SimilarityPennerConeMetric>(SimilarityPennerConeMetric(*this));
    }
    std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& reduced_metric_coords) const override;
    std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override;

    // Constraint methods
    bool constraint(
        VectorX& constraint,
        MatrixX& J_constraint,
        bool need_jacobian,
        bool only_free_vertices) const override;
    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<Optimization::ProjectionParameters> proj_params =
            nullptr) const override;

    // Discrete metric methods
    void make_discrete_metric() override;

    // One form getters and setters
    void set_one_form(const VectorX& one_form)
    {
        assert(is_closed_one_form(*this, one_form));
        m_one_form = one_form;
    }
    void set_one_form_direction(const VectorX& one_form_direction)
    {
        assert(is_closed_one_form(*this, one_form_direction));
        m_one_form_direction = one_form_direction;
    }
    const VectorX& get_one_form() const { return m_one_form; }
    const VectorX& get_one_form_direction() const { return m_one_form_direction; }

    std::tuple<VectorX, VectorX, std::vector<bool>> get_integrated_metric_coordinates(
        std::vector<bool> cut_h = {}) const;

    VectorX reduce_one_form(const VectorX& one_form) const;

    SimilarityPennerConeMetric scale_by_one_form() const;

    void make_delaunay(std::vector<int>& flip_seq);

    void separate_coordinates(
        const VectorX& reduced_metric_coords,
        VectorX& metric_coords,
        VectorX& harmonic_form_coords) const;

private:
    VectorX m_harmonic_form_coords;
    VectorX m_one_form;
    VectorX m_one_form_direction;
};

void similarity_corner_angles(
   const SimilarityPennerConeMetric& similarity_metric,
   VectorX& he2angle,
   VectorX& he2cot);

void MakeSimilarityDelaunay(
    SimilarityPennerConeMetric& m,
    DelaunayStats& delaunay_stats,
    SolveStats<Scalar>& solve_stats,
    bool Ptolemy = true);

} // namespace Holonomy
} // namespace Penner