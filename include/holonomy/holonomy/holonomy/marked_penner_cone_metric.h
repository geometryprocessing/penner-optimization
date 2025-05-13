#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/core/homology_basis.h"

#include "optimization/core/cone_metric.h"

namespace Penner {
namespace Holonomy {

// TODO Refactor this and cone metric for more minimal inheritance

/**
 * @brief Check if a mesh is valid
 * 
 * @param m: mesh to check
 * @return true if the mesh is valid
 * @return false otherwise
 */
bool is_valid_mesh(const Mesh<Scalar>& m);

/**
 * @brief Class to represent a mesh with a Penner metric and homology basis markings
 */
class MarkedPennerConeMetric : public Optimization::PennerConeMetric
{
public:
    // Additional constraints for homology loops
    std::vector<Scalar> kappa_hat;

    // TODO move
    VectorX original_coords;

    /**
     * @brief Construct a new Marked Penner Cone Metric object with given metric coordinates
     * and dual loop markings with holonomy constraints.
     *
     * @param m: mesh connectivity
     * @param metric_coords: initial metric coordinates
     * @param homology_basis_loops: homology basis loops for the surface
     * @param kappa: holonomy constraints on the basis loops
     */
    MarkedPennerConeMetric(
        const Mesh<Scalar>& m,
        const VectorX& metric_coords,
        const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
        const std::vector<Scalar>& kappa);

    MarkedPennerConeMetric(const MarkedPennerConeMetric& marked_metric);
    void operator=(const MarkedPennerConeMetric& m);

    /**
     * @brief Reset the connectivity and dual loops to that of another mesh of the same size.
     *
     * @param m: mesh with the same element counts as the current mesh
     */
    void reset_marked_metric(const MarkedPennerConeMetric& m);

    /**
     * @brief Change the metric of the given mesh given new coordinates on the original
     * connectivity.
     *
     * The new metric is assumed to be defined on the same initial connectivity as the current
     * metric but with potentially new metric coordinates.
     *
     * TODO: Move to base class
     *
     * @param m: mesh used to initialize the current mesh
     * @param metric_coords: new metric coordinates
     * @param need_jacobian: (optional) track change of metric jacobian if true
     * @param do_repeat_flips: (optional) repeat flips to restore current connectivity if true
     */
    virtual void change_metric(
        const MarkedPennerConeMetric& m,
        const VectorX& metric_coords,
        bool need_jacobian = true,
        bool do_repeat_flips = false);

    /**
     * @brief Get number of homology basis loops
     *
     * @return homology basis loop count
     */
    int n_homology_basis_loops() const { return m_homology_basis_loops.size(); }

    /**
     * @brief Get the homology basis loops
     *
     * @return const reference to the homology basis loops
     */
    const std::vector<std::unique_ptr<DualLoop>>& get_homology_basis_loops() const
    {
        return m_homology_basis_loops;
    }

    ///////////////////////////
    // Virtual Method Overrides
    ///////////////////////////

    /**
     * @brief Clone the differentiable cone metric
     *
     * @return pointer to a copy of the cone metric
     */
    virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override
    {
        return std::make_unique<MarkedPennerConeMetric>(MarkedPennerConeMetric(*this));
    }

    /**
     * @brief Clone the differentiable cone metric with new metric coordinates
     *
     * TODO: Make this non-virtual and implement with change metric
     *
     * @return pointer to a copy of the cone metric with new metric coordinates
     */
    virtual std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& reduced_metric_coords) const override;

    // TODO: Remove scale conformally from cone metric interface

    // TODO Use full holonomy constraints
    virtual bool constraint(
        VectorX& constraint,
        MatrixX& J_constraint,
        bool need_jacobian,
        bool only_free_vertices) const override;

    // TODO Use Newton
    virtual std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<Optimization::ProjectionParameters> proj_params =
            nullptr) const override;

    // Flip method
    virtual bool flip_ccw(int _h, bool Ptolemy = true) override;

    virtual VectorX constraint(const VectorX& angles);

    virtual MatrixX constraint_jacobian(const VectorX& cotangents);

    Scalar max_constraint_error() const;

    virtual std::unique_ptr<MarkedPennerConeMetric> clone_marked_metric() const
    {
        return std::make_unique<MarkedPennerConeMetric>(MarkedPennerConeMetric(*this));
    }

    virtual void write_status_log(std::ostream& stream, bool write_header=false);

protected:
    std::vector<std::unique_ptr<DualLoop>> m_homology_basis_loops;
    DualLoopManager m_dual_loop_manager;
    void reset_connectivity(const MarkedPennerConeMetric& m);
    void reset_markings(const MarkedPennerConeMetric& m);
    void copy_connectivity(const MarkedPennerConeMetric& m);
    void copy_metric(const MarkedPennerConeMetric& m);
    void copy_holonomy(const MarkedPennerConeMetric& m);
};


} // namespace Holonomy
} // namespace Penner