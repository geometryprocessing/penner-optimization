#pragma once

#include "feature/core/boundary_path.h"
#include "feature/core/common.h"
#include "holonomy/core/dual_loop.h"
#include "holonomy/core/homology_basis.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "holonomy/holonomy/constraint.h"

namespace Penner {
namespace Feature {

// TODO Refactor this and cone metric for more minimal inheritance

/**
 * @brief Class to represent a mesh with a Penner metric, homology basis markings, and length
 * constraints on the boundary
 */
class DirichletPennerConeMetric : public Holonomy::MarkedPennerConeMetric
{
public:
    DirichletPennerConeMetric();

    /**
     * @brief Construct a new Dirichlet Penner Cone Metric object
     *
     * @param m: mesh topology and cone constraints
     * @param metric_coords: initial metric coordinates
     * @param homology_basis_loops: system of dual loops on the surface
     * @param kappa: target holonomy for the loops
     * @param boundary_paths: list of all constrained boundary paths on the surface
     * @param boundary_constraint_system: system of boundary path constraints
     * @param ell: target values for boundary path system
     */
    DirichletPennerConeMetric(
        const Mesh<Scalar>& m,
        const VectorX& metric_coords,
        const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
        const std::vector<Scalar>& kappa,
        const std::vector<BoundaryPath>& boundary_paths,
        const MatrixX& boundary_constraint_system,
        const VectorX& ell);

    /**
     * @brief Construct a new Dirichlet Penner Cone Metric object with relaxed angle constraints
     *
     * @param m: mesh topology and cone constraints
     * @param metric_coords: initial metric coordinates
     * @param homology_basis_loops: system of dual loops on the surface
     * @param kappa: target holonomy for the loops
     * @param boundary_paths: list of all constrained boundary paths on the surface
     * @param boundary_constraint_system: system of boundary path constraints
     * @param ell: target values for boundary path system
     * @param angle_constraint_syste: system of reduced angle constraints
     */
    DirichletPennerConeMetric(
        const Mesh<Scalar>& m,
        const VectorX& metric_coords,
        const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
        const std::vector<Scalar>& kappa,
        const std::vector<BoundaryPath>& boundary_paths,
        const MatrixX& boundary_constraint_system,
        const VectorX& ell,
        const MatrixX& angle_constraint_syste);

    /**
     * @brief Add Dirichlet conditions to a marked metric.
     *
     * @param marked_metric: metric with holonomy constraints
     * @param boundary_paths: list of all constrained boundary paths on the surface
     * @param boundary_constraint_system: system of boundary path constraints
     * @param ell: target values for boundary path system
     */
    DirichletPennerConeMetric(
        const Holonomy::MarkedPennerConeMetric& marked_metric,
        const std::vector<BoundaryPath>& boundary_paths,
        const MatrixX& boundary_constraint_system,
        const VectorX& ell);

    /**
     * @brief Copy constructor.
     *
     * @param dirichlet_metric: metric to copy
     */
    DirichletPennerConeMetric(const DirichletPennerConeMetric& dirichlet_metric);

    /**
     * @brief Assignment operator.
     *
     * @param dirichlet_metric: metric to copy
     */
    void operator=(const DirichletPennerConeMetric& dirichlet_metric);

    // constraints for boundary edge path systems
    VectorX ell_hat;

    // if true, use relaxed system for constraints
    bool use_relaxed_system;

    // flip method
    virtual bool flip_ccw(int _h, bool Ptolemy = true) override;

    // constraint overrides

    virtual bool constraint(
        VectorX& constraint,
        MatrixX& J_constraint,
        bool need_jacobian,
        bool only_free_vertices) const override;

    virtual VectorX constraint(const VectorX& angles) override;

    virtual MatrixX constraint_jacobian(const VectorX& cotangents) override;

    virtual std::unique_ptr<Holonomy::MarkedPennerConeMetric> clone_marked_metric() const override
    {
        return std::make_unique<DirichletPennerConeMetric>(DirichletPennerConeMetric(*this));
    }

    virtual std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(const VectorX& metric_coords) const override;

    virtual std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<Optimization::ProjectionParameters> proj_params) const override;

    virtual void write_status_log(std::ostream& stream, bool write_header = false) override;

    /**
     * @brief Get the number of boundary edge paths
     *
     * @return boundary path count
     */
    int n_boundary_paths() const { return m_boundary_paths.size(); }

    /**
     * @brief Get the number of boundary edge segments
     *
     * @return boundary segment count
     */
    int n_boundary_segments();

    /**
     * @brief Get the number of boundary edge constraints
     *
     * @return boundary constraint count
     */
    int n_boundary_constraints() const { return ell_hat.size(); }

    /**
     * @brief Get the boundary paths object
     * 
     * Checks for any invalidated boundary paths and rebuilds them.
     *
     * @return read-only boundary paths
     */
    const std::vector<BoundaryPath>& get_boundary_paths();

    /**
     * @brief Get a list of all starting vertices of the boundary edge paths.
     *
     * @return starting vertices for the boundary edge paths
     */
    std::vector<int> get_path_starting_vertices() const;

    /**
     * @brief Get the boundary constraint system matrix
     *
     * @return read-only boundary constraint matrix
     */
    const MatrixX& get_boundary_constraint_system() const { return m_boundary_constraint_system; }

    /**
     * @brief Set an explicit angle constraint system to use in place of per-vertex constraints.
     *
     * @param angle_constraint_system: matrix mapping vertices to angle constraints with RHS Th_hat
     */
    void set_angle_constraint_system(const MatrixX& angle_constraint_system)
    {
        m_relaxed_angle_constraint_system = angle_constraint_system;
    }

    /**
     * @brief Get the angle constraint system matrix
     *
     * @return read-only angle constraint matrix
     */
    const MatrixX& get_angle_constraint_system() const
    {
        if (use_relaxed_system)
        {
            return m_relaxed_angle_constraint_system;
        }
        else 
        {
            return m_full_angle_constraint_system;
        }
    }

    /**
     * @brief Serialize mesh data to output stream in human readable format.
     *
     * @param out: output stream
     */
    void serialize(std::ostream& output) const;
private:

    /**
     * @brief Rebuild all boundary paths in the current connectivity
     *
     */
    void reset_boundary_paths();

    /**
     * @brief Rebuild boundary path in the current connectivity with the given method
     *
     * @param bd_index: index of the boundary to update
     */
    void reset_boundary_path(int bd_index);

    void copy_feature(const DirichletPennerConeMetric& m);

    std::vector<BoundaryPath> m_boundary_paths;
    std::vector<bool> m_is_boundary_path_valid;
    std::vector<int> h2bd;
    MatrixX m_boundary_constraint_system;
    MatrixX m_full_angle_constraint_system;
    MatrixX m_relaxed_angle_constraint_system;
};


/**
 * @brief Generate boundary length consistency constraints from a cut mesh and gluing map.
 * 
 * @param marked_metric: cut metric with angle constraints
 * @param vtx_reindex: vertex reindexing from halfedge to VF mesh
 * @param V_map: identification map from cut VF vertices to the glued mesh vertices
 * @return metric with angle and length constraints
 */
DirichletPennerConeMetric generate_dirichlet_metric_from_mesh(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map);

} // namespace Feature
} // namespace Penner