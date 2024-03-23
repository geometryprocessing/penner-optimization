#pragma once

#include <memory>

#include "common.hh"
#include "embedding.hh"
#include "flip_matrix_generator.hh"

namespace CurvatureMetric {

/// Representation of a differentiable intrinsic metric on a mesh with angle constraints at cones.
///
/// This abstract representation supports coordinates for the space of intrinsic metrics on a fixed
/// topology and cone vertex set with the following two properties:
///   1. The coordinates are defined per halfedge
///   2. A change of coordinates is defined for edge flips
///   3. There exists a computable sequence of flips that make the coordinates Euclidean log edge lengths
/// The two metric coordinates with these properties are Euclidean log length coordinates (on a convex
/// domain determined by triangle inequalities) and Penner coordinates (which correspond to hyperbolic
/// length coordinates).
///
/// Given a choice of coordinates, this class also must support methods to:
///   - Change coordinates with edge flips to a discrete metric (i.e., with log edge length coordinates)
///   - Get the per-halfedge metric coordinates
///   - Get an independent vector of reduced metric coordinates
///   - Compute per-corner angles (if the metric is a discrete metric)
///   - Flip an arbitrary edge counterclockwise
///   - Clone the mesh, possibly with new halfedge coordinates or conformal scaling at vertices
///   - Computation of the cone angle constraint error with reduced coordinate Jacobian
///   - Compute a new metric satisfying the constraints
///   - Compute the change of coordinates flip Jacobian
///
class DifferentiableConeMetric : public Mesh<Scalar>
{
public:
    DifferentiableConeMetric(const Mesh<Scalar>& m);

    std::vector<int> he2e; // map from halfedge to edge
    std::vector<int> e2he; // map from edge to halfedge

    // **************************************************************************
    // Metric access methods: need to access metric coordinates and corner angles
    // **************************************************************************

    /// Get a reduced vector of independent metric coordinates
    ///
    /// The size of this vector is determined by the choice of metric representation
    ///
    /// @return reduced metric coordinate vector
    virtual VectorX get_reduced_metric_coordinates() const = 0;

    /// Get the per-halfedge metric coordinates
    ///
    /// @return per-halfedge metric coordinate vector
    virtual VectorX get_metric_coordinates() const;

    /// Compute the per-corner triangle angles and their cotangents
    ///
    /// Corner angles are indexed by the halfedge opposite the corner.
    ///
    /// @param[out] he2angle: per-corner angle vector
    /// @param[out] he2cot: per-corner angle cotangent vector
    virtual void get_corner_angles(VectorX& he2angle, VectorX& he2cot) const;

    // **********************************************************
    // Flip method: need to be able to flip edge counterclockwise
    // **********************************************************

    /// Flip a halfedge with a given index counterclockwise.
    ///
    /// @param[in] halfedge_index: halfedge to flip counterclockwise
    virtual bool flip_ccw(int halfedge_index, bool Ptolemy=true) = 0;

    // ****************************************************************************
    // Clone methods: need to be able to clone metric and change metric coordinates
    // ****************************************************************************

    /// Clone the current metric exactly as is
    ///
    /// @return abstract base class pointer to a clone of the current metric
    virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const = 0;

    /// Clone the current mesh with new metric coordinates
    ///
    /// @param[in] metric_coords: new metric coordinates for the mesh
    /// @return abstract base class pointer to a clone of the current mesh with new coordinates
    virtual std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const = 0;

    /// Clone the current mesh with new conformally equivalent metric coordinates determined by scale factors
    ///
    /// @param[in] u: per-vertex scale factors
    /// @return abstract base class pointer to a clone of the current mesh with new coordinates obtained
    ///     by conformal scaling
    virtual std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const = 0;

    // ****************************************************************************************
    // Constraint methods: need to have a differentiable constraint and method to project to it
    // ****************************************************************************************

    /// Project the metric to the constraints Th_hat, conformally or otherwise.
    ///
    /// @param[out] solve_stats: (optional) statistics for the projection method
    /// @param[in] proj_params: (optional) parameters for the projection method
    /// @return pointer to abstract base class with new constrained metric
    virtual std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const = 0;
    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const;

    /// Compute the current differentiable discrepancy between the vertex cone angles and the target angles.
    ///
    /// @param[out] constraint: per-vertex cone angle constraint error (theta - Th_hat)
    /// @param[out] J_constraint: optional Jacobian of the constraint with respect to reduced metric coordinates
    /// @param[in] need_jacobian: (optional) if true, compute the Jacobian
    /// @param[in] only_free_vertices: (optional) if true, only compute the angle error at vertices that are not
    ///     marked as fixed
    virtual bool constraint(
        VectorX& constraint,
        MatrixX& J_constraint,
        bool need_jacobian = true,
        bool only_free_vertices = true) const;

    // ********************************************************************************
    // Discrete metric methods: need differentiable method to flip to a discrete metric
    // ********************************************************************************

    /// Flip the mesh to a connectivity where the metric coordinates are discrete, meaning the
    /// metric coordinates are Euclidean log edge lengths (scaled by 2 for Penner theory reasons).
    virtual void make_discrete_metric() = 0;

    /// Get the change of coordinates Jacobian matrix for the per-halfedge metric coordinates in
    /// the current connectivity in terms of the reduced coordinates for the original connectivity.
    ///
    /// This matrix 
    ///
    /// @return change of coordinates Jacobian matrix 
    virtual MatrixX get_transition_jacobian() const = 0;

    // **********************************************************************
    // General methods: non-virtual methods that don't need to be overwritten
    // **********************************************************************

    /// Determine if the mesh is currently a discrete metric, meaning the metric coordinates are
    /// Euclidean log edge lengths (scaled by 2 for Penner theory reasons).
    ///
    /// @return true if the metric coordinates correspond to a discrete metric
    bool is_discrete_metric() const { return m_is_discrete_metric; };

    /// Determine how many flips have been performed since the metric was initialized
    ///
    /// @return number of flips
    int num_flips() { return m_flip_seq.size(); };

    /// Get the full sequence of flips that have been performed since the metric was initialized
    ///
    /// @return sequence of edge flip indices
    const std::vector<int>& get_flip_sequence() const { return m_flip_seq; }

    /// Undo all flips that have occurred since the metric was initialized
    void undo_flips();

    /// Get the number of reduced independent metric coordinates
    ///
    /// @return number of reduced coordinates
    int n_reduced_coordinates() const;

    /// Convert a Jacobian matrix in terms of the per-halfedge metric coordinates of the current mesh
    /// to a Jacobian in terms of the reduced metric coordinates of the initial metric connectivity.
    ///
    /// The input matrix is assumed to be in IJV format.
    ///
    /// @param[in] I: vector of row indices
    /// @param[in] J: vector of column indices
    /// @param[in] V: vector of values
    /// @param[in] num_rows: number of rows of the Jacobian
    /// @return Jacobian in reduced coordinates for the original mesh
    MatrixX change_metric_to_reduced_coordinates(
        const std::vector<int>& I,
        const std::vector<int>& J,
        const std::vector<Scalar>& V,
        int num_rows) const;

    /// Convert a Jacobian matrix in terms of the per-halfedge metric coordinates of the current mesh
    /// to a Jacobian in terms of the reduced metric coordinates of the initial metric connectivity.
    ///
    /// The input matrix is assumed to be in IJV format.
    ///
    /// @param[in] tripletList: vector of ijv triplets
    /// @param[in] num_rows: number of rows of the Jacobian
    /// @return Jacobian in reduced coordinates for the original mesh
    MatrixX change_metric_to_reduced_coordinates(
        const std::vector<Eigen::Triplet<Scalar>>& tripletList,
        int num_rows) const;

    /// Convert a Jacobian matrix in terms of the per-halfedge metric coordinates of the current mesh
    /// to a Jacobian in terms of the reduced metric coordinates of the initial metric connectivity.
    ///
    /// @param[in] halfedge_jacobian: Jacobian in halfedge coordinates for the current mesh
    /// @return Jacobian in reduced coordinates for the original mesh
    MatrixX change_metric_to_reduced_coordinates(const MatrixX& halfedge_jacobian) const;

    virtual ~DifferentiableConeMetric() = default;

protected:
    bool m_is_discrete_metric;
    std::vector<int> m_flip_seq;
    MatrixX m_identification;
};

/// Differentiable cone metric using Penner coordinates. The advantage of this choice is any
/// per-edge coordinates defines a valid metric, and any metric can be defined by some Penner
/// coordinates. That is, they provide a parameterization of the space of cone metrics with fixed
/// topology and vertex set. The disadvantage is Penner coordinates are less intuitive than
/// Euclidean edge lengths and make length constraints difficult to impose.

/// The reduced coordinates are simply per-edge values, and the flip change of coordinates are given
/// by the rational Ptolemy formula. The metric is discrete if and only if the coordinates are
/// ideal Delaunay, and the projection to constraints is obtained by conformal scaling.
class PennerConeMetric : public DifferentiableConeMetric
{
public:
    PennerConeMetric(const Mesh<Scalar>& m, const VectorX& metric_coords);

    // ****************************************************
    // Override methods: methods overridden from base class
    // ****************************************************

    // Metric access methods
    VectorX get_reduced_metric_coordinates() const override;

    // Flip method
    bool flip_ccw(int halfedge_index, bool Ptolemy=true) override;

    // Clone methods
    std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override;
    std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const override;
    std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override;

    // Constraint methods
    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const override;

    // Discrete metric methods
    void make_discrete_metric() override;
    MatrixX get_transition_jacobian() const override;

    // ********************************************
    // Unique methods: methods unique to this class
    // ********************************************

    /// Reset the flip sequence, treating the current metric and mesh as the base
    void reset();

protected:
    std::vector<int> m_embed;
    std::vector<int> m_proj;
    MatrixX m_projection;
    bool m_need_jacobian;
    FlipMatrixGenerator m_transition_jacobian_lol;

    VectorX reduce_metric_coordinates(const VectorX& metric_coords) const;
    void expand_metric_coordinates(const VectorX& metric_coords);
    MatrixX get_flip_jacobian() const { return m_transition_jacobian_lol.build_matrix(); }
};

/// Differentiable cone metric using log Euclidean coordinates. The advantage of this choice is any
/// that these coordinates are familiar and allow direct length constraints. The disadvantage is they
/// are limited by the triangle inequality, so only some coordinates define a valid metric, and flips
/// are nontrivial functions determined by the law of cosines.
///
/// The reduced coordinates are simply per-edge values, and the flip change of coordinates are given
/// by the Euclidean law of cosines formula. The metric is always discrete, and the projection to constraints
/// is obtained by conformal scaling.
///
/// WARNING: Currently, the transition Jacobian is not computed and the current mesh is always treated as the
/// base mesh, in contrast with the Penner cone metric where the initial connectivity is used as the base.
class DiscreteMetric : public DifferentiableConeMetric
{
public:
    DiscreteMetric(const Mesh<Scalar>& m, const VectorX& log_length_coords);

    // ****************************************************
    // Override methods: methods overridden from base class
    // ****************************************************

    // Metric access methods
    VectorX get_reduced_metric_coordinates() const override;

    // Flip method
    bool flip_ccw(int halfedge_index, bool Ptolemy=false) override;

    // Clone methods
    std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override;
    std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(
        const VectorX& metric_coords) const override;
    std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override;

    // Constraint methods
    std::unique_ptr<DifferentiableConeMetric> project_to_constraint(
        SolveStats<Scalar>& solve_stats,
        std::shared_ptr<ProjectionParameters> proj_params = nullptr) const override;

    // Discrete metric methods
    void make_discrete_metric() override;
    MatrixX get_transition_jacobian() const override;

protected:
    std::vector<int> m_embed;
    std::vector<int> m_proj;
    MatrixX m_projection;

    VectorX reduce_metric_coordinates(const VectorX& metric_coords) const;
    void expand_metric_coordinates(const VectorX& metric_coords);
};

} // namespace CurvatureMetric
