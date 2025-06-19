
#pragma once

#include "feature/core/common.h"
#include "feature/experimental/boundary_constraint_generator.h"
#include "holonomy/core/dual_loop.h"

namespace Penner {
namespace Feature {

// The validity check approach for meshes with multiple components is proving to be very
// difficult and likely a dead end for practical research.
class BoundaryConstraintValidator : public BoundaryConstraintGenerator 
{
public:
    /**
     * @brief Construct a new Boundary Constraint Generator object, marking all boundary components
     * as boundary features with no opposite features or junctions and trivial target lengths.
     *
     * @param m: underyling mesh
     */
    BoundaryConstraintValidator(Mesh<Scalar> m);

    // path attributes
    int count_paths() const { return m_path_holonomy.size(); }

    /**
     * @brief Split boundary paths segments and store boundary path pairings with holonomy.
     *
     * @param basis_loops: symmetric loops connecting boundary components
     * @param kappa_hat: loop holonomies
     */
    void add_boundary_path_segments(
        const std::vector<std::unique_ptr<DualLoop>>& basis_loops,
        const std::vector<Scalar>& kappa_hat
    );

    /**
     * @brief Construct a validity constraint system (Ax = b) for the features.
     *
     * A solution to the boundary constraint system exists if and only if there are
     * positive lengths satisfying length equality constraints and also a closure condition,
     * which states that the layout of each polygonal boundary component must be closed
     * (possibly after adding a virtual vertex to compensate for missing geodesic curvature).
     *
     * @return constraint validity matrix (A)
     * @return constraint validity vector (b)
     */
    std::tuple<MatrixX, VectorX> build_boundary_validity_system() const;

    // TODO: component attributes
    std::tuple<MatrixX, VectorX> build_component_validity_system() const;
    void view_component_layout(const VectorX& ell) const;
    // int component(int feature_index) const { return m_component[feature_index]; }
    // std::vector<int> features(int component_index) const { return m_features[component_index]; }

protected:

    // feature segment attributes (hidden from interface)
    int path(int segment_index) const { return m_path[segment_index]; }
    int reverse_path(int segment_index) const { return m_reverse_path[segment_index]; }

    // path attributes (hidden from interface)
    // a path is an abstract nonseparating path between two boundary components with some target holonomy
    int path_start_segment(int path_index) const { return m_path_start_segment[path_index]; }
    int path_end_segment(int path_index) const { return m_path_end_segment[path_index]; }
    int path_reverse_start_segment(int path_index) const { return m_path_reverse_start_segment[path_index]; }
    int path_reverse_end_segment(int path_index) const { return m_path_reverse_end_segment[path_index]; }
    Scalar path_holonomy(int path_index) const { return m_path_holonomy[path_index]; }

    // atomic operation with path management
    int split_segment(int segment_index);

    // useful predicates
    bool is_path_start(int segment_index) const { return (path(segment_index) >= 0); }
    bool is_reverse_path_start(int segment_index) const { return (reverse_path(segment_index) >= 0); }

    // path helper functions
    // TODO make class
    void add_feature(
        int feature_index,
        Vector2& d,
        Scalar& total_curvature,
        std::vector<Eigen::Triplet<Scalar>>& tripletList) const;
    void add_free_connection(
        const std::array<int, 5>& virtual_segments,
        bool reverse,
        Vector2& d,
        std::vector<Vector2>& segment_directions,
        std::vector<int>& segment_indices) const;
    void add_fixed_connection(
        const std::vector<int>& virtual_segments,
        bool is_counterclockwise,
        bool reverse,
        Vector2& d,
        std::vector<Vector2>& segment_directions,
        std::vector<int>& segment_indices) const;

    std::tuple<int, std::vector<Vector2>, std::vector<int>> build_component_layout() const;

private:
    // segment attributes
    std::vector<int> m_path; // TODO finish
    std::vector<int> m_reverse_path; // TODO finish

    // path attributes
    std::vector<int> m_path_start_segment;
    std::vector<int> m_path_end_segment;
    std::vector<int> m_path_reverse_start_segment;
    std::vector<int> m_path_reverse_end_segment;
    std::vector<Scalar> m_path_holonomy;
        

    // component attributes
    // TODO
    // std::vector<int> m_component;
    // std::vector<std::vector<int>> m_features;
};


} // namespace Feature
} // namespace Penner