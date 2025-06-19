
#pragma once

#include "feature/core/common.h"
#include "holonomy/core/dual_loop.h"

namespace Penner {
namespace Feature {

// Idea: Want to unify setup for building constraints for meshes where we care about each boundary
// edge length (e.g., feature alignment) and where we only care about the sum between cones (e.g.,
// polygon constraints for disks). Also want to generally keep track of boundary edges for checking
// validity or for local modification. To do this, we build a meta mesh of the features and
// junctions of the components, and we allow additional boundary vertices to be marked as junctions
// (in the limit, all boundary vertices).
//
// We can then support features of the following type:
// 1) Consistency of opposite boundary features
// 2) Fixed length (either of boundary or both opposite boundary features)
//
// This in turn allows for the following setups:
// a) Polygon boundary constraints are fixed length coarse boundary features with uniform length.
// b) Disk boundary constraints are fixed length fine boundary features with uniform length.
// c) Isometric boundary constraints are fixed length fine boundary features with initial length.
// d) Feature alignment constraints are consistent length fine boundary features.

class BoundaryConstraintGenerator
{
protected:
    // declare iterator class
    class SegmentIterator;

public:
    /**
     * @brief Construct a new Boundary Constraint Generator object, marking all boundary components
     * as boundary features with no opposite features or junctions and trivial target lengths.
     *
     * @param m: underyling mesh
     */
    BoundaryConstraintGenerator(Mesh<Scalar> m);

    // feature attributes
    int count_features() const { return m_next_feature.size(); }
    int count_segments() const { return m_next_segment.size(); }
    int count_halfedges() const { return m_mesh.n_halfedges(); }
    int next_feature(int feature_index) const
    {
        assert(is_valid_feature_index(feature_index));
        return m_next_feature[feature_index];
    }
    int prev_feature(int feature_index) const
    {
        assert(is_valid_feature_index(feature_index));
        return m_prev_feature[feature_index];
    }
    int to(int feature_index) const
    {
        assert(is_valid_feature_index(feature_index));
        return m_to[feature_index];
    }

    // junction attributes
    int count_junctions() const { return m_out.size(); }
    int out(int junction_index) const { return m_out[junction_index]; }

    /**
     * @brief Enumerate the cones on a given feature.
     *
     * @param feature_index: index of the feature
     * @return list of cones (in order) around the feature
     */
    std::vector<Scalar> enumerate_feature_cones(int feature_index) const;

    /**
     * @brief Enumerate the cones in the interior of the mesh
     *
     * @return list of cones (in arbitrary order) in the interior of the mesh
     */
    std::vector<Scalar> enumerate_interior_cones() const;

    /**
     * @brief Mark all cone vertices as junctions.
     *
     */
    void mark_cones_as_junctions();

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
     * @brief Pair all boundary segments identified by a vertex map for a cut mesh
     *
     * @param vtx_reindex: reindexing from mesh to vertex map indices
     * @param V_map: map from cut vertices to original vertices
     */
    void pair_boundary_edges(const std::vector<int>& vtx_reindex, const Eigen::VectorXi& V_map);

    /**
     * @brief Distribute a target feature length across segments
     *
     * @param feature_index: index of the feature to distribute
     * @param target_feature_length: total target feature length
     */
    void distribute_feature_length(int feature_index, Scalar target_feature_length);

    /**
     * @brief Set all feature lengths to a uniform value
     *
     * @param target_feature_length: target feature length
     */
    void set_uniform_feature_lengths(Scalar target_feature_length);

    /// Build a segment iterator for a feature.
    ///
    /// @param feature_index: feature to iterate over
    /// @return iterator for the given feature
    SegmentIterator get_feature_iterator(int feature_index) const
    {
        return SegmentIterator(*this, feature_start(feature_index));
    }

    /**
     * @brief Compute the total length of the given boundary feature
     *
     * @param feature_index: index of the given feature
     * @return length of the boundary
     */
    Scalar compute_feature_length(int feature_index) const;

    /**
     * @brief Get the underlying mesh object
     *
     * @return reference to the mesh
     */
    const Mesh<Scalar>& get_mesh() const { return m_mesh; }

    /**
     * @brief Get the target length for a given segment
     *
     * @param segment_index: index of the segment
     * @return target length of the segment
     */
    Scalar get_target_length(int segment_index) const { return m_target_length[segment_index]; }

    /**
     * @brief Set the target length for a given segment
     *
     * @param segment_index: index of the segment
     * @param target_length: new target length of the segment
     */
    void set_target_length(int segment_index, Scalar target_length)
    {
        m_target_length[segment_index] = target_length;
    }

    /**
     * @brief Generate a list of halfedges in the feature.
     *
     * @param feature_index: index of the feature
     * @return halfedges in the feature
     */
    std::vector<int> compute_feature_halfedges(int feature_index) const;

    /**
     * @brief Construct a matrix system mapping mesh halfedges to boundary features
     *
     * @return matrix mapping halfedges to features
     * @return vector of constraints for features
     */
    std::tuple<MatrixX, VectorX> build_boundary_constraint_system() const;

    /**
     * @brief View the current boundary constraint system
     *
     * @param V: mesh vertices
     * @param vtx_reindex: map from mesh vertices to VF vertices
     */
    void view(const Eigen::MatrixXd& V, const std::vector<int>& vtx_reindex);

protected:
    bool is_valid_feature_index(int feature_index) const
    {
        return ((feature_index >= 0) && (feature_index < count_features()));
    }

    bool is_valid_halfedge_index(int halfedge_index) const
    {
        return ((halfedge_index >= 0) && (halfedge_index < count_halfedges()));
    }

    // feature segment attributes (hidden from interface)
    // a segment is a halfedge in a feature edge (which is necessarily a boundary edge)
    int feature_start(int feature_index) const { return m_feature_start[feature_index]; }
    int feature_end(int feature_index) const { return m_feature_end[feature_index]; }
    int feature(int segment_index) const { return m_feature[segment_index]; }
    int halfedge(int segment_index) const { return m_halfedge[segment_index]; }
    int next_segment(int segment_index) const { return m_next_segment[segment_index]; }
    int prev_segment(int segment_index) const { return m_prev_segment[segment_index]; }
    int pair(int segment_index) const { return m_pair[segment_index]; }
    int segment(int halfedge_index) const { return m_segment[halfedge_index]; }

    // useful predicates
    bool is_feature_end(int segment_index) const { return (feature_end(feature(segment_index)) == segment_index); }

    // feature creation atomic operations
    void add_junction_after_segment(int segment_index);
    int split_segment(int segment_index);

    class SegmentIterator
    {
    public:
        SegmentIterator(const BoundaryConstraintGenerator& parent, int segment_index)
            : m_parent(parent)
        {
            m_current_segment_index = segment_index;
        }

        SegmentIterator& operator++()
        {
            if (!is_end()) {
                m_current_segment_index = m_parent.next_segment(m_current_segment_index);
            }

            return *this;
        }

        SegmentIterator operator++(int)
        {
            SegmentIterator temp = *this;
            ++*this;
            return temp;
        }

        SegmentIterator& operator--()
        {
            if (!is_end()) {
                m_current_segment_index = m_parent.prev_segment(m_current_segment_index);
            }

            return *this;
        }

        SegmentIterator operator--(int)
        {
            SegmentIterator temp = *this;
            --*this;
            return temp;
        }

        int operator*() { return m_parent.halfedge(m_current_segment_index); }

        bool is_end() { return (m_current_segment_index < 0); }

    private:
        const BoundaryConstraintGenerator& m_parent;
        int m_current_segment_index;
    };

private:
    // mesh copy
    Mesh<Scalar> m_mesh;

    // feature attributes
    std::vector<int> m_feature_start;
    std::vector<int> m_feature_end;
    std::vector<int> m_next_feature;
    std::vector<int> m_prev_feature;
    std::vector<int> m_to;
    std::vector<Scalar> m_target_length;

    // junction attributes
    std::vector<int> m_out;

    // feature segment attributes
    std::vector<int> m_feature;
    std::vector<int> m_halfedge;
    std::vector<int> m_next_segment;
    std::vector<int> m_prev_segment;
    std::vector<int> m_pair; // TODO, make feature attribute

    // halfedge attributes
    std::vector<int> m_segment;
};


} // namespace Feature
} // namespace Penner