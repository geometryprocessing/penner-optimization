// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/dual_segment.h"

/**
 * @brief Methods to manage dual loops on a surface. Includes classes to represent dual loops on a surface
 * and data structures to quickly query intersections of the loops with edges of the underlying mesh.
 * 
 */

namespace Penner {
namespace Holonomy {

/**
 * @brief Representation of a dual segment as two halfedges in a face at the start
 * and end of the dual segment.
 *
 */
typedef std::array<int, 2> DualSegment;

/**
 * @brief Utility class to map from halfedges to dual loop segments that intersect it,
 * using a dense vector representation,
 *
 * WARNING: Memory usage is O(|E|), so it scales poorly for large meshes with many edges
 *
 */
class DenseHalfedgeMap {
public:
    /**
     * @brief Construct a halfedge map with a fixed capacity.
     *
     * @param num_halfedges: total number of halfedges in the mesh
     */
    DenseHalfedgeMap(int num_halfedges);

    /**
     * @brief Remove all segment entries from all halfedges.
     */
    void clear();

    /**
     * @brief Add entry for a segment in a halfedge.
     *
     * @param halfedge_index: halfedge containing segment
     * @param segment_index: segment intersecting halfedge
     */
    void add_segment(int halfedge_index, int segment_index);

    /**
     * @brief Remove all segments from a given halfedge
     *
     * @param halfedge_index: halfedge in mesh
     */
    void erase_entry(int halfedge_index);

    /**
     * @brief Get all segments in the given halfedge
     *
     * @param halfedge_index: index of a halfedge in the mesh
     * @return segments in the halfedge.
     */
    const std::vector<int>& get_halfedge_segments(int halfedge_index);
private:
    std::vector<int> m_h_num_segments;
    std::vector<int> m_h_first_segment;
    std::vector<int> m_h_bucket;
    std::vector<std::vector<int>> m_h2segments;

    // Pre-allocated temporary data structures
    std::vector<int> m_empty_list;
    std::vector<int> m_temp_list;
};

/**
 * @brief Utility class to map from halfedges to dual loop segments that intersect it,
 * using a sparse map representation,
 *
 */
class SparseHalfedgeMap {
public:
    /**
     * @brief Construct an empty halfedge map.
     */
    SparseHalfedgeMap();

    /**
     * @brief Remove all segment entries from all halfedges.
     */
    void clear();

    /**
     * @brief Add entry for a segment in a halfedge.
     *
     * @param halfedge_index: halfedge containing segment
     * @param segment_index: segment intersecting halfedge
     */
    void add_segment(int halfedge_index, int segment_index);

    /**
     * @brief Remove all segments from a given halfedge.
     *
     * @param halfedge_index: halfedge in mesh
     */
    void erase_entry(int halfedge_index);

    /**
     * @brief Get all segments in the given halfedge.
     *
     * @param halfedge_index: index of a halfedge in the mesh
     * @return segments in the halfedge, or an empty list if none exist
     */
    const std::vector<int>& get_halfedge_segments(int halfedge_index);
private:
    std::unordered_map<int, std::vector<int>> m_h2segments;
    std::vector<int> m_empty_list;
};

// By default, use sparse representation
typedef SparseHalfedgeMap HalfedgeMap;

/**
 * @brief Interface for dual loop tracking on a mesh
 *
 */
class DualLoop
{
public:
    /**
     * @brief Update the dual loop after a counter-clockwise flip in the underlying mesh.
     *
     * @param m: underlying mesh before the flip
     * @param halfedge_index: halfedge to be flipped
     */
    virtual void update_under_ccw_flip(const Mesh<Scalar>& m, int halfedge_index) = 0;

    /**
     * @brief Generate the ordered sequence of face indices the dual loop traverses.
     *
     * @param m: underlying mesh
     * @return sequence of face indices in traversal order
     */
    virtual std::vector<int> generate_face_sequence(const Mesh<Scalar>& m) const = 0;

    /**
     * @brief Create a heap-allocated copy of this dual loop.
     *
     * @return owning pointer to a cloned dual loop of the same concrete type
     */
    virtual std::unique_ptr<DualLoop> clone() const = 0;
    virtual ~DualLoop() = default;

    /**
     * @brief Iterator to iterate over the segments of the dual loop
     *
     */
    class DualSegmentIterator
    {
    public:
        /**
         * @brief Construct an iterator positioned at the start or end of the loop.
         *
         * @param parent: dual loop being iterated
         * @param is_start: true to position at the first segment, false for the end sentinel
         */
        DualSegmentIterator(const DualLoop& parent, bool is_start = true)
            : m_parent(parent)
            , m_is_start(is_start)
        {
            m_start_segment_index = m_parent.get_start_segment_index();
            m_current_segment_index = m_start_segment_index;
        }

        /**
         * @brief Advance the iterator to the next segment (pre-increment).
         *
         * @return reference to this iterator after advancing
         */
        DualSegmentIterator& operator++()
        {
            m_is_start = false;
            m_current_segment_index = m_parent.get_next(m_current_segment_index);
            return *this;
        }

        /**
         * @brief Advance the iterator to the next segment (post-increment).
         *
         * @return copy of this iterator before advancing
         */
        DualSegmentIterator operator++(int)
        {
            DualSegmentIterator temp = *this;
            ++*this;
            return temp;
        }

        /**
         * @brief Check equality with another iterator.
         *
         * Two iterators are equal when they point to the same segment index and
         * both share the same start/non-start state (to distinguish begin from end
         * on a closed loop).
         *
         * @param rhs: iterator to compare against
         * @return true if iterators are equal
         */
        bool is_equal(const DualSegmentIterator& rhs) const
        {
            return (
                (m_is_start == rhs.m_is_start) &&
                (m_current_segment_index == rhs.m_current_segment_index));
        }

        friend bool operator==(const DualSegmentIterator& lhs, const DualSegmentIterator& rhs)
        {
            return (lhs.is_equal(rhs));
        }

        friend bool operator!=(const DualSegmentIterator& lhs, const DualSegmentIterator& rhs)
        {
            return (!(lhs == rhs));
        }

        /**
         * @brief Check whether the iterator has traversed the entire loop.
         *
         * @return true if the iterator has wrapped back to the start segment
         */
        bool is_end()
        {
            return ((!m_is_start) && (m_current_segment_index == m_start_segment_index));
        }

        /**
         * @brief Dereference the iterator to obtain the current dual segment.
         *
         * @return dual segment at the current position
         */
        DualSegment operator*() { return m_parent.get_dual_segment(m_current_segment_index); }

    private:
        const DualLoop& m_parent;
        bool m_is_start;
        int m_start_segment_index;
        int m_current_segment_index;
    };

    /**
     * @brief Construct an iterator to the beginning of the dual loop
     *
     * @return loop start iterator
     */
    DualSegmentIterator begin() const { return DualSegmentIterator(*this, true); }

    /**
     * @brief Construct an iterator to one past the end of the dual loop
     *
     * @return loop end iterator
     */
    DualSegmentIterator end() const { return DualSegmentIterator(*this, false); }

protected:
    /**
     * @brief Return the index of the segment following the given one in the loop.
     *
     * @param segment_index: index of the current segment
     * @return index of the next segment
     */
    virtual int get_next(int segment_index) const = 0;

    /**
     * @brief Return the index of the canonical first segment used to initialize iteration.
     *
     * @return starting segment index
     */
    virtual int get_start_segment_index() const = 0;

    /**
     * @brief Return the dual segment stored at the given index.
     *
     * @param segment_index: index of the segment to retrieve
     * @return dual segment (pair of halfedge indices bounding a face crossing)
     */
    virtual DualSegment get_dual_segment(int segment_index) const = 0;
};

/**
 * @brief Representation of a dual loop on a mesh. This data structure supports iteration over the
 * dual loop, local updates for flips in the underlying mesh, and conversions to and from sequences
 * of faces on the mesh constituting a closed dual loop.
 *
 */
class DualLoopConnectivity : public DualLoop
{
public:
    /**
     * @brief Construct a trivial dual loop connectivity
     */
    DualLoopConnectivity();

    /**
     * @brief Construct a new dual loop connectivity from a sequence of dual segments
     *
     * @param dual_loop_segments: sequence of continuous dual segments on the mesh
     */
    DualLoopConnectivity(const std::vector<DualSegment>& dual_loop_segments);

    /**
     * @brief Construct a new dual loop connectivity from a sequence of faces on a mesh.
     *
     * @param[in] m: mesh
     * @param[in] dual_loop_faces: sequence of faces in the mesh (must be adjacent)
     */
    DualLoopConnectivity(const Mesh<Scalar>& m, const std::vector<int>& dual_loop_faces);

    std::unique_ptr<DualLoop> clone() const override
    {
        return std::make_unique<DualLoopConnectivity>(*this);
    }

    /**
     * @brief Update dual loop connectivity after a flip in the underlying mesh.
     *
     * @param m: underlying mesh (before the flip)
     * @param halfedge_index: halfedge to flip
     */
    void update_under_ccw_flip(const Mesh<Scalar>& m, int halfedge_index) override;

    /**
     * @brief Generate the sequence of faces the dual loop traverses.
     *
     * @param m: underlying mesh
     * @return sequence of faces in the dual loop
     */
    std::vector<int> generate_face_sequence(const Mesh<Scalar>& m) const override;

    /**
     * @brief Clear all internal data.
     *
     */
    void clear();

    /**
     * @brief Enable dual loop validity checks (with a large runtime cost);
     *
     */
    void enable_validity_checks() { m_check_validity = true; }

protected:
    /**
     * @brief Return the total number of allocated segment slots (including deleted ones).
     *
     * @return number of segment slots
     */
    int count_segment_indices() const { return m_next.size(); }

    /**
     * @brief Return the index of the segment following the given one in the loop.
     *
     * @param segment_index: index of the current segment
     * @return index of the next segment
     */
    int get_next(int segment_index) const override
    {
        assert(is_valid_segment_index(segment_index));
        return m_next[segment_index];
    }

    /**
     * @brief Return the index of the segment preceding the given one in the loop.
     *
     * @param segment_index: index of the current segment
     * @return index of the previous segment
     */
    int get_prev(int segment_index) const
    {
        assert(is_valid_segment_index(segment_index));
        return m_prev[segment_index];
    }

    /**
     * @brief Return the start halfedge of the given segment.
     *
     * @param segment_index: index of the segment
     * @return halfedge index at which the segment enters its face
     */
    int get_start(int segment_index) const
    {
        assert(is_valid_segment_index(segment_index));
        return m_start[segment_index];
    }

    /**
     * @brief Return the end halfedge of the given segment.
     *
     * @param segment_index: index of the segment
     * @return halfedge index at which the segment exits its face
     */
    int get_end(int segment_index) const
    {
        assert(is_valid_segment_index(segment_index));
        return m_end[segment_index];
    }

    /**
     * @brief Check whether a segment slot has been freed.
     *
     * @param segment_index: index of the segment slot
     * @return true if the slot is deleted and available for reuse
     */
    bool is_deleted(int segment_index) const { return m_is_deleted[segment_index]; }

    // Index based segment management
    int get_start_segment_index() const override;
    DualSegment get_dual_segment(int segment_index) const override;

    /**
     * @brief Check whether a segment index is within bounds and not deleted.
     *
     * @param segment_index: index to validate
     * @return true if the index refers to a live segment
     */
    bool is_valid_segment_index(int segment_index) const;

    /**
     * @brief Verify that the next/prev linkage is internally consistent.
     *
     * @return true if all segments form a well-linked circular list
     */
    bool is_valid_connectivity() const;

    /**
     * @brief Verify that the dual loop correctly overlays the given mesh.
     *
     * Checks that every segment's halfedges belong to a common face and that
     * consecutive segments share an edge crossing.
     *
     * @param m: underlying mesh
     * @return true if the dual loop is geometrically valid on the mesh
     */
    bool is_valid_dual_loop(const Mesh<Scalar>& m) const;

private:
    // Segment connectivity
    std::vector<int> m_next;
    std::vector<int> m_prev;

    // Dual segment information
    std::vector<int> m_start;
    std::vector<int> m_end;

    // Track garbage collection
    std::vector<bool> m_is_deleted;
    std::deque<int> m_free_indices;

    // Maps from mesh halfedges to segments starting at them
    HalfedgeMap m_halfedge_map;

    bool m_check_validity = false;

    /**
     * @brief Split a segment at an interior halfedge, inserting a new segment.
     *
     * Used when a flip introduces an edge that bisects an existing segment.
     *
     * @param segment_index: segment to split
     * @param halfedge_index: new halfedge crossing the segment
     * @param opposite_halfedge: opposite halfedge of the crossing edge
     */
    void split_segment(int segment_index, int halfedge_index, int opposite_halfedge);

    /**
     * @brief Swap the endpoints of two adjacent segments across a flipped edge.
     *
     * @param first_segment_index: first segment involved in the flip
     * @param second_segment_index: second segment involved in the flip
     * @param halfedge_index: halfedge that was flipped
     * @param opposite_halfedge: opposite halfedge of the flipped edge
     */
    void flip_segments(
        int first_segment_index,
        int second_segment_index,
        int halfedge_index,
        int opposite_halfedge);

    /**
     * @brief Merge two consecutive segments into one, removing the shared crossing.
     *
     * @param first_segment_index: segment whose end will be extended
     * @param second_segment_index: segment to be absorbed and deleted
     */
    void combine_segments(int first_segment_index, int second_segment_index);

    /**
     * @brief Allocate a new segment slot, reusing a freed index if available.
     *
     * @return index of the newly created segment slot
     */
    int create_segment_index();

    /**
     * @brief Mark a segment slot as deleted and return it to the free list.
     *
     * @param segment_index: index of the segment to free
     */
    void delete_segment_index(int segment_index);

    // TODO Add garbage collector for resizing
};

/**
 * @brief Minimal representation for a dual loop as a list of dual segments
 *
 */
class DualLoopList : public DualLoop
{
public:
    /**
     * @brief Construct a new dual loop list from a sequence of dual segments
     *
     * @param m: underlying mesh
     * @param dual_loop_segments: sequence of continuous dual segments on the mesh
     */
    DualLoopList(const std::vector<DualSegment>& dual_loop_segments)
        : m_dual_path(dual_loop_segments)
    {}

    /**
     * @brief Construct a new dual loop list from a sequence of faces on a mesh.
     *
     * @param[in] m: mesh
     * @param[in] dual_loop_faces: sequence of faces in the mesh (must be adjacent)
     */
    DualLoopList(const Mesh<Scalar>& m, const std::vector<int>& dual_loop_faces)
        : m_dual_path(build_dual_path_from_face_sequence(m, dual_loop_faces))
    {}

    std::unique_ptr<DualLoop> clone() const override
    {
        return std::make_unique<DualLoopList>(*this);
    }

    void update_under_ccw_flip(const Mesh<Scalar>& m, int halfedge_index) override
    {
        update_dual_loop_under_ccw_flip(m, halfedge_index, m_dual_path);
    }

    std::vector<int> generate_face_sequence(const Mesh<Scalar>& m) const override
    {
        return build_face_sequence_from_dual_path(m, m_dual_path);
    }

protected:
    int get_next(int segment_index) const override
    {
        return (segment_index + 1) % count_segment_indices();
    }

    int get_start_segment_index() const override { return 0; }

    DualSegment get_dual_segment(int segment_index) const override
    {
        return m_dual_path[segment_index];
    }

private:
    std::vector<DualSegment> m_dual_path;

    int count_segment_indices() const { return m_dual_path.size(); }
};

/**
 * @brief Manager that maps mesh edges to the dual loops crossing them.
 *
 * Provides O(1) lookup of which dual loops intersect a given edge, and supports
 * bulk registration of all edges crossed by a loop.
 */
class DualLoopManager{
public:
    /**
     * @brief Construct a manager sized for the given number of edges.
     *
     * @param num_edges: total number of edges in the mesh
     */
    DualLoopManager(int num_edges);

    /**
     * @brief Remove all edge-to-loop associations.
     */
    void clear();

    /**
     * @brief Record that a dual loop crosses a given edge.
     *
     * @param edge_index: edge intersected by the loop
     * @param loop_index: index of the dual loop
     */
    void add_loop(int edge_index, int loop_index);

    /**
     * @brief Register all edges crossed by a dual loop.
     *
     * Iterates over every segment of @p dual_loop, determines which mesh edge each
     * segment crosses, and calls add_loop() for each edge-loop pair.
     *
     * @param loop_index: index of the dual loop being registered
     * @param m: underlying mesh
     * @param dual_loop: dual loop whose edge crossings will be recorded
     */
    void register_loop_edges(int loop_index, const Mesh<Scalar>& m, const DualLoop& dual_loop);

    /**
     * @brief Remove all loop associations for a given edge.
     *
     * @param edge_index: edge whose loop entries will be cleared
     */
    void erase_entry(int edge_index);

    /**
     * @brief Get all dual loops crossing a given edge.
     *
     * @param edge_index: index of the edge to query
     * @return loop indices crossing the edge
     */
    const std::vector<int>& get_edge_loops(int edge_index);
private:
    std::vector<int> m_e_num_loops;
    std::vector<int> m_e_first_loop;
    std::vector<int> m_e_bucket;
    std::vector<std::vector<int>> m_e2loops;

    // Pre-allocated temporary data structures
    std::vector<int> m_empty_list;
    std::vector<int> m_temp_list;
};

} // namespace Holonomy
} // namespace Penner