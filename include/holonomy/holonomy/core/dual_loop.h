#pragma once

#include "holonomy/core/common.h"
#include "holonomy/core/dual_segment.h"

namespace Penner {
namespace Holonomy {

typedef std::array<int, 2> DualSegment;

class DenseHalfedgeMap {
public:
    DenseHalfedgeMap(int num_halfedges);
    void clear();
    void add_segment(int halfedge_index, int segment_index);
    void erase_entry(int halfedge_index);
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

class SparseHalfedgeMap {
public:
    SparseHalfedgeMap();
    void clear();
    void add_segment(int halfedge_index, int segment_index);
    void erase_entry(int halfedge_index);
    const std::vector<int>& get_halfedge_segments(int halfedge_index);
private:
    std::unordered_map<int, std::vector<int>> m_h2segments;
    std::vector<int> m_empty_list;
};

typedef SparseHalfedgeMap HalfedgeMap;

/**
 * @brief Interface for dual loop tracking on a mesh
 *
 */
class DualLoop
{
public:
    virtual void update_under_ccw_flip(const Mesh<Scalar>& m, int halfedge_index) = 0;
    virtual std::vector<int> generate_face_sequence(const Mesh<Scalar>& m) const = 0;
    virtual std::unique_ptr<DualLoop> clone() const = 0;
    virtual ~DualLoop() = default;

    /**
     * @brief Iterator to iterate over the segments of the dual loop
     *
     */
    class DualSegmentIterator
    {
    public:
        DualSegmentIterator(const DualLoop& parent, bool is_start = true)
            : m_parent(parent)
            , m_is_start(is_start)
        {
            m_start_segment_index = m_parent.get_start_segment_index();
            m_current_segment_index = m_start_segment_index;
        }

        DualSegmentIterator& operator++()
        {
            m_is_start = false;
            m_current_segment_index = m_parent.get_next(m_current_segment_index);
            return *this;
        }

        DualSegmentIterator operator++(int)
        {
            DualSegmentIterator temp = *this;
            ++*this;
            return temp;
        }

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

        bool is_end()
        {
            return ((!m_is_start) && (m_current_segment_index == m_start_segment_index));
        }

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
    virtual int get_next(int segment_index) const = 0;
    virtual int get_start_segment_index() const = 0;
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
    // Connectivity getters
    int count_segment_indices() const { return m_next.size(); }

    int get_next(int segment_index) const override
    {
        assert(is_valid_segment_index(segment_index));
        return m_next[segment_index];
    }

    int get_prev(int segment_index) const
    {
        assert(is_valid_segment_index(segment_index));
        return m_prev[segment_index];
    }

    int get_start(int segment_index) const
    {
        assert(is_valid_segment_index(segment_index));
        return m_start[segment_index];
    }

    int get_end(int segment_index) const
    {
        assert(is_valid_segment_index(segment_index));
        return m_end[segment_index];
    }

    bool is_deleted(int segment_index) const { return m_is_deleted[segment_index]; }

    // Index based segment management
    int get_start_segment_index() const override;
    DualSegment get_dual_segment(int segment_index) const override;

    // Validity tests
    bool is_valid_segment_index(int segment_index) const;
    bool is_valid_connectivity() const;
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

    // Atomic loop change operations
    void split_segment(int segment_index, int halfedge_index, int opposite_halfedge);
    void flip_segments(
        int first_segment_index,
        int second_segment_index,
        int halfedge_index,
        int opposite_halfedge);
    void combine_segments(int first_segment_index, int second_segment_index);

    // Index memory management
    int create_segment_index();
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

class DualLoopManager{
public:
    DualLoopManager(int num_edges);
    void clear();
    void add_loop(int edge_index, int loop_index);
    void register_loop_edges(int loop_index, const Mesh<Scalar>& m, const DualLoop& dual_loop);
    void erase_entry(int edge_index);
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