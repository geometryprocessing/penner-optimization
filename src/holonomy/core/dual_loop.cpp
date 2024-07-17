#include "holonomy/core/dual_loop.h"
#include <algorithm>
#include <stdexcept>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

DualLoopManager::DualLoopManager(int num_edges)
{
    m_e_num_loops = std::vector<int>(num_edges, 0);
    m_e_first_loop = std::vector<int>(num_edges, -1);
    m_e_bucket = std::vector<int>(num_edges, -1);
    m_e2loops.clear();

    m_empty_list = {};
    m_temp_list = {-1};
}

void DualLoopManager::clear()
{
    m_e2loops.clear();
    m_e_num_loops.clear();
    m_e_first_loop.clear();
    m_e_bucket.clear();

    m_empty_list = {};
    m_temp_list = {-1};
}

// Add a map from a edge index to a loop index
void DualLoopManager::add_loop(int edge_index, int loop_index)
{
    // If the edge has no loops intersecting it, record the new loop in the first loop array
    if (m_e_num_loops[edge_index] == 0) {
        m_e_num_loops[edge_index] = 1;
        m_e_first_loop[edge_index] = loop_index;
    }
    // If the edge already has a single outgoing loop, move it and the new loop to the
    // unordered map
    else if (m_e_num_loops[edge_index] == 1) {
        // Check if trying to add a duplicate entry
        if (m_e_first_loop[edge_index] == loop_index) return;

        // Get bucket for the edge index (only allocate if needed)
        if (m_e_bucket[edge_index] < 0) {
            m_e_bucket[edge_index] = m_e2loops.size();
            m_e2loops.push_back({});
        }

        // Copy to the bucket
        int bucket_index = m_e_bucket[edge_index];
        m_e2loops[bucket_index].clear();
        m_e2loops[bucket_index].push_back(m_e_first_loop[edge_index]);
        m_e2loops[bucket_index].push_back(loop_index);
        m_e_num_loops[edge_index] = 2;
    }
    // If the edge already has multiple outgoing loops, add the new loop to the map
    else {
        int bucket_index = m_e_bucket[edge_index];
        const auto& bucket = m_e2loops[bucket_index];

        // Only add loop if not already in map
        if (std::find(bucket.begin(), bucket.end(), loop_index) == bucket.end())
        {
            m_e2loops[bucket_index].push_back(loop_index);
            m_e_num_loops[edge_index] += 1;
        }
    }
}

void DualLoopManager::register_loop_edges(
    int loop_index,
    const Mesh<Scalar>& m,
    const DualLoop& dual_loop)
{
    // Get edge maps
    std::vector<int> he2e, e2he;
    build_edge_maps(m, he2e, e2he);

    // Add all loops adjacent to the dual loop
    for (const auto& dual_segment : dual_loop) {
        int hij = dual_segment[0];
        for (int h : {hij, m.n[hij], m.n[m.n[hij]]}) {
            add_loop(he2e[h], loop_index);
        }
    }
}

// Remove all the recorded outgoing loops for a edge from the edge-to-loop map
void DualLoopManager::erase_entry(int edge_index)
{
    // Set the number of outgoing loops as zero
    m_e_num_loops[edge_index] = 0;
}

// Get list of loops that start at a given edge
const std::vector<int>& DualLoopManager::get_edge_loops(int edge_index)
{
    // Use preallocated empty list for the one-to-zero case
    if (m_e_num_loops[edge_index] == 0) {
        return m_empty_list;
    }
    // Overwrite and use preallocated single edge list for the one-to-one case
    // WARNING: dangerous for parallelism
    else if (m_e_num_loops[edge_index] == 1) {
        m_temp_list[0] = m_e_first_loop[edge_index];
        return m_temp_list;
    }
    // Use stored list for the one-to-many case
    else {
        int bucket_index = m_e_bucket[edge_index];
        assert(bucket_index >= 0);
        return m_e2loops[bucket_index];
    }
}

DenseHalfedgeMap::DenseHalfedgeMap(int num_halfedges)
{
    m_h_num_segments = std::vector<int>(num_halfedges, 0);
    m_h_first_segment = std::vector<int>(num_halfedges, -1);
    m_h_bucket = std::vector<int>(num_halfedges, -1);
    m_h2segments.clear();

    m_empty_list = {};
    m_temp_list = {-1};
}

void DenseHalfedgeMap::clear()
{
    m_h2segments.clear();
    m_h_num_segments.clear();
    m_h_first_segment.clear();
    m_h_bucket.clear();

    m_empty_list = {};
    m_temp_list = {-1};
}

// Add a map from a halfedge index to a segment index
void DenseHalfedgeMap::add_segment(int halfedge_index, int segment_index)
{
    // We need to map halfedges to segments that start at this halfedge. In the vast majority of
    // cases, this is one-to-zero or one-to-one, but for some halfedges there may be many outgoing
    // segments. Using a fully general map data structure is slow due to cache incoherence.
    //
    // To balance generality with efficiency we, explicitly track the number of outgoing segments,
    // use a fixed size vector to record the map from halfedges to the first outgoing segment,
    // and only switch to an unordered map if a one-to-many case is encountered.

    // If the halfedge has no outgoing segments yet, record the new segment in the first segment
    // array
    if (m_h_num_segments[halfedge_index] == 0) {
        m_h_num_segments[halfedge_index] = 1;
        m_h_first_segment[halfedge_index] = segment_index;
    }
    // If the halfedge already has a single outgoing segment, move it and the new segment to the
    // unordered map
    else if (m_h_num_segments[halfedge_index] == 1) {
        // Get bucket for the halfedge index (only allocate if needed)
        if (m_h_bucket[halfedge_index] < 0) {
            m_h_bucket[halfedge_index] = m_h2segments.size();
            m_h2segments.push_back({});
        }

        // Copy to the bucket
        int bucket_index = m_h_bucket[halfedge_index];
        m_h2segments[bucket_index].clear();
        m_h2segments[bucket_index].push_back(m_h_first_segment[halfedge_index]);
        m_h2segments[bucket_index].push_back(segment_index);
        m_h_num_segments[halfedge_index] = 2;
    }
    // If the halfedge already has multiple outgoing segments, add the new segment to the map
    else {
        int bucket_index = m_h_bucket[halfedge_index];
        m_h2segments[bucket_index].push_back(segment_index);
        m_h_num_segments[halfedge_index] += 1;
    }
}

// Remove all the recorded outgoing segments for a halfedge from the halfedge-to-segment map
void DenseHalfedgeMap::erase_entry(int halfedge_index)
{
    // Set the number of outgoing segments as zero
    m_h_num_segments[halfedge_index] = 0;
}

// Get list of segments that start at a given halfedge
const std::vector<int>& DenseHalfedgeMap::get_halfedge_segments(int halfedge_index)
{
    // Use preallocated empty list for the one-to-zero case
    if (m_h_num_segments[halfedge_index] == 0) {
        return m_empty_list;
    }
    // Overwrite and use preallocated single edge list for the one-to-one case
    // WARNING: dangerous for parallelism
    else if (m_h_num_segments[halfedge_index] == 1) {
        m_temp_list[0] = m_h_first_segment[halfedge_index];
        return m_temp_list;
    }
    // Use stored list for the one-to-many case
    else {
        int bucket_index = m_h_bucket[halfedge_index];
        assert(bucket_index >= 0);
        return m_h2segments[bucket_index];
    }
}

SparseHalfedgeMap::SparseHalfedgeMap()
{
    m_empty_list = {};
}

void SparseHalfedgeMap::clear()
{
    m_h2segments.clear();

    m_empty_list = {};
}

void SparseHalfedgeMap::add_segment(int halfedge_index, int segment_index)
{
    m_h2segments[halfedge_index].push_back(segment_index);
}

void SparseHalfedgeMap::erase_entry(int halfedge_index)
{
    m_h2segments.erase(halfedge_index);
}

const std::vector<int>& SparseHalfedgeMap::get_halfedge_segments(int halfedge_index)
{
    const auto& itr = m_h2segments.find(halfedge_index);
    if (itr != m_h2segments.end()) {
        return itr->second;
    } else {
        return m_empty_list;
    }
}


DualLoopConnectivity::DualLoopConnectivity()
    : m_halfedge_map()
{
    clear();
}

DualLoopConnectivity::DualLoopConnectivity(
    const std::vector<DualSegment>& dual_loop_segments)
    : m_halfedge_map()
{
    clear();

    // Resize segment arrays
    int num_segments = dual_loop_segments.size();
    m_next.resize(num_segments);
    m_prev.resize(num_segments);
    m_start.resize(num_segments);
    m_end.resize(num_segments);

    // Initialize trivial free and deleted index data
    m_is_deleted = std::vector<bool>(num_segments, false);
    m_free_indices.clear();

    for (int i = 0; i < num_segments; ++i) {
        // Find the edge adjacent to the next face in the sequence
        int j = (i + 1) % num_segments; // next periodic index

        // Set connectivity for the current edge (just a simple periodic offset by 1)
        m_next[i] = j;
        m_prev[j] = i;

        // Set dual loop halfedge indices for the current segment
        m_start[i] = dual_loop_segments[i][0];
        m_end[i] = dual_loop_segments[i][1];

        // Map halfedge to segment starting at this halfedge
        m_halfedge_map.add_segment(m_start[i], i);
    }

}

DualLoopConnectivity::DualLoopConnectivity(
    const Mesh<Scalar>& m,
    const std::vector<int>& dual_loop_faces)
    : DualLoopConnectivity(build_dual_path_from_face_sequence(m, dual_loop_faces))
{
    assert(is_valid_dual_loop(m));
}

void DualLoopConnectivity::update_under_ccw_flip(const Mesh<Scalar>& m, int halfedge_index)
{
    assert(is_valid_dual_loop(m));

    // Get halfedges and faces in the flipped quad
    int hij = halfedge_index;
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    int hji = m.opp[hij];
    int hil = m.n[hji];
    int hlj = m.n[hil];

    // Clear current halfedge to segment maps for flipped edge
    m_halfedge_map.erase_entry(hij);
    m_halfedge_map.erase_entry(hji);

    //    Before
    //      vk
    // hki /   \ hjk
    //    / hij \ .
    //   vi --- vj
    //    \ hji / .
    // hil \   / hlj
    //      vl
    //
    //     After
    //      vk
    // hki / | \ hjk
    //    /  |  \ .
    //   vi  |  vj
    //    \  |  / .
    // hil \ | / hlj
    //      vl

    // Adjust portions of the dual loop that intersect the quad
    // There are three possible cases (up to rotational symmetry)
    //   1. The dual loop enters and leaves the quad from the same triangle
    //   2. The dual loop crosses the diagonal and exits the opposite side of the quad
    //   3. The dual loop crosses the diagonal and exits the same side of the quad
    // For each possible initial edge, we handle the above three cases in this order, which
    // necessitate the following corresponding operations
    //   1. Split one segment into two
    //   2. Flip (or leave unchanged) the edge between two segments
    //   3. Combine two segments into one

    // Dual loop segments entering quad from hil
    for (int segment_index : m_halfedge_map.get_halfedge_segments(hil)) {
        int next_segment_index = get_next(segment_index);
        if (get_end(segment_index) == hlj) {
            split_segment(segment_index, hij, hji);
        } else if (get_start(next_segment_index) == hij && get_end(next_segment_index) == hjk) {
            flip_segments(segment_index, next_segment_index, hij, hji);
        } else if (get_start(next_segment_index) == hij && get_end(next_segment_index) == hki) {
            combine_segments(segment_index, next_segment_index);
        } else {
            throw std::runtime_error("Invalid dual loop flip encountered");
        }
    }

    // Dual loop edge enters quad from hlj
    for (int segment_index : m_halfedge_map.get_halfedge_segments(hlj)) {
        int next_segment_index = get_next(segment_index);
        if (get_end(segment_index) == hil) {
            split_segment(segment_index, hji, hij);
        } else if (get_start(next_segment_index) == hij && get_end(next_segment_index) == hki) {
            flip_segments(segment_index, next_segment_index, hji, hij);
        } else if (get_start(next_segment_index) == hij && get_end(next_segment_index) == hjk) {
            combine_segments(segment_index, next_segment_index);
        } else {
            throw std::runtime_error("Invalid dual loop flip encountered");
        }
    }

    // Dual loop edge enters quad from hjk
    for (int segment_index : m_halfedge_map.get_halfedge_segments(hjk)) {
        int next_segment_index = get_next(segment_index);
        if (get_end(segment_index) == hki) {
            split_segment(segment_index, hji, hij);
        } else if (get_start(next_segment_index) == hji && get_end(next_segment_index) == hil) {
            flip_segments(segment_index, next_segment_index, hji, hij);
        } else if (get_start(next_segment_index) == hji && get_end(next_segment_index) == hlj) {
            combine_segments(segment_index, next_segment_index);
        } else {
            throw std::runtime_error("Invalid dual loop flip encountered");
        }
    }

    // Dual loop edge enters quad from hki
    for (int segment_index : m_halfedge_map.get_halfedge_segments(hki)) {
        int next_segment_index = get_next(segment_index);
        if (get_end(segment_index) == hjk) {
            split_segment(segment_index, hij, hji);
        } else if (get_start(next_segment_index) == hji && get_end(next_segment_index) == hlj) {
            flip_segments(segment_index, next_segment_index, hij, hji);
        } else if (get_start(next_segment_index) == hji && get_end(next_segment_index) == hil) {
            combine_segments(segment_index, next_segment_index);
        } else {
            throw std::runtime_error("Invalid dual loop flip encountered");
        }
    }

    assert(is_valid_connectivity()); // can only check connectivity before mesh flip complete
}

std::vector<int> DualLoopConnectivity::generate_face_sequence(const Mesh<Scalar>& m) const
{
    // Resize face loop to the size of the dual loop
    std::vector<int> dual_loop_faces;
    dual_loop_faces.reserve(count_segment_indices());

    // Map dual segments to faces
    for (const auto& dual_segment : *this) {
        dual_loop_faces.push_back(compute_dual_segment_face(m, dual_segment));
    }

    return dual_loop_faces;
}

void DualLoopConnectivity::clear()
{
    m_next.clear();
    m_prev.clear();
    m_start.clear();
    m_end.clear();
    m_is_deleted.clear();
    m_free_indices.clear();

    m_halfedge_map.clear();
}

// Get the first undeleted segment index
int DualLoopConnectivity::get_start_segment_index() const
{
    // Find index that is not deleted
    int num_indices = count_segment_indices();
    for (int i = 0; i < num_indices; ++i) {
        if (!is_deleted(i)) return i;
    }

    // Return invalid index otherwise
    return -1;
}

// Construct the dual segment corresponding to a given dual segment index
DualSegment DualLoopConnectivity::get_dual_segment(int segment_index) const
{
    assert(is_valid_segment_index(segment_index));
    DualSegment dual_segment = {get_start(segment_index), get_end(segment_index)};
    return dual_segment;
}

// Check if a segment index is valid
bool DualLoopConnectivity::is_valid_segment_index(int segment_index) const
{
    // Segment index should be in the bounds for the segment arrays
    int num_segments = m_next.size();
    if (segment_index < 0) return false;
    if (segment_index >= num_segments) return false;

    // Corresponding segment should not be deleted
    if (m_is_deleted[segment_index]) return false;

    return true;
}

// Check if the dual loop connectivity is valid (but not consistent with the underlying mesh)
bool DualLoopConnectivity::is_valid_connectivity() const
{
    if (!m_check_validity) return true; // option to skip validity check for debugging
    int num_segments = m_next.size();

    // Check next and prev are inverse
    for (int i = 0; i < num_segments; ++i) {
        if (m_is_deleted[i]) continue;

        // previous segment is valid
        if (!is_valid_segment_index(m_prev[i])) {
            spdlog::error("Segment {} has invalid previous segment", i);
            return false;
        }

        // next segment is valid
        if (!is_valid_segment_index(m_next[i])) {
            spdlog::error("Segment {} has invalid next segment", i);
            return false;
        }

        // prev-next is identity
        if (m_next[m_prev[i]] != i) {
            spdlog::error("Segment i = {} does not satisfy next[prev[i]] = i", i);
            return false;
        }

        // next-prev is identity
        if (m_prev[m_next[i]] != i) {
            spdlog::error("Segment i = {} does not satisfy prev[next[i]] = i", i);
            return false;
        }
    }

    // Check free indices and deleted indices are the same
    int num_free_indices = m_free_indices.size();
    for (int i : m_free_indices) {
        if (!m_is_deleted[i]) {
            spdlog::error("Free segment index {} is not deleted", i);
            return false;
        }
    }
    if (std::count(m_is_deleted.begin(), m_is_deleted.end(), true) != num_free_indices) {
        spdlog::error("Inconsistent number of deleted and free indices");
        return false;
    }

    return true;
}

// Check if the dual loop connectivity is valid and consistent with the underlying mesh
bool DualLoopConnectivity::is_valid_dual_loop(const Mesh<Scalar>& m) const
{
    if (!m_check_validity) return true;

    int num_segments = m_next.size();

    // Check connectivity conditions
    if (!is_valid_connectivity()) return false;

    // Check start and end halfedges are valid
    int num_halfedges = m.n_halfedges();
    for (int i = 0; i < num_segments; ++i) {
        if (m_is_deleted[i]) continue;

        // start halfedge is valid
        int h_start = m_start[i];
        if ((h_start < 0) || (h_start >= num_halfedges)) {
            spdlog::error("Start {} of segment {} is invalid", h_start, i);
            return false;
        }

        // end halfedge is valid
        int h_end = m_end[i];
        if ((h_end < 0) || (h_end >= num_halfedges)) {
            spdlog::error("End {} of segment {} is invalid", h_end, i);
            return false;
        }

        // start and end halfedges in same face
        if (m.f[h_start] != m.f[h_end]) {
            spdlog::error("Segment {} = ({}, {}) is not contained in a face", i, h_start, h_end);
            return false;
        }
    }

    // Check halfedge to segment map is valid
    // TODO
    // for (int i = 0; i < num_halfedges; ++i) {
    //    const auto& segments = itr.second;
    //    for (int segment_index : segments) {
    //        if (!is_valid_segment_index(segment_index)) return false;
    //        if (m_start[segment_index] != h) return false;
    //    }
    //}

    return true;
}

// Split one segment into two with new splitting edge having the given halfedge indices
void DualLoopConnectivity::split_segment(
    int segment_index,
    int halfedge_index,
    int opposite_halfedge)
{
    assert(is_valid_connectivity());
    assert(is_valid_segment_index(segment_index));

    // Get the next segment and a new segment
    int next_segment_index = m_next[segment_index];
    int new_segment_index = create_segment_index();

    // Get the halfedges at the start and end of the current segment
    int h_start = m_start[segment_index];
    int h_end = m_end[segment_index];

    // Connect new segment in the loop
    m_next[segment_index] = new_segment_index;
    m_prev[new_segment_index] = segment_index;
    m_next[new_segment_index] = next_segment_index;
    m_prev[next_segment_index] = new_segment_index;

    // Set the start and end halfedge of the current segment
    m_start[segment_index] = h_start;
    m_end[segment_index] = halfedge_index;

    // Set the start and end halfedge of the new segment
    m_start[new_segment_index] = opposite_halfedge;
    m_end[new_segment_index] = h_end;

    // Add the new segment to the halfedge map
    m_halfedge_map.add_segment(opposite_halfedge, new_segment_index);

    assert(is_valid_connectivity());
}

// Change the halfedge indices for the start and end of the given (adjacent) segments.
// This is needed for flipping an edge in a loop that enters and leaves a flip
// quad on opposite sides.
void DualLoopConnectivity::flip_segments(
    int first_segment_index,
    int second_segment_index,
    int halfedge_index,
    int opposite_halfedge)
{
    assert(is_valid_connectivity());
    assert(is_valid_segment_index(first_segment_index));
    assert(is_valid_segment_index(second_segment_index));
    assert(m_next[first_segment_index] == second_segment_index);

    // Overwrite shared edge indices
    m_end[first_segment_index] = halfedge_index;
    m_start[second_segment_index] = opposite_halfedge;

    // Add the second segment to the halfedge map
    m_halfedge_map.add_segment(opposite_halfedge, second_segment_index);

    assert(is_valid_connectivity());
}

// Combine two (adjacent) segments into a single segment
void DualLoopConnectivity::combine_segments(int first_segment_index, int second_segment_index)
{
    assert(is_valid_connectivity());
    assert(is_valid_segment_index(first_segment_index));
    assert(is_valid_segment_index(second_segment_index));
    assert(m_next[first_segment_index] == second_segment_index);

    // Get the next segment in the loop
    int next_segment_index = m_next[second_segment_index];

    // Get the start and ending halfedge of the combined segment
    int h_start = m_start[first_segment_index];
    int h_end = m_end[second_segment_index];

    // Remove segment segment from the loop
    m_next[first_segment_index] = next_segment_index;
    m_prev[next_segment_index] = first_segment_index;

    // Set the start and end halfedge of the first segment to the combined values
    m_start[first_segment_index] = h_start;
    m_end[first_segment_index] = h_end;

    // Remove the second segment
    delete_segment_index(second_segment_index);

    assert(is_valid_connectivity());
}

// Get a free segment index
int DualLoopConnectivity::create_segment_index()
{
    int segment_index = -1;

    // Get a free index if one exist
    if (!m_free_indices.empty()) {
        segment_index = m_free_indices.back();
        m_free_indices.pop_back();
        m_is_deleted[segment_index] = false;
    }
    // Allocate more space if not
    else {
        segment_index = m_next.size();
        m_next.push_back(-1);
        m_prev.push_back(-1);
        m_start.push_back(-1);
        m_end.push_back(-1);
        m_is_deleted.push_back(false);
    }

    return segment_index;
}

// Mark a segment index as deleted and free for reuse
void DualLoopConnectivity::delete_segment_index(int segment_index)
{
    m_free_indices.push_back(segment_index);
    m_is_deleted[segment_index] = true;
}


} // namespace Holonomy
} // namespace Penner