#pragma once
#include "resolution/core/common.h"
#include "resolution/hierarchy/voronoi.h"
#include "resolution/core/geodesics.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace Penner
{
namespace Resolution
{
    
/**
 * @brief Representation of a connected piece-wise geodesic path on a halfedge mesh, with indexed wedges
 * between adjacent edges.
 * 
 */
class GeodesicPath
{
public:
    std::vector<int> to; // wedge index at tip of segment
    std::vector<int> in; // segment index into wedge
    std::vector<int> next; // next segment
    std::vector<int> prev; // previous segment
    std::vector<int> halfedge; // halfedge on base mesh
    std::vector<Scalar> lengths; // history of path lengths
    std::vector<Scalar> sizes; // history of path sizes

    /**
     * @brief Construct a new geodesic path object from a halfedge path
     * 
     * @param path: continuous sequence of halfedges on a base mesh
     */
    GeodesicPath(const std::vector<int>& path);

    /**
     * @brief Get the halfedge sequence of the geodesic path
     * 
     * @return sequence of halfedges on the base mesh
     */
    std::vector<int> get_path_halfedges() const;

    /**
     * @brief Determine if the path is a single edge
     * 
     * @return true iff the path length is one
     */
    bool is_edge() { return (start == end); }

    /**
     * @brief Get the index of the first segment in the path
     * 
     * @return start index
     */
    int get_start() const { return start; }

    /**
     * @brief Get the index of the last segment in the path
     * 
     * @return end index
     */
    int get_end() const { return end; }

    /**
     * @brief Set the start segment of the path
     * 
     * @param index: new path start index
     */
    void set_start(int index) { start = index; }

    /**
     * @brief Set the end segment of the path
     * 
     * @param index: new path end index
     */
    void set_end(int index) { end = index; }

    /**
     * @brief Determine if a segment is the start of the path.
     * 
     * @param index: index of segment to check
     * @return true iff the index is the start index
     */
    bool is_start(int index) { return (index == start); }

    /**
     * @brief Determine if a segment is the end of the path.
     * 
     * @param index: index of segment to check
     * @return true iff the index is the end index
     */
    bool is_end(int index) { return (index == end); }

    /**
     * @brief Determine if the wedge is still valid
     * 
     * @param index: wedge index to check
     * @return true iff the 
     */
    bool is_wedge_valid(int index) const { return m_is_wedge_valid[index]; }

    /**
     * @brief Create a isolated segment with given halfedge
     * 
     * @param h: halfedge on the base mesh
     * @return new segment index
     */
    int create_segment(int h);

    /**
     * @brief Create a wedge at the tip of a segment with a given index
     * 
     * @param index: index of the segment with new wedge at the tip
     * @return index of the new wedge
     */
    int create_wedge(int index);

    /**
     * @brief Create a segment after the one with the given index and a wedge in between.
     * 
     * @param index: previous segment for new segment
     * @param h: halfedge on the base mesh
     * @return index of the new segment
     */
    int create_next_segment(int index, int h);

    /**
     * @brief Mark the wedge as invalid.
     * 
     * @param index: wedge to mark as invalid
     */
    void mark_wedge_invalid(int index) { m_is_wedge_valid[index] = false; }

    /**
     * @brief Determine if the geodesic path is a valid path on the base mesh
     * 
     * @param m: base mesh
     * @return true iff the path is valid and continuous on the base mesh
     */
    bool is_valid(Mesh<Scalar>& m);

private:
    int start;
    int end;

    std::vector<bool> m_is_wedge_valid;
};

/**
 * @brief Class for flipping a network of edge paths to geodesics.
 * 
 * In some cases, flipping all paths to geodesics is not possible. By default, paths are only
 * allowed to intersect at their endpoints. Curves are never allowed to intersect transversely,
 * but tangential intersections are possible.
 * 
 * 
 */
class GeodesicFlipper
{
public:
    Scalar wedge_eps = 1e-5;
    Scalar infinity = 1e10;
    int max_flipouts = -1;
    bool track_deltas = false;
    //Scalar flip_angle = 0.9 * PI;
    Scalar flip_angle = PI;
    Scalar path_curvature_threshold = 0.;

    /**
     * @brief Construct a new Geodesic Flipper object with the given halfedge
     * paths as the initial curve network.
     * 
     * @param overlay_mesh: base mesh
     * @param paths: halfedge paths forming the initial curve network
     * @param allow_intersections: (optional) allow tangential intersections if true
     */
    GeodesicFlipper(
        VoronoiOverlayMesh& overlay_mesh,
        const std::vector<std::vector<int>>& paths,
        bool allow_intersections=false
    );

    /**
     * @brief Construct a new Geodesic Flipper object with the given halfedge
     * paths as the initial curve network.
     * 
     * @param mesh: base mesh
     * @param paths: halfedge paths forming the initial curve network
     * @param allow_intersections: (optional) allow tangential intersections if true
     */
    GeodesicFlipper(
        const Mesh<Scalar>& mesh,
        const std::vector<std::vector<int>>& paths,
        bool allow_intersections=false
    );

    /**
     * @brief Count the number of faces in the overlay mesh.
     * 
     * @return overlay face count
     */
    int count_overlay_faces() const { return overlay_faces; }

    /**
     * @brief Count the number of flips done to flip to geodesic.
     * 
     * @return total flip count
     */
    int count_flips() const { return flip_sequence.size(); }

    /**
     * @brief Basic method to flip the curve network to geodesics.
     * 
     */
    void flip_to_geodesics();

    /**
     * @brief Check if the given wedge in a path is acute and add it to the priority queue if so.
     * 
     * @param path_index: path index
     * @param wedge_index: index of the wedge in the path
     */
    void check_wedge(int path_index, int wedge_index);

    /**
     * @brief Check all wedges.
     * 
     */
    void check_all_wedges();

    /**
     * @brief Check if a wedge can be shorten with flipout, and do so if it can.
     * 
     * @return true iff the wedge is shortened
     */
    bool process_wedge();

    /**
     * @brief Flip out acute edges in the wedge between the two halfedges.
     * 
     * @param hij: first halfedge of the wedge
     * @param hjk: second halfedge of the wedge
     * @param is_left_joint: is wedge to left or right of the two edges
     * @return number of flips done
     */
    int flip_out(int hij, int hjk, bool is_left_joint);

    /**
     * @brief Flip an acute angle on the left of the wedge between the two halfedges.
     * 
     * @param hij: first halfedge of the wedge
     * @param hjk: second halfedge of the wedge
     * @return true iff an edge is flipped
     */
    bool flip_left_acute_angles(int hij, int hjk);

    /**
     * @brief Flip an acute angle on the right of the wedge between the two halfedges.
     * 
     * @param hij: first halfedge of the wedge
     * @param hjk: second halfedge of the wedge
     * @return true iff an edge is flipped
     */
    bool flip_right_acute_angles(int hij, int hjk);

    /**
     * @brief Flip the given halfedge
     * 
     * @param hij: halfedge to flip
     */
    void flip_halfedge(int hij);

    /**
     * @brief Shorten a path at a given wedge by replacing the wedge with a shorter path.
     * 
     * WARNING: assumes the wedge is already flipped out
     * 
     * @param path_index: path index
     * @param wedge_index: index of the wedge in the path
     * @param is_left_joint: is wedge to left or right of the two edges
     */
    void shorten_wedge(int path_index, int wedge_index, bool is_left_joint);

    /**
     * @brief Undo a given number of flips
     * 
     * @param num_flips: number of flips to do
     */
    void undo_flips(int num_flips);

    /**
     * @brief Compute the angle of the wedge clockwise between two halfedges
     * 
     * @param hij: first halfedge of the wedge
     * @param hjk: second halfedge of the wedge
     * @return wedges angle
     */
    Scalar compute_clockwise_wedge_angle(int hij, int hjk);

    /**
     * @brief Compute the angle of the wedge counterclockwise between two halfedges
     * 
     * @param hij: first halfedge of the wedge
     * @param hjk: second halfedge of the wedge
     * @return wedges angle
     */
    Scalar compute_ccw_wedge_angle(int hij, int hjk);

    /**
     * @brief Compute the curvature of the wedge, defined as the maximum defect from Pi
     * 
     * @param hij: first halfedge of the wedge
     * @param hjk: second halfedge of the wedge
     * @param use_signed: use signed defect instead (using clockwise wedge angles)
     * @return wedge curvature
     */
    Scalar compute_wedge_curvature(int hij, int hjk, bool use_signed=false);

    /**
     * @brief Compute the total curvature of the path.
     * 
     * @param path_index: path to compute curvature for
     * @param use_signed: use signed defect (using clockwise wedge angles)
     * @return path curvature
     */
    Scalar compute_path_curvature(int path_index, bool use_signed=false);

    /**
     * @brief Compute all path curvatures.
     * 
     * @param use_signed: use signed defect (using clockwise wedge angles)
     * @return list of curvatures of all paths
     */
    std::vector<Scalar> compute_path_curvatures(bool use_signed=false);

    /**
     * @brief Generate surface points for all paths on the surface
     * 
     * WARNING: computationally intensive as a reversed overlay mesh needs to be computed
     */
    void generate_surface_points();

    /**
     * @brief Get a boolean mask of paths that intersect other paths
     * 
     * @return mask of intersecting paths
     */
    std::vector<bool> get_intersecting_paths();

    /**
     * @brief Add the curve network to the polyscope viewer.
     * 
     * @param overlay_mesh: base mesh
     */
    void add_to_viewer(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief View the curve network.
     * 
     * @param overlay_mesh: base mesh
     */
    void view(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief Flip the curve network to geodesics while taking screenshots of the curve network.
     * 
     * @param overlay_mesh: base mesh
     * @param frame_delta: number of wedge shortens between screenshots
     */
    void animate_flip_to_geodesics(VoronoiOverlayMesh& overlay_mesh, int frame_delta);

    // Initialize the stack of vertices to process with all vertices
    typedef std::tuple<Scalar, int, int, bool> WeightedWedge;
    struct WedgeCompare
    {
      bool operator() (const WeightedWedge& left, const WeightedWedge& right)
      {
        return std::get<0>(left) > std::get<0>(right);
      };
    };
    typedef std::priority_queue<WeightedWedge, std::vector<WeightedWedge>, WedgeCompare> PriorityQueue;

    std::vector<Scalar> deltas;
    std::vector<GeodesicPath> paths;
    std::vector<Scalar> path_length;
    std::vector<Scalar> path_curvature;

    std::vector<std::vector<SurfacePoint>> surface_points;
    std::vector<std::vector<std::array<int, 2>>> edge_splits;
private:
    //OverlayMesh<Scalar> m_o;
    Mesh<Scalar> m_o;
    std::vector<int> edge_path_counts;
    std::vector<std::deque<std::pair<int, int>>> adjacent_paths;
    std::vector<int> h2e, e2h;
    bool keep_simple = true;

    // Predicates
    bool is_farthest_left(int path_index, int index);
    bool is_farthest_right(int path_index, int index);
    bool is_left_wedge_empty(int hij, int hjk);
    bool is_right_wedge_empty(int hij, int hjk);
    bool is_left_wedge_simple(int hij, int hjk);
    bool is_right_wedge_simple(int hij, int hjk);
    bool is_simple();
    bool is_wedge_simple(int path_index, int wedge_index, bool is_left_joint);
    bool is_left_joint_flexible(int path_index, int wedge_index);
    bool is_right_joint_flexible(int path_index, int wedge_index);
    bool is_joint_flexible(int path_index, int wedge_index, bool is_left_joint);
    bool is_vertex_isolated(int vi);

    void add_adjacent_halfedge(int path_index, int index, bool on_left);
    void update_face_angles(int face_index);
    
    bool is_valid();

    VectorX angles;
    std::vector<int> flip_sequence;
    PriorityQueue wedges_to_process;
    int overlay_faces = 0;
};

}
}