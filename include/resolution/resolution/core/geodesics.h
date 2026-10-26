#pragma once
#include "resolution/core/common.h"
#include "resolution/hierarchy/voronoi.h"

#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/meshio.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

/**
 * @brief Methods to insert geodesic paths into a halfedge mesh.
 * 
 * TODO: Rename this file and main claiss to clarify that it is inserting geodesic paths.
 * 
 */

namespace Penner
{
namespace Resolution
{

/**
 * @brief Convert a collection of polyline paths into a flat node-edge representation with component
 * labels.
 *
 * Each path in @p polyline becomes a sequence of directed edges connecting consecutive nodes.
 * Nodes from different paths are stored contiguously; the returned label array identifies which
 * path each edge belongs to.
 *
 * @tparam Pos3D position type (e.g. Eigen::Vector3d or a 2D UV coordinate)
 * @param polyline: collection of paths, each given as an ordered list of positions
 * @return array of node positions
 * @return array of edges, each an index pair {src, dst} into the node array
 * @return per-edge path label (index into @p polyline)
 */
template <typename Pos3D>
std::tuple<std::vector<Pos3D>, std::vector<std::array<int, 2>>, std::vector<int>>
convert_polyline(const std::vector<std::vector<Pos3D>>& polyline)
{
    std::vector<Pos3D> nodes;
    std::vector<std::array<int, 2>> edges;
    std::vector<int> labels;
    int n_paths = polyline.size();
    for (int i = 0; i < n_paths; ++i)
    {
        int n_path_nodes = polyline[i].size();
        for (int j = 0; j < n_path_nodes; ++j)
        {
            int n = nodes.size();
            edges.push_back({n, n + 1});
            nodes.push_back(polyline[i][j]);
            labels.push_back(i);
        }
        nodes.push_back(polyline[i].back());
    }

    return std::make_tuple(nodes, edges, labels);
}

/**
 * @brief A point on a triangle mesh surface, represented in one of three local coordinate frames.
 *
 * - Vertex: the point coincides with mesh vertex @p index.
 * - Edge:   the point lies on the edge between vertices @p edge_base and @p edge_tip at parameter
 *           @p t (i.e. position = (1-t)*base + t*tip). @p index is the edge index.
 * - Face:   the point lies in the interior of face @p index. @p t and the edge fields are unused.
 */
struct SurfacePoint {
    enum class Type { Vertex, Edge, Face };
    Type type;
    int index;
    int edge_base;
    int edge_tip;
    Scalar t;
};

/**
 * @brief Inserts a collection of piecewise-linear surface paths as edges in a halfedge.
 *
 * Given an overlay mesh and initial path descriptions as SurfacePoints, PathStraightener
 * iteratively inserts the paths by:
 *   1. Splitting overlay edges at path crossing points.
 *   2. Tracing paths between the inserted vertices.
 *   3. Separating crossing or overlapping paths (if necessary).
 */
class PathStraightener
{
public:
    using EdgeType = SimplificationOverlayMesh::EdgeType;
    using VertexType = SimplificationOverlayMesh::VertexType;

    /**
     * @brief Construct a PathStraightener for the given paths
     *
     * @param overlay_mesh: Voronoi overlay mesh; modified in place during straightening
     * @param paths: initial path descriptions as sequences of SurfacePoints
     */
    PathStraightener(
        VoronoiOverlayMesh& overlay_mesh,
        const std::vector<std::vector<SurfacePoint>>& paths
    );

    /**
     * @brief Construct a PathStraightener, marking which paths are known to intersect one another.
     *
     * Intersecting paths require extra separation steps; non-intersecting paths can be
     * inserted independently.
     *
     * @param overlay_mesh: Voronoi overlay mesh; modified in place during straightening
     * @param paths: initial path descriptions as sequences of SurfacePoints
     * @param is_intersecting_path: per-path flag; true if the path crosses other paths
     */
    PathStraightener(
        VoronoiOverlayMesh& overlay_mesh,
        const std::vector<std::vector<SurfacePoint>>& paths,
        const std::vector<bool>& is_intersecting_path);

    /**
     * @brief Split overlay edges at points where paths cross them.
     *
     * @param overlay_mesh: overlay mesh to modify
     */
    void split_edges(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief Split overlay edges and record which halfedges were split per path.
     *
     * @param overlay_mesh: overlay mesh to modify
     * @param edges_to_split: output; for each path, the list of {halfedge, halfedge} pairs split
     */
    void split_edges(
        VoronoiOverlayMesh& overlay_mesh,
        std::vector<std::vector<std::array<int, 2>>>& edges_to_split);

    /**
     * @brief Create edges between inserted vertices introduced by split_edges
     *
     * @param overlay_mesh: overlay mesh to modify
     */
    void join_edges(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief Retrace all redundant paths to ensure they only intersect at endpoints.
     * 
     * @param overlay_mesh: overlay mesh carrying the current metric
     */
    void retrace_paths(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief Run the full straightening pipeline: split, join, retrace
     *
     * @param overlay_mesh: overlay mesh to operate on
     */
    void run(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief Visualize the current state of the paths on the overlay mesh.
     *
     * @param overlay_mesh: overlay mesh for geometric context
     */
    void view(VoronoiOverlayMesh& overlay_mesh);

    /**
     * @brief Remove path @p e from the mesh, allowing it to be retraced.
     *
     * @param overlay_mesh: overlay mesh
     * @param e: path index to remove
     */
    void remove_path(VoronoiOverlayMesh& overlay_mesh, int e);

    /**
     * @brief Retrace path @p e.
     *
     * @param overlay_mesh: overlay mesh carrying the metric
     * @param e: path index to retrace
     * @param f: optional starting face hint; -1 uses automatic detection
     */
    void retrace_path(VoronoiOverlayMesh& overlay_mesh, int e, int f=-1);

    /**
     * @brief Retrace path @p e while enforcing consistent orientation with adjacent paths.
     *
     * @param overlay_mesh: overlay mesh carrying the metric
     * @param e: path index to retrace
     */
    void retrace_consistent_path(VoronoiOverlayMesh& overlay_mesh, int e);

    /**
     * @brief Check whether path @p e is topologically and geometrically valid.
     *
     * @param overlay_mesh: overlay mesh providing connectivity
     * @param e: path index to validate
     * @return true if the path is valid
     */
    bool is_valid_path(VoronoiOverlayMesh& overlay_mesh, int e);

    /**
     * @brief Compute a spanning subset of paths (a basis for the path complex).
     *
     * Returns the indices of paths that form a spanning set, eliminating redundant ones.
     *
     * @param overlay_mesh: overlay mesh for connectivity queries
     * @return indices of spanning paths
     */
    std::vector<int> compute_spanning_paths(VoronoiOverlayMesh& overlay_mesh);

private:
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<std::vector<SurfacePoint>> polyline_pts;
    std::vector<bool> is_redundant_path;

    std::vector<int> v_new2old, v_old2new;
    Eigen::SparseMatrix<int> vv2he_o;
    std::vector<std::vector<int>> split_halfedges;
    std::vector<std::vector<int>> path_halfedges;

    std::vector<std::vector<int>> redundant_edges;
    std::vector<std::vector<int>> redundant_vertices;
    std::vector<int> halfedge_counts;

    std::vector<int> generate_face_chain(const std::vector<std::array<int, 2>>& faces) const;
    int rotate_halfedge_to_tip(const VoronoiOverlayMesh& overlay_mesh, int hij, int vk) const;
    int get_start_halfedge(const VoronoiOverlayMesh& overlay_mesh, int e);

    /**
     * @brief Attempt to separate a simple (non-self-intersecting) path @p e from others.
     *
     * @param overlay_mesh: overlay mesh to modify
     * @param e: path index to separate
     * @return true if the separation succeeded, false if the path could not be resolved
     */
    bool separate_simple_path(VoronoiOverlayMesh& overlay_mesh, int e);

    /**
     * @brief Locally separate any remaining crossing or co-incident paths.
     *
     * WARNING: currently fails for complex intersections; use retrace_paths instead
     *
     * @param overlay_mesh: overlay mesh to modify
     */
    void separate_paths(VoronoiOverlayMesh& overlay_mesh);
};

/**
 * @brief Get the 3D position of a SurfacePoint on an overlay mesh.
 *
 * @param V_o: vertex positions of the original mesh (#V x 3)
 * @param overlay_mesh: overlay mesh providing barycentric embedding
 * @param pt: surface point to evaluate
 * @return 3D position of @p pt
 */
Eigen::Vector3d get_surface_point_position(
    Eigen::MatrixXd& V_o,
    VoronoiOverlayMesh& overlay_mesh,
    SurfacePoint& pt);

/**
 * @brief Get the 3D positions for all SurfacePoints in a polyline.
 *
 * @param overlay_mesh: overlay mesh providing barycentric embedding
 * @param polyline_pts: per-path sequences of SurfacePoints
 * @return per-path sequences of 3D positions
 */
std::vector<std::vector<Eigen::Vector3d>> get_surface_point_positions(
    VoronoiOverlayMesh& overlay_mesh,
    std::vector<std::vector<SurfacePoint>>& polyline_pts
);

/**
 * @brief Convert a geometry-central SurfacePoint to the SurfacePoint representation.
 * 
 * WARNING: Face points are an error state, and all geometric data is forgotten.
 *
 * @param pt: geometry-central surface point
 * @return equivalent SurfacePoint
 */
SurfacePoint convert_surface_point(const geometrycentral::surface::SurfacePoint& pt);

/**
 * @brief Compute geodesic path points for a set of vertex-to-vertex paths on an overlay mesh.
 *
 * Each path in @p paths is a sequence of overlay mesh vertex indices. The returned points
 * describe the piecewise-geodesic interpolation of each path as SurfacePoints, suitable for
 * subsequent straightening via PathStraightener.
 *
 * @param overlay_mesh: Voronoi overlay mesh carrying metric and connectivity
 * @param paths: paths to process, each given as an ordered list of overlay vertex indices
 * @return per-path lists of SurfacePoints tracing the geodesic
 */
std::vector<std::vector<SurfacePoint>> generate_geodesic_path_points(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<int>>& paths);

/**
 * @brief Embed geodesic paths into a VoronoiOverlayMesh by tracing and straightening them.
 *
 * Each path in @p paths is a sequence of overlay vertex indices. The function traces a
 * piecewise-linear path between consecutive vertices and straightens it into a geodesic
 * embedded in the overlay mesh connectivity.
 * 
 * WARNING: Replaced with custom geodesic tracing to avoid any intersections.
 *
 * @param overlay_mesh: Voronoi overlay mesh; modified to contain the embedded paths
 * @param paths: paths to embed, each given as an ordered list of overlay vertex indices
 */
// TODO: replace voronoi with basic overlay once separate retracing step
void add_geodesic_paths(
    VoronoiOverlayMesh& overlay_mesh,
    const std::vector<std::vector<int>>& paths);

}
}
