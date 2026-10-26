#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/halfedge.h"
#include "resolution/hierarchy/overlay.h"
#include "resolution/hierarchy/simplifier.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"
#include <queue>
#include <igl/Timer.h>
#include <igl/doublearea.h>

#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "main_helper.h"

#include <nlohmann/json.hpp>

namespace Penner {
namespace Resolution {

//SymDir::Parameters read_parameters(json& config);

class VoronoiOverlayMesh : public SimplificationOverlayMesh
{
public:
    bool use_geodesic_refinement = false; // straighten paths to geodesics when adding parametrization refinement
    bool simplify_parameterization = true; // simplify the parametrized mesh generated from the overlay
    bool simplify_param_overlay = false; // simplify the parametrization connectivity before tracing new edges 
    bool show_trace_failure = false; // open viewer when a tracing operation fails
    bool optimize_layout = false; // use uv optimization at the coarse level
    std::string output_dir = "";
    std::string name =  "";
    std::string symdir_json = "scripts/json/symdir.json";
    Scalar path_curvature_threshold = 0.;
    Scalar infinity = 1e20;

  VoronoiOverlayMesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& mesh);

  // ---------------
  // Voronoi Regions
  // ---------------

  /**
   * @brief Grow Voronoi regions from the set of initial seeds using a multi-source Dijkstra,
   * adding additional seeds as necessary to ensure the tiles are simply connected
   * 
   * @param seeds: initial seed set
   * @return true if succeeds
   */
  bool grow_voronoi_regions(const std::vector<int>& seeds);

  /**
   * @brief Grow Voronoi regions from the set of initial seeds using a multi-source Dijkstra,
   * adding additional seeds as necessary to ensure the tiles are simply connected and to create
   * an approximately uniform sampling of vertices in the mesh.
   * 
   * @param seeds: initial seed set
   * @param ratio: target ratio of simplified to original vertices
   * @param normal_scale: if positive, use curvature adaptive weights that prioritize high curvature regions
   * @param dual_length_ratio: minimum allowed ratio between the sum of Voronoi boundary edges
   */
  void grow_uniform_voronoi_regions(
      std::vector<int>& seeds,
      Scalar ratio,
      Scalar normal_scale=0.,
      Scalar dual_length_ratio=0.1);

  /**
   * @brief Set a maximum voronoi distance, inserting additional seeds as necessary.
   * 
   * @param max_distance: maximum allowed distance between seeds
   */
  void set_max_voronoi_distance(Scalar max_distance);

  /**
   * @brief Set a maximum voronoi distance for a subset of vertices, inserting additional
   * seeds as necessary.
   * 
   * @param max_distance: maximum allowed distance between seeds
   * @param is_allowed: is vertex allowed as a seed vertex
   */
  void set_constained_max_voronoi_distance(
      Scalar max_distance,
      const std::vector<bool>& is_allowed);

  /**
   * @brief Set a maximum weighted voronoi distance, inserting additional seeds as necessary.
   * 
   * @param max_distance: maximum allowed weighted distance between seeds
   * @param weights: per-vertex weights to scaled distance by
   */
  void set_weighted_max_voronoi_distance(
      Scalar max_distance,
      const VectorX& weights);

  /**
   * @brief Add additional Voronoi seeds to avoid badly conditioned dual Delaunay edges
   * 
   * @param dual_length_ratio: minimum allowed ratio between the sum of Voronoi boundary edges
   * adjacent to a vertex relative to the entire perimeter
   */
  void make_voronoi_uniform(Scalar dual_length_ratio);


  //bool grow_voronoi_regions_recursive();
  //void make_voronoi_simple();

  // ----------------
  // Voronoi Pipeline
  // ----------------

  /**
   * @brief Build the basic voronoi connectivity from 
   *   1. split_voronoi_edges
   *   2. split_voronoi_faces
   *   3. find_voronoi_dual_edges
   *   4. propagate_voronoi_vertices_to_faces
   */
  void build_voronoi_connectivity();

  /**
   * @brief Split edges between vertex regions, and store the maximum distance of the inserted vertex
   * to a neighboring seed
   * 
   */
  void split_voronoi_edges();

  /**
   * @brief Split voronoi faces to connect split voronoi edges, marking the inserted edges as
   * embedded Voronoi region edges.
   * 
   * Also begins to build the Delaunay connectivity (dual to the Voronoi connectivity), excluding
   * the opposite relations.
   * 
   * WARNING: assumes split_voronoi_edges has been called
   */
  void split_voronoi_faces();

  /**
   * @brief Label the Voronoi region boundary edges, and find the midpoints of these edges
   * 
   * Also completes the Delaunay connectity by computing opposite halfedges, and copies it to the
   * simplified mesh embedding overlay data
   * 
   * WARNING: assumes split_voronoi_faces has been called
   */
  void find_voronoi_dual_edges();

  /**
   * @brief Label faces with nearest voronoi seed.
   * 
   * WARNING: assumes find_voronoi_dual_edges has been called
   */
  void propagate_voronoi_vertices_to_faces();

  // -----------------
  // Delaunay Pipeline
  // -----------------

  /**
   * @brief Trace paths from voronoi region centers to voronoi boundary midpoints, marking
   * traced edges as part of the simplified embedded edge.
   * 
   */
  void trace_delaunay_edges();

  /**
   * @brief Iteratively remove and retrace delaunay edge paths, removing the restriction that 
   * they pass through the Voronoi boundary midpoint.
   * 
   */
  void retrace_edges();

  /**
   * @brief Use an intrinsic geodesic flipping method to straighten embedded simplified edge paths.
   * 
   */
  void straighten_edges();
  
  // -----------
  // Frame Field
  // -----------

  /**
   * @brief Sample field on an embedded connectivity at all faces of the overlay mesh
   * 
   * @param reference_field_overlay: reference field on the faces of the embedded mesh
   * @param thetas_overlay: field angle relative to the reference field on the faces of the embedded mesh
   * @param period_jumps_overlay: period jump opposite corners of the embedded mesh
   * @param overlay: embedded mesh connectivity
   * @return sampled field data on the fine overlay connectivity
   */
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXi>
  sample_field_from_overlay(
      const Eigen::MatrixXd& reference_field_overlay,
      const Eigen::VectorXd& thetas_overlay,
      const Eigen::VectorXi& period_jumps_overlay,
      const OverlayStructure& overlay) const;

  /**
   * @brief Comb cross field directions so the period jumps are only nontrivial on a given cut
   * 
   * WARNING: assumes the cut is a cut to disk with all singularities on the cut
   * 
   * @param is_overlay_cut: mask for overlay cut edges where period jumps are allowed
   * @param start_halfedge_index: comb field starting from the face adjacent to this halfedge
   * @param thetas: field angle relative to the reference field on the faces of the overlay mesh
   * @param period_jumps: period jump opposite corners of the overlay mesh
   * @param overlay: embedded overlay connectivity
   */
  void comb_field_to_overlay_seam(
      const std::vector<bool> is_overlay_cut,
      int start_halfedge_index,
      Eigen::VectorXd& thetas,
      Eigen::VectorXi& period_jumps,
      const OverlayStructure& overlay) const;
  
  /**
   * @brief Compute the principal direction of a frame field defined an overlay mesh
   * using the face planes of an embedded mesh.
   * 
   * WARNING: assumes field directions are sampled from the embedded mesh geometry
   * 
   * @param V_overlay: vertices of the embedded mesh
   * @param F_overlay: faces of the embedded mesh
   * @param reference_field: reference field on the faces of the overlay mesh
   * @param thetas: field angles on the faces of the overlay mesh
   * @param overlay: embedded mesh connectivity
   * @return principal field directions on the overlay mesh
   */
  Eigen::MatrixXd compute_overlay_principal_direction(
      const Eigen::MatrixXd& V_overlay,
      const Eigen::MatrixXi& F_overlay,
      const Eigen::MatrixXd& reference_field,
      const Eigen::VectorXd& thetas,
      const OverlayStructure& overlay) const;

  /**
   * @brief Average field directions on the faces of the overlay mesh to the parent
   * face of an embedded mesh.
   * 
   * The average is area weighted, and projected orthogonally onto the tangent plane
   * of the embedded mesh
   * 
   * @param V: embedded mesh vertices
   * @param F: embedded mesh faces
   * @param field_direction: per-overlay-face principal field direction
   * @param overlay: embedded mesh connectivity
   * @return averaged principal field direction on the embedded mesh (i.e.., u gradient)
   * @return rotated principal field direction (i.e., v gradient)
   */
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> average_overlay_field(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& F,
      const Eigen::MatrixXd& field_direction,
      const OverlayStructure& overlay) const;

  /**
   * @brief Average field directions on the faces of the overlay mesh to the parent
   * face of the simplified mesh embedded mesh.
   * 
   * The average is area weighted, and projected orthogonally onto the tangent plane
   * of the embedded mesh
   * 
   * @param field_direction: per-overlay-face principal field direction
   * @return averaged principal field direction on the embedded mesh (i.e.., u gradient)
   * @return rotated principal field direction (i.e., v gradient)
   */
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> transfer_overlay_field_to_simplified(
    const Eigen::MatrixXd& field_direction
  ) const;

  /**
   * @brief Sample the principal directions of the original field on the current
   * overlay mesh, combed using a cut on the simplified mesh
   * 
   * @return combed field direction on the overlay mesh
   */
  Eigen::MatrixXd transfer_original_field_to_overlay() const;

  // ---------------
  // Parametrization
  // ---------------

  /**
   * @brief Generate a parametrization on the simplified connectivity
   * 
   * @param homology_basis_loops 
   * @param kappa 
   */
  void parameterize_with_overlay(
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa);

  /**
   * @brief Optimize the parametrization associated with the simplified embedded mesh.
   * 
   * Uses the UV optimization parameter json file at symdir_json.
   * 
   * WARNING: assumes initial simplified parametrization has been computed
   * 
   */
  void optimize_simplified_parameterization();

  /**
   * @brief Create a geometric embedding for the simplified connectivity using Tutte maps,
   * and use it to transfer the parametrization on the simplified mesh to the overlay connectivity
   * 
   * Optional side effects include:
   *   - straightening geodesic paths
   *   - refining to ensure the embedded meshes are three connected
   *   - rebuilding the original mesh embedding with harmonic maps
   *
   * 
   * If simplify_parametrization is enabled, the overlay mesh is simplified after the
   * parametrization is transferred
   * 
   * @return overlay vertices
   * @return overlay faces
   * @return overlay uv vertices
   * @return overlay parametrization faces
   * @return map from overlay to original faces
   * @return map from overlay to original corners
   * @return per-corner mask of original mesh edges
   */
  std::tuple<
    Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi,
    std::vector<int>, Eigen::MatrixXi, Eigen::MatrixXi>
  transfer_simplified_parameterization();

  // -----------------
  // Atomic Operations
  // -----------------

  // need to override to handle additional bookkeeping
  virtual int create_face() override;
  virtual int create_edge() override;
  virtual int create_vertex() override;
  virtual int split_edge(int halfedge_index) override;
  virtual int split_face(int h, int g) override;
  virtual int add_center_to_face(int face_index) override;
  virtual bool is_regular_face(int face_index) const override;
  virtual bool is_regular_edge(int edge_index) const override;
  virtual void view() const override;
  virtual void collect_garbage() override;

  // ---------------
  // Data Structures
  // ---------------

  /**
   * @brief Data structure for discrete voronoi regions (dual to a Delauany triangulation)
   * 
   * The voronoi regions are rooted at primal vertex seeds, and are composed of unions of
   * vertex regions (which ensures their dual has triangular faces)
   * 
   */
  struct VoronoiStructure {
    Scalar infinity = 1e20;
    Scalar max_distance = infinity;

    VoronoiStructure(int num_vertices, int num_edges, int num_faces)
    : face_region(num_faces, -1)
    , edge_region(num_edges, -1)
    , edge_type(num_edges, EdgeType::EF)
    , vertex_region(num_vertices, -1)
    , vertex_type(num_vertices, VertexType::VF)
    , distance(num_vertices, infinity)
    , dual_halfedge(2 * num_edges, -1)
    {}

    void reset()
    {
      int num_faces = face_region.size();
      int num_edges = edge_region.size();
      int num_vertices = vertex_region.size();
      face_region = std::vector<int>(num_faces, -1);
      edge_region = std::vector<int>(num_edges, -1);
      edge_type = std::vector<EdgeType>(num_edges, EdgeType::EF);
      vertex_region = std::vector<int>(num_vertices, -1);
      vertex_type = std::vector<VertexType>(num_vertices, VertexType::VF);
      distance = std::vector<Scalar>(num_vertices, infinity);
      dual_halfedge = std::vector<int>(2 * num_edges, -1);
      mesh = Connectivity();
      dual_vertices = {};
    }

    void collect_garbage(
      const std::vector<int>& he_new2old,
      const std::vector<int>& v_new2old,
      const std::vector<int>& e_new2old,
      const std::vector<int>& f_new2old,
      const std::vector<int>& he_old2new,
      const std::vector<int>& v_old2new
    ) {
      face_region = vector_compose(face_region, f_new2old);
      edge_region = vector_compose(edge_region, e_new2old);
      edge_type = vector_compose(edge_type, e_new2old);
      vertex_region = vector_compose(vertex_region, v_new2old);
      vertex_type = vector_compose(vertex_type, v_new2old);
      distance = vector_compose(distance, v_new2old);
      dual_halfedge = vector_compose(dual_halfedge, he_new2old);

      dual_vertices = vector_compose(v_old2new, dual_vertices);
      dual_halfedge_starts = vector_compose(he_old2new, dual_halfedge_starts);
      dual_edge_midpoints = vector_compose(v_old2new, dual_edge_midpoints);
    }

    std::vector<int> face_region; // face of M containing the given face
    std::vector<int> edge_region; // face or edge of M containing the given edge
    std::vector<EdgeType> edge_type; // label whether the edge is in an edge or face of M
    std::vector<int> vertex_region; // face, edge, or vertex of M containing the given vertex
    std::vector<VertexType> vertex_type; // label whether the vertex is interior, boundary, or intersection
    std::vector<Scalar> distance; // distance from seed
    std::vector<int> dual_halfedge; // index for dual halfedge edges

    std::vector<int> dual_vertices;
    std::vector<int> dual_halfedge_starts;
    std::vector<int> dual_edge_midpoints;

    Connectivity mesh;
  };


  /**
   * @brief Utility class to trace paths on a voronoi mesh between two vertices
   * 
   */
  class PathTracer
  {
  public:
    VoronoiOverlayMesh& parent;
    PathTracer(VoronoiOverlayMesh& voronoi)
    : parent(voronoi) {}

    bool trace_path(
        int first_vertex,
        int second_vertex,
        int halfedge_index,
        int edge_index,
        std::vector<int>& halfedges,
        int face_index);

    typedef std::pair<int, Scalar> WeightedVertex;
    const Scalar inf_cost = 1e20;
      bool show_failure = true;
    // Initialize the stack of vertices to process with all vertices
    struct VertexCompare 
    {
      bool operator() (const WeightedVertex& left, const WeightedVertex& right)
      {
        return left.second > right.second;
      };
    };
    typedef std::priority_queue<WeightedVertex, std::vector<WeightedVertex>, VertexCompare> PriorityQueue;

  private:

    virtual void initialize() {}
    virtual void clean() {}
    virtual void set_vertex_cost(int vertex, Scalar value) = 0;
    virtual Scalar get_vertex_cost(int vertex)  = 0;
    virtual void set_halfedge_from_vertex(int vertex, int halfedge) = 0;
    virtual int get_halfedge_from_vertex(int vertex) = 0;
    virtual void add_vertex_to_queue(int vertex) = 0;
    virtual WeightedVertex get_vertex_to_process() = 0;
    virtual bool is_vertex_queue_empty() = 0;
    virtual bool is_processed_vertex(int vertex) = 0;
    virtual void mark_processed_vertex(int vertex) = 0;
  };

  class PathTracerList : public PathTracer
  {
  public:
    PathTracerList(VoronoiOverlayMesh& voronoi)
    : PathTracer(voronoi)
    {}

  private:
    std::vector<bool> m_is_processed_vertex;
    std::vector<int> m_halfedge_from_vertex;
    std::vector<Scalar> m_vertex_cost;
    PriorityQueue m_vertices_to_process;

    void initialize() override;

    void set_vertex_cost(int vertex, Scalar value) override
    {
      m_vertex_cost[vertex] = value;
    }

    Scalar get_vertex_cost(int vertex) override
    {
      return m_vertex_cost[vertex];
    }

    void set_halfedge_from_vertex(int vertex, int halfedge) override
    {
      m_halfedge_from_vertex[vertex] = halfedge;
    }

    int get_halfedge_from_vertex(int vertex) override
    {
      return m_halfedge_from_vertex[vertex];
    }

    void add_vertex_to_queue(int vertex) override
    {
      m_vertices_to_process.push(std::make_pair(vertex, get_vertex_cost(vertex)));
    }

    WeightedVertex get_vertex_to_process() override
    {
      WeightedVertex weighted_vertex =  m_vertices_to_process.top();
      m_vertices_to_process.pop();
      return weighted_vertex;
    }

    bool is_vertex_queue_empty() override
    {
      return m_vertices_to_process.empty();
    }

    bool is_processed_vertex(int vertex) override
    {
      return m_is_processed_vertex[vertex];
    }

    void mark_processed_vertex(int vertex) override
    {
      m_is_processed_vertex[vertex] = true;
    }

  };

  class PathTracerCommonList : public PathTracer
  {
  public:
    PathTracerCommonList(VoronoiOverlayMesh& voronoi)
    : PathTracer(voronoi)
    , m_is_processed_vertex(0)
    , m_halfedge_from_vertex(0)
    , m_vertex_cost(0)
    , m_modified_vertices(0)
    , m_vertices_to_process()
    {}

  private:
    std::vector<bool> m_is_processed_vertex;
    std::vector<int> m_halfedge_from_vertex;
    std::vector<Scalar> m_vertex_cost;
    PriorityQueue m_vertices_to_process;
    std::vector<int> m_modified_vertices;

    void initialize() override;
    void clean() override;

    void set_vertex_cost(int vertex, Scalar value) override
    {
      m_vertex_cost[vertex] = value;
    }

    Scalar get_vertex_cost(int vertex) override
    {
      return m_vertex_cost[vertex];
    }

    void set_halfedge_from_vertex(int vertex, int halfedge) override
    {
      m_halfedge_from_vertex[vertex] = halfedge;
    }

    int get_halfedge_from_vertex(int vertex) override
    {
      return m_halfedge_from_vertex[vertex];
    }

    void add_vertex_to_queue(int vertex) override
    {
      m_vertices_to_process.push(std::make_pair(vertex, get_vertex_cost(vertex)));
      m_modified_vertices.push_back(vertex);
    }

    WeightedVertex get_vertex_to_process() override
    {
      WeightedVertex weighted_vertex =  m_vertices_to_process.top();
      m_vertices_to_process.pop();
      return weighted_vertex;
    }

    bool is_vertex_queue_empty() override
    {
      return m_vertices_to_process.empty();
    }

    bool is_processed_vertex(int vertex) override
    {
      return m_is_processed_vertex[vertex];
    }

    void mark_processed_vertex(int vertex) override
    {
      m_is_processed_vertex[vertex] = true;
    }

  };

  VoronoiStructure voronoi;
  PathTracerCommonList path_tracer;
  std::vector<BaryCoords> he_uv;
  std::vector<Scalar> he_length;
  std::vector<Scalar> u_simp, v_simp;
  std::vector<bool> is_cut_simp;

  // vertices for simplified mesh overaly
  Eigen::MatrixXd simplified_V;
  std::vector<Eigen::Vector3d> PD1_o;
  std::vector<Eigen::Vector3d> PD2_o;
  
  /**
   * @brief Trace an embedded edge path from the first vertex to the second, marking the
   * parent edge and halfedge of the traced edges on the path
   * 
   * @param first_vertex: start vertex of the path
   * @param second_vertex: end vertex of the path
   * @param halfedge_index: halfedge parent for halfedges in the path
   * @param edge_index: edge parent for edges in the path
   * @param halfedges: output list of halfedges (in order) in the path
   * @param face_index: (optional) if positive, only allow edges interior to the given simplified face index
   * @return true iff the path is successfully traced
   */
  bool trace_path(
    int first_vertex,
    int second_vertex,
    int halfedge_index,
    int edge_index,
    std::vector<int>& halfedges,
    int face_index=-1);
    
  /**
   * @brief Generate centers of dual faces, where three voronoi regions meet
   * 
   * @return list of faces in the mesh where three voronoi regions meet
   */
  std::vector<int> generate_dual_faces() const;

  std::tuple<VectorX, VectorX> integrate_simplified_function_onto_original(
    const VectorX& simplified_function);


  std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
  generate_voronoi_boundary(
      const Eigen::MatrixXd& V,
      const std::vector<int>& vtx_reindex) const;

  std::vector<bool> compute_layout();

  //std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<bool>> compute_length_layout();

  // --------------
  // Viewer Helpers
  // --------------

  void add_voronoi_to_viewer(
      const std::string& mesh_handle,
      const VoronoiStructure& voronoi_structure) const;
  void add_voronoi_edges_to_viewer(
      const Eigen::MatrixXd& V,
      const std::vector<int>& vtx_reindex) const;
  void add_voronoi_points_to_viewer(
      const Eigen::MatrixXd& V,
      const std::vector<int>& vtx_reindex) const;
      
  nlohmann::json log;
private:

  igl::Timer timer;

  // helpers for the voronoi structure in complex operations
  void split_edge_voronoi(
      int halfedge_index,
      int edge_index,
      VoronoiStructure& voronoi_structure);
  int split_face_voronoi(
      int e,
      int face_index,
      VoronoiStructure& voronoi_structure);
  void add_center_to_face_voronoi(
      int vertex_index,
      int face_index,
      VoronoiStructure& voronoi_structure);

  // Warning: buggy
  void parameterize_with_layout(
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa
  );

};



}
}
