#pragma once

#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

  /**
   * @brief Representation of a halfedge mesh that supports standard connectivity modifications:
   *   - Split Edge
   *   - Split Vertex
   *   - Split Face
   *   - Join Vertices
   *   - Join Faces
   * 
   * Additional specific operations, such as to remove degree a one vertex, are supported.
   * 
   * There are also methods for common triangle mesh operations:
   *   - Collapse Edge
   *   - Flip Edge
   *   - Split Edge (while preserving triangular faces)
   * 
   * Many operations also feature a "conservative" implementation that only uses an atomic splice
   * operation. These methods are generally less efficient and less memory aware, and are primarily
   * intended for debugging the faster direct methods.
   */
class HalfedgeMesh
{
public:
  std::vector<int> opp; // Opposite halfedge of halfedge
  std::vector<int> n;   // Next Halfedge of Halfedge, in Counterclockwise Order
  std::vector<int> to;  // To Vertex of Halfedge
  std::vector<int> face;   // Face of Halfedge
  std::vector<int> h2e; // map from halfedges to edges

  std::vector<int> prev;   // Previous Halfedge of Halfedge, in Counterclockwise Order
  std::vector<int> out; // One outgoing Halfedge of vertex;
  std::vector<int> f2h;   // One Halfedge of Face
  std::vector<int> e2h; // map from edges to halfedges

  bool skip_garbage_collection = false;

  /**
   * @brief Construct a new trivial Halfedge Mesh.
   * 
   */
  HalfedgeMesh();

  /**
   * @brief Construct a new Halfedge Mesh with given partial halfedge data (the rest are inferred)
   * 
   * @param opp_: opposite halfedge of halfedge
   * @param n_: next Halfedge of Halfedge
   * @param to_: to Vertex of Halfedge
   * @param face_: face of Halfedge
   * @param h2e_: map from halfedges to edges
   */
  HalfedgeMesh(
    const std::vector<int>& opp_,
    const std::vector<int>& n_,
    const std::vector<int>& to_,
    const std::vector<int>& face_,
    const std::vector<int>& h2e_
  );

  /**
   * @brief Get the number of halfedges in the mesh
   * 
   * @return halfedge count
   */
  int count_halfedges() const {return m_halfedge_count;}

  /**
   * @brief Get the number of faces in the mesh
   * 
   * @return face count
   */
  int count_faces() const {return m_face_count;}

  /**
   * @brief Get the number of edges in the mesh
   * 
   * @return edge count
   */
  int count_edges() const {return m_edge_count;}

  /**
   * @brief Get the number of vertices in the mesh
   * 
   * @return vertex count
   */
  int count_vertices() const {return m_vertex_count;}

  /**
   * @brief Number of edges adjacent to a vertex.
   * 
   * @param vertex_index: index of vertex
   * @return degree of vertex
   */
  int vertex_degree(int vertex_index) const;

  /**
   * @brief Number of edges adjacent to a face.
   * 
   * @param face_index: index of face
   * @return degree of face
   */
  int face_degree(int face_index) const;

  /**
   * @brief Predicate for a degree one vertex
   * 
   * @param vertex_index: index of vertex
   * @return true iff the vertex is degree one
   */
  bool is_vertex_degree_one(int vertex_index) const;

  /**
   * @brief Predicate for a degree two vertex
   * 
   * @param vertex_index: index of vertex
   * @return true iff the vertex is degree two
   */
  bool is_vertex_degree_two(int vertex_index) const;

  /**
   * @brief Get a halfedge in a given face with a given vertex at the tip
   * 
   * WARNING: this halfedge may not be unique
   * 
   * @param face_index: index of the face
   * @param vertex_index: index of the vertex
   * @return halfedge index
   */
  int halfedge_from_face_vertex(int face_index, int vertex_index) const;

  /**
   * @brief Relabel the tip of halfedge with the given vertex
   * 
   * @param halfedge_index: index of a halfedge with tip to reindex
   * @param vertex_index: index of the vertex to put at the tip of the halfedge
   */
  void relabel_tip(int halfedge_index, int vertex_index);

  /**
   * @brief Relabel the face of a halfedge.
   * 
   * @param halfedge_index: index of a halfedge in the face to relabel
   * @param face_index: index of the face label
   */
  void relabel_face(int halfedge_index, int face_index);

  /**
   * @brief Create and allocate space for a new halfedge index
   * 
   * @return index of new halfedge
   */
  virtual int create_halfedge();

  /**
   * @brief Create and allocate space for a new vertex index
   * 
   * @return index of new vertex
   */
  virtual int create_vertex();

  /**
   * @brief Create and allocate space for a new edge index
   * 
   * @return index of new edge
   */
  virtual int create_edge();

  /**
   * @brief Create and allocate space for a new face index
   * 
   * @return index of new face
   */
  virtual int create_face();

  //TODO: Rename to make_bubble
  /**
   * @brief Create a new edge on a bubble with two halfedges, two vertices, and one adjacent face
   * 
   * @return index of the new edge
   */
  int make_edge();

  /**
   * @brief Delete an edge and its associated halfedges
   * 
   * @param edge_index: index of edge to delete
   */
  void delete_edge(int edge_index);

  /**
   * @brief Delete an edge without deleting any associated halfedges
   * 
   * @param edge_index: index of edge to delete
   */
  void delete_dangling_edge(int edge_index);

  /**
   * @brief Delete a vertex
   * 
   * @param vertex_index: index of vertex to delete
   */
  void delete_vertex(int vertex_index);

  /**
   * @brief Delete a face
   * 
   * @param face_index: index of face to delete
   */
  void delete_face(int face_index);

  /**
   * @brief Delete an edge on a bubble with two adjacent vertices and one adjacent face, inverse to make_edge
   * 
   * TODO: replace with edge index
   * 
   * @param halfedge_index: halfedge on the bubble to delete
   */
  void delete_bubble(int halfedge_index);

  /**
   * @brief Perform the atomic halfedge splice operation, swapping the next halfedge of two halfedges.
   * 
   * Resuses existing face and vertex indices to avoid memory usage and fragmentation.
   * 
   * TODO: replace bool with relevant int or void
   * 
   * @param h: first halfedge to splice
   * @param g: second halfedge to splice
   * @return true as always valid
   */
  bool splice_halfedges(int h, int g);

  /**
   * @brief Perform the atomic halfedge splice operation, swapping the next halfedge of two halfedges.
   * 
   * Does not reuse any existing face or vertex indices to avoid unintended references.
   * 
   * TODO: replace bool with relevant int or void
   * 
   * @param h: first halfedge to splice
   * @param g: second halfedge to splice
   * @return true as always valid
   */
  bool splice_halfedges_conservative(int h, int g);

  /**
   * @brief Determine if the endpoints of the halfedge can be joined.
   * 
   * @param halfedge_index: halfedge to join
   * @return true iff the endpoints can be joined
   */
  bool is_join_halfedge_endpoints_valid(int halfedge_index) const;

  /**
   * @brief Remove an edge, joining the endpoints.
   * 
   * TODO: replace bool with relevant int or void
   * 
   * @param halfedge_index: halfedge to join
   * @return true as always succeeds
   */
  bool join_halfedge_endpoints(int halfedge_index);

  /**
   * @brief Remove an edge, joining the endpoints, using atomic splice operations
   * 
   * TODO: replace bool with relevant int or void
   * 
   * @param halfedge_index: halfedge to join
   * @return true as always succeeds
   */
  bool join_halfedge_endpoints_conservative(int halfedge_index);

  /**
   * @brief Split one vertex into two, creating a new edge between them at the tips of the given halfedges.
   * 
   * TODO: make conservative and fast version, and a test for validity
   * 
   * WARNING: g must be distinct from h but have the same tip
   * 
   * @param h: first halfedge with vertex at tip
   * @param g: second halfedge with vertex at tip
   * @return one halfedge of the new edge
   */
  int split_vertex(int h, int g);

  // TODO: add remove degree two vertex method

  /**
   * @brief Split one edge into two, creating one new vertex and no new faces.
   * 
   * @param halfedge_index: index of halfedge in edge to split
   * @return index of halfedge pointing to new vertex in same face as the original halfedge
   */
  virtual int split_edge(int halfedge_index);

  /**
   * @brief Split one edge into two, creating one new vertex and no new faces, using atomic
   * splice operations.
   * 
   * @param halfedge_index: index of halfedge in edge to split
   * @return index of halfedge pointing to new vertex in same face as the original halfedge
   */
  int split_edge_conservative(int halfedge_index);

  /**
   * @brief Split one edge in a triangle mesh into two, creating one new vertex and two faces.
   * 
   * @param halfedge_index: index of halfedge in edge to split
   * @return index of halfedge pointing to new vertex in same face as the original halfedge
   */
  int split_triangle_mesh_edge(int halfedge_index);

  /**
   * @brief Determine if an edge can be removed, joining the adjacent faces.
   * 
   * @param halfedge_index: halfedge to remove
   * @return true iff the adjacent faces can be joined
   */
  bool is_join_edge_faces_valid(int halfedge_index) const;

  /**
   * @brief Remove an edge, joining the adjacent faces.
   * 
   * @param halfedge_index: halfedge to remove
   * @return index of face containing the now removed edge
   */
  virtual int join_edge_faces(int halfedge_index);

  /**
   * @brief Remove an edge, joining the adjacent faces, using atomic splice operations.
   *
   * @param halfedge_index: halfedge to remove
   * @return index of face containing the now removed edge
   */
  int join_edge_faces_conservative(int halfedge_index);

  /**
   * @brief Split the face containing the two halfedges with an edge connecting the tips
   * of the halfedges
   * 
   * TODO: add validity check
   * 
   * @param h: first halfedge in face
   * @param g: second halfedge in face
   * @return halfedge of new edge
   */
  virtual int split_face(int h, int g);

  /**
   * @brief Split the face containing the two halfedges with an edge connecting the tips
   * of the halfedges, using atomic splice operations.
   * 
   * TODO: add validity check
   * 
   * @param h: first halfedge in face
   * @param g: second halfedge in face
   * @return halfedge of new edge
   */
  int split_face_conservative(int h, int g);

  /**
   * @brief Remove a degree one vertex and the single adjacent edge
   * 
   * @param vertex_index: vertex to remove
   * @return index of vertex at root of trimmed edge
   */
  int remove_degree_one_vertex(int vertex_index);

  /**
   * @brief Remove a degree one vertex and the single adjacent edge, using atomic splice
   * operations
   * 
   * @param vertex_index: vertex to remove
   * @return index of vertex at root of trimmed edge
   */
  int remove_degree_one_vertex_conservative(int vertex_index);

  /**
   * @brief Triangulate a face, connecting each edge to a new vertex at the center
   * 
   * @param face_index face to triangulate with new center
   * @return new center vertex
   */
  virtual int add_center_to_face(int face_index);

  // TODO: continue refactor from here

  std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> generate_halfedge_faces() const;
  void reindex_vertices(const std::vector<int>& vtx_reindex);

  bool is_collapse_edge_valid(int halfedge_index) const;

  // WARNING: must remove edges adjacent to tip vertex for overlay
  virtual int collapse_edge(int halfedge_index);
  int collapse_edge_conservative(int halfedge_index);

  // more restrictive condition for halfedge collapse that prevents degenerate triangles
  std::set<int> get_collapse_link(int halfedge_index) const;
  bool is_collapse_link_valid(int halfedge_index) const;

  virtual bool triangulate_face(int face_index);

  virtual int flip_edge(int halfedge_index);

  void triangulate();

  Eigen::SparseMatrix<int> generate_VV_to_halfedge_map() const;

  bool is_simplicial() const;
  virtual bool is_valid() const;
  virtual void collect_garbage();
  void collect_garbage(
      const std::vector<int>& he_new2old,
      const std::vector<int>& e_new2old,
      const std::vector<int>& v_new2old,
      const std::vector<int>& f_new2old,
      const std::vector<int>& he_old2new,
      const std::vector<int>& e_old2new,
      const std::vector<int>& v_old2new,
      const std::vector<int>& f_old2new
  );

  std::tuple<std::vector<int>, std::vector<int>> get_face_reductions() const;
  std::tuple<std::vector<int>, std::vector<int>> get_vertex_reductions() const;
  std::tuple<std::vector<int>, std::vector<int>> get_halfedge_reductions() const;
  std::tuple<std::vector<int>, std::vector<int>> get_edge_reductions() const;
  bool is_halfedge_deleted(int halfedge_index) const { return m_is_halfedge_deleted[halfedge_index]; }
  bool is_edge_deleted(int edge_index) const { return m_is_edge_deleted[edge_index]; }
  bool is_vertex_deleted(int vertex_index) const { return m_is_vertex_deleted[vertex_index]; }
  bool is_face_deleted(int face_index) const { return m_is_face_deleted[face_index]; }

protected:
  void delete_halfedge(int halfedge_index);

  void set_next(int h, int hn)
  {
    n[h] = hn;
    prev[hn] = h;
  }


  std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> reduce_halfedge_maps() const;
  std::tuple<std::vector<int>, std::vector<int>> reduce_face_maps() const;
  std::tuple<std::vector<int>, std::vector<int>> reduce_vertex_maps() const;
  std::tuple<std::vector<int>, std::vector<int>> reduce_edge_maps() const;

private:
  std::vector<bool> m_is_halfedge_deleted; // flag for deleted halfedges
  std::vector<bool> m_is_vertex_deleted; // flag for deleted vertices
  std::vector<bool> m_is_edge_deleted; // flag for deleted edges
  std::vector<bool> m_is_face_deleted; // flag for delete faces 
  int m_halfedge_count;
  int m_face_count;
  int m_edge_count;
  int m_vertex_count;
};

/**
 * @brief Generate VF faces from minimal halfedge data
 * 
 * @param next: next halfedge in mesh
 * @param f2h: map from face to an incident halfedge
 * @param to: vertex index at tip of halfedge
 * @return: list of list of vertices in faces
 * @return: map from triangle corners to opposite halfedge
 */
std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> generate_connectivity_faces(
    const std::vector<int>& next,
    const std::vector<int>& f2h,
    const std::vector<int>& to);

std::tuple<HalfedgeMesh, std::vector<int>, std::vector<std::pair<int, int>>> generate_halfedge(const Eigen::MatrixXi& F);

}
}