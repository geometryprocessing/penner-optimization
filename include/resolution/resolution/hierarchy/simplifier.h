#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/overlay.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"



namespace Penner {
namespace Resolution {

/**
 * @brief Abstract class to handle simplification of an overlay mesh with both 3D and 
 * parametric data embedded in the surface to construct a VF mesh.
 * 
 * The approach is bottom up, starting from the original triangulation and only refining
 * edges as necessary to ensure the parametrization is locally injective
 * 
 */
class Simplifier
{
  public:
    Mesh<Scalar> m;
    std::vector<bool> is_cut_orig;
    std::vector<int> to;
    std::vector<Eigen::Vector2d> he_uv;

    std::vector<int> start_halfedge;
    std::vector<int> end_halfedge;

    std::vector<bool> is_removed_vertex;
    std::vector<bool> is_refined_halfedge;
    std::vector<bool> is_refined_face;
    double area_threshold;
    bool use_angles = true;

    Simplifier(
      const Mesh<Scalar>& mesh,
      const std::vector<bool>& is_cut,
      const std::vector<int>& to_,
      const std::vector<Eigen::Vector2d>& he_uv_
    )
    : m(mesh)
    , is_cut_orig(is_cut)
    , to(to_)
    , he_uv(he_uv_)
    {
        int num_halfedges = m.n_halfedges();
        int num_faces = m.h.size();
        is_refined_halfedge = std::vector<bool>(num_halfedges, false);
        is_refined_face = std::vector<bool>(num_faces, false);

        // actually angle threshold
        area_threshold = 1e-4;
    }

    /**
     * @brief Refine halfedges as necessary to ensure the parametrization does not have
     * inverted triangles.
     * 
     */
    void refine_halfedges();

    /**
     * @brief Attempt to remove inserted vertices from the parametrization without destroying
     * local injectivity and inserting flipped triangles.
     * 
     * WARNING: assumes refine_halfedges has already been called
     * 
     * Note that this method uses a top-down heuristic for removing vertices and is not
     * generally optimal
     * 
     */
    void simplify_vertices();

    /**
     * @brief Get the vertices at the tips of the halfedges returned by get_halfedges.
     * 
     * @param halfedge_index: embedded halfedge index
     * @return list of vertex tips in the embedded halfedge
     */
    std::vector<int> get_halfedge_vertices(int halfedge_index) const;

    /**
     * @brief Get the uv and 3D vertex indices on the boundary of the face with current
     * refinement state.
     * 
     * @param face_index: embedded face index
     * @param uv_vertices: uv vertex indices on the boundary
     * @param vertices: vertex indices on the boundary
     */
    void get_face_vertex_indices(
        int face_index,
        std::vector<int>& uv_vertices,
        std::vector<int>& vertices) const;

    /**
     * @brief Get the overlay halfedges in the embedded halfedge
     * 
     * @param halfedge_index: embedded halfedge index
     * @return list of overlay halfedges in the embedded edge path
     */
    virtual std::vector<int> get_halfedges(int halfedge_index) const = 0;

    /**
     * @brief Determine if the face is self overlapping and can be parametrized without
     * inverted triangles.
     * 
     * @param face_index: embedded face index
     * @return true iff the face is self overlapping
     */
    virtual bool is_face_self_overlapping(int face_index) const = 0;
};

class SimplificationOverlaySimplifier : public Simplifier
{
public:
    typedef SimplificationOverlayMesh::EdgeType EdgeType;
    SimplificationOverlayMesh& parent;
    Eigen::MatrixXd V_o;

    SimplificationOverlaySimplifier(
      SimplificationOverlayMesh& overlay_mesh,
      const std::vector<bool>& is_cut,
      const std::vector<Eigen::Vector2d>& he_uv_
    )
    : parent(overlay_mesh)
    , Simplifier(overlay_mesh.orig.m, is_cut, overlay_mesh.to, he_uv_)
    {
      V_o = parent.generate_overlay_vertices(
        parent.V_orig,
        parent.vtx_reindex_orig,
        parent.orig);
      start_halfedge = parent.find_start_halfedges(parent.orig);
      end_halfedge = parent.find_end_halfedges(parent.orig);
      is_removed_vertex = std::vector<bool>(parent.out.size(), false);
    }

    // Override core functions
    std::vector<int> get_halfedges(int halfedge_index) const override;
    bool is_face_self_overlapping(int face_index) const override;

    /**
     * @brief Get the positions of the vertices on the boundary of the given face.
     * 
     * @param face_index: embedded face index
     * @param uv_vertices: uv vertex positions on the boundary
     * @param vertices: vertex positions on the boundary
     */
    void get_face_vertices(
        int face_index,
        std::vector<Eigen::Vector2d>& uv_vertices,
        std::vector<Eigen::Vector3d>& vertices) const;

    /**
     * @brief Triangulate the embedded mesh face (if possible) and get the subfaces
     * 
     * @param face_index: embedded face index
     * @param face_triangles: list of 3D sub-faces, defined by 3 vertex indices
     * @param uv_face_triangles: list of uv sub-faces, defined by 3 vertex indices
     * @param face_indices: list of sub-faces, using local sequential boundary indexing
     * @return true if the face was triangulated, and false if not
     */
    bool triangulate_face(
        int face_index,
        std::vector<std::array<int, 3>>& face_triangles,
        std::vector<std::array<int, 3>>& uv_face_triangles,
        std::vector<std::array<int, 3>>& face_indices) const;

    /**
     * @brief Build the refined parametrized mesh with maps to the parent elements in the
     * unrefined mesh.
     * 
     * @return refined mesh vertices
     * @return refined mesh faces 
     * @return refined mesh uv vertices
     * @return refined mesh uv faces
     * @return map from refined to original faces
     * @return map from refined to original halfedges (indexed by opposite corner)
     * @return mask for halfedges in original mesh (indexed by opposite corner)
     * > 
     */
    std::tuple<
    Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi,
    std::vector<int>, Eigen::MatrixXi, Eigen::MatrixXi
    >
    build_refined_mesh();


private:
    // experimental: may be buggy
    void mark_cut_faces(const std::vector<bool>& is_cut_halfedge);
    void view_parameterization();
};

/**
 * @brief Note currently used
 * 
 */
class OverlaySimplifier : public Simplifier
{
public:
    OverlayMesh<Scalar> m_o;

    OverlaySimplifier(
      const OverlayMesh<Scalar>& m_o_,
      const Mesh<Scalar>& m_,
      const std::vector<bool>& is_cut,
      const std::vector<Eigen::Vector2d>& he_uv_
    )
    : m_o(m_o_)
    , Simplifier(m_, is_cut, m_o_.to, he_uv_)
    {
      is_removed_vertex = std::vector<bool>(m_o.out.size(), false);

      // find start and end halfedges
      start_halfedge = std::vector<int>(m.n_halfedges(), -1);
      end_halfedge = std::vector<int>(m.n_halfedges(), -1);
      for (int hij = 0; hij < m_o.n_halfedges(); ++hij)
      {
          // only look for original edges
          if (m_o.edge_type[hij] == CURRENT_EDGE)
              continue;


          int vi = m_o.to[m_o.opp[hij]];
          int vj = m_o.to[hij];
          int Hij = m_o.origin_of_origin[hij];
          if (m_o.edge_type[hij] == ORIGINAL_EDGE)
              Hij = m_o.origin[hij];
          if (m_o.vertex_type[vi] == ORIGINAL_VERTEX)
          {
              start_halfedge[Hij] = hij;
          }
          if (m_o.vertex_type[vj] == ORIGINAL_VERTEX)
          {
              end_halfedge[Hij] = hij;
          }
      }
    }

    void get_face_vertices(
        int face_index,
        std::vector<Eigen::Vector2d>& uv_vertices) const;
    bool triangulate_face(
        int face_index,
        std::vector<std::array<int, 3>>& face_triangles,
        std::vector<std::array<int, 3>>& uv_face_triangles) const;

    std::vector<int> get_halfedges(int halfedge_index) const override;
    bool is_face_self_overlapping(int face_index) const override;

    std::tuple<OverlayMesh<Scalar>,  std::vector<int>, std::vector<Eigen::Vector2d>> get_refined_mesh();
};

}
}