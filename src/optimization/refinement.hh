#include "common.hh"

/// @file refinement.hh
///
/// Methods to refine a triangulation with an accompanying overlay layout sufficiently
/// to ensure the parametrization does not have inverted elements.

namespace CurvatureMetric {

/// A class to represent a mesh that supports an overlay refinement scheme.
class RefinementMesh
{
public:
	typedef int Index;

	/// Constructor for the refinement mesh from an overlay VF mesh and a corresponding
	/// parameterization with overlay face to original face and endpoint annotation.
	///
	/// @param[in] V: overlay mesh vertices
	/// @param[in] F: overlay mesh faces
	/// @param[in] uv: overlay uv vertices
	/// @param[in] F_uv: overlay uv faces
	/// @param[in] Fn_to_F: map from overlay faces to original face indices
	/// @param[in] endpoints: map from overlay vertices to endpoints of the edge containing
	///     the vertex in the original mesh, or (-1, -1) if the vertex is from the original
	RefinementMesh(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& uv,
		const Eigen::MatrixXi& F_uv,
		const std::vector<int>& Fn_to_F,
		const std::vector<std::pair<int, int>>& endpoints
	);

	/// Get a VF representation of the refinement mesh with uv coordinates with overlay face
	/// to original face and endpoint annotation.
	///
	/// @param[in] V: refinement mesh vertices
	/// @param[in] F: refinement mesh faces
	/// @param[in] uv: refinement uv vertices
	/// @param[in] F_uv: refinement uv faces
	/// @param[in] Fn_to_F: map from refinement faces to original face indices
	/// @param[in] endpoints: map from refinement vertices to endpoints of the edge containing
	///     the vertex in the original mesh, or (-1, -1) if the vertex is from the original
	void
	get_VF_mesh(
		Eigen::MatrixXd& V,
		Eigen::MatrixXi& F,
		Eigen::MatrixXd& uv,
		Eigen::MatrixXi& F_uv,
		std::vector<int>& Fn_to_F,
		std::vector<std::pair<int, int>>& endpoints
	) const;

	std::tuple<
		Eigen::MatrixXd, // V
		Eigen::MatrixXi, // F
		Eigen::MatrixXd, // uv
		Eigen::MatrixXi, // F_uv
		std::vector<int>, // Fn_to_F
		std::vector<std::pair<int, int>> // endpoints
	>
	get_VF_mesh() const;

	/// Get the number of vertices in the mesh
	///
	/// @return number of vertices
	int
	n_vertices() const
	{
		return m_V.rows();
	}

	/// Get the number of uv vertices in the mesh
	///
	/// @return number of uv vertices
	int
	n_uv_vertices() const
	{
		return m_uv.rows();
	}

	/// Get the number of faces in the mesh
	///
	/// @return number of faces
	int
	n_faces() const
	{
		return h.size();
	}

	/// Get the number of halfedges in the mesh
	///
	/// @return number of halfedges
	int
	n_halfedges() const
	{
		return n.size();
	}

	/// Get the number of edges/vertices of a given face
	///
	/// @param[in] face_index: polygon face index
	/// @return number of edges/vertices in the given face
	int
	compute_face_size(Index face_index) const;

	/// Iterator to move around a face
	class FaceIterator
	{
	public:
		/// Constructor
		FaceIterator(
			const RefinementMesh& parent,
			Index face_index)
			: m_parent(parent)
		{
			// Get starting halfedge for the face
			m_current_h = m_start_h = parent.h[face_index];

			// Set the number of times a loop has been made around the face to 0
			m_num_loops = 0;
		}

		/// Iterate halfedge (prefix)
		FaceIterator& operator++()
		{
			// Iterate to next halfedge
			m_current_h = m_parent.n[m_current_h];

			// Check if a full loop has been made
			if (m_current_h == m_start_h)
			{
				++m_num_loops;
			}

			return *this;
		}

		/// Iterate halfedge (postfix)
		FaceIterator operator++(int)
		{
			FaceIterator temp = *this;
			++*this;
			return temp;
		}

		/// Check if full loop made
		bool done()
		{ 
			// Done if a full loop has been made
			return (m_num_loops > 0);
		}

		/// Get current halfedge
		Index operator*() { return m_current_h; }

	private:
		const RefinementMesh& m_parent;
		Index m_current_h;
		Index m_start_h;
		int m_num_loops;
	};

	/// Get an iterator for the given face index
	///
	/// @param[in] face_index: face to iterate over
	/// @return iterator for the face
	FaceIterator get_face_iterator(Index face_index) const;
	
	/// Get the position in space of a vertex
	///
	/// @param[in] vertex_index: index of the vertex in the mesh
	/// @return vertex position
	VectorX
	get_vertex(
		Index vertex_index
	) const;

	/// Get the parametric domain coordinates of a vertex
	///
	/// @param[in] vertex_index: index of the vertex in the mesh
	/// @return parametric domain coordinates of the vertex
	VectorX
	get_uv_vertex(
		Index vertex_index
	) const;

	/// Get the positions in space of the vertices of a face
	///
	/// @param[in] face_index: index of the face in the mesh
	/// @return list of face vertex positions 
	void
	get_face_vertices(
		Index face_index,
		std::vector<VectorX>& vertices
	) const;

	/// Get the parametric domain coordinates of the vertices of a face
	///
	/// @param[in] face_index: index of the face in the mesh
	/// @return list of face vertex parametric coordinates
	void
	get_face_uv_vertices(
		Index face_index,
		std::vector<VectorX>& uv_vertices
	) const;

	/// Clear the data of the mesh
	void clear();

	/// Return true iff the mesh is empty.
	///
	/// @return true iff the mesh is empty
	bool empty();

	/// Viewer for the refinement mesh
	void view_refinement_mesh() const;

	/// Viewer for a face of the mesh in space
	///
	/// @param[in] face_index: index of the face in the mesh
	void view_face(Index face_index) const;

	/// Viewer for a face of the mesh in the parametric domain
	///
	/// @param[in] face_index: index of the face in the mesh
	void view_uv_face(Index face_index) const;

	private:
	// Connectivity
	std::vector<Index> n;
	std::vector<Index> prev;
	std::vector<Index> opp;
	std::vector<Index> to;
	std::vector<Index> f;
	std::vector<Index> h;
	std::vector<bool> is_bnd_loop;

	// Special uv connectivity attribute
	std::vector<Index> uv_to;

	// Vertex attributes
	Eigen::MatrixXd m_V;
	Eigen::MatrixXd m_uv;
	std::vector<std::pair<int, int>> m_endpoints;

	// Halfedge attributes
	std::vector<std::deque<Index>> h_v_points;
	std::vector<std::deque<Index>> h_uv_points;

	// Inserted halfedges pointing to inserted vertices (stored for simplification)
	std::vector<Index> m_inserted_vertex_halfedges;

	// Original connectivity
	std::vector<std::vector<std::array<int, 3>>> m_overlay_face_triangles;
	std::vector<std::vector<std::array<int, 3>>> m_overlay_uv_face_triangles;

	// Constructor helper functions

	void
	build_connectivity(
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXi& F_uv,
		const std::vector<int>& Fn_to_F,
		const std::vector<std::pair<int, int>>& endpoints
	);

	void
	build_vertex_points(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXd& uv
	);

	void
	refine_mesh();

	bool
	simplify_vertex(
		Index halfedge_index
	);

	void
	simplify_mesh();

	// Mesh manipulation

	bool
	refine_halfedge(
		Index halfedge_index
	);

	bool
	refine_face(
		Index face_index
	);


	// Mesh predicates

	bool
	is_self_overlapping_face(
		Index face_index
	) const;

	bool
	triangulate_face(
		Index face_index,
		std::vector<std::array<int, 3>>& face_triangles,
		std::vector<std::array<int, 3>>& uv_face_triangles
	) const;

	bool is_valid_refinement_mesh() const;

};

}
