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

RefinementMesh(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& uv,
	const Eigen::MatrixXi& F_uv,
	const std::vector<int>& Fn_to_F,
	const std::vector<std::pair<int, int>>& endpoints
);

void
get_VF_mesh(
	Eigen::MatrixXd& V,
	Eigen::MatrixXi& F,
	Eigen::MatrixXd& uv,
	Eigen::MatrixXi& F_uv
) const;

void view_original_mesh() const;

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

	FaceIterator operator++(int)
	{
		FaceIterator temp = *this;
		++*this;
		return temp;
	}

	bool done()
	{ 
		// Done if a full loop has been made
		return (m_num_loops > 0);
	}

	Index operator*() { return m_current_h; }

private:
	const RefinementMesh& m_parent;
	Index m_current_h;
	Index m_start_h;
	int m_num_loops;
};

FaceIterator get_face_iterator(Index face_index) const;
 
VectorX
get_vertex(
	Index vertex_index
) const;

VectorX
get_uv_vertex(
	Index vertex_index
) const;

void
get_face_vertices(
	Index face_index,
	std::vector<VectorX>& vertices
) const;

void
get_face_uv_vertices(
	Index face_index,
	std::vector<VectorX>& uv_vertices
) const;

private:
// Connectivity
std::vector<Index> n;
std::vector<Index> prev;
std::vector<Index> opp;
std::vector<Index> to;
//std::vector<Index> out; FIXME
std::vector<Index> f;
std::vector<Index> h;
std::vector<bool> is_bnd_loop;

// Special uv connectivity attribute
std::vector<Index> uv_to;

// Vertex attributes
Eigen::MatrixXd m_V;
Eigen::MatrixXd m_uv;

// Halfedge attributes
std::vector<std::vector<Index>> h_v_points;
std::vector<std::vector<Index>> h_uv_points;

// Constructor helper functions

void
build_connectivity(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& uv,
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

void
simplify_mesh();

// Mesh manipulation

void
refine_halfedge(
	Index halfedge_index
);

void
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


// Viewers

void view_face(Index face_index) const;

void view_uv_face(Index face_index) const;

};

/// Given three vertices in the plane, determine if the triangle they form
/// has negative orientation.
///
/// @param[in] vertices: three triangle vertices
/// @return true iff the triangle is inverted in the uv plane
bool
is_inverted_triangle(
	const std::array<VectorX, 3>& vertices
);

/// Given a list of vertices in the plane, determine if the polygon they
/// determine is self-overlapping.
///
/// Also generates a table indicating if the subpolygons with vertices
/// (i,...,j) are self overlapping and the corresponding splitting vertices.
///
/// @param[in] uv_vertices: list of vertices in the uv plane
/// @param[in] vertices: list of vertices (for minimum face area computation)
/// @param[out] is_self_overlapping_subpolygon: table of subpolygon self
/// 		overlapping predicate truth values
/// @param[out] splitting_vertices: table of splitting vertices for self
///     overlapping subpolygons
/// @param[out] min_face_areas: table of minimum areas of subpolygon areas
/// @return true iff the polygon is self-overlapping
bool
is_self_overlapping_polygon(
	const std::vector<VectorX>& uv_vertices,
	const std::vector<VectorX>& vertices,
	std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	std::vector<std::vector<int>>& splitting_vertices,
	std::vector<std::vector<Scalar>>& min_face_areas
);

/// Given a table indicating if the subpolygons of a polygon with vertices
/// (i,...,j) are self overlapping and the corresponding splitting vertices,
/// construct a triangulation of the full polygon.
///
/// The tables can be generated by is_self_overlapping_polygon.
///
/// @param[in] is_self_overlapping_subpolygon: table of subpolygon self
/// 		overlapping predicate truth values
/// @param[in] splitting_vertices: table of splitting vertices for self
///     overlapping subpolygons
/// @param[in] min_face_areas: table of minimum areas of subpolygon areas
/// @param[out] faces: list of triangles that triangulate the polygon
void
triangulate_self_overlapping_polygon(
	const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	const std::vector<std::vector<int>>& splitting_vertices,
	const std::vector<std::vector<Scalar>>& min_face_areas,
	std::vector<std::array<int, 3>>& faces
);


}
