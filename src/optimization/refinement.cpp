#include "refinement.hh"
#include "area.hh"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "viewers.hh"
#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>
#include <igl/facet_components.h>
#include <igl/flipped_triangles.h>
#include <igl/boundary_facets.h>
#include <igl/remove_duplicate_vertices.h>
#include <stack>
#include <set>

namespace CurvatureMetric {

RefinementMesh::RefinementMesh(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& uv,
	const Eigen::MatrixXi& F_uv,
	const std::vector<int>& Fn_to_F,
	const std::vector<std::pair<int, int>>& endpoints
) 
{
  Eigen::VectorXi flipped_f;
  igl::flipped_triangles(uv, F_uv, flipped_f);
  spdlog::info("{} flipped elements in overlay mesh", flipped_f.size());

	// Build initial topology with refinement data
	build_vertex_points(V, uv);
	build_connectivity(V, F, uv, F_uv, Fn_to_F, endpoints);
	if (!is_valid_refinement_mesh())
	{
		spdlog::error("Initial refinement mesh is invalid");
		return;
	}

	// Refine (and simplify) mesh as necessary
	refine_mesh();
	simplify_mesh();
	if (!is_valid_refinement_mesh())
	{
		spdlog::error("Final refinement mesh is invalid");
		return;
	}
}

void
RefinementMesh::get_VF_mesh(
	Eigen::MatrixXd& V,
	Eigen::MatrixXi& F,
	Eigen::MatrixXd& uv,
	Eigen::MatrixXi& F_uv
) const {
	// Ensure mesh is triangulated
	// TODO

	// Build vertices
	V = m_V;
	uv = m_uv;
	// TODO Remove unreferenced

	// Build triangle faces
	std::vector<std::array<int, 3>> mesh_triangles(0);
	std::vector<std::array<int, 3>> uv_mesh_triangles(0);
	mesh_triangles.reserve(n_faces());
	uv_mesh_triangles.reserve(n_faces());
	for (Index fi = 0; fi < n_faces(); ++fi)
	{
		// Skip boundary faces
		if (is_bnd_loop[fi]) continue;
		
		// Triangulate face
		std::vector<std::array<int, 3>> face_triangles;
		std::vector<std::array<int, 3>> uv_face_triangles;
		triangulate_face(
			fi,
			face_triangles,
			uv_face_triangles
		);
		mesh_triangles.insert(
			mesh_triangles.end(),
			face_triangles.begin(),
			face_triangles.end()
		);
		uv_mesh_triangles.insert(
			uv_mesh_triangles.end(),
			uv_face_triangles.begin(),
			uv_face_triangles.end()
		);

		// Print triangles if more than 1
		if (face_triangles.size() > 1)
		{
			for (size_t j = 0; j < face_triangles.size(); ++j)
			spdlog::info(
				"Triangle {} for face {} is ({}, {}, {})",
				j,
				fi,
				face_triangles[j][0],
				face_triangles[j][1],
				face_triangles[j][2]
			);
		}
	}

	// Copy vectors of mesh triangles to the face matrices
	int num_triangles = mesh_triangles.size();
	F.resize(num_triangles, 3);
	F_uv.resize(num_triangles, 3);
	for (int fi = 0; fi < num_triangles; ++fi)
	{
		for (int j = 0; j < 3; ++j)
		{
			F(fi, j) = mesh_triangles[fi][j];
			F_uv(fi, j) = uv_mesh_triangles[fi][j];
		}
	}
}

/// Get the number of edges of a given face
///
/// @param[in] face_index: polygon face index
/// @return number of edges in the given face
int
RefinementMesh::compute_face_size(RefinementMesh::Index face_index) const
{
	// Count number of edges by iterating around the face
	int face_size = 0;
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		++face_size;
	}

	return face_size;
}

RefinementMesh::FaceIterator
RefinementMesh::get_face_iterator(Index face_index) const
{
  return FaceIterator(*this, face_index);
}

VectorX
RefinementMesh::get_vertex(
	RefinementMesh::Index vertex_index
) const {
	return m_V.row(vertex_index);
}

VectorX
RefinementMesh::get_uv_vertex(
	RefinementMesh::Index vertex_index
) const {
	return m_uv.row(vertex_index);
}

void
RefinementMesh::get_face_vertices(
	RefinementMesh::Index face_index,
	std::vector<VectorX>& vertices
) const {
	vertices.clear();
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hij = *iter;
		Index vj = to[hij];
		vertices.push_back(get_vertex(vj));
	}
}

void
RefinementMesh::get_face_uv_vertices(
	RefinementMesh::Index face_index,
	std::vector<VectorX>& uv_vertices
) const {
	uv_vertices.clear();
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hij = *iter;
		Index uvij = uv_to[hij];
		uv_vertices.push_back(get_uv_vertex(uvij));
	}
}

void
build_face(
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXi& F_uv,
	const std::vector<int>& subfaces,
	const std::vector<std::pair<int, int>>& endpoints,
	std::array<int, 3>& vertices,
	std::array<std::vector<int>, 3>& edge_v_points,
	std::array<int, 3>& uv_vertices,
	std::array<std::vector<int>, 3>& edge_uv_points
) {
	// Build a VF mesh for the subdivided faces
	int num_subfaces = subfaces.size();
	Eigen::MatrixXi F_face(num_subfaces, 3);
	Eigen::MatrixXi F_uv_face(num_subfaces, 3);
	for (int fi = 0; fi < num_subfaces; ++fi)
	{
		F_face.row(fi) = F.row(subfaces[fi]);
		F_uv_face.row(fi) = F_uv.row(subfaces[fi]);
	}

	// Get the (ordered) boundary loop of the faces
	std::vector<int> loop, uv_loop;
	igl::boundary_loop(F_face, loop);
	if (static_cast<int>(loop.size()) != (F_face.rows() + 2))
	{
		spdlog::error("Loop of size {} for face {}", loop.size(), F_face);
	}

	// Get the three original vertices of the loop
	int count = 0;
	int num_boundary_vertices = loop.size();
	std::array<int, 3> vertex_indices;
	for (int i = 0; i < num_boundary_vertices; ++i)
	{
		int vi = loop[i];

		// Check if original vertex
		spdlog::trace(
			"Endpoints for boundary vertex {} are {} and {}",
			vi,
			endpoints[vi].first,
			endpoints[vi].second
		);
		if ((endpoints[vi].first == -1) || (endpoints[vi].second == -1))
		{
			// Exit with error if more than 3 vertices found
			if (count >= 3)
			{
				spdlog::error("More than 3 boundary vertices found in curved triangle {}", F_face);
			  break;
			}

			// Add vertex and increment counter
			vertex_indices[count] = i;
			vertices[count] = vi;
			++count;
		}
	}

	// Exit with error if less than 3 vertices found
	if (count < 3)
	{
		spdlog::error("Less than 3 vertices found in curved triangle {}", F_face);
		return;
	}

	// Build a map from boundary vertices to corners
	std::vector<std::pair<int, int>> loop_to_corner(num_boundary_vertices);
	for (int fi = 0; fi < num_subfaces; ++fi)
	{
		for (int j = 0; j < 3; ++j)
		{
			for (int k = 0; k < num_boundary_vertices; ++k)
			{
				if (F_face(fi, j) == loop[k])
				{
					loop_to_corner[k] = std::make_pair(fi, j);
				}
			}
		}
	}

	// Get the three boundary vertices for the uv map
	for (int i = 0; i < 3; ++i)
	{
		int fi = loop_to_corner[vertex_indices[i]].first;
		int j = loop_to_corner[vertex_indices[i]].second;
		uv_vertices[i] = F_uv_face(fi, j);
	}

	// The edge opposite corner 0 begins ccw from corner 1
	int edge = 0;
	int i = (vertex_indices[1] + 1) % num_boundary_vertices;

	// Cycle around the boundary to find the other boundary edge vertices
	while (i != vertex_indices[1])
	{
		// Increment the edge if one of the corner vertices is found
		if ((i == vertex_indices[0]) || (i == vertex_indices[1]) || (i == vertex_indices[2]))
		{
			++edge;
			i = (i + 1) % num_boundary_vertices;
			continue;
		}

		// Add the 3D vertex to the corresponding edge
		int vi = loop[i];
		edge_v_points[edge].push_back(vi);

		// Add the uv vertex to the corresponding edge
		int fi = loop_to_corner[i].first;
		int j = loop_to_corner[i].second;
		int uvi = F_uv_face(fi, j);
		edge_uv_points[edge].push_back(uvi);

		// Increment the cyclic index
		i = (i + 1) % num_boundary_vertices;
	}
}

void
build_faces(
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXi& F_uv,
	const std::vector<std::vector<int>>& F_to_Fn,
	const std::vector<std::pair<int, int>>& endpoints,
	Eigen::MatrixXi& F_orig,
	std::vector<std::array<std::vector<int>, 3>>& corner_v_points,
	Eigen::MatrixXi& F_uv_orig,
	std::vector<std::array<std::vector<int>, 3>>& corner_uv_points
) {
	int num_faces = F_to_Fn.size();
	F_orig.resize(num_faces, 3);
	corner_v_points.resize(num_faces);
	F_uv_orig.resize(num_faces, 3);
	corner_uv_points.resize(num_faces);
	for (int fi = 0; fi < num_faces; ++fi)
	{
		// Build face
		std::array<int, 3> vertices;
		std::array<int, 3> uv_vertices;
		build_face(
			F,
			F_uv,
			F_to_Fn[fi],
			endpoints,
			vertices,
			corner_v_points[fi],
			uv_vertices,
			corner_uv_points[fi]
		);

		// Copy vertices to face matrices
		for (int j = 0; j < 3; ++j)
		{
			F_orig(fi, j) = vertices[j];
			F_uv_orig(fi, j) = uv_vertices[j];
		}
	}
}

void
RefinementMesh::build_connectivity(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& uv,
	const Eigen::MatrixXi& F_uv,
	const std::vector<int>& Fn_to_F,
	const std::vector<std::pair<int, int>>& endpoints
) {
	// Combine (almost) identical uv vertices
	// TODO Make sure to check for validity
	//Eigen::MatrixXd uv;
	//Eigen::VectorXi SVI, SVJ;
	//igl::remove_duplicate_vertices(uv_in, 1e-12, uv, SVI, SVJ);
	//Eigen::MatrixXi F_uv(F_uv_in.rows(), 3);
	//for (int fi = 0; fi < F_uv_in.rows(); ++fi)
	//{
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		F_uv(fi, j) = SVJ[F_uv_in(fi, j)];
	//	}
	//}

	// Check if uv faces are a single component
	int num_components = count_components(F_uv);
	if (num_components != 1)
	{
		spdlog::error("uv face connectivity has {} components", num_components);
	}

	// Make lists of new faces per original face
	int num_new_faces = Fn_to_F.size();

	// Get the number of faces
	int num_faces = 0;
	for (int i = 0; i < num_new_faces; ++i)
	{
		num_faces = std::max(num_faces, Fn_to_F[i] + 1);
	}

	// Get the map from faces to list of new faces
	std::vector<std::vector<int>> F_to_Fn(num_faces, std::vector<int>(0));
	for (int i = 0; i < num_new_faces; ++i)
	{
		int fi = Fn_to_F[i];
		F_to_Fn[fi].push_back(i);
	}

	// For each original face, get the overlay vertices corresponding to the face
	Eigen::MatrixXi F_orig, F_uv_orig;
	std::vector<std::array<std::vector<int>, 3>> corner_v_points, corner_uv_points;
	build_faces(
		F,
		F_uv,
		F_to_Fn,
		endpoints,
		F_orig,
		corner_v_points,
		F_uv_orig,
		corner_uv_points
	);

	// TODO View flipped triangles
	//view_flipped_triangles(V, F, uv, F_uv);
	view_flipped_triangles(V, F_orig, uv, F_uv_orig);
  Eigen::VectorXi flipped_f;
  igl::flipped_triangles(uv, F_uv_orig, flipped_f);
  spdlog::info("{} flipped elements", flipped_f.size());

	// TODO Check topology directly here before building halfedge
	Eigen::VectorXi face_components;
	igl::facet_components(F_uv_orig, face_components);
	Eigen::MatrixXd V_cut, V_orig_cut;
	cut_mesh_along_parametrization_seams(V, F, uv, F_uv, V_cut);
	cut_mesh_along_parametrization_seams(V, F_orig, uv, F_uv_orig, V_orig_cut);
	Eigen::MatrixXi E;
	Eigen::VectorXi J, K;
	igl::boundary_facets(F_uv, E, J, K);
	VectorX boundary_edges;
	boundary_edges.setZero(num_new_faces * 3);
	for (int i = 0; i < J.size(); ++i)
	{
		int j = J[i];		
		int k = (K[i] + 1)%3;		
		boundary_edges[3 * j + k] = 1.0;
	}
	polyscope::init();
	polyscope::registerSurfaceMesh("original mesh", V_orig_cut, F_uv_orig)
		->addVertexParameterizationQuantity("uv", uv);
	polyscope::registerSurfaceMesh("mesh", V_cut, F_uv)
		->addVertexParameterizationQuantity("uv", uv);
	polyscope::getSurfaceMesh("original mesh")
		->addFaceScalarQuantity("components", face_components);
	polyscope::registerSurfaceMesh2D("original uv mesh", uv, F_uv_orig);
	polyscope::registerSurfaceMesh2D("uv mesh", uv, F_uv);
	polyscope::getSurfaceMesh("original uv mesh")
		->addFaceScalarQuantity("components", face_components);
	polyscope::getSurfaceMesh("uv mesh")
		->addHalfedgeScalarQuantity("boundary", boundary_edges);
	polyscope::show();
	polyscope::removeStructure("mesh");
	polyscope::removeStructure("uv mesh");
	polyscope::removeStructure("original mesh");
	polyscope::removeStructure("original uv mesh");

	// Build halfedges for the faces
	std::vector<int> next_he;
	std::vector<int> opposite;
	std::vector<int> vtx_reindex;
	std::vector<int> bnd_loops;
	std::vector<std::vector<int>> corner_to_he;
	std::vector<std::pair<int, int>> he_to_corner;
	Connectivity C;
	FV_to_NOB(F_orig, next_he, opposite, bnd_loops, vtx_reindex, corner_to_he, he_to_corner);
	NOB_to_connectivity(next_he, opposite, bnd_loops, C);

	// Check if uv faces is manifold
	if (!igl::is_edge_manifold(F_uv_orig))
	{
		spdlog::error("uv original face connectivity is not manifold");
	}

	// Check if uv faces are a single component
	int num_orig_components = count_components(F_uv_orig);
	if (num_orig_components != 1)
	{
		spdlog::error("uv original face connectivity has {} components", num_orig_components);
	}

	// Build halfedges for the uv faces
	//std::vector<int> next_he_uv;
	//std::vector<int> opp_uv;
	//std::vector<int> bnd_loops_uv;
	//std::vector<int> vtx_reindex_uv;
	//std::vector<std::vector<int>> corner_to_he_uv;
	//std::vector<std::pair<int, int>> he_to_corner_uv;
	//Connectivity C_uv;
	//FV_to_NOB(F_uv_orig, next_he_uv, opp_uv, bnd_loops_uv, vtx_reindex_uv, corner_to_he_uv, he_to_corner_uv);
	//NOB_to_connectivity(next_he_uv, opp_uv, bnd_loops_uv, C_uv);

	n = C.n;
	prev = C.prev;
	opp = C.opp;
	f = C.f;
	h = C.h;

	// Build boundary loop mask
	is_bnd_loop = std::vector<bool>(h.size(), false);
	for (auto fi : bnd_loops)
	{
		is_bnd_loop[fi] = true;
	}


	// Reindex the to arrays
	int num_halfedges = C.n.size();;
	to.resize(num_halfedges);
	uv_to.resize(num_halfedges);
	h_v_points.resize(num_halfedges);
	h_uv_points.resize(num_halfedges);

	// TODO This currently cycles faces by 1; harmless, but may be confusing
	for (int hi = 0; hi < num_halfedges; ++hi)
	{
		// Reindex to with the vertex
		to[hi] = vtx_reindex[C.to[hi]];

		// Build corner vertex points for halfedges with known opposite corner vertices
		if (is_bnd_loop[f[hi]]) continue;
		std::pair<int, int> ci = he_to_corner[hi];
		if ((ci.first >=0) && (ci.second >= 0))
		{
			// TODO Check same instead of overwrite
			//to[hi] = F_orig(ci.first, (ci.second + 2)%3); // offset of 2 as corner is opposite edge
			uv_to[hi] = F_uv_orig(ci.first, (ci.second + 2)%3); // offset of 2 as corner is opposite edge

			// Build halfedge refinement points
			h_v_points[hi] = corner_v_points[ci.first][ci.second];
			h_uv_points[hi] = corner_uv_points[ci.first][ci.second];
		}

		// FIXME
		//if (!corner_v_points[ci.first][ci.second].empty())
		//{
		//	spdlog::info("Face {} has nontrivial refinement", fi);
		//}
	}

	// FIXME
	//int num_vertices = C.out.size();;
	//out.resize(num_vertices);
	//for (int vi = 0; vi < num_vertices; ++vi)
	//{
	//	out[vtx_reindex[vi]] = C.out[vi];
	//}
}

void
RefinementMesh::build_vertex_points(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& uv
) {
	// Simply copy the vertices
	m_V = V;
	m_uv = uv;
}

void
RefinementMesh::refine_mesh()
{
	// Build list of inverted faces
	std::stack<Index> invalid_faces;
	int num_faces = n_faces();
	for (int fijk = 0; fijk < num_faces; ++fijk)
	{
		// Skip boundary loops
		if (is_bnd_loop[fijk]) continue;

		std::array<VectorX, 3> triangle;
		Index hij = h[fijk];
		for (int l = 0; l < 3; ++l)
		{
			// Get vertex at tip of halfedge
			Index vj = uv_to[hij];
			triangle[l] = get_uv_vertex(vj);

			// Increment halfedge
			hij = n[hij];
		}

		// Check that face is a triangle
		if (hij != h[fijk])
		{
			spdlog::error("Can only refine a triangle mesh");
			return;
		}

		// Check if the face is inverted
		if (is_inverted_triangle(triangle))
		{
			spdlog::info("Inverted face {}, {}, {}", triangle[0], triangle[1], triangle[2]); // FIXME
			invalid_faces.push(fijk);
		}
	}

	// Refine faces until all are valid
	while (!invalid_faces.empty())
	{
		// Get invalid face
		Index current_face = invalid_faces.top();
		invalid_faces.pop();
		spdlog::info("Processing face {}", current_face);

		// Get (unique) list of adjacent faces
		std::set<Index> adjacent_faces;
		for (auto iter = get_face_iterator(current_face); !iter.done(); ++iter)
		{
			Index hij = *iter;
			Index hji = opp[hij];
			Index adjacent_face = f[hji];
			adjacent_faces.insert(adjacent_face);
		}

		// Refine the given face
		refine_face(current_face);

		// Check if the adjacent faces are valid
		// WARNING: Assumes the face indices are fixed
		for (auto face_index : adjacent_faces)
		{
			if (!is_self_overlapping_face(face_index))
			{
				invalid_faces.push(face_index);
			}
		}
	}
}

void
RefinementMesh::simplify_mesh()
{
	// TODO
}

// Mesh manipulation

void
RefinementMesh::refine_halfedge(
	Index halfedge_index
) {
	spdlog::trace("Refining edge for halfedge {}", halfedge_index);

	// Get the number of vertices to add (and do nothing if none)
	int num_new_vertices = h_v_points[halfedge_index].size();
	if (num_new_vertices == 0) return;
	spdlog::trace("Refining edge with {} vertices", num_new_vertices);

	// Get initial next and prev halfedges
	Index hij = halfedge_index;
	Index hji = opp[hij];
	Index next_h = n[hij];
	Index prev_ho = prev[hji];
	spdlog::trace("Refining edge with halfedges ({}, {})", hij, hji);

	// Get the faces adjacent to the edge
	Index f0 = f[hij];
	Index f1 = f[hji];

	// Get the endpoint vertices
	Index vi = to[hji];
	Index vj = to[hij];

	// Get the uv to vertices
	Index uvij = is_bnd_loop[f0] ? -1 : uv_to[hij];
	Index uvji = is_bnd_loop[f1] ? -1 : uv_to[hji];

	// TODO
	// Iteratively push new vertices and halfedges to the mesh
	Index hik = hij;
	Index hki = hji;
	Index uvkj = uvij;
	Index uvki = uvji;
	for (int k = 0; k < num_new_vertices; ++k)
	{
		// Get new vertex and halfedge indices
		Index hkj = n.size();
		Index hjk = hkj + 1;
		Index vk = h_v_points[hij][k];
		if (vk != h_v_points[hji][num_new_vertices - 1 - k])
		{
			spdlog::warn(
				"Inconsistent halfedge vertices {} and {}",
				vk,
				h_v_points[hji][num_new_vertices - 1 - k]
			);
		}
		Index uvik = is_bnd_loop[f0] ? -1 : h_uv_points[hij][k];
		Index uvjk = is_bnd_loop[f1] ? -1 : h_uv_points[hji][num_new_vertices - 1 - k];

		// Update next
		n.push_back(0);
		n.push_back(0);
		n[hkj] = next_h;
		n[hjk] = hki;
		n[hik] = hkj;
		n[prev_ho] = hjk;

		// Update prev
		prev.push_back(0);
		prev.push_back(0);
		prev[hkj] = hik;
		prev[hjk] = prev_ho;
		prev[hki] = hjk;
		prev[next_h] = hkj;

		// Update opp
		opp.push_back(0);
		opp.push_back(0);
		opp[hkj] = hjk;
		opp[hjk] = hkj;

		// Update to
		to.push_back(0);
		to.push_back(0);
		to[hik] = vk; 
		to[hkj] = vj; 
		to[hjk] = vk; 
		to[hki] = vi; 

		// Update face
		f.push_back(0);
		f.push_back(0);
		f[hkj] = f0;
		f[hjk] = f1;

		// Update halfedge
		h[f0] = hik;
		h[f1] = hki;

		// Update uv to
		uv_to.push_back(0);
		uv_to.push_back(0);
		uv_to[hik] = uvik; 
		uv_to[hkj] = uvkj; 
		uv_to[hjk] = uvjk; 
		uv_to[hki] = uvki; 

		// Update halfedge points with trivial data
		h_v_points.push_back(std::vector<Index>(0));
		h_v_points.push_back(std::vector<Index>(0));

		// Iterate vertex i to k and k to j
		hik = hkj;
		hki = hjk;
		vi = vk;
		uvki = uvjk;
		// vj and uvkj do not change
	}

	// Remove halfedge points
	h_v_points[hij].clear();
	h_v_points[hji].clear();
	h_uv_points[hij].clear();
	h_uv_points[hji].clear();
}

void
RefinementMesh::refine_face(
	Index face_index
) {
	// Get all edges of the face (before changing connectivity)
	std::vector<Index> face_halfedges(0);
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hij = *iter;
		face_halfedges.push_back(hij);
	}

	// Now refine all edges of the face
	// WARNING Assumes refine_halfedge only changes the halfedge and its opposite
	for (auto hij: face_halfedges)
	{
		refine_halfedge(hij);
	}
}

// Mesh predicates

Scalar
compute_face_area(
	const std::array<VectorX, 3>& vertices
) {
	// Get edge lengths for triangle
	Scalar li = (vertices[1] - vertices[0]).norm();
	Scalar lj = (vertices[2] - vertices[1]).norm();
	Scalar lk = (vertices[0] - vertices[2]).norm();

	// Compute area from lengths
	return sqrt(area_squared(li, lj, lk));
}

bool
is_inverted_triangle(
	const std::array<VectorX, 3>& vertices
) {
	// Build matrix of triangle homogenous coordinates
	Eigen::Matrix<Scalar, 3, 3> tri_homogenous_coords;
	tri_homogenous_coords.col(0) << vertices[0][0], vertices[0][1], 1.0;
	tri_homogenous_coords.col(1) << vertices[1][0], vertices[1][1], 1.0;
	tri_homogenous_coords.col(2) << vertices[2][0], vertices[2][1], 1.0;

	// Triangle is flipped iff the determinant is negative
	double det = tri_homogenous_coords.determinant();
	return (det < 0.0);
}


bool
is_self_overlapping_polygon(
	const std::vector<VectorX>& uv_vertices,
	const std::vector<VectorX>& vertices,
	std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	std::vector<std::vector<int>>& splitting_vertices,
	std::vector<std::vector<Scalar>>& min_face_areas 
) {
	is_self_overlapping_subpolygon.clear();
	splitting_vertices.clear();
	min_face_areas.clear();
	if (vertices.size() != uv_vertices.size())
	{
		spdlog::error("Inconsistent uv and 3D vertices");
		return false;
	}

	// Build a table to record if the subpolygon between vertex i and j is self
	// overlapping and the corresponding splitting vertex indices
	int face_size = uv_vertices.size();
	is_self_overlapping_subpolygon.resize(face_size);
	splitting_vertices.resize(face_size);
	min_face_areas.resize(face_size);
	for (int i = 0; i < face_size; ++i)
	{
		is_self_overlapping_subpolygon[i] = std::vector<bool>(face_size, false);
		splitting_vertices[i] = std::vector<int>(face_size, -1);
		min_face_areas[i] = std::vector<Scalar>(face_size, 0.0);
	}

	// Set diagonal and superdiagonal to true as trivial polygons with less than 3 faces can be
	// triangulated with positive orientation (by convention)
	for (int i = 0; i < face_size; ++i)
	{
		is_self_overlapping_subpolygon[i][i] = true;
		is_self_overlapping_subpolygon[i][(i + 1)%face_size] = true;
	}

	// Check for degenerate (line, point, empty) polygons
	if (face_size < 3)
	{
		spdlog::warn("Checking if trivial polygon is self overlapping");
		return true;
	}

	// Build table iteratively by distance between vertices in ccw order
	for (int d = 2; d < face_size; ++d)
	{
		for (int i = 0; i < face_size; ++i)
		{
			// Checking for subpolygon between i and the vertex d away ccw around
			// the polygon
			int j = (i + d) % face_size;

			// Look for a splitting vertex k between i and j
			for (int k = (i + 1) % face_size; k != j; k = (k + 1) % face_size)
			{
				// Check if triangle T_ikj is positively oriented
				std::array<VectorX, 3> uv_triangle = { uv_vertices[i], uv_vertices[k], uv_vertices[j] };
				std::array<VectorX, 3> triangle = { vertices[i], vertices[k], vertices[j] };
				if (is_inverted_triangle(uv_triangle)) continue;

				// Check if the two subpolygons (i, k) and (k, j) are self overlapping
				if (!is_self_overlapping_subpolygon[i][k]) continue;
				if (!is_self_overlapping_subpolygon[k][j]) continue;

				// Otherwise, k is a splitting vertex and (i, j) is self overlapping
				is_self_overlapping_subpolygon[i][j] = true;

				// Compute minimum of uv and 3D triangle areas for the subpolygon ij with
				// splitting vertex k
				Scalar uv_triangle_area = compute_face_area(uv_triangle);
				Scalar triangle_area = compute_face_area(triangle);
				Scalar min_area = std::min(uv_triangle_area, triangle_area);
				if (k != (i + 1) % face_size)
				{
					min_area = std::min(min_area, min_face_areas[i][k]);
				}
				if (j != (k + 1) % face_size)
				{
					min_area = std::min(min_area, min_face_areas[k][j]);
				}

				// Set splitting vertex, overwriting existing values iff it increase the min area
				if ((splitting_vertices[i][j] < 0) || (min_face_areas[i][j] < min_area))
				{
					splitting_vertices[i][j] = k;
					min_face_areas[i][j] = min_area;
				}
			}
		}
	}

	// Check if a subdiagonal element is true (and thus the polygon is self overlapping)
	for (int j = 0; j < face_size; ++j)
	{
		int i = (j + 1) % face_size; // j = i - 1
		if (is_self_overlapping_subpolygon[i][j])
		{
			return true;
		}
	}

	// Polygon is not self overlapping otherwise
	return false;
}

// Helper function to triangulate subpolygons
void
triangulate_self_overlapping_subpolygon(
	const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	const std::vector<std::vector<int>>& splitting_vertices,
	int start_vertex,
	int end_vertex,
	std::vector<std::array<int, 3>>& faces
) {
	int face_size = is_self_overlapping_subpolygon.size();
	spdlog::trace(
		"Triangulating subpolygon ({}, {}) of polygon of size {}",
		start_vertex,
		end_vertex,
		face_size
	);

	// Get starting edge (j,i)
	int i = start_vertex;
	int j = end_vertex;

	// Do not triangulate edges or vertices
	if ((i == j) || ((i + 1) % face_size == j))
	{
		spdlog::trace("Base case vertex or edge");
		return;
	}

	// Add splitting face Tikj
	int k = splitting_vertices[i][j];
	faces.push_back( {i, k, j} );
	spdlog::trace("Adding triangle ({}, {}, {})", i, k, j);

	// Triangulate subpolygon (i, k)
	spdlog::trace("Recursing to subpolygon ({}, {})", i, k);
	triangulate_self_overlapping_subpolygon(
		is_self_overlapping_subpolygon,
		splitting_vertices,
		i,
		k,
		faces
	);

	// Triangulate subpolygon (k, j)
	spdlog::trace("Recursing to subpolygon ({}, {})", k, j);
	triangulate_self_overlapping_subpolygon(
		is_self_overlapping_subpolygon,
		splitting_vertices,
		k,
		j,
		faces
	);
}

void
triangulate_self_overlapping_polygon(
	const std::vector<std::vector<bool>>& is_self_overlapping_subpolygon,
	const std::vector<std::vector<int>>& splitting_vertices,
	const std::vector<std::vector<Scalar>>& min_face_areas,
	std::vector<std::array<int, 3>>& faces
) {
	faces.clear();
	int face_size = is_self_overlapping_subpolygon.size();
	if (face_size < 3)
	{
		spdlog::warn("Triangulated trivial face");
		return;
	}

	// Find (j, i) so that the triangulation of the polygon containing it has maximal min area
	int optimal_j = -1;
	Scalar max_min_area = -1.0;
	for (int j = 0; j < face_size; ++j)
	{
		int i = (j + 1) % face_size; // j = i - 1
		if ((is_self_overlapping_subpolygon[i][j]) && (min_face_areas[i][j] > max_min_area))
		{
			optimal_j = j;
			max_min_area = min_face_areas[i][j];
		}
	}

	// Call recursive subroutine on the whole polygon for the optimal edge (j, i)
	int j = optimal_j;
	int i = (j + 1) % face_size; // j = i - 1
	triangulate_self_overlapping_subpolygon(
		is_self_overlapping_subpolygon,
		splitting_vertices,
		i,
		j,
		faces
	);

	// Check triangulation is the right size
	int num_tri = faces.size();
	if (face_size != num_tri + 2)
	{
		spdlog::error(
			"Invalid triangulation of size {} for polygon of size {}",
			num_tri,
			face_size
		);
		faces.clear();
	}
}

bool
RefinementMesh::is_self_overlapping_face(
	RefinementMesh::Index face_index
) const {
	// Boundary faces are self overlapping
	if (is_bnd_loop[face_index]) return true;

	// Get vertices of the face
	std::vector<VectorX> uv_vertices, vertices;
	get_face_uv_vertices(face_index, uv_vertices);
	get_face_vertices(face_index, vertices);

	// Determine if the face is self overlapping
	std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
	std::vector<std::vector<int>> splitting_vertices;
	std::vector<std::vector<Scalar>> min_face_areas;
	return is_self_overlapping_polygon(
		uv_vertices,
		vertices,
		is_self_overlapping_subpolygon,
		splitting_vertices,
		min_face_areas
	);
}

bool
RefinementMesh::triangulate_face(
	RefinementMesh::Index face_index,
	std::vector<std::array<int, 3>>& face_triangles,
	std::vector<std::array<int, 3>>& uv_face_triangles
) const {
	spdlog::trace("Triangulating face {}", face_index);
	face_triangles.clear();
	uv_face_triangles.clear();

	// Get uv vertices of the face
	std::vector<int> vertex_indices;
	std::vector<int> uv_vertex_indices;
	std::vector<VectorX> vertices;
	std::vector<VectorX> uv_vertices;
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hij = *iter;
		Index vj = to[hij];
		Index uvij = uv_to[hij];
		vertex_indices.push_back(vj);
		uv_vertex_indices.push_back(uvij);
		vertices.push_back(get_vertex(vj));
		uv_vertices.push_back(get_uv_vertex(uvij));
	}

	// FIXME
	if (vertex_indices.size() > 3)
	{
		spdlog::trace("Triangulating face {} of size {}", face_index, vertex_indices.size());
		spdlog::trace("Face vertices are {}", formatted_vector(vertex_indices));
	}

	// Determine if the face is self overlapping
	std::vector<std::vector<bool>> is_self_overlapping_subpolygon;
	std::vector<std::vector<int>> splitting_vertices;
	std::vector<std::vector<Scalar>> min_face_areas;
	bool is_self_overlapping = is_self_overlapping_polygon(
		uv_vertices,
		vertices,
		is_self_overlapping_subpolygon,
		splitting_vertices,
		min_face_areas
	);

	// Return false if the face is not self overlapping
	if (!is_self_overlapping)
	{
		spdlog::warn("Face {} is not self overlapping", face_index);
		return false;
	}

	// Triangulate the face
	std::vector<std::array<int, 3>> polygon_faces;
	triangulate_self_overlapping_polygon(
		is_self_overlapping_subpolygon,
		splitting_vertices,
		min_face_areas,
		polygon_faces
	);

	// Convert polygon face indices to global indices for V and uv
	int num_polygon_faces = polygon_faces.size();
	face_triangles.resize(num_polygon_faces);
	uv_face_triangles.resize(num_polygon_faces);
	for (int fi = 0; fi < num_polygon_faces; ++fi)
	{
		for (int j = 0; j < 3; ++j)
		{
			face_triangles[fi][j] = vertex_indices[polygon_faces[fi][j]];
			uv_face_triangles[fi][j] = uv_vertex_indices[polygon_faces[fi][j]];
		}
	}

	// Optionally view the triangulated face
	bool view_triangulated_face = false;
	if ((view_triangulated_face) && (num_polygon_faces > 1))
	{
		// Print triangulation table
		spdlog::info("SO table");
		for (size_t i = 0; i < is_self_overlapping_subpolygon.size(); ++i)
		{
			spdlog::info("row {}: {}", i, formatted_vector(is_self_overlapping_subpolygon[i]));
		}
		spdlog::info("splitting vertices table");
		for (size_t i = 0; i < splitting_vertices.size(); ++i)
		{
			spdlog::info("row {}: {}", i, formatted_vector(splitting_vertices[i]));
		}
		spdlog::info("min face area table");
		for (size_t i = 0; i < min_face_areas.size(); ++i)
		{
			spdlog::info("row {}: {}", i, formatted_vector(min_face_areas[i]));
		}

		// Print vertices
		for (size_t vi = 0; vi < vertices.size(); ++vi)
		{
			spdlog::info("Vertex {} with index {} at {}", vi, vertex_indices[vi], vertices[vi]);
		}

		// Print faces
		for (int fi = 0; fi < num_polygon_faces; ++fi)
		{
			spdlog::info(
				"Local face {} is {}, {}, {}",
				fi,
				polygon_faces[fi][0],
				polygon_faces[fi][1],
				polygon_faces[fi][2]
			);
			spdlog::info(
				"Global face {} is {}, {}, {}",
				fi,
				face_triangles[fi][0],
				face_triangles[fi][1],
				face_triangles[fi][2]
			);
		}

		// Open viewer
		polyscope::init();
		
		polyscope::registerPointCloud("face vertices", vertices);
		polyscope::registerSurfaceMesh("face mesh", m_V, face_triangles);
		polyscope::show();
		polyscope::removeStructure("face mesh");
		polyscope::removeStructure("face vertices");

		// Print faces
		for (int fi = 0; fi < num_polygon_faces; ++fi)
		{
			spdlog::info(
				"Layout face {} is {}, {}, {}",
				fi,
				uv_face_triangles[fi][0],
				uv_face_triangles[fi][1],
				uv_face_triangles[fi][2]
			);
		}

		polyscope::registerPointCloud2D("face layout vertices", uv_vertices);
		polyscope::registerSurfaceMesh2D("face layout mesh", m_uv, uv_face_triangles);
		polyscope::show();
		polyscope::removeStructure("face layout mesh");
		polyscope::removeStructure("face layout vertices");
	}


	return true;
}

bool RefinementMesh::is_valid_refinement_mesh() const
{
	// Check halfedge topology
	int num_halfedges = n_halfedges();
	for (int hij = 0; hij < num_halfedges; ++hij)
	{
		// Check next and prev are inverse
		if ((n[prev[hij]] != hij) || (prev[n[hij]] != hij))
		{
			spdlog::error("next and prev are not inverses");
			return false;
		}

		// Check opp is an order 2 involution
		if ((opp[hij] == hij) || (opp[opp[hij]] != hij))
		{
			spdlog::error("opp is not an order 2 involution");
			return false;
		}

		// Check that to is invariant under vertex circulation
		if (to[opp[n[hij]]] != to[hij])
		{
			spdlog::error("to is not invariant under vertex circulation");
			return false;
		}

		// Check that face is invariant under face circulation
		if (f[n[hij]] != f[hij])
		{
			spdlog::error("f is not invariant under face circulation");
			return false;
		}
	}

	// Check face topology
	int num_faces = n_faces();
	for (int fi = 0; fi < num_faces; ++fi)
	{
		// Check that h is a left inverse for face
		if (f[h[fi]] != fi)
		{
			spdlog::error("h is not a left inverse for f");
			return false;
		}
	}

	// TODO Optional: check that to and face do not identify any distinct
	// orbits of the vertex and face circulators

	// TODO Check uv and halfedge refinement attributes

	// Valid otherwise
	return true;
}

void
RefinementMesh::view_original_mesh() const
{
	polyscope::init();

	// Build mesh
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd uv;
	Eigen::MatrixXi F_uv;
	get_VF_mesh(V, F, uv, F_uv);

  // Get the flipped elements
  Eigen::VectorXi flipped_f;
  igl::flipped_triangles(uv, F_uv, flipped_f);
  spdlog::info("{} flipped elements", flipped_f.size());

	// Cut mesh
	Eigen::MatrixXd V_cut;
	cut_mesh_along_parametrization_seams(V, F, uv, F_uv, V_cut);

	// Register meshes
	polyscope::registerSurfaceMesh("original mesh", V_cut, F_uv)
		->addVertexParameterizationQuantity("uv", uv);
	polyscope::registerSurfaceMesh2D("layout mesh", uv, F_uv);

	// Show meshes
	polyscope::show();
	polyscope::removeStructure("original mesh");
	polyscope::removeStructure("layout mesh");
}

void
RefinementMesh::view_face(
	Index face_index
) const
{
	polyscope::init();

	// Register corner and halfedge vertices
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hi = *iter;
		Index vi = to[hi];
		polyscope::registerPointCloud(
			"corner vertex " + std::to_string(hi),
			get_vertex(vi).transpose()
		);

		int num_h_points = h_v_points[hi].size();
		Eigen::MatrixXd h_V(num_h_points, 3);
		for (int k = 0; k < num_h_points; ++k)
		{
			Index vk = h_v_points[hi][k];
			h_V.row(k) = get_vertex(vk);
		}
		polyscope::registerPointCloud("halfedge vertices " + std::to_string(hi), h_V);
	}
	
	// Show meshes
	polyscope::show();

	// Remove meshes
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hi = *iter;
		polyscope::removeStructure("corner vertex " + std::to_string(hi));
		polyscope::removeStructure("halfedge vertices " + std::to_string(hi));
	}
}


void
RefinementMesh::view_uv_face(
	Index face_index
) const
{
	polyscope::init();

	// Register corner and halfedge vertices
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hi = *iter;
		Index uvi = uv_to[hi];
		polyscope::registerPointCloud(
			"uv corner vertex " + std::to_string(hi),
			get_uv_vertex(uvi).transpose()
		);

		int num_h_points = h_v_points[hi].size();
		Eigen::MatrixXd h_uv(num_h_points, 3);
		for (int k = 0; k < num_h_points; ++k)
		{
			Index uvk = h_uv_points[hi][k];
			h_uv.row(k) = get_uv_vertex(uvk);
		}
		polyscope::registerPointCloud("halfedge uv vertices " + std::to_string(hi), h_uv);
	}
	
	// Show meshes
	polyscope::show();

	// Remove meshes
	for (auto iter = get_face_iterator(face_index); !iter.done(); ++iter)
	{
		Index hi = *iter;
		polyscope::removeStructure("uv corner vertex " + std::to_string(hi));
		polyscope::removeStructure("halfedge uv vertices " + std::to_string(hi));
	}
}


}
