#include "resolution/core/triangulation.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner
{
namespace Resolution
{

Vector2 rotate_counterclockwise(const Vector2&p)
{
    return {-p[1], p[0]};
}

bool does_half_plane_contain(
	const Vector2& p,
	const Vector2& x0,
	const Vector2& x1,
	Scalar epsilon)
{
    Vector2 d = p - x0; // direction to p
    Vector2 edge = (x1 - x0).normalized(); // direction along edge
    Vector2 edge_perp = rotate_counterclockwise(edge);
    return (edge_perp.dot(d) > (-epsilon));
}

Scalar compute_triangle_area(
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2)
{
    Vector2 a = x1 - x0;
    Vector2 b = x2 - x0;
		Vector2 a_perp = rotate_counterclockwise(a);
		return 0.5 * a_perp.dot(b);
}

Scalar cross(const Vector2& a, const Vector2& b)
{
		return (a[0] * b[1]) - (a[1] * b[0]);
}

Scalar orient(
	const Vector2& x0,
	const Vector2& x1,
	const Vector2& x2)
{
	return cross(x1 - x0, x2 - x0);
}

Scalar compute_triangle_angle(
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2)
{
    Vector2 a = (x1 - x0).normalized();
    Vector2 b = (x2 - x0).normalized();
		return atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
}

Scalar compute_triangle_angle(
    const Vector3& x0,
    const Vector3& x1,
    const Vector3& x2)
{
    Vector3 a = (x1 - x0);
    Vector3 b = (x2 - x0);
		return atan2(a.cross(b).norm(), a.dot(b));
}

bool do_edges_intersect(
	const Vector2& x0,
	const Vector2& x1,
	const Vector2& y0,
	const Vector2& y1)
{
	Scalar ox0 = orient(y0, y1, x0);
	Scalar ox1 = orient(y0, y1, x1);
	Scalar oy0 = orient(x0, x1, y0);
	Scalar oy1 = orient(x0, x1, y1);
	return (((ox0 * ox1) < 0.) && ((oy0 * oy1) < 0.));
}

bool does_oriented_triangle_contain(
    const Vector2& p,
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2,
		Scalar epsilon)
{
    bool c0 = (does_half_plane_contain(p, x0, x1, epsilon));
    bool c1 = (does_half_plane_contain(p, x1, x2, epsilon));
    bool c2 = (does_half_plane_contain(p, x2, x0, epsilon));
    return (c0 && c1 && c2);
}

bool does_triangle_contain(
    const Vector2& p,
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2,
		Scalar epsilon)
{
	if (does_oriented_triangle_contain(p, x0, x1, x2, epsilon)) return true;
	if (does_oriented_triangle_contain(p, x0, x2, x1, epsilon)) return true;
	return false;
}


EarClipper::EarClipper(const std::vector<Vector2>& polygon_vertices)
: vertices(polygon_vertices)
{
	int num_vertices = vertices.size();

	// build linked list data structure
	next.resize(num_vertices);
	prev.resize(num_vertices);
	for (int i = 0; i < num_vertices; ++i)
	{
		int ni = (i + 1) % num_vertices;
		next[i] = ni;
		prev[ni] = i;
	}

	// determine which vertices are reflex (start with all true to avoid need to check later)
	is_reflex_vertex = std::vector<bool>(num_vertices, true);
	for (int i = 0; i < num_vertices; ++i)
	{
		update_reflex_vertex(i);
	}
}

void EarClipper::triangulate_fast()
{
	int i = 0;

	// clip ears until only one triangle remains
	int num_edges = vertices.size();
	int count = 0;
	while (num_edges > 3)
	{
		int ni = next[i];
		if (is_vertex_ear(i))
		{
			clip_ear(i);
			--num_edges;
			count = 0;
		}
		i = ni;
		++count;

		// check if failed to clip ear
		if (count > num_edges)
		{
			spdlog::error("failed to clip ear of {}-gon", num_edges);
			return;
		}
	}

	// add final triangle
	triangles.push_back({i, next[i], next[next[i]]});
}

// TODO: get maximum area ear
void EarClipper::triangulate_optimal()
{
	int i = 0;

	// clip ears until only one triangle remains
	int num_edges = vertices.size();
	int count = 0;
	while (num_edges > 3)
	{
		int start = i;
		Scalar min_angle = 2 * PI;
		int ear_candidate = i;
		do 
		{
			if (is_vertex_ear(i))
			{
				Scalar angle = compute_triangle_angle(vertices[i], vertices[next[i]], vertices[prev[i]]);
				if (angle < min_angle)
				{
					min_angle = angle;
					ear_candidate = i;
				}
			}

			i = next[i];
		} while (i != start);

		int ni = next[ear_candidate];
		clip_ear(ear_candidate);
		--num_edges;
		i = ni;
	}

	// add final triangle
	triangles.push_back({i, next[i], next[next[i]]});
}

void EarClipper::triangulate()
{
	triangulate_optimal();
}

void EarClipper::update_reflex_vertex(int i)
{
	// vertex cannot become reflex during clipping
	if (!is_reflex_vertex[i]) return;

	int pi = prev[i]; 
	int ni = next[i]; 
	// use negative offset to mark nearly colinear vertices as reflex
	is_reflex_vertex[i] = !does_half_plane_contain(vertices[pi], vertices[i], vertices[ni], -epsilon);
}

bool EarClipper::does_corner_contain(const Vector2& p, int i) const
{
	int pi = prev[i];
	int ni = next[i];
	return does_triangle_contain(p, vertices[pi], vertices[i], vertices[ni], epsilon);
}

bool EarClipper::is_vertex_ear(int i) const
{
	// vertex must be convex
	if (is_reflex_vertex[i]) return false;

	int ni = next[i];
	int pi = prev[i];
	int ci = next[ni];

	// check if any vertex is contained in the corner of vertex i
	while (ci != pi)
	{
		if ((is_reflex_vertex[ci]) && (does_corner_contain(vertices[ci], i))) return false;
		ci = next[ci];
	}

	return true;
}

void EarClipper::clip_ear(int i)
{
	// add clipped ear to triangulation
	int pi = prev[i];
	int ni = next[i];
	edges.push_back({pi, ni});
	triangles.push_back({pi, i, ni});

	// remove ear from the remaining polygon 
	next[i] = i;
	prev[i] = i;
	next[pi] = ni;
	prev[ni] = pi;

	// check if adjacent vertices are now reflex
	update_reflex_vertex(pi);
	update_reflex_vertex(ni);
}

void EarClipper::view() const
{
	int num_vertices = vertices.size();
	std::vector<std::array<int, 2>> boundary;
	boundary.reserve(num_vertices);
	for (int i = 0; i < num_vertices; ++i)
	{
		int ni = (i + 1) % num_vertices;
		boundary.push_back({i, ni});
	}

	int num_triangles = triangles.size();
	std::vector<Scalar> areas(num_triangles);
	for (int i = 0; i < num_triangles; ++i)
	{
		auto [vi, vj, vk] = triangles[i];
		areas[i] = compute_triangle_area(vertices[vi], vertices[vj], vertices[vk]);
	}

#ifdef ENABLE_VISUALIZATION
	polyscope::init();
	polyscope::registerPointCloud2D("vertices", vertices);
	polyscope::registerCurveNetwork2D("boundary", vertices, boundary);
	polyscope::registerCurveNetwork2D("edges", vertices, edges);
	polyscope::registerSurfaceMesh2D("triangles", vertices, triangles);
	polyscope::getSurfaceMesh("triangles")->addFaceScalarQuantity("area", areas);
	polyscope::show();
#endif
}

} // namespace Resolution
} // namespace Penner