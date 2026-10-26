#pragma once
#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

  /**
   * @brief Rotate the vector p 90 degrees counterclockwise.
   * 
   * @param p: vector in plane
   * @return rotated vector
   */
Vector2 rotate_counterclockwise(const Vector2&p);

/**
 * @brief Determine if the halfplane to the left of the edge x1 - x0 contains the point p.
 * 
 * @param p: point to check
 * @param x0: base of half plane edge
 * @param x1: tip of half plane edge
 * @param epsilon: (optional) epsilon offset for half plane
 * @return true iff the half plane (strictly) contains p
 */
bool does_half_plane_contain(
  const Vector2& p,
  const Vector2& x0,
  const Vector2& x1,
  Scalar epsilon=0.);

/**
 * @brief Determine if the triangle bounded by x0, x1, x2 contains the point p
 * 
 * @param p: point to check
 * @param x0: first vertex
 * @param x1: second vertex
 * @param x2: third vertex
 * @param epsilon: (optional) epsilon offset for triangle boundaries
 * @return true iff the triangle (strictly) contains p
 */
bool does_triangle_contain(
    const Vector2& p,
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2,
    Scalar epsilon=0.);

/**
 * @brief Compute the (signed) area of a triangle bounded by three vertices
 * 
 * @param x0: first vertex
 * @param x1: second vertex
 * @param x2: third vertex
 * @return triangle area
 */
Scalar compute_triangle_area(
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2);

/**
 * @brief Compute the angle of a triangle corner
 * 
 * @param x0: first vertex (where angle measured)
 * @param x1: second vertex
 * @param x2: third vertex
 * @return triangle area
 */
Scalar compute_triangle_angle(
    const Vector2& x0,
    const Vector2& x1,
    const Vector2& x2);
Scalar compute_triangle_angle(
    const Vector3& x0,
    const Vector3& x1,
    const Vector3& x2);

class EarClipper {
public:
  Scalar epsilon = 1e-10;
  EarClipper(const std::vector<Vector2>& polygon_vertices);

  void triangulate();
  void triangulate_fast();
  void triangulate_optimal();

  void clip_ear(int i);
  void update_reflex_vertex(int i);

  bool is_vertex_ear(int i) const;
  bool does_corner_contain(const Vector2& p, int i) const;

  std::vector<std::array<int, 2>> edges;
  std::vector<std::array<int, 3>> triangles;

  void view() const;

private:
  std::vector<Vector2> vertices;
  std::vector<int> next;
  std::vector<int> prev;
  std::vector<bool> is_reflex_vertex;
};
}
}