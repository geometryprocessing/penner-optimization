#include "holonomy/core/common.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Compute the triangle quality as the ratio of outradius to inradius.
 * 
 * @param lij: first edge length
 * @param ljk: second edge length
 * @param lki: third edge length
 * @return triangle quality measure
 */
Scalar compute_triangle_quality(Scalar lij, Scalar ljk, Scalar lki);
 
 /**
  * @brief Compute the mesh per-face triangle qualities
  * 
  * @param cone_metric: mesh with metric
  * @return: per-face triangle quality measures
  */
VectorX compute_mesh_quality(const DifferentiableConeMetric& cone_metric);

/**
  * @brief Compute the minimum corner angle of the mesh
  * 
  * @param cone_metric: mesh with metric
  * @return: minimum corner angle
 */
Scalar compute_min_angle(const DifferentiableConeMetric& cone_metric);

} // namespace Holonomy
} // namespace Penner