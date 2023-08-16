#include "common.hh"

/// @file viewers.hh
///
/// Some simple viewers to be used to analyze and debug the optimization pipeline.

namespace CurvatureMetric {

/// View the triangles in the mesh with inverted elements.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: mesh uv vertices
/// @param[in] F_uv: mesh uv faces
void
view_flipped_triangles(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& F_uv
);

}
