#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/halfedge.h"
#include "resolution/hierarchy/overlay.h"
#include "resolution/experimental/simplification.h"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace Penner {
namespace Resolution {


class IntrinsicSimplificationOverlayMesh : public SimplificationOverlayMesh
{
public:
  IntrinsicSimplificationMesh M; // simplified mesh (TODO: maybe remove this)

  /**
   * @brief Construct a new Simplification Overlay Mesh object to simplify the given mesh with an overlay
   * 
   * @param mesh 
   */
  IntrinsicSimplificationOverlayMesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex,
    const Mesh<Scalar>& mesh);
  void reset();
  void mark_vertices();
  bool find_edge(int edge_index);
};

}
}

  /**
   * @brief Representation of a regular polygon for overaly refinement.
   * 
  class RegularOverlayChart
  {
  public:
    RegularOverlayChart(int num_edges);

  private:
    void make_edge_between_tip_vertices(int first_halfedge, int second_halfedge);
  };
   */