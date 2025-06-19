#pragma once

#include "feature/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"
#include "feature/core/boundary_path.h"
#include "feature/feature/features.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "feature/surgery/refinement.h"

namespace Penner {
namespace Feature {

class MarkedRefinementMesh : public IntrinsicRefinementMesh
{
public:
    // TODO
    MarkedRefinementMesh(const Mesh<Scalar>& m)
    : IntrinsicRefinementMesh(m)
    {}

    /**
     * @brief Generate a marked metric with the given mesh and boundary loops.
     *
     * @param kappa_hat: holonomy constraints for the refined metric
     * @return refined metric
     */
    MarkedPennerConeMetric generate_marked_metric(const std::vector<Scalar>& kappa_hat) const;

    /**
     * @brief Generate a dirichlet metric with the given mesh and boundary loops
     *
     * @param kappa_hat: holonomy constraints for the refined metric
     * @param start_vertices: vertices to start th
     * @param boundary_constraint_system
     * @param ell
     * @return refined metric with boundary constraints
     */
    MarkedPennerConeMetric generate_dirichlet_metric(
        const std::vector<Scalar>& kappa_hat,
        const std::vector<int>& start_vertices,
        const MatrixX& boundary_constraint_system,
        const VectorX ell) const;

private:
    // TODO: support full intrinsic refinement for, e.g., intrinsic Delaunay refinement
    std::vector<std::vector<int>> m_dual_loops;
    std::vector<std::vector<int>>& get_dual_loops() { return m_dual_loops; }
    const std::vector<std::vector<int>>& get_dual_loops() const { return m_dual_loops; }

    virtual int refine_single_face(int face_index) override;
    virtual int refine_single_halfedge(int halfedge_index) override;

};

} // namespace Feature
} // namespace Penner