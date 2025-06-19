#pragma once

#include "feature/core/common.h"

#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "feature/feature/gluing.h"
#include <queue>


namespace Penner {
namespace Feature {

class ConePerturber
{
public:
    /**
     * @brief Construct a trivial Cone Perturber object
     * 
     */
    ConePerturber() {}

    /**
     * @brief Change the period jump for the halfedge, shifting curvature from the tip to the base.
     * 
     * @param m: mesh
     * @param rotation_form: per-halfedge rotation form
     * @param hij: halfedge to perturb
     */
    void perturb_cone(DirichletPennerConeMetric& m, VectorX& rotation_form, int hij);

    /**
     * @brief Perturb cones on the boundary of a cut mesh to avoid near degenerate cones without
     * (if possible) changing sum of sector cone angles.
     * 
     * WARNING: In some cases, it is not possible to avoid changing sum of sector angles.
     * 
     * @param m: cut mesh
     * @param rotation_form: per-halfedge rotation form
     * @param vtx_reindex: map from halfedge to VF vertex indices
     * @param V_map: identification map from cut VF vertices to the glued mesh vertices
     * @param min_angle: (optional) minimum allowed cone angle
     */
    void perturb_boundary_cones(
        DirichletPennerConeMetric& m,
        VectorX& rotation_form,
        const std::vector<int>& vtx_reindex,
        const Eigen::VectorXi& V_map,
        Scalar min_angle = M_PI);

    /**
     * @brief Applies a heuristic to find a cone pair and remove it.
     * 
     * WARNING: Only attempts for possible local perturbations along a single edge.
     *
     * @param m: mesh
     * @param rotation_form: per-halfedge rotation form
     * @param vtx_reindex: map from halfedge to VF vertex indices
     * @param V_map: identification map from cut VF vertices to the glued mesh vertices
     * @param min_angle: (optional) minimum allowed cone angle
     * @return true iff curvature was moved from a positive to negative adjacent cone
     */
    bool remove_cone_pair(
        DirichletPennerConeMetric& m,
        VectorX& rotation_form,
        const std::vector<int>& vtx_reindex,
        const Eigen::VectorXi& V_map,
        Scalar min_angle = M_PI);


private:


    std::queue<int> find_degenerate_cones(Mesh<Scalar>& m, Scalar min_angle = M_PI);
    std::array<int, 2> find_adjacent_boundary_edges(Mesh<Scalar>& m, int vi);
    int find_max_adjacent_cone(Mesh<Scalar>& m, int vi);

};

} // namespace Feature
} // namespace Penner
