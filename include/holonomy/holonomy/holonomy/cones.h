#pragma once

#include "holonomy/core/common.h"
#include "holonomy/holonomy/marked_penner_cone_metric.h"

namespace Penner {
namespace Holonomy {

/**
 * @brief Compute the cones from a rotation form on an intrinsic mesh.
 * 
 * @param m: mesh with metric
 * @param rotation_form: per-halfedge rotation form
 * @return per-vertex cones corresponding to the rotation form
 */
std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form);

/**
 * @brief Compute the cones from a rotation form on an extrinsic mesh with reindexed vertices.
 * 
 * @param m: mesh with metric
 * @param vtx_reindex: map from halfedge to VF vertex indices
 * @param rotation_form: per-halfedge rotation form
 * @param has_boundary: (optional) if true, treat mesh as a doubled mesh with boundary
 * @return per-vertex cones corresponding to the rotation form
 */
std::vector<Scalar> generate_cones_from_rotation_form(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const VectorX& rotation_form,
    bool has_boundary=false);

/**
 * @brief Determine if a mesh cone prescription corresponds to a trivial torus
 * 
 * @param m: mesh with cone constraints
 * @return true if the cones correspond to a trivial torus
 * @return false otherwise
 */
bool is_trivial_torus(const Mesh<Scalar>& m);

/**
 * @brief Check if the cones are valid for seamless holonomy constraints
 * 
 * Checks both for invalid cones that cannot be satisfied independently (i.e., a negative
 * or zero cone) and for cones that cannot be satisfied with seamless holonomy constraints
 * (i.e., a torus with a pair of cones).
 * 
 * WARNING: Don't check for trivial torus constraints, but must be accounted for by removing
 * holonomy constraints as the trivial torus only supports trivial topology.
 * 
 * @param Th_hat: per-vertex cone angles
 * @return true if the cones are valid for seamless holonomy constraints
 * @return false otherwise
 */
bool validate_cones(const Mesh<Scalar>& m);

/**
 * @brief Given target cone angles, fix any problems that prevent them from being valid
 * for seamless holonomy constraints.
 * 
 * @param m: mesh with cone constraints
 * @param min_cone_index: replace cones smaller index
 */
void fix_cones(Mesh<Scalar>& m, int min_cone_index=1);

void add_random_cone_pair(Mesh<Scalar>& m, bool only_interior=true);

std::tuple<int, int> get_constraint_outliers(
    MarkedPennerConeMetric& marked_metric,
    bool use_interior_vertices=true,
    bool use_flat_vertices=true);
std::tuple<int, int> add_optimal_cone_pair(MarkedPennerConeMetric& marked_metric);

void make_interior_free(Mesh<Scalar>& m);

std::pair<int, int> count_cones(const Mesh<Scalar>& m);

} // namespace Holonomy
} // namespace Penner