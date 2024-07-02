
#pragma once

#include "holonomy/core/common.h"
#include "holonomy/similarity/similarity_penner_cone_metric.h"
#include "optimization/parameterization/interpolation.h"

#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace PennerHolonomy {

/**
 * @brief Generate a parameterization for a VF mesh with a similarity metric structure.
 * 
 * The integrated scaled metric is used, so the parameterization may have different edge
 * lengths across the parameterization cut.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param Th_hat: per-vertex cone angles
 * @param initial_similarity_metric: similarity metric structure for the mesh with the given cone angles
 * @param cut_h: (optional) cut to disk for the mesh for the parameterization
 * @return overlay mesh for the metric flipped to a Delaunay connectivity
 * @return overlay mesh vertices
 * @return overlay mesh faces
 * @return overlay mesh uv vertices
 * @return overlay mesh uv faces
 * @return cut for the mesh
 * @return cut for the overlay mesh
 * @return map from overlay face indices to faces in the original mesh
 * @return map from overlay vertices to endpoint vertices in the original mesh
 */
std::
    tuple<
        CurvatureMetric::OverlayMesh<Scalar>, // m_o
        Eigen::MatrixXd, // V_o
        Eigen::MatrixXi, // F_o
        Eigen::MatrixXd, // uv_o
        Eigen::MatrixXi, // FT_o
        std::vector<bool>, // is_cut_h
        std::vector<bool>, // is_cut_o
        std::vector<int>, // Fn_to_F
        std::vector<std::pair<int, int>> // endpoints_o
        >
    generate_VF_mesh_from_similarity_metric(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<Scalar>& Th_hat,
        const SimilarityPennerConeMetric& initial_similarity_metric,
        std::vector<bool> cut_h);

} // namespace PennerHolonomy
