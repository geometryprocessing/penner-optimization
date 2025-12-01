#pragma once

#include "feature/core/common.h"
#include "feature/dirichlet/dirichlet_penner_cone_metric.h"
#include "holonomy/interface.h"

namespace Penner {
namespace Feature {

class CutMetricGenerator
{
public:
    /**
     * @brief Initialize a metric generator from a cut mesh with marked corners for
     * halfedges to avoid in basis loop computations.
     *
     * @param V_cut: cut mesh vertices
     * @param F_cut: cut mesh faces
     * @param marked_metric_params: parameters for metric generation
     * @param marked_corners: marked corners for mesh generation
     */
    CutMetricGenerator(
        const Eigen::MatrixXd& V_cut,
        const Eigen::MatrixXi& F_cut,
        MarkedMetricParameters marked_metric_params,
        std::vector<std::pair<int, int>> marked_corners);

    /**
     * @brief Generate field for the cut mesh.
     *
     * @param V_cut: cut mesh vertices
     * @param F_cut: cut mesh faces
     * @param V_map: identification map from cut VF vertices to the glued mesh vertices
     * @param direction: per-face reference tangent direction matrix
     * @param is_fixed_direction: per-face mask for salient directions
     */
    void generate_fields(
        const Eigen::MatrixXd& V_cut,
        const Eigen::MatrixXi& F_cut,
        const Eigen::VectorXi& V_map,
        const Eigen::MatrixXd& direction,
        const std::vector<bool>& is_fixed_direction);

    /**
     * @brief Set the metric fields from a given field.
     *
     * @param F_cut: cut mesh faces
     * @param face_reference_field: per-face reference tangent direction matrix
     * @param face_theta: offset angles of a representative cross field direction relative to the
     * reference
     * @param corner_kappa: per-corner rotation angle of the reference direction field across the
     * opposite edge
     * @param corner_period_jump: per-corner period jump of the cross field across the opposite edge
     */
    void set_fields(
        const Eigen::MatrixXi& F_cut,
        const Eigen::MatrixXd& face_reference_field,
        const Eigen::VectorXd& face_theta,
        const Eigen::MatrixXd& corner_kappa,
        const Eigen::MatrixXi& corner_period_jump);

    /**
     * @brief Build a union cut halfedge mesh with vertex and face reindexing.
     *
     * @return cut union mesh
     * @return map from cut halfedge mesh vertices to cut VF vertices
     * @return map from cut halfedge mesh faces to VF faces
     */
    std::tuple<Mesh<Scalar>, std::vector<int>, std::vector<int>> get_union_mesh() const;

    /**
     * @brief Build a union cut halfedge metric with angle constraints and vertex and face
     * reindexing.
     *
     * @return cut union metric
     * @return map from cut halfedge mesh vertices to cut VF vertices
     * @return map from cut halfedge mesh faces to VF faces
     * @return union metric rotation form
     * @return union metric target cones
     */
    std::tuple<
        MarkedPennerConeMetric,
        std::vector<int>,
        std::vector<int>,
        VectorX,
        std::vector<Scalar>>
    get_union_metric(MarkedMetricParameters marked_metric_params);

    /**
     * @brief Build a union cut halfedge metric with feature constraints and vertex and face
     * reindexing.
     *
     * @return cut union metric with dirichlet constraints
     * @return map from cut halfedge mesh vertices to cut VF vertices
     * @return map from cut halfedge mesh faces to VF faces
     * @return union metric rotation form
     * @return union metric target cones
     */
    std::tuple<
        DirichletPennerConeMetric,
        std::vector<int>,
        std::vector<int>,
        VectorX,
        std::vector<Scalar>>
    get_aligned_metric(const Eigen::VectorXi& V_map, MarkedMetricParameters marked_metric_params);

    /**
     * @brief Build a union cut halfedge metric with feature constraints after applying heuristics
     * to avoid unsolvable cone prescriptions.
     *
     * @return cut union metric with dirichlet constraints
     * @return map from cut halfedge mesh vertices to cut VF vertices
     * @return map from cut halfedge mesh faces to VF faces
     * @return union metric rotation form
     * @return union metric target cones
     */
    std::tuple<
        DirichletPennerConeMetric,
        std::vector<int>,
        std::vector<int>,
        VectorX,
        std::vector<Scalar>>
    get_fixed_aligned_metric(
        const Eigen::VectorXi& V_map,
        MarkedMetricParameters marked_metric_params);

    /**
     * @brief Get the global union mesh field.
     *
     * @return: per-face reference tangent direction matrix
     * @return: offset angles of a representative cross field direction relative to the reference
     * @return: per-corner rotation angle of the reference direction field across the opposite edge
     * @return: per-corner period jump of the cross field across the opposite edge
     */
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXi> get_field()
        const;

    /**
     * @brief Get the fixed faces for the field.
     * 
     * @return mask for fixed faces
     */
    std::vector<bool> get_fixed_faces() const;

private:
    // VF data
    Eigen::MatrixXi F_mask;
    Eigen::VectorXi components;

    // local mesh data
    std::vector<Mesh<Scalar>> meshes;
    std::vector<std::vector<int>> vtx_reindexes;
    std::vector<std::vector<int>> marked_halfedges;
    std::vector<MarkedPennerConeMetric> marked_metrics;
    std::vector<Eigen::VectorXi> vertex_maps;
    std::vector<std::vector<int>> face_maps;
    std::vector<VectorX> rotation_forms;

    // cross field data
    Eigen::VectorXi reference_corner;
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    std::vector<bool> is_fixed_face;

    void generate_marked_metrics(MarkedMetricParameters marked_metric_params);

    void optimize_fields(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXi& F_cut);

};

} // namespace Feature
} // namespace Penner
