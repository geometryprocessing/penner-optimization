#include "holonomy/similarity/layout.h"

#include "optimization/parameterization/layout.h"
#include "optimization/parameterization/translation.h"
#include "conformal_ideal_delaunay/ConformalInterface.hh"

namespace Penner {
namespace Holonomy {

// Use interpolation method that also tracks the flips in the marked metric
// TODO Replace with method in Penner code that tracks flip sequence
void interpolate_penner_coordinates(
    const Mesh<Scalar>& mesh,
    const SimilarityPennerConeMetric& initial_marked_metric,
    SimilarityPennerConeMetric& marked_metric,
    Optimization::InterpolationMesh& interpolation_mesh,
    Optimization::InterpolationMesh& reverse_interpolation_mesh)
{
    marked_metric = initial_marked_metric;

    // Get forward Euclidean interpolation mesh
    VectorX trivial_scale_factors;
    trivial_scale_factors.setZero(mesh.n_ind_vertices());
    bool is_hyperbolic = false;
    interpolation_mesh =
        Optimization::InterpolationMesh(mesh, trivial_scale_factors, is_hyperbolic);

    // Get initial reflection structure
    Mesh<Scalar>& mc = interpolation_mesh.get_mesh();
    std::vector<char> initial_type = mc.type;
    std::vector<int> initial_R = mc.R;

    spdlog::trace("Making surface Delaunay");
    std::vector<int> euclidean_flip_sequence;
    interpolation_mesh.convert_to_delaunay_hyperbolic_surface(euclidean_flip_sequence);
    VectorX initial_metric_coords = interpolation_mesh.get_halfedge_metric_coordinates();

    // Get Euclidean Delaunay reflection structure
    std::vector<char> eucl_del_type = mc.type;
    std::vector<int> eucl_del_R = mc.R;

    // Copy flip sequence with ptolemy flips and new metric to get metric coordinates
    spdlog::trace("Getting flipped metric coordinates");
    for (const auto& h : euclidean_flip_sequence) {
        marked_metric.flip_ccw(-h - 1);
    }
    VectorX flipped_metric_coords = marked_metric.get_metric_coordinates();

    // Compute translations for reparametrization
    VectorX translations;
    Optimization::compute_as_symmetric_as_possible_translations(
        mc,
        flipped_metric_coords,
        initial_metric_coords,
        translations);
    SPDLOG_INFO("Translations in range [{}, {}]", translations.minCoeff(), translations.maxCoeff());

    // Change the metric and reparameterize
    spdlog::trace("Changing underlying metric");
    interpolation_mesh.change_hyperbolic_surface_metric(
        flipped_metric_coords,
        trivial_scale_factors,
        translations);

    // Make delaunay with new metric
    spdlog::trace("Making new surface Delaunay");
    std::vector<int> flip_sequence;
    marked_metric.make_delaunay(flip_sequence);
    interpolation_mesh.follow_flip_sequence(flip_sequence);

    // Build a clean overlay mesh with the final metric
    spdlog::trace("Building reverse interpolation mesh");
    Mesh<Scalar> m_layout = interpolation_mesh.get_mesh();
    is_hyperbolic = true;
    reverse_interpolation_mesh =
        Optimization::InterpolationMesh(m_layout, trivial_scale_factors, is_hyperbolic);

    // Undo the flips to make the hyperbolic surface with new metric Delaunay
    reverse_interpolation_mesh.reverse_flip_sequence(flip_sequence);
    reverse_interpolation_mesh.get_mesh().type = eucl_del_type;
    reverse_interpolation_mesh.get_mesh().R = eucl_del_R;

    // Undo the reparametrization
    VectorX inverse_translations = -translations;
    reverse_interpolation_mesh.change_hyperbolic_surface_metric(
        initial_metric_coords,
        trivial_scale_factors,
        inverse_translations);

    // Generate reverse map for Euclidean flips
    reverse_interpolation_mesh.force_convert_to_euclidean_surface();
    reverse_interpolation_mesh.reverse_flip_sequence(euclidean_flip_sequence);
    reverse_interpolation_mesh.get_mesh().type = initial_type;
    reverse_interpolation_mesh.get_mesh().R = initial_R;
}

std::
    tuple<
        OverlayMesh<Scalar>, // m_o
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
        std::vector<bool> cut_h)
{
    // Get mesh with vertex reindexing
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m =
        FV_to_double(V, F, V, F, Th_hat, vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops);

    // Find boundary halfedges
    std::vector<bool> is_bd(m.n_ind_vertices(), false);
    for (int i = 0; i < m.n_halfedges(); i++) {
        if ((m.type[i] == 1) && (m.type[m.opp[i]] == 2))
        {
            is_bd[m.v_rep[m.to[i]]] = true;
        }
    }

    // Compute interpolation overlay mesh
    // TODO: Use consistent interpolation code from the Penner codebase
    Eigen::MatrixXd V_overlay;
    Optimization::InterpolationMesh interpolation_mesh, reverse_interpolation_mesh;
    SimilarityPennerConeMetric similarity_metric = initial_similarity_metric;
    spdlog::trace("Interpolating penner coordinates");
    interpolate_penner_coordinates(
        m,
        initial_similarity_metric,
        similarity_metric,
        interpolation_mesh,
        reverse_interpolation_mesh);
    spdlog::trace("Interpolating vertex positions");
    interpolate_vertex_positions(
        V,
        vtx_reindex,
        interpolation_mesh,
        reverse_interpolation_mesh,
        V_overlay);
    OverlayMesh<Scalar> m_o = interpolation_mesh.get_overlay_mesh();

    // Scale the overlay mesh and make tufted
    auto [metric_coords, u_integral, is_cut_integral] =
        similarity_metric.get_integrated_metric_coordinates();
    Mesh<Scalar>& mc = m_o.cmesh();
    for (int h = 0; h < metric_coords.size(); ++h) {
        mc.l[h] = exp(metric_coords[h] / 2.0);
    }
    Optimization::make_tufted_overlay(m_o);

    // Get endpoints
    std::vector<std::pair<int, int>> endpoints;
    find_origin_endpoints(m_o, endpoints);

    // Convert overlay mesh to transposed vector format
    std::vector<std::vector<Scalar>> V_overlay_vec(3);
    for (int i = 0; i < 3; ++i) {
        V_overlay_vec[i].resize(V_overlay.rows());
        for (int j = 0; j < V_overlay.rows(); ++j) {
            V_overlay_vec[i][j] = V_overlay(j, i);
        }
    }

    // Get layout topology from original mesh
    std::vector<bool> is_cut = Optimization::compute_layout_topology(m, cut_h);

    // Convert overlay mesh to VL format
    spdlog::trace("Getting layout");
    std::vector<Scalar> u(m.n_ind_vertices(), 0.0);
    return Optimization::consistent_overlay_mesh_to_VL(
        m_o,
        vtx_reindex,
        is_bd,
        u,
        V_overlay_vec,
        endpoints,
        is_cut,
        is_cut_integral);
}

} // namespace Holonomy
} // namespace Penner