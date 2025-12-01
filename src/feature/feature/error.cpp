#include "feature/feature/error.h"
#include <igl/predicates/predicates.h>
#include "feature/core/common.h"
#include "feature/core/component_mesh.h"
#include "feature/dirichlet/constraint.h"
#include "holonomy/holonomy/constraint.h"
#include "optimization/core/constraint.h"

#include "util/vf_mesh.h"

#include <igl/internal_angles.h>

namespace Penner {
namespace Feature {

double compute_edge_alignment(const Eigen::Vector2d& d)
{
    // compute angle between the direction and an arbitrary diagonal
    double diagonal_angle = atan2(d[1], d[0]) + (2. * M_PI) + (M_PI / 4.);

    // compute signed angle between the nearest axis
    double signed_axis_angle =
        (double)(Holonomy::pos_fmod(diagonal_angle, M_PI / 2.) - (M_PI / 4.));

    // compute normalized absolute value
    return abs(signed_axis_angle) / (M_PI / 4.);
}

double compute_corner_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::pair<int, int>& corner)
{
    int fijk = corner.first;
    int k = corner.second;
    int i = (k + 1) % 3;
    int j = (k + 2) % 3;

    // measure the uv alignment of the given edge
    int uvi = F_uv(fijk, i);
    int uvj = F_uv(fijk, j);
    Eigen::Vector2d UVi = uv.row(uvi);
    Eigen::Vector2d UVj = uv.row(uvj);
    return compute_edge_alignment(UVj - UVi);
}


Eigen::MatrixXd compute_mask_uv_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_aligned)
{
    int num_faces = F_uv.rows();
    Eigen::MatrixXd uv_alignment = Eigen::MatrixXd::Zero(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int k = 0; k < 3; ++k) {
            // skip edges that are not supposed to be aligned
            if (!F_is_aligned(fijk, k)) continue;
            uv_alignment(fijk, k) = compute_corner_alignment(uv, F_uv, {fijk, k});
        }
    }

    return uv_alignment;
}

std::vector<double> compute_uv_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::pair<int, int>>& aligned_halfedges)
{
    int num_corners = aligned_halfedges.size();
    std::vector<double> uv_alignment(num_corners);
    for (int i = 0; i < num_corners; ++i) {
        uv_alignment[i] = compute_corner_alignment(uv, F_uv, aligned_halfedges[i]);
    }

    return uv_alignment;
}

// compute the alignment of an edge as max of both halfedges
double compute_face_edge_alignment(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const FaceEdge& edge)
{
    double right_alignment = compute_corner_alignment(uv, F_uv, edge.right_corner());
    double left_alignment = compute_corner_alignment(uv, F_uv, edge.left_corner());
    return max(right_alignment, left_alignment);
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>
prune_misaligned_corners(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::pair<int, int>>& corners)
{
    std::vector<std::pair<int, int>> pruned_corners = {};
    std::vector<std::pair<int, int>> misaligned_corners = {};
    pruned_corners.reserve(corners.size());
    misaligned_corners.reserve(corners.size());

    // search for misalignment
    for (const auto& corner : corners) {
        if (compute_corner_alignment(uv, F_uv, corner) > 1e-10) {
            misaligned_corners.push_back(corner);
        } else {
            pruned_corners.push_back(corner);
        }
    }
    spdlog::info(
        "{} aligned and {} misaligned corners",
        pruned_corners.size(),
        misaligned_corners.size());

    return std::make_tuple(pruned_corners, misaligned_corners);
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>
prune_misaligned_edges(const Eigen::MatrixXd& uv, const std::vector<std::pair<int, int>>& edges)
{
    std::vector<std::pair<int, int>> pruned_edges = {};
    std::vector<std::pair<int, int>> misaligned_edges = {};
    pruned_edges.reserve(edges.size());
    misaligned_edges.reserve(edges.size());

    // search for misalignment
    for (const auto& edge : edges) {
        auto [uvi, uvj] = edge;
        Eigen::Vector2d UVi = uv.row(uvi);
        Eigen::Vector2d UVj = uv.row(uvj);
        if (compute_edge_alignment(UVj - UVi) > 1e-10) {
            misaligned_edges.push_back(edge);
        } else {
            pruned_edges.push_back(edge);
        }
    }

    spdlog::info(
        "{} aligned and {} misaligned edges",
        pruned_edges.size(),
        misaligned_edges.size());
    return std::make_tuple(edges, misaligned_edges);
}

std::tuple<std::vector<FaceEdge>, std::vector<FaceEdge>> prune_misaligned_face_edges(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<FaceEdge>& edges,
    Scalar feature_threshold)
{
    std::vector<FaceEdge> aligned_edges = {};
    std::vector<FaceEdge> misaligned_edges = {};
    aligned_edges.reserve(edges.size());
    misaligned_edges.reserve(edges.size());

    // search for misalignment
    for (const auto& edge : edges) {
        if (compute_face_edge_alignment(uv, F_uv, edge) > feature_threshold) {
            misaligned_edges.push_back(edge);
        } else {
            aligned_edges.push_back(edge);
        }
    }
    spdlog::info(
        "{} aligned and {} misaligned edges",
        aligned_edges.size(),
        misaligned_edges.size());

    return std::make_tuple(aligned_edges, misaligned_edges);
}

std::vector<double> compute_feature_alignment(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<std::array<int, 2>>& E)
{
    // generate corner data for edges (consistent with uv connectivity)
    int num_edges = E.size();
    std::vector<std::pair<int, int>> left_corners(num_edges);
    std::vector<std::pair<int, int>> right_corners(num_edges);
    Eigen::SparseMatrix<int> vv2f = generate_VV_to_face_map(F);
    for (int e = 0; e < num_edges; ++e) {
        auto [vi, vj] = E[e];
        int f = vv2f.coeffRef(vi, vj) - 1;
        int fo = vv2f.coeffRef(vj, vi) - 1;
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;
            if ((F(f, i) == vi) && (F(f, j) == vj)) {
                left_corners[e] = {f, k};
            }
            if ((F(fo, j) == vi) && (F(fo, i) == vj)) {
                right_corners[e] = {fo, k};
            }
        }
    }
    
    // compute the alignment of the individual halfedges
    std::vector<double> left_alignment = compute_uv_alignment(uv, F_uv, left_corners);
    std::vector<double> right_alignment = compute_uv_alignment(uv, F_uv, right_corners);

    // compute the edge alignment as the max of the two halfedge alignments
    std::vector<double> uv_alignment(num_edges);
    for (int e = 0; e < num_edges; ++e) {
        uv_alignment[e] = max(left_alignment[e], right_alignment[e]);
    }

    return uv_alignment;
}

bool check_seamless_cones(
    const std::vector<Scalar>& Th_hat,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT)
{
    spdlog::info("Checking seamless cones");

    // compute cone angles from the seamless map
    VectorX cone_angles = compute_cone_angles(V, F, uv, FT);

    // check cones for original vertices
    int num_original_vertices = Th_hat.size();
    int num_refined_vertices = cone_angles.size();
    bool consistent_cones = true;
    for (int vi = 0; vi < num_original_vertices; ++vi) {
        if (!float_equal(cone_angles[vi], Th_hat[vi])) {
            spdlog::info(
                "Inconsistent cone at {} with angle {} instead of {}",
                vi,
                cone_angles[vi],
                Th_hat[vi]);
            consistent_cones = false;
        }
    }

    // check inserted cones are flat
    for (int vi = num_original_vertices; vi < num_refined_vertices; ++vi) {
        if (!float_equal<Scalar>(cone_angles[vi], 2. * M_PI)) {
            spdlog::info("Inserted cone {} with angle {}", vi, cone_angles[vi]);
            consistent_cones = false;
        }
    }

    return consistent_cones;
}

// TODO: also need to consider holonomy and Dirichlet constraints (which are per component)
VectorX compute_component_error(
    const MarkedPennerConeMetric& marked_metric,
    const Eigen::VectorXi& vertex_component)
{
    // Make mesh into discrete metric
    spdlog::debug("Making metric discrete");
    MarkedPennerConeMetric marked_metric_copy = marked_metric;
    marked_metric_copy.make_discrete_metric();

    // Generate corner angles
    spdlog::debug("Computing corner angles");
    VectorX he2angle;
    VectorX cotangents;
    marked_metric_copy.get_corner_angles(he2angle, cotangents);

    // Generate cones and cone errors
    spdlog::debug("Computing cones and errors");
    VectorX constraint = compute_vertex_constraint(marked_metric_copy, he2angle);

    // view mesh vertex components
    VectorX vertex_constraint = vector_compose(constraint, marked_metric.v_rep);
    VectorX component_error =
        compute_vertex_component_max(marked_metric, vertex_component, vertex_constraint);

    return component_error;
}

int check_flip(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& FT)
{
    int fl = 0;
    for (int i = 0; i < FT.rows(); i++) {
        Eigen::Matrix<double, 1, 2> a_db(uv(FT(i, 0), 0), uv(FT(i, 0), 1));
        Eigen::Matrix<double, 1, 2> b_db(uv(FT(i, 1), 0), uv(FT(i, 1), 1));
        Eigen::Matrix<double, 1, 2> c_db(uv(FT(i, 2), 0), uv(FT(i, 2), 1));
        if (igl::predicates::orient2d(a_db, b_db, c_db) != igl::predicates::Orientation::POSITIVE) {
            fl++;
        }
    }
    return fl;
}

Eigen::MatrixXd compute_height(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& FT)
{
    int num_faces = FT.rows();
    Eigen::MatrixXd height = Eigen::MatrixXd::Zero(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int k = 0; k < 3; ++k) {
            int i = (k + 1) % 3;
            int j = (i + 1) % 3;

            // get the triangle vertex positions
            Eigen::Matrix<double, 1, 2> uvi = uv.row(FT(fijk, i));
            Eigen::Matrix<double, 1, 2> uvj = uv.row(FT(fijk, j));
            Eigen::Matrix<double, 1, 2> uvk = uv.row(FT(fijk, k));

            // get the base edge and next edge
            Eigen::Matrix<double, 1, 2> eij = uvj - uvi;
            Eigen::Matrix<double, 1, 2> ejk = uvk - uvj;

            // get the length of the component of the next edge orthogonal to the base (i.e., the height)
            Eigen::Matrix<double, 1, 2> eij_perp(-eij[1], eij[0]);
            double lij = eij.norm();
            height(fijk, k) = (lij > 0) ? (eij_perp.dot(ejk) / lij) : lij;
        }
    }

    return height;
}

} // namespace Feature
} // namespace Penner
