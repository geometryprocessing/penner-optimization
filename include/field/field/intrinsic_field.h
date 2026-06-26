// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

/**
 * @brief Methods to generate a smooth cross field on a halfedge mesh using only
 * the intrinsic metric, with curvature alignment for extrinsic geoemtry
 * 
 */

#pragma once

#include "metric/cone_metric.h"

// TODO: This needs a lot of cleanup. To make it clean:
//   - The rounder should be moved to another file, and the CoMISo changes made in a fork
//   - The facet principal curvature should be in another file
//   - The double code should be much cleaner, e.g., by making a derived class

namespace Penner {
namespace Field {

/**
 * @brief Parameters for cross field generation
 *
 */
struct FieldParameters
{
    int min_cone = 0; // minimum allowed cone angle in the cross field
    bool fix_cone_pair = false; // collapse infeasible cone pair on a torus
    bool collapse_cones = false; // collapse as many cones as possible TODO
    bool use_roundings = true; // round away from zero
    bool use_principal_directions = false; // round away from zero
    Scalar min_cone_pair_distance = 0.; // minimum relative cone pair distance for fitting
    Scalar rel_anisotropy=0.9;
    Scalar abs_anisotropy=0.2;
};

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
compute_facet_principal_curvature(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int radius=5);

class IntrinsicNRosyField
{
public:
    IntrinsicNRosyField() {
        use_trivial_boundary = false;
    };
    IntrinsicNRosyField(const FieldParameters& field_params) {
        min_cone = field_params.min_cone;
        use_roundings = field_params.use_roundings;
        fix_cones = field_params.fix_cone_pair;
    };
    VectorX run(const Mesh<Scalar>& m);
    VectorX run_with_viewer(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXd& V);

    void set_fixed_directions(
        const Mesh<Scalar>& m,
        const std::vector<Scalar>& target_theta,
        const std::vector<bool>& is_fixed);

    void get_field(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXi& F,
        const std::vector<int>& face_reindex,
        Eigen::VectorXi& reference_corner,
        Eigen::VectorXd& face_angle,
        Eigen::MatrixXd& corner_kappa,
        Eigen::MatrixXi& corner_period_jump) const;
    void get_field(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXi& F,
        Eigen::VectorXi& reference_corner,
        Eigen::VectorXd& face_angle,
        Eigen::MatrixXd& corner_kappa,
        Eigen::MatrixXi& corner_period_jump) const;

    void get_halfedge_field(
        const Mesh<Scalar>& m,
        Eigen::VectorXd& face_theta,
        Eigen::VectorXd& halfedge_kappa,
        Eigen::VectorXi& halfedge_period_jump);

    void get_fixed_faces(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        std::vector<bool>& is_fixed) const;

    void set_min_cones(const std::vector<int>& _min_cones)
    {
        min_cones = _min_cones;
    }

    void set_field(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXi& F, 
        const std::vector<int>& face_reindex,
        const Eigen::VectorXd& face_theta,
        const Eigen::MatrixXd& corner_kappa,
        const Eigen::MatrixXi& corner_period_jump);
    void set_field(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXi& F, 
        const Eigen::VectorXd& face_theta,
        const Eigen::MatrixXd& corner_kappa,
        const Eigen::MatrixXi& corner_period_jump);
    void set_theta(
        const Mesh<Scalar>& m,
        const Eigen::VectorXd& face_theta);

    Scalar min_angle = 0.;

    void set_halfedge_field(
        const Mesh<Scalar>& m,
        const Eigen::VectorXd& face_theta,
        const Eigen::VectorXd& halfedge_kappa,
        const Eigen::VectorXi& halfedge_period_jump);

    Eigen::MatrixXd generate_reference_field(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXd& V) const;


    /**
     * @brief Build system Ax - b measuring rotation across edges in terms of face angles
     * theta for current period jumps.
     * 
     * @param m: underlying halfedge
     * @return matrix mapping face angles to their difference across an oriented edge
     * @return rotation across the edge induced by the base field and current period jumps
     */
    std::tuple<MatrixX, VectorX> build_theta_system(const Mesh<Scalar>& m) const;

    void optimize_theta(const Mesh<Scalar>& m, Scalar reg_factor=0.);
    void smooth_theta(const Mesh<Scalar>& m, int iterations=0);

    void set_fixed_directions(const std::vector<bool>& is_fixed_direction)
    {
        is_face_fixed = is_fixed_direction;
    }

    int min_cone = 1;
    bool use_roundings = true;
    bool fix_cones = false;

    void move_cone(const Mesh<Scalar>& m, int origin_v, int destination_v, int size);
    void initialize(const Mesh<Scalar>& m);
    void solve(const Mesh<Scalar>& m);
    void compute_principal_matchings(const Mesh<Scalar>& m);
    void fix_inconsistent_matchings(const Mesh<Scalar>& m);
    void remove_greedy_cone_pairs(const Mesh<Scalar>& m);
    void fix_cone_pair(const Mesh<Scalar>& m);
    void fix_zero_cones(const Mesh<Scalar>& m);
    void collapse_adjacent_cones(const Mesh<Scalar>& m);
    void collapse_nearby_cones(const Mesh<Scalar>& m);

    void concentrate_curvature(const Mesh<Scalar>& m);
    void remove_close_cone_pairs(const Mesh<Scalar>& m, Scalar rel_edge_length=1e-3);
    void infer_field_from_rotation_form(const Mesh<Scalar>& m, const VectorX& rotation_form);
    VectorX compute_rotation_form(const Mesh<Scalar>& m);
    void set_reference_halfedge(
        const Mesh<Scalar>& m,  
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXi& F, 
        const std::vector<int>& face_reindex,
        const Eigen::VectorXi& reference_corner);
void update_viewer(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V) const;
void view(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V);
void initialize_priority_kappa(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex);
void initialize_double_priority_kappa(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex);

    bool use_trivial_boundary;
private:
    // Local frames
    VectorX theta; // per-face angle from local frame to face vector
    VectorX kappa; // per-halfedge angle between reference frames
    std::vector<int> face_reference_halfedge; // index of reference halfedges
    std::vector<bool> is_face_fixed;
    bool constrain_bd;

    // Period jumps
    std::vector<int> he2e;
    std::vector<int> e2he;
    Eigen::VectorXi period_jump;
    VectorX period_value;
    std::vector<bool> is_period_jump_fixed;

    // Metric information
    VectorX he2angle;
    VectorX he2cot;

    // MI system
    std::vector<int> face_var_id;
    std::vector<int> halfedge_var_id;
    MatrixX A;
    VectorX b;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C;
    std::vector<int> min_cones;

    Scalar compute_angle_to_reference(const Mesh<Scalar>& m, const VectorX& he2angle, int h) const;
    Scalar compute_angle_between_frames(const Mesh<Scalar>& m, const VectorX& he2angle, int h) const;
    std::vector<int> generate_base_cones(const Mesh<Scalar>& m) const;
    std::vector<int> generate_kappa_cones(const Mesh<Scalar>& m) const;
    std::vector<int> generate_cones(const Mesh<Scalar>& m) const;
    bool has_cone_pair(const Mesh<Scalar>& m) const;
    bool has_zero_cone(const Mesh<Scalar>& m) const;
    int get_max_cone(const std::vector<int>& cones) const;
    int get_zero_cone(const std::vector<int>& cones) const;
    int compute_total_defect(const Mesh<Scalar>& m) const;

    void initialize_local_frames(const Mesh<Scalar>& m);
    void initialize_kappa(const Mesh<Scalar>& m);
    void initialize_period_jump(const Mesh<Scalar>& m);
    void initialize_mixed_integer_system(const Mesh<Scalar>& m);

    void initialize_double_local_frames(const Mesh<Scalar>& m);
    void initialize_double_kappa(const Mesh<Scalar>& m);
    void initialize_double_period_jump(const Mesh<Scalar>& m);
    void initialize_double_mixed_integer_system(const Mesh<Scalar>& m);


    void set_period_jump(const Mesh<Scalar>& m, int hij, Scalar jump_value);
};

std::vector<int> generate_min_cones(const Mesh<Scalar>& m, int min_cone=1);
std::vector<int> build_double_dual_bfs_forest(const Mesh<Scalar>& m, const std::vector<int> roots);

} // namespace Field
} // namespace Penner