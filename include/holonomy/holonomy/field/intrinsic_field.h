#pragma once

#include "holonomy/core/common.h"

// TODO: This needs a lot of cleanup. To make it clean:
//   - The rounder should be moved to another file, and the CoMISo changes made in a fork
//   - The facet principal curvature should be in another file
//   - The double code should be much cleaner, e.g., by making a derived class

namespace Penner {
namespace Holonomy {

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

    Scalar min_angle = 0.;

    void initialize(const Mesh<Scalar>& m);
    void solve(const Mesh<Scalar>& m);
    void compute_principal_matchings(const Mesh<Scalar>& m);
    void fix_inconsistent_matchings(const Mesh<Scalar>& m);
    void fix_cone_pair(const Mesh<Scalar>& m);
    void fix_zero_cones(const Mesh<Scalar>& m);
    VectorX compute_rotation_form(const Mesh<Scalar>& m);
    void set_reference_halfedge(
        const Mesh<Scalar>& m,  
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXi& F, 
        const std::vector<int>& face_reindex,
        const Eigen::VectorXi& reference_corner);
void view(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V) const;
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

    void initialize_local_frames(const Mesh<Scalar>& m);
    void initialize_kappa(const Mesh<Scalar>& m);
    void initialize_period_jump(const Mesh<Scalar>& m);
    void initialize_mixed_integer_system(const Mesh<Scalar>& m);

    void initialize_double_local_frames(const Mesh<Scalar>& m);
    void initialize_double_kappa(const Mesh<Scalar>& m);
    void initialize_double_period_jump(const Mesh<Scalar>& m);
    void initialize_double_mixed_integer_system(const Mesh<Scalar>& m);

};

std::vector<int> generate_min_cones(const Mesh<Scalar>& m);

} // namespace Holonomy
} // namespace Penner
