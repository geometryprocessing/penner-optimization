#pragma once

#include "holonomy/core/common.h"

namespace PennerHolonomy {

class IntrinsicNRosyField
{
public:
    IntrinsicNRosyField() {};
    VectorX run(const Mesh<Scalar>& m);
    VectorX run_with_viewer(
        const Mesh<Scalar>& m,
        const std::vector<int>& vtx_reindex,
        const Eigen::MatrixXd& V);

    Scalar min_angle = 1e-3;

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

    Scalar compute_angle_to_reference(const Mesh<Scalar>& m, const VectorX& he2angle, int h) const;
    Scalar compute_angle_between_frames(const Mesh<Scalar>& m, const VectorX& he2angle, int h) const;

    void initialize_local_frames(const Mesh<Scalar>& m);
    void initialize_period_jump(const Mesh<Scalar>& m);
    void initialize_mixed_integer_system(const Mesh<Scalar>& m);

    void initialize_double_local_frames(const Mesh<Scalar>& m);
    void initialize_double_period_jump(const Mesh<Scalar>& m);
    void initialize_double_mixed_integer_system(const Mesh<Scalar>& m);

    void solve(const Mesh<Scalar>& m);
    VectorX compute_rotation_form(const Mesh<Scalar>& m);
};



} // namespace PennerHolonomy
