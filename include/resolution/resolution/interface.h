#pragma once

#include "resolution/core/common.h"
#include "resolution/hierarchy/halfedge.h"
#include "resolution/hierarchy/voronoi.h"
#include "util/io.h"
#include <nlohmann/json.hpp>

namespace Penner {
namespace Resolution {

struct ScalableConfig {
    bool   retrace_delaunay = true;
    bool   use_geodesic_edges = true;
    bool   optimize_layout = true;
    bool   use_uniform_overlay = true;
    bool   use_harmonic_overlay = true;
    double ratio = 0.02;
    double normal_scale = 1.;
    double dual_length_ratio = 0.1;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ScalableConfig,
    retrace_delaunay,
    use_geodesic_edges,
    optimize_layout,
    use_uniform_overlay,
    use_harmonic_overlay,
    ratio,
    normal_scale,
    dual_length_ratio)


// Reject unknown keys and values of the wrong type; missing keys keep their defaults.
ScalableConfig config_from_json(const nlohmann::json& config);
ScalableConfig default_medium_config();
ScalableConfig default_coarse_config();

/**
 * @brief Generate a parametrization satisfying seamless constraints inferred from
 * a smooth cross-field, optimized with a MIQ method on a medium-resolution mesh, using
 * a scalable parametrization on a coarse-resolution mesh 
 * 
 * The parametrization algorithm is based on a modified newton's method.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param field_params: (optional) parameters for field generation
 * @param alg_params: (optional) parameters for modified Newton method
 * @return parametrized VF mesh with uv coordinates
 * @return cross field direction for u coordinate gradient
 * @return cross field direction for v coordinate gradient
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXd>
parametrize_scalable(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    ScalableConfig medium_params = default_medium_config(),
    ScalableConfig coarse_params = default_coarse_config(),
    Field::FieldParameters field_params=Field::FieldParameters(),
    NewtonParameters alg_params=NewtonParameters());

/**
 * @brief Generate a parametrization satisfying seamless constraints inferred from the given cross field using a scalable
 * multiresolution approach with a coarse-resolution mesh.
 * 
 * Since the mesh is refined by the parametrization, the cross field also needs to be refined.
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param theta: field angles relative to the reference directions
 * @param kappa: angles across edges between reference directions
 * @param period_jump: jump in period across edges
 * @return parametrized VF mesh with uv coordinates
 * @return refined cross field
 * 
 */
std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
generate_scalable_parametrization(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXi& period_jump,
    ScalableConfig coarse_params = default_coarse_config(),
    NewtonParameters alg_params=NewtonParameters());


class ResolutionMetricGenerator
{
public:
    std::string symdir_json;

    ResolutionMetricGenerator(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F);

    void set_field(
        const Eigen::MatrixXd& reference_field,
        const Eigen::VectorXd& theta,
        const Eigen::MatrixXd& kappa,
        const Eigen::MatrixXi& period_jump
    );

    void run(
        ScalableConfig coarse_params = default_coarse_config(),
        NewtonParameters alg_params = NewtonParameters());

    void optimize();

    std::tuple<
        Eigen::MatrixXd,
        Eigen::MatrixXi,
        Eigen::MatrixXd,
        Eigen::MatrixXi>
    get_parametrized_mesh() const
    {
        return std::make_tuple(V_o, F_o, uv_o, FT_o);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> get_combed_field() const;

    std::tuple<
        Eigen::MatrixXd,
        Eigen::VectorXd,
        Eigen::MatrixXd,
        Eigen::MatrixXi>
    get_refined_field() const;

private:
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;

    Eigen::MatrixXd V_o;
    Eigen::MatrixXi F_o;
    Eigen::MatrixXd uv_o;
    Eigen::MatrixXi FT_o;
    std::vector<int> fn_to_f_o;
    Eigen::MatrixXi hn_to_h_o;
    Eigen::MatrixXi is_edge_o;

    Eigen::MatrixXd PD1, PD2;

    Eigen::MatrixXd reference_field_r;
    Eigen::VectorXd theta_r;
    Eigen::MatrixXd kappa_r;
    Eigen::MatrixXi period_jump_r;

    nlohmann::json results;

};

} // namespace Resolution
} // namespace Penner