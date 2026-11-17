#include "resolution/interface.h"
#include "resolution/optimization/field.h"
#include "optimization/metric_optimization/uv_optimization.h"

namespace Penner {
namespace Resolution {

// A value may replace a default only if it has the same JSON type
// (an integer is accepted where the default is a double).
static bool same_kind(const nlohmann::json& default_value, const nlohmann::json& value)
{
    if (default_value.is_number_float()) return value.is_number();
    if (default_value.is_number_integer()) return value.is_number_integer();
    return default_value.type() == value.type();
}
    
ScalableConfig config_from_json(const nlohmann::json& config)
{
    const nlohmann::json defaults = ScalableConfig{};
    for (const auto& [key, value] : config.items()) {
        if (!defaults.contains(key)) throw std::runtime_error("unknown config key: " + key);
        if (!same_kind(defaults[key], value))
            throw std::runtime_error("config key " + key + ": expected " +
                (defaults[key].is_number_integer() ? "integer" : defaults[key].type_name()) +
                ", got " + value.dump());
    }
    return config.get<ScalableConfig>();
}

ScalableConfig default_medium_config()
{
    ScalableConfig c;
    c.use_geodesic_edges = false;
    c.use_uniform_overlay = false;
    c.use_harmonic_overlay = false;
    c.ratio = 0.1;

    return c;
}

ScalableConfig default_coarse_config()
{
    ScalableConfig c;
    c.use_geodesic_edges = true;
    c.use_uniform_overlay = true;
    c.use_harmonic_overlay = true;
    c.ratio = 0.02;

    return c;
}

ResolutionMetricGenerator::ResolutionMetricGenerator(
    const Eigen::MatrixXd& V_,
    const Eigen::MatrixXi& F_)
    : V(V_)
    , F(F_)
{
    std::vector<Scalar> Th_hat(V.rows(), 2 * PI);
    std::tie(m, vtx_reindex) = Holonomy::generate_mesh(V, F, V, F, Th_hat);
}

void ResolutionMetricGenerator::set_field(
    const Eigen::MatrixXd& reference_field_,
    const Eigen::VectorXd& theta_,
    const Eigen::MatrixXd& kappa_,
    const Eigen::MatrixXi& period_jump_
)
{
    reference_field = reference_field_;
    theta = theta_;
    kappa = kappa_;
    period_jump = period_jump_;

    MarkedMetricParameters marked_metric_params;
    marked_metric_params.use_log_length = true;
    marked_metric_params.use_connectivity = true;

    // build metric with constraints
    marked_metric_params.max_loop_constraints = 0;
    std::tie(marked_metric, vtx_reindex, rotation_form, Th_hat) = generate_metric_from_field(V, F, theta, kappa, period_jump, marked_metric_params);
}

VoronoiOverlayMesh ResolutionMetricGenerator::build_coarse_mesh(ScalableConfig coarse_params)
{
    igl::Timer method_timer;
    results["config"] = coarse_params;

    // get initial cones
    std::vector<int> cones = {};
    for (int vi = 0; vi < marked_metric.n_vertices(); ++vi)
    {
        if (!float_equal(Th_hat[vtx_reindex[marked_metric.v_rep[vi]]], 2 * PI)) cones.push_back(vi);
    }

    VoronoiOverlayMesh voronoi_mesh(V, vtx_reindex, m);
    voronoi_mesh.set_original_field(reference_field, theta, period_jump);
    voronoi_mesh.symdir_json = symdir_json;
    voronoi_mesh.use_geodesic_refinement = coarse_params.use_geodesic_edges;
    voronoi_mesh.use_uniform_overlay_edges = coarse_params.use_uniform_overlay;
    voronoi_mesh.use_harmonic_overlay_vertices = coarse_params.use_harmonic_overlay;
    voronoi_mesh.optimize_layout = coarse_params.optimize_layout;
    voronoi_mesh.F_orig = F;

    // start from single seed
    std::vector<int> seeds = cones;
    if (seeds.empty()) seeds.push_back(0);

    // build voronoi regions
    method_timer.start();
    voronoi_mesh.grow_uniform_voronoi_regions(
        seeds,
        coarse_params.ratio,
        coarse_params.normal_scale,
        coarse_params.dual_length_ratio);
    results["1.1-voronoi-grow-time"] = method_timer.getElapsedTime();

    // build voronoi topology
    method_timer.start();
    voronoi_mesh.build_voronoi_connectivity();
    results["1.2-voronoi-boundary-time"] = method_timer.getElapsedTime();
    results["1-voronoi-boundary-faces"] = voronoi_mesh.count_faces();

    // trace dual Delauany topology
    method_timer.start();
    voronoi_mesh.trace_delaunay_edges();
    for (int i = 0; i < cones.size(); ++i)
    {
        int vi = cones[i];
        voronoi_mesh.simp.m.Th_hat[i] = marked_metric.Th_hat[vi];
    }
    voronoi_mesh.voronoi.reset(); // reset to allow overlay path tracing
    if (coarse_params.retrace_delaunay) voronoi_mesh.retrace_edges();
    results["2.1-delaunay-edge-time"] = method_timer.getElapsedTime();

    // build face emebddings, and simplifiy refinement
    results["faces-delaunay-full"] = voronoi_mesh.count_faces();
    method_timer.start();
    voronoi_mesh.flood_faces();
    voronoi_mesh.remove_excess_refinement();
    results["faces-delaunay-simp"] = voronoi_mesh.count_faces();
    results["2.2-delaunay-simplify-time"] = method_timer.getElapsedTime();

    // optionallly straighten edges
    if (coarse_params.use_geodesic_edges)
    {
        method_timer.start();
        voronoi_mesh.straighten_edges();
        results["2.3-delaunay-geodesics-time"] = method_timer.getElapsedTime();
    }
    else
    {
        method_timer.start();
        voronoi_mesh.set_uniform_edge_coordinates(voronoi_mesh.orig);
        voronoi_mesh.find_tutte_coordinates(voronoi_mesh.orig);
        results["2.3-delaunay-uniform-time"] = method_timer.getElapsedTime();
    }
    results["2-delaunay-faces"] = voronoi_mesh.count_faces();
    results["coarse-faces"] = voronoi_mesh.simp.m.n_faces();

    return voronoi_mesh;
}

void ResolutionMetricGenerator::run(
    ScalableConfig coarse_params,
    NewtonParameters alg_params)
{
    igl::Timer timer, method_timer;
    timer.start();
    VoronoiOverlayMesh voronoi_mesh = build_coarse_mesh(coarse_params);
 

    // lift basis loops from marked metric to simplified mesh
    method_timer.start();
    std::vector<std::unique_ptr<DualLoop>> simp_basis_loops = {};
    std::vector<Scalar> simp_kappa = {};
    lift_basis_loops(voronoi_mesh, marked_metric, rotation_form, simp_basis_loops, simp_kappa);
    results["3-dual-loop-time"] = method_timer.getElapsedTime();

    // parameterize the overlay mesh with given holonomy signature on the simplified mesh
    voronoi_mesh.log = results;
    voronoi_mesh.parameterize_with_overlay(simp_basis_loops, simp_kappa);

    // get uv coordinates for the simplified parameterization
    std::tie(V_o, F_o, uv_o, FT_o, fn_to_f_o, hn_to_h_o, is_edge_o) = voronoi_mesh.transfer_simplified_parameterization();
    results = voronoi_mesh.log;
    results["4-param-faces"] = voronoi_mesh.count_faces();
    method_timer.stop();
    voronoi_mesh.collect_garbage();

    // check if combed field already generate, or infer from refined field if not
    if (voronoi_mesh.PD1_o.size() == 0)
    {
        spdlog::warn("not implemented");
    }

    // get map from faces to overlay faces
    spdlog::info("using already combed field");
    Eigen::MatrixXd PD1, PD2;
    int num_ref_faces = F_o.rows();
    PD1.resize(num_ref_faces, 3);
    PD2.resize(num_ref_faces, 3);
    std::vector<int> f_to_fo(F.rows());
    for (int f = 0; f < voronoi_mesh.orig.face_parent.size(); ++f)
    {
        if (voronoi_mesh.is_face_deleted(f)) continue;
        f_to_fo[voronoi_mesh.orig.face_parent[f]] = f;
    }

    // write target gradient directions
    for (int f = 0; f < fn_to_f_o.size(); ++f)
    {
        int fo = f_to_fo[fn_to_f_o[f]];
        PD1.row(f) = voronoi_mesh.PD1_o[fo];
        PD2.row(f) = voronoi_mesh.PD2_o[fo];
    }

    std::tie(reference_field_r, theta_r, kappa_r, period_jump_r) = refine_overlay_field(
        voronoi_mesh,    
        vtx_reindex,
        F,
        fn_to_f_o,
        hn_to_h_o,
        is_edge_o,
        reference_field,
        theta,
        kappa,
        period_jump);
        
    timer.stop();
}

void ResolutionMetricGenerator::optimize()
{
#if USE_UV_OPTIMIZATION
    std::ifstream js_in(symdir_json);
    nlohmann::json config = nlohmann::json::parse(js_in);
    config["model"] = "mesh";
    SymDir::Parameters uv_param = Optimization::read_parameters(config);
    uv_o = Optimization::optimize_seamless_parameterization(
        V_o,
        F_o, 
        uv_o,
        FT_o,
        PD1,
        PD2,
        uv_param);
#else
    spdlog::warn("uv optimization disabled");
#endif
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> ResolutionMetricGenerator::get_combed_field() const
{
    return std::make_tuple(PD1, PD2);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::MatrixXi>
ResolutionMetricGenerator::get_refined_field() const
{
    return std::make_tuple(reference_field_r, theta_r, kappa_r, period_jump_r);
}

    
}
}