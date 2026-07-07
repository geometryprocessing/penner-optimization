#include <igl/readOBJ.h>
#include "feature/interface.h"
#include "feature/core/io.h"
#include "field/frame_field.h"
#include "util/io.h"
#include "util.h"
#include "holonomy/core/viewer.h"
#include "feature/surgery/cut_metric_generator.h"
#include "util/vf_mesh.h"

#include <CLI/CLI.hpp>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/internal_angles.h>

#include "polyscope/surface_mesh.h"

using namespace Penner;
using namespace Penner::Field;
using namespace Penner::Holonomy;
using namespace Penner::Feature;

#if USE_UV_OPTIMIZATION 
#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "main_helper.h"

SymDir::Parameters read_parameters(const nlohmann::json& config)
{
    SymDir::Parameters param;
    param.max_iters = config["max_iters"]; // iterations
    param.max_time = config["max_time"]; // time in seconds
    param.smooth_only_iters = config["smooth_only_iters"];
    param.E_target = config["E_target"]; // Energy target
    param.ls_iters = config["ls_iters"]; // param for linesearch in smoothing operation
    param.do_newton = config["do_newton"]; // do newton/gd steps for smoothing operation
    param.local_smooth = config["local_smooth"];
    param.global_smooth = config["global_smooth"];
    param.elen_alpha = config["elen_alpha"];
    param.do_projection = config["do_projection"];
    param.with_cons = config["with_cons"];
    param.Lp = config["Lp"];
    param.save_meshes = config["save_meshes"];
    param.do_feature_alignment = config["do_feature_alignment"]; // align feature edges
    param.symdir_weight = config["symdir_weight"];
    param.alignment_weight = config["alignment_weight"];
    param.degenerate_weight = config["degenerate_weight"];
    param.fix_misaligned = config["fix_misaligned"];
    param.use_rref = config["use_rref"];
    // param.solver_type = config["solver_type"];
    param.cg_rel_err = config["cg_rel_err"];
    param.cg_iters = config["cg_iters"];
    
    param.percentages = config["percentages"].get<std::vector<double>>();
    param.percentage_target = config["percentage_target"];
    param.percentage_target_value = config["percentage_target_value"];
    param.save_percentages_meshes = config["save_percentages_meshes"];

    param.E_abs_err = config["E_abs_err"];
    param.E_rel_err = config["E_rel_err"];
    param.diff_err = config["diff_err"];
    param.grad_abs_err = config["grad_abs_err"];
    param.grad_rel_err = config["grad_rel_err"];
    
    param.precompute_seamless = config["precompute_seamless"];
    param.projected_newton = config["projected_newton"];
    param.soft_max = config["soft_max"];
    param.t = config["t"];
    param.precompute_seamless = config["precompute_seamless"];
    
    param.percentage_target_converge = config["percentage_target_converge"];
    param.max_grad_abs_converge = config["max_grad_abs_converge"];
    param.max_grad_rel_converge = config["max_grad_rel_converge"];
    param.energy_diff_converge = config["energy_diff_converge"];
    param.use_worst_n_energy_in_ls = config["use_worst_n_energy_in_ls"];
    param.E_abs_converge = config["E_abs_converge"];
    param.E_rel_converge = config["E_rel_converge"];

    param.last_screenshot_after_optimization = config["last_screenshot_after_optimization"];
    param.screenshot_interval = config["screenshot_interval"];
    param.output_dir_for_screenshots = config["output_dir_for_screenshots"];
    param.uv_scale_for_screenshots = config["uv_scale_for_screenshots"];
    param.angle_to_rotate_model_for_screenshots = config["angle_to_rotate_model_for_screenshots"];
    param.screenshot_during_optimization = config["screenshot_during_optimization"];

    param.degenerate_vertices_preconditioner = config["degenerate_vertices_preconditioner"];
    param.precond_dim = config["precond_dim"];
    param.triangle_threshold = config["triangle_threshold"];

    return param;
}   

Eigen::MatrixXd optimize_aligned_parameterization(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXi& F_init,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& thetas,
    const Eigen::MatrixXi& period_jumps,
    const Eigen::MatrixXi& FE_init,
    const Eigen::MatrixXi& ME,
    const nlohmann::json& config,
    bool fix_boundary
) {
    SymDir::Parameters param = read_parameters(config);
    param.fix_boundary = fix_boundary;

    igl::Timer timer;
    double time = 0;
    timer.start();
	MeshCutter meshcutter(V_init, uv, F_init, F);
	auto [V, EE] = meshcutter.cut_mesh();
    Eigen::MatrixXi FE(0, 0);
    time = timer.getElapsedTime();
    if (param.do_feature_alignment)
    {
        // Loading the feature edge constraints
        FE = meshcutter.reindex_feature_edges(FE_init);
    }
    double cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Initial constraints error {}", cons_residual);

    Eigen::MatrixXi new_F;
    Eigen::MatrixXd new_V, new_uv;
    SymDir::ExtremeOpt extremeopt(V, F);
    extremeopt.m_params = param;
    
    extremeopt.create_mesh(V, F, uv);
    extremeopt.set_v_map(F_init, F);

    if (extremeopt.m_params.with_cons)
    {
        std::vector<std::vector<int>> EE_e = transform_EE(F, EE);
        std::vector<std::vector<int>> FE_e;
        if (extremeopt.m_params.do_feature_alignment) {
            FE_e = transform_FE(F, FE);
        }
        extremeopt.init_constraints(EE_e);
        extremeopt.EE = EE;
        extremeopt.FE = FE;
        extremeopt.ME = ME;
    }

    //extremeopt.view();
    extremeopt.comb_matchings(reference_field, thetas, period_jumps);

    Eigen::MatrixXi F_opt = F;
    Eigen::MatrixXd uv_opt;
    extremeopt.do_optimization_without_log();
    extremeopt.export_mesh(V, F_opt, uv_opt);

    return uv_opt;
}

#endif

Eigen::MatrixXi tag_cone_corners(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const std::vector<FaceEdge>& feature_face_edges
) {
    // get mask of feature corners
    int num_faces = FT.rows();
    int num_vertices = V.rows();
    Eigen::MatrixXi is_feature = compute_mask_from_face_edges(num_faces, feature_face_edges);

    // get triangle adjacency for mesh
    Eigen::MatrixXi TT, TTi;
	igl::triangle_triangle_adjacency(F, TT, TTi);

    // get unioned vertices, split by features
    int num_halfedges = 3 * num_faces;
    UnionFind cut_vertices(num_halfedges);
    //Eigen::MatrixXi is_boundary = Eigen::MatrixXi::Zero(num_faces, 3);
    Eigen::VectorXi is_boundary = Eigen::VectorXi::Zero(num_vertices);
    for (int f = 0; f < num_faces; ++f) {
        for (int i = 0 ; i < 3; ++i)
        {
            int l = (i + 2) % 3;

            // mark feature edges
            if (is_feature(f, l))
            {
                //is_boundary(f, i) = 1;
                is_boundary(F(f, i)) = 1;
            }
            // union tips across edge otherwise
            else
            {
                // halfedge with F(f, i) at tip
                int hki = 3 * f + i;

                // get halfedge rotated clockwise, using libigls halfedge base indexing
                int f_opp = TT(f, i);
                int j = TTi(f, i); // index for halfedge opposite hij
                int k = (j + 1) % 3; // index for halfedge in f pointing to vi
                int hji = 3 * f_opp + k;
                cut_vertices.union_sets(hki, hji);
            }
        }
    }

    // compute cone angles of unioned vertices
    int num_cut_vertices = cut_vertices.count_sets();
    std::vector<int> set_index = cut_vertices.index_sets();
    Eigen::MatrixXd corner_angles;
    igl::internal_angles(uv, FT, corner_angles);
    VectorX cone_angles = VectorX::Zero(num_cut_vertices);
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        for (int i = 0; i < 3; ++i)
        {
            int vi = set_index[3 * fijk + i];
            cone_angles[vi] += corner_angles(fijk, i);
        }
    }

    Eigen::MatrixXi vertex_indices(num_faces, 3);
    Eigen::MatrixXd halfedge_tip_angles = Eigen::MatrixXd::Zero(num_faces, 3);
    Eigen::MatrixXi is_cone = Eigen::MatrixXi::Zero(num_faces, 3);
    for (int f = 0; f < num_faces; ++f) {
        for (int i = 0 ; i < 3; ++i)
        {
            int h = 3 * f + i;
            int v = set_index[h];
            vertex_indices(f, i) = v;
            halfedge_tip_angles(f, i) += cone_angles[v];
        }
    }

    int num_uv_vertices = uv.rows();
    std::vector<bool> is_uv_cone(num_uv_vertices, false);
    Eigen::VectorXi is_uv_cone_mask = Eigen::VectorXi::Zero(num_uv_vertices);
    for (int f = 0; f < num_faces; ++f) {
        for (int i = 0 ; i < 3; ++i)
        {
            int v = F(f, i);
            int vt = FT(f, i);
            if ((!is_boundary[v]) && (!float_equal(halfedge_tip_angles(f, i), 2 * PI)))
            {
                is_cone(f, i) = 1;
                is_uv_cone[vt] = true;
                is_uv_cone_mask[vt] = 1;
            }
            else if ((is_boundary[v]) && (!float_equal(halfedge_tip_angles(f, i), PI)))
            {
                is_cone(f, i) = 1;
                is_uv_cone[vt] = true;
                is_uv_cone_mask[vt] = 1;
            }
        }
    }

    bool show_uv_cones = false;
    if (show_uv_cones)
    {
        polyscope::init();

        // closed mesh
        std::string mesh_handle = "mesh";
        polyscope::registerSurfaceMesh(mesh_handle, V, F);
        polyscope::getSurfaceMesh(mesh_handle)
            ->addHalfedgeScalarQuantity(
                "vertex indices",
                vertex_indices.transpose().reshaped())
            ->setEnabled(true);
        polyscope::getSurfaceMesh(mesh_handle)
            ->addHalfedgeScalarQuantity(
                "3D vertex indices",
                F.transpose().reshaped());
        polyscope::getSurfaceMesh(mesh_handle)
            ->addHalfedgeScalarQuantity(
                "cone angles",
                halfedge_tip_angles.transpose().reshaped());
        polyscope::getSurfaceMesh(mesh_handle)
            ->addVertexScalarQuantity(
                "is boundary",
                is_boundary.transpose().reshaped())
            ->setColorMap("coolwarm");
        polyscope::getSurfaceMesh(mesh_handle)
            ->addHalfedgeScalarQuantity(
                "is cone",
                is_cone.transpose().reshaped())
            ->setColorMap("coolwarm");

        // cut mesh along seams
        Eigen::MatrixXd V_cut;
        cut_mesh_along_parametrization_seams(V, F, uv, FT, V_cut);
        mesh_handle = "cut mesh";
        polyscope::registerSurfaceMesh(mesh_handle, V_cut, FT);
        polyscope::getSurfaceMesh(mesh_handle)
            ->addVertexScalarQuantity(
                "is uv cone",
                is_uv_cone_mask.transpose().reshaped())
            ->setEnabled(true);

        polyscope::show();
    }

    return is_cone;
}


int main(int argc, char* argv[])
{
    spdlog::set_level(spdlog::level::info);

    // Get command line arguments
    CLI::App app{"Generate a feature aligned parametrization."};
    std::string mesh = "";
    std::string input_dir = "./";
    std::string output_dir = "./";
    std::filesystem::path current_dir = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path input_json = current_dir / "symdir.json";
    bool use_uniform_bc = false;
    bool optimize = false;
    bool use_free_cones = false;

    int full_itr = 100;
    NewtonParameters alg_params;
    int max_itr = alg_params.max_itr;
    spdlog::level::level_enum log_level = spdlog::level::info;

    // IO Parameters
    bool use_existing_field = false;
    bool show_field = false;
    bool show_parameterization = false;
    app.add_option("--name", mesh, "Mesh name (without obj suffix, e.g., fandisk)")->required();
    app.add_option("-i,--input", input_dir, "Input directory")->check(CLI::ExistingDirectory)->required();
    app.add_option("-o,--output", output_dir, "Output directory");
    app.add_flag("--use_existing_field", use_existing_field, "Use precomputed field at the input directory");
    app.add_flag("--use_uniform_bc", use_uniform_bc, "Use uniform barycentric coordinates");
    app.add_flag("--use_free_cones", use_free_cones, "Use free cones and remove holonomy constraints");
    app.add_flag("--optimize", optimize, "Optimize uv coordinates");
    app.add_flag("--show_field", show_field, "Show field constraints");
    app.add_flag("--show_parameterization", show_parameterization, "Show aligned parameterization");
    app.add_option("--full_itr", full_itr, "Initial iterations of full (potentially unsatisfiable) constraints");
    app.add_option("--log_level", log_level, "Level of logging")
        ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));

    CLI11_PARSE(app, argc, argv);
    spdlog::set_level(log_level);

    // Marked Metric Parameters
    add_newton_parameters(app, alg_params);
    CLI11_PARSE(app, argc, argv);

    std::filesystem::create_directory(output_dir);

    // create filepaths for input data
    std::string mesh_filename = join_path(input_dir, mesh + ".obj");
    std::string feature_filename = join_path(input_dir, mesh + "_features");
    std::string hard_feature_filename = join_path(input_dir, mesh + "_hard_features");
    std::string field_filename = join_path(input_dir, mesh + ".ffield");

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("optimizing mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    // Get features and field
    std::vector<VertexEdge> feature_edges, hard_feature_edges;
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    if (use_existing_field)
    {
        spdlog::info("loading feature edges");
        feature_edges = load_feature_edges(feature_filename);
        hard_feature_edges = load_feature_edges(hard_feature_filename);

        spdlog::info("loading constraints");
        std::tie(reference_field, theta, kappa, period_jump) = Penner::Field::load_frame_field(field_filename);
    }
    else {
        // refine input mesh
        std::tie(V, F, feature_edges, hard_feature_edges) = generate_refined_feature_mesh(V, F, false);
        FeatureFinder feature_finder(V, F);
        feature_finder.mark_features(feature_edges);
        auto[V_cut, F_cut, V_map, F_is_feature] = feature_finder.generate_feature_cut_mesh();

        int radius = 5;
        Scalar rel_anisotropy=0.9;
        Scalar abs_anisotropy=0.2;
        Scalar bb_diag = igl::bounding_box_diagonal(V);
        auto [direction, is_fixed_direction] = Penner::Field::compute_field_direction(
            V_cut,
            F_cut,
            radius,
            abs_anisotropy / bb_diag,
            rel_anisotropy);
        MarkedMetricParameters marked_metric_params;
        marked_metric_params.remove_trivial_torus = false; // FIXME
        marked_metric_params.use_log_length = true;
        marked_metric_params.use_initial_zero = false;
        CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, {});
        cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction);
        std::tie(reference_field, theta, kappa, period_jump) = cut_metric_generator.get_field();
    }
    if (show_field) view_cross_field(V, F, reference_field, theta, kappa, period_jump);

    // get optimized metric
    spdlog::info("projecting to feature constraints");
    alg_params.output_dir = output_dir;
    alg_params.error_eps = 1e-10;
    alg_params.solver = "ldlt";
    alg_params.do_reduction = true;
    MarkedMetricParameters marked_metric_params;
    if (use_free_cones)
    {
        // TODO migrate some of this code to AlignedMetricGenerator
        marked_metric_params.use_free_cones = true;
        marked_metric_params.max_boundary_constraints = 0;
        marked_metric_params.max_loop_constraints = 0;
    }
    AlignedMetricGenerator aligned_metric_generator(
        V,
        F,
        feature_edges,
        hard_feature_edges,
        reference_field,
        theta,
        kappa,
        period_jump,
        marked_metric_params);

    // run iterations of fully optimized method
    alg_params.max_itr = full_itr;
    aligned_metric_generator.optimize_full(alg_params);

    // only works for feature alignment
    if (!use_free_cones)
    {
        // run iterations of relaxed optimization
        alg_params.max_itr = max_itr;
        aligned_metric_generator.optimize_relaxed(alg_params);
    }
    else
    {
        aligned_metric_generator.is_axis_aligned = false;
    }

    aligned_metric_generator.parameterize(false, use_uniform_bc);
    auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r] = aligned_metric_generator.get_parameterization();
    auto [feature_face_edges, misaligned_edges] = aligned_metric_generator.get_refined_features();
    auto feature_edges_r = compute_face_edge_endpoints(feature_face_edges, F_r);
    auto misaligned_edges_r = compute_face_edge_endpoints(misaligned_edges, F_r);
    auto [reference_field_r, theta_r, kappa_r, period_jump_r] = aligned_metric_generator.get_refined_field();

    // for free cones, mark all feature edges
    if (use_free_cones)
    {
        feature_edges_r = feature_edges;
    }

    // Optionally optimize parameterization 
    if (optimize)
    {
#if USE_UV_OPTIMIZATION
        std::ifstream js_in(input_json);
        nlohmann::json config = nlohmann::json::parse(js_in);
        config["model"] = mesh;

        // get feature edges
        int num_features = feature_edges_r.size();
        Eigen::MatrixXi FE(num_features, 2);
        for (int eij = 0; eij < num_features; ++eij)
        {
            FE(eij, 0) = feature_edges_r[eij][0];
            FE(eij, 1) = feature_edges_r[eij][1];
        }

        // get misaliged edges
        int num_misaligned = misaligned_edges_r.size();
        Eigen::MatrixXi ME(num_misaligned, 2);
        for (int eij = 0; eij < num_misaligned; ++eij)
        {
            ME(eij, 0) = misaligned_edges_r[eij][0];
            ME(eij, 1) = misaligned_edges_r[eij][1];
        }

        bool fix_boundary = use_free_cones; // fix boundary if using free cones
        uv_r = optimize_aligned_parameterization(
            V_r,
            F_r,
            uv_r,
            FT_r,
            reference_field_r,
            theta_r,
            period_jump_r,
            FE,
            ME,
            config,
            fix_boundary);
#else
        spdlog::warn("uv optimization disabled");
#endif
    }

    if (show_parameterization) view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "refined mesh", true);

    std::string output_filename = join_path(output_dir, mesh+"_opt.obj");
    write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);
    write_mesh_edges(output_filename, feature_edges_r);
    output_filename = join_path(output_dir, mesh+".ffield");
    Penner::Field::write_frame_field(output_filename,  reference_field_r, theta_r, kappa_r, period_jump_r);
    output_filename = join_path(output_dir, mesh+"_fn_to_f");
    write_vector(fn_to_f_r, output_filename);

    // get uv cone vertices
    Eigen::MatrixXi is_cone_corner = tag_cone_corners(V_r, F_r, uv_r, FT_r, feature_face_edges);
    //std::vector<int> uv_cone_vertices;
    //convert_boolean_array_to_index_vector(is_cone_uv, uv_cone_vertices);
    output_filename = join_path(output_dir, mesh+"_uv_cone_corners");
    write_integer_matrix(is_cone_corner, output_filename, " ");

    //std::string output_filename = join_path(output_dir, "optimized_corner_coords");
    //write_matrix(opt_corner_coords, output_filename, " ");



}
