# Script to project a marked metric to holonomy constraints with feature alignment

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
import numpy as np
import pandas as pd
import penner
import igl
import optimization_scripts.script_util as script_util

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Skip meshes that are already processed with low error
    if os.path.exists(os.path.join(output_dir, name + "_refined_with_uv.obj")):
        try:
            # get uv iteration data
            uv_data_dir = os.path.join(output_dir, 'uv_analysis.csv')
            uv_data = pd.read_csv(uv_data_dir)

            thres = 1e-10
            if (float(uv_data['seamless_error']) < thres) and (float(uv_data['length_error']) < thres):
                print("Skipping processed mesh")
                return
        except:
            print("Could not open uv analysis; proceeding with layout")

    # Get logger
    log_path = os.path.join(output_dir, name+'_feature_overlay.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building overlay for {}".format(name))

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['field_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # load frame field
    try:
        reference_field, theta, kappa, period_jump = feature.load_frame_field(os.path.join(args['field_dir'], name + ".ffield"))
    except:
        logger.info("Could not open frame field")
        return

    # load feature edges
    try:
        feature_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_features"), dtype=int)
        spanning_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_hard_features"), dtype=int)
    except:
        logger.info("Could not open feature edge data")
        return

    # load metric
    try:
        mesh_metric_dir = script_util.get_mesh_output_directory(args['lambdas_dir'], m)
        corner_coords = np.loadtxt(os.path.join(mesh_metric_dir, name + '_metric_coords'), dtype=float)

    except:
        logger.info("Could not open metric data")
        return

    # generate hard feature constraints
    logger.info("Creating feature findere")
    feature_finder = feature.FeatureFinder(V, F)
    feature_finder.mark_features(feature_edges)
    hard_feature_finder = feature.FeatureFinder(V, F)
    hard_feature_finder.mark_features(spanning_edges)

    # generate cut mesh and feature masks
    logger.info("Cutting mesh along features")
    V_cut, F_cut, V_map, F_is_hard_feature = hard_feature_finder.generate_feature_cut_mesh()
    V_cut, F_cut, V_map, F_is_feature = feature_finder.generate_feature_cut_mesh()

    # build metric with full constraints and marked feature corners
    logger.info("Building embedding metric")
    marked_metric_params = penner.MarkedMetricParameters()
    marked_metric_params.remove_trivial_torus = False
    marked_metric_params.use_log_length = True
    marked_metric_params.remove_loop_constraints = True
    cut_metric_generator = feature.CutMetricGenerator(V_cut, F_cut, marked_metric_params, [])
    cut_metric_generator.set_fields(F_cut, reference_field, theta, kappa, period_jump)
    embedding_metric, vtx_reindex, face_reindex, rotation_form, Th_hat = cut_metric_generator.get_fixed_aligned_metric(V_map, marked_metric_params)

    # build metric with penner coordinates
    logger.info("Computing penner coordinates")
    dirichlet_metric = feature.DirichletPennerConeMetric(embedding_metric)
    opt_marked_metric = feature.DirichletPennerConeMetric(embedding_metric)
    metric_coords = feature.transfer_corner_function_to_halfedge(
        dirichlet_metric,
        vtx_reindex,
        V_map,
        F,
        corner_coords)
    opt_marked_metric.change_metric(embedding_metric, metric_coords, True, False)

    logger.info("Generating parametrized mesh")
    if (args['use_robust_overlay']):
        V_o, F_o, uv_o, FT_o, fn_to_f, endpoints = feature.parameterize_cut_mesh_mpfr(
            embedding_metric,
            dirichlet_metric,
            opt_marked_metric,
            V_cut,
            vtx_reindex,
            face_reindex,
            args['use_uniform_bc'],
            "")
    else:
        V_o, F_o, uv_o, FT_o, fn_to_f, endpoints = feature.parameterize_cut_mesh(
            embedding_metric,
            dirichlet_metric,
            opt_marked_metric,
            V_cut,
            vtx_reindex,
            face_reindex,
            args['use_uniform_bc'],
            "")

    F_o_is_hard_feature = feature.generate_overlay_cut_mask(F_o, endpoints, F_cut, F_is_hard_feature)
    F_o_is_feature = feature.generate_overlay_cut_mask(F_o, endpoints, F_cut, F_is_feature)
    uv_o = feature.align_to_hard_features(uv_o, FT_o, F_o_is_hard_feature)
    stitch_res = feature.stitch_cut_overlay(
        V_o,
        F_o,
        uv_o,
        FT_o,
        fn_to_f,
        endpoints,
        F_o_is_feature,
        V_map,
        args['use_uniform_bc'])
    V_s, F_s, uv_s, FT_s, fn_to_f_s, endpoints_s, F_s_is_feature = stitch_res

    # refine mesh
    logger.info("Running refinement")
    refinement_mesh = penner.RefinementMesh(V_s, F_s, uv_s, FT_s, fn_to_f_s, endpoints_s)
    V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r = refinement_mesh.get_VF_mesh()
    if args['use_connected_layout']:
        uv_r, FT_r = feature.generate_connected_parameterization(V_r, F_r, uv_r, FT_r)
    F_r_is_feature = feature.generate_overlay_cut_mask(F_r, endpoints_r, F, F_is_feature)

    # Write combined refined mesh with uv
    uv_mesh_path = os.path.join(output_dir, name + '_refined_with_uv.obj')
    logger.info("Saving refined uv mesh at {}".format(uv_mesh_path))
    penner.write_obj_with_uv(uv_mesh_path, V_r, F_r, uv_r, FT_r)
    feature_corners = feature.compute_mask_corners(F_r_is_feature)
    feature_face_edges = feature.compute_face_edges_from_corners(F_r, feature_corners)
    feature_face_edges, misaligned_face_edges = feature.prune_misaligned_face_edges(
        uv_r,
        FT_r,
        feature_face_edges,
        args['feature_threshold'])
    feature_edges = feature.compute_face_edge_endpoints(feature_face_edges, F_r)
    feature_error = feature.compute_mask_uv_alignment(uv_r, FT_r, F_r_is_feature)

    # save misaligned edges to file
    misaligned_edges = feature.compute_face_edge_endpoints(misaligned_face_edges, FT_r)
    misaligned_path = os.path.join(output_dir, name + '_refined_with_uv_misaligned_edges')
    np.savetxt(misaligned_path, misaligned_edges, fmt='%i')

    # run basic analysis
    uv_embed = np.zeros((len(uv_r), 3))
    uv_embed[:, :2] = uv_r[:, :2]
    uv_areas = 0.5 * igl.doublearea(uv_embed, FT_r)
    corner_angles = igl.internal_angles(uv_embed, FT_r)
    height = feature.compute_height(uv_r, FT_r)

    # write uv analysis
    analysis_dict = {} 
    uv_length_error, uv_angle_error, uv_length, uv_angle = feature.compute_seamless_error(F_r, uv_r, FT_r)
    analysis_dict['length_error'] = np.max(uv_length_error)
    analysis_dict['seamless_error'] = np.max(uv_angle_error)
    analysis_dict['feature_error'] = np.max(feature_error)
    analysis_dict['min_area'] = np.min(uv_areas)
    analysis_dict['height'] = np.min(height)
    analysis_dict['min_angle'] = np.min(corner_angles)
    analysis_dict['max_angle'] = np.max(corner_angles)
    analysis_dict['num_flipped'] = feature.check_flip(uv_r, FT_r)
    analysis_df = pd.DataFrame(analysis_dict, index=[name,])
    csv_path = os.path.join(output_dir, 'uv_analysis.csv')
    logger.info(f"Saving analysis table to {csv_path}")
    analysis_df.to_csv(csv_path)

    # write feature edges
    with open(uv_mesh_path, "a") as uv_mesh_file:
        for feature_edge in feature_edges:
            vi = feature_edge[0]
            vj = feature_edge[1]
            uv_mesh_file.write("l {} {}\n".format(vi + 1, vj + 1))

    # write seams and features
    global_output_dir = os.path.join(args['output_dir'], "output")
    seam_path = os.path.join(global_output_dir, name + '_seams.obj')
    feature.write_seams(seam_path, V_r, F_r, FT_r, F_r_is_feature)
    feature_path = os.path.join(global_output_dir, name + '_features.obj')
    feature.write_features(feature_path, V_r, F_r, F_r_is_feature)

    # refine cross field
    reference_field_r, theta_r, kappa_r, period_jump_r = feature.refine_frame_field(
        F_r,
        FT_r,
        fn_to_f_r,
        endpoints_r,
        F,
        reference_field,
        theta,
        kappa,
        period_jump)

    # Write fn_to_f to file
    face_map_path = os.path.join(output_dir, name + '_fn_to_f')
    logger.info("Saving new to old face map at {}".format(face_map_path))
    np.savetxt(face_map_path, fn_to_f_r, fmt='%i')

    # Write endpoints to file
    endpoints_path = os.path.join(output_dir, name + '_endpoints')
    logger.info("Saving endpoints at {}".format(endpoints_path))
    np.savetxt(endpoints_path, endpoints_r, fmt='%i')

    output_filename = os.path.join(output_dir, m + ".ffield")
    feature.write_frame_field(output_filename, reference_field_r, theta_r, kappa_r, period_jump_r);
    

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = penner.NewtonParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--use_connected_layout",  help="regenerate layout with one connected component",
                                                     type=bool, default=False)
    parser.add_argument("--use_uniform_bc",  help="use uniform edge barycentric coordinates",
                                                     type=bool, default=False)
    parser.add_argument("--use_robust_overlay",  help="use multiprecision for overlay",
                                                     type=bool, default=False)
    parser.add_argument("--feature_threshold",  help="threshold for flagging feature edges",
                                                     type=float, default=1e-10)
    parser.add_argument("--lambdas_dir",
                        help="directory for metrics")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Optimize angles with relaxed alignment")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
