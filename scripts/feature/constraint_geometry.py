# Script to generate constraint geometry for a mesh with features and field

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

    # Get logger
    log_path = os.path.join(output_dir, name+'_create_constraints.log')
    logger = script_util.get_logger(log_path)
    logger.info("Projecting {} to constraints".format(name))

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['field_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # load feature edges
    try:
        feature_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_features"), dtype=int)
        spanning_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_hard_features"), dtype=int)
    except:
        logger.info("Could not open feature edge data")
        return

    # load frame field
    try:
        reference_field, theta, kappa, period_jump = penner.load_frame_field(os.path.join(args['field_dir'], name + ".ffield"))
    except:
        logger.info("Could not open frame field")
        return

    # generate hard feature constraints
    logger.info("Creating feature findere")
    feature_finder = penner.FeatureFinder(V, F)
    feature_finder.mark_features(feature_edges)
    hard_feature_finder = penner.FeatureFinder(V, F)
    if (args['feature'] == "forest_dihedral_angle"):
        hard_feature_finder.mark_features(spanning_edges)
    elif (args['feature'] == "minimal_dihedral_angle"):
        hard_feature_finder.mark_features(spanning_edges)
        hard_feature_finder.prune_greedy()

    # generate cut mesh and feature masks
    logger.info("Cutting mesh along features")
    V_cut, F_cut, V_map, F_is_hard_feature = hard_feature_finder.generate_feature_cut_mesh()
    V_cut, F_cut, V_map, F_is_feature = feature_finder.generate_feature_cut_mesh()

    # find feature corners
    F_is_soft_feature = penner.mask_difference(F_is_feature, F_is_hard_feature)
    relaxed_corners = penner.compute_mask_corners(F_is_soft_feature)

    # build metric with full constraints and marked feature corners
    logger.info("Building embedding metric")
    marked_metric_params = penner.MarkedMetricParameters()
    marked_metric_params.remove_trivial_torus = False
    marked_metric_params.use_log_length = True
    cut_metric_generator = penner.CutMetricGenerator(V_cut, F_cut, marked_metric_params, relaxed_corners)
    cut_metric_generator.set_fields(F_cut, reference_field, theta, kappa, period_jump)
    embedding_metric, vtx_reindex, face_reindex, rotation_form, Th_hat = cut_metric_generator.get_fixed_aligned_metric(V_map, marked_metric_params)

    # save cones
    cone_positions, cone_values = penner.generate_glued_cone_vertices(V, embedding_metric, vtx_reindex, V_map)
    global_output_dir = os.path.join(args['output_dir'], "constraint_output")
    os.makedirs(global_output_dir, exist_ok=True)
    cones_path = os.path.join(global_output_dir, name + '_cones.obj')
    penner.write_obj_with_uv(cones_path, cone_positions, [], [], [])
    cones_path = os.path.join(global_output_dir, name + '_cone_indices.obj')
    penner.write_obj_with_uv(cones_path, cone_values, [], [], [])

    # save features
    feature_path = os.path.join(global_output_dir, name + '_features.obj')
    with open(feature_path, "w") as uv_mesh_file:
        for v in V:
            uv_mesh_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for feature_edge in feature_edges:
            vi = feature_edge[0]
            vj = feature_edge[1]
            uv_mesh_file.write("l {} {}\n".format(vi + 1, vj + 1))

    feature_path = os.path.join(global_output_dir, name + '_hard_features.obj')
    with open(feature_path, "w") as uv_mesh_file:
        for v in V:
            uv_mesh_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for feature_edge in spanning_edges:
            vi = feature_edge[0]
            vj = feature_edge[1]
            uv_mesh_file.write("l {} {}\n".format(vi + 1, vj + 1))

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = penner.NewtonParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    parser.add_argument("--feature",      help="relaxed feature to use",
                                                     default="forest_dihedral_angle")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate constraint geometry")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
