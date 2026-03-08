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

    # Create output directory for the field 
    output_dir = args['field_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_generate_feature_field.log')
    logger = script_util.get_logger(log_path)
    logger.info("Generate feature field for {}".format(name))

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # Skip meshes that are already processed
    if (not args['regenerate_field']) and (os.path.exists(os.path.join(output_dir, name + ".ffield"))):
        print("Skipping processed field")
        return

    # generate feature cut mesh
    logger.info("Finding features")
    V, F, feature_edges, spanning_edges = penner.generate_refined_feature_mesh(V, F, args['use_minimal_forest'])
    feature_finder = penner.FeatureFinder(V, F)
    feature_finder.mark_features(feature_edges)

    # generate cut mesh and feature masks
    logger.info("Cutting mesh along features")
    V_cut, F_cut, V_map, F_is_feature = feature_finder.generate_feature_cut_mesh()

    # initialize cut intrinsic metric for field generation
    logger.info("Building cut metric")
    marked_metric_params = penner.MarkedMetricParameters()
    marked_metric_params.remove_trivial_torus = False
    marked_metric_params.use_log_length = True
    marked_metric_params.remove_loop_constraints = True
    cut_metric_generator = penner.CutMetricGenerator(V_cut, F_cut, marked_metric_params, [])

    # build cross field aligned to salient curvature directions and features
    logger.info("Building fields")
    bb_diag = igl.bounding_box_diagonal(V)
    threshold = 0.9
    direction, is_fixed_direction = penner.compute_field_direction(V_cut, F_cut, 5, 0.2 / bb_diag, threshold)
    cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction)
    reference_field, theta, kappa, period_jump = cut_metric_generator.get_field()

    # write refined mesh
    mesh_filename = os.path.join(output_dir, name + '.obj')
    igl.write_obj(mesh_filename, V, F)

    # write feature and spanning edges
    edge_filename = os.path.join(output_dir, name + '_features')
    np.savetxt(edge_filename, feature_edges, fmt='%i')
    edge_filename = os.path.join(output_dir, name + '_hard_features')
    np.savetxt(edge_filename, spanning_edges, fmt='%i')

    # write field
    field_filename = os.path.join(output_dir, name + '.ffield')
    penner.write_frame_field(field_filename, reference_field, theta, kappa, period_jump)


def run_many(args):
    script_util.run_many(run_one, args)


def add_arguments(parser):
    alg_params = penner.NewtonParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--feature",      help="relaxed feature to use",
                                                     default="forest_dihedral_angle")
    parser.add_argument("--field_dir",
                        help="directory to write refined fields")
    parser.add_argument("--regenerate_field",  help="overwrite existing field",
                                                     type=bool, default=False)
    parser.add_argument("--use_minimal_forest",  help="use minimal spanning forest",
                                                     type=bool, default=False)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Optimize angles with relaxed alignment")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
