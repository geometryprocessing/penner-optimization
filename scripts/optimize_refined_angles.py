# Script to project a refined marked metric to holonomy constraints

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import penner
import igl
import optimization_scripts.script_util as script_util

def optimize_refined_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_optimize_refined.log')
    logger = script_util.get_logger(log_path)
    logger.info("Projecting {} to constraints with intrinsic refinement".format(name))

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # Generate initial similarity metric
    marked_metric_params = penner.MarkedMetricParameters()
    marked_metric_params.use_initial_zero = args['use_initial_zero']
    marked_metric_params.remove_loop_constraints = args['remove_loop_constraints']
    try:
        marked_metric, Th_hat, rotation_form = penner.generate_refined_marked_metric(V, F, args['min_angle'], marked_metric_params)
    except:
        logger.info("Could not build refined metric")
        return

    # Initialize parameters
    alg_params = penner.NewtonParameters()
    alg_params.error_eps = args['conf_error_eps']
    alg_params.max_itr = args['conf_max_itr']
    alg_params.do_reduction = args['do_reduction']
    alg_params.reset_lambda = args['reset_lambda']
    alg_params.lambda0 = args['lambda_init']
    alg_params.max_time = args['max_time']
    alg_params.output_dir = output_dir
    alg_params.log_level = 6
    alg_params.error_log = True

    # Project to constraint, undoing flips to restore initial connectivity
    logger.info("Optimizing metric")
    try:
        marked_metric = penner.optimize_metric_angles(marked_metric, alg_params)
        marked_metric.undo_flips()
    except:
        logger.info("Could not optimize metric")
        return

    # Return if no output needed
    if args['skip_output']:
        return

    # Save metric coordinate information
    output_path = os.path.join(output_dir, name + '_metric_coords')
    logger.info("Saving metric coordinates at {}".format(output_path))
    np.savetxt(output_path, marked_metric.get_reduced_metric_coordinates())


def optimize_refined_many(args):
    script_util.run_many(optimize_refined_one, args)

def add_optimize_refined_arguments(parser):
    alg_params = penner.AlgorithmParameters()
    ls_params = penner.LineSearchParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--conf_error_eps",      help="maximum error for conformal projection",
                                                     type=float, default=alg_params.error_eps)
    parser.add_argument("--max_time",      help="maximum time for projection",
                                                     type=float, default=1e10)
    parser.add_argument("--min_angle",      help="minimum angle for refinement",
                                                     type=float, default=25)
    parser.add_argument("-m", "--conf_max_itr",   help="maximum number of iterations for the conformal method",
                                                     type=int, default=alg_params.max_itr)
    parser.add_argument("--do_reduction",      help="do reduction for conformal step",
                                                     type=bool, default=ls_params.do_reduction)
    parser.add_argument("--reset_lambda",      help="reset lambda for each conformal step",
                                                     type=bool, default=ls_params.reset_lambda)
    parser.add_argument("--lambda_init",      help="initial lambda",
                                                     type=bool, default=ls_params.lambda0)
    parser.add_argument("--use_initial_zero",      help="use zero vector for initial metric coordinates",
                                                     action="store_true")
    parser.add_argument("--remove_loop_constraints",      help="remove holonomy constraints",
                                                     action="store_true")
    parser.add_argument("--skip_output",      help="don't write metric output if true",
                                                     action="store_true")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run optimization method")
    add_optimize_refined_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    optimize_refined_many(args)
