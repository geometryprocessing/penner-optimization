# Script to project a marked metric to holonomy constraints with feature alignment

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import holonomy_py as holonomy
import optimization_py as opt 
import igl
import math
import optimization_scripts.script_util as script_util

def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_optimize_aligned_angles.log')
    logger = script_util.get_logger(log_path)
    logger.info("Projecting {} to constraints".format(name))

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # Skip meshes that are already processed
    try:
        final_metric = np.loadtxt(os.path.join(output_dir, name + "_metric_coords"), dtype=float)
        if (len(final_metric) > 0):
            print("Skipping processed mesh")
        return
    except:
        pass

    # cut mesh along features
    feature_finder = holonomy.FeatureFinder(V, F)
    feature_finder.mark_dihedral_angle_features(60.)
    feature_finder.prune_small_features(5)
    if (args['prune_junctions']):
        feature_finder.prune_junctions()
        feature_finder.prune_closed_loops()
        feature_finder.prune_small_features(1)
    else:
        feature_finder.prune_small_components(5)
    V_cut, F_cut, V_map = feature_finder.generate_feature_cut_mesh()

    # Generate initial similarity metric
    marked_metric_params = holonomy.MarkedMetricParameters()
    marked_metric_params.use_initial_zero = args['use_initial_zero']
    marked_metric_params.remove_loop_constraints = args['remove_holonomy_constraints']
    if (args['remove_boundary_constraints']):
        logger.info("Generating union metric")
        dirichlet_metric, _ = holonomy.generate_union_metric(V_cut, F_cut, marked_metric_params)
    else:
        logger.info("Generating dirichlet metric")
        dirichlet_metric, _ = holonomy.generate_aligned_metric(V_cut, F_cut, V_map, marked_metric_params)

    # Refine initial metric to avoid spanning triangles
    if (False):
        refinement_mesh = holonomy.IntrinsicRefinementMesh(dirichlet_metric, [])
        refinement_mesh.refine_spanning_faces()
        starting_vertices = dirichlet_metric.get_path_starting_vertices()
        dirichlet_metric = refinement_mesh.generate_dirichlet_metric(
            dirichlet_metric.kappa_hat,
            starting_vertices,
            dirichlet_metric.get_boundary_constraint_system(),
            dirichlet_metric.ell_hat)
    
    # Initialize parameters
    alg_params = holonomy.NewtonParameters()
    alg_params.error_eps = args['conf_error_eps']
    alg_params.max_itr = args['conf_max_itr']
    alg_params.do_reduction = args['do_reduction']
    alg_params.reset_lambda = args['reset_lambda']
    alg_params.lambda0 = args['lambda_init']
    alg_params.solver = args['solver']
    alg_params.max_time = args['max_time']
    alg_params.output_dir = output_dir
    alg_params.log_level = 0
    alg_params.error_log = True

    # Project to constraint, undoing flips to restore initial connectivity
    logger.info("Optimizing metric")
    dirichlet_metric = holonomy.optimize_metric_angles(dirichlet_metric, alg_params)

    for i in np.arange(5):
        if (dirichlet_metric.max_constraint_error() < 1e-12):
            break
        holonomy.add_optimal_cone_pair(dirichlet_metric)
        dirichlet_metric = holonomy.optimize_metric_angles(dirichlet_metric, alg_params)

    # Save metric coordinate information
    output_path = os.path.join(output_dir, name + '_metric_coords')
    logger.info("Saving metric coordinates at {}".format(output_path))
    np.savetxt(output_path, dirichlet_metric.get_reduced_metric_coordinates())

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = holonomy.NewtonParameters()
    ls_params = opt.LineSearchParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--conf_error_eps",      help="maximum error for conformal projection",
                                                     type=float, default=alg_params.error_eps)
    parser.add_argument("--max_time",      help="maximum time for projection",
                                                     type=float, default=1e10)
    parser.add_argument("-m", "--conf_max_itr",   help="maximum number of iterations for the conformal method",
                                                     type=int, default=alg_params.max_itr)
    parser.add_argument("--do_reduction",      help="do reduction for conformal step",
                                                     type=bool, default=ls_params.do_reduction)
    parser.add_argument("--reset_lambda",      help="reset lambda for each conformal step",
                                                     type=bool, default=ls_params.reset_lambda)
    parser.add_argument("--lambda_init",      help="initial lambda",
                                                     type=bool, default=ls_params.lambda0)
    parser.add_argument("--solver",      help="solver to use for matrix inversion",
                                                     default=alg_params.solver)
    parser.add_argument("--remove_holonomy_constraints",      help="remove holonomy constraints",
                                                     action="store_true")
    parser.add_argument("--remove_boundary_constraints",      help="remove boundary constraints",
                                                     action="store_true")
    parser.add_argument("--use_initial_zero",      help="use zero vector for initial metric coordinates",
                                                     action="store_true")
    parser.add_argument("--prune_junctions",      help="remove junctions and closed loops from features",
                                                     action="store_true")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Optimize angles with alignment")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
