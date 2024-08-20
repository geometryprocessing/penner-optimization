# Script to project a marked metric to holonomy constraints

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import penner
import igl
import math
import optimization_scripts.script_util as script_util

def constrain_similarity_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_optimize_angles.log')
    logger = script_util.get_logger(log_path)
    logger.info("Projecting {} to constraints".format(name))

    # Skip meshes that are already processed
    try:
        final_metric = np.loadtxt(os.path.join(output_dir, name + "_metric_coords"), dtype=float)
        if (len(final_metric) > 0):
            print("Skipping processed mesh")
        return
    except:
        pass

    # get triangle mesh
    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
        if (len(V) < 4):
            logger.info("Skipping single triangle mesh")
            return
    except:
        logger.info("Could not open mesh data")
        return

    # get precomputed form, or generate on the fly
    if args['fit_field']:
        logger.info("Fitting cross field")
        field_params = penner.FieldParameters()
        field_params.min_angle = np.pi
        rotation_form, Th_hat = penner.generate_intrinsic_rotation_form(V, F, field_params)
    else:
        try:
            Th_hat = np.loadtxt(os.path.join(args['input_dir'], name + "_Th_hat"), dtype=float)
            rotation_form = np.loadtxt(os.path.join(args['input_dir'], name + "_kappa_hat"), dtype=float)
        except:
            logger.info("Could not open rotation form")
            return

    # save form to output file
    output_path = os.path.join(output_dir, name + '_Th_hat')
    np.savetxt(output_path, Th_hat)
    output_path = os.path.join(output_dir, name + '_kappa_hat')
    np.savetxt(output_path, rotation_form)

    # Generate initial similarity metric
    free_cones = []
    marked_metric_params = penner.MarkedMetricParameters()
    marked_metric_params.use_initial_zero = args['use_initial_zero']
    marked_metric_params.remove_loop_constraints = args['remove_holonomy_constraints']
    marked_metric_params.free_interior = args['free_interior']
    marked_metric, _ = penner.generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params)

    # optionally fix cones
    if args['do_fix_cones']:
        logger.info("Fixing cones")
        penner.fix_cones(marked_metric, args['min_cone_index'])

    # Optionally refine initial metric to avoid spanning triangles
    if (args['refine']):
        refinement_mesh = penner.IntrinsicRefinementMesh(marked_metric)
        refinement_mesh.refine_spanning_faces()
        marked_metric = refinement_mesh.generate_marked_metric(marked_metric.kappa_hat)
        if (args['free_interior']):
            penner.make_interior_free(marked_metric)

    # Optionally make initial mesh delaunay
    flip_seq = np.array([])
    if (args['use_delaunay']):
        logger.info("Using Delaunay connectivity")
        # Flip to delaunay connectivity
        marked_metric.make_discrete_metric()
        flip_seq = np.array(marked_metric.get_flip_sequence())

        # Build new mesh with Delaunay connectivity
        reduced_metric_coords = marked_metric.get_reduced_metric_coordinates()
        marked_metric = marked_metric.set_metric_coordinates(reduced_metric_coords)

    # Regularize the mesh until it has good triangle quality    
    if (args['regularize']):
        logger.info("Regularizing")
        reduced_metric_coords = marked_metric.get_reduced_metric_coordinates()

        #  Compute quality metrics
        mesh_quality = penner.compute_mesh_quality(marked_metric)
        logger.info("Initial quality is {}".format(np.max(mesh_quality)))

        discrete_metric = marked_metric.clone_cone_metric()
        discrete_metric.make_discrete_metric()
        min_angle = (360 / (2 * math.pi)) * penner.compute_min_angle(discrete_metric)
        logger.info("Initial min angle is {}".format(min_angle))

        num_edges = marked_metric.n_edges()
        average_initial_coord = np.average(reduced_metric_coords[:num_edges])
        logger.info("Average metric coordinate is {}".format(average_initial_coord))
        changed = False
        while ((np.max(mesh_quality) > args['max_triangle_quality']) or (min_angle < args['min_angle'])):
            logger.info("Reducing coordinate norm")

            if (args['max_triangle_quality'] <= 2) or (args['min_angle'] >= 60):
                reduced_metric_coords = 0. * reduced_metric_coords
            else:
                reduced_metric_coords = 0.9 * reduced_metric_coords

            marked_metric = marked_metric.set_metric_coordinates(reduced_metric_coords)
            discrete_metric = marked_metric.clone_cone_metric()
            discrete_metric.make_discrete_metric()
            min_angle = (360 / (2 * math.pi)) * penner.compute_min_angle(discrete_metric)
            mesh_quality = penner.compute_mesh_quality(marked_metric)
            logger.info("Quality is {}".format(np.max(mesh_quality)))
            logger.info("min angle is {}".format(min_angle))

            changed = True
        if changed:
            reduced_metric_coords[:num_edges] += (average_initial_coord - np.average(reduced_metric_coords[:num_edges]))
            marked_metric = marked_metric.set_metric_coordinates(reduced_metric_coords)
            mesh_quality = penner.compute_mesh_quality(marked_metric)
            logger.info("Final quality is {}".format(np.max(mesh_quality)))
            logger.info("Final average is {}".format(np.average(reduced_metric_coords)))

    # Initialize parameters
    alg_params = penner.NewtonParameters()
    alg_params.error_eps = args['conf_error_eps']
    alg_params.max_itr = args['conf_max_itr']
    alg_params.do_reduction = args['do_reduction']
    alg_params.reset_lambda = args['reset_lambda']
    alg_params.lambda0 = args['lambda_init']
    alg_params.max_time = args['max_time']
    alg_params.solver = args['solver']
    alg_params.output_dir = output_dir
    alg_params.log_level = 6
    alg_params.error_log = True

    # Project to constraint, undoing flips to restore initial connectivity
    if (args['optimization_method'] == 'metric'):
        logger.info("Optimizing metric")
        marked_metric = penner.optimize_metric_angles(marked_metric, alg_params)
    elif (args['optimization_method'] == 'metric_subspace'):
        logger.info("Optimizing metric subspace")
        subspace_basis = penner.compute_jump_newton_optimization_basis(marked_metric)
        marked_metric = penner.optimize_subspace_metric_angles(marked_metric, subspace_basis, alg_params)

    # try adding cones if failure
    for i in np.arange(args['cone_pair_corrections']):
        if (marked_metric.max_constraint_error() < 1e-12):
            break
        penner.add_optimal_cone_pair(marked_metric)
        marked_metric = penner.optimize_metric_angles(marked_metric, alg_params)

    # Undo flips
    if flip_seq.size != 0:
        for h in flip_seq[::-1]:
            marked_metric.flip_ccw(h, True)
            marked_metric.flip_ccw(h, True)
            marked_metric.flip_ccw(h, True)

    # Return if no output needed
    if args['skip_output']:
        return

    # Save metric coordinate information
    output_path = os.path.join(output_dir, name + '_metric_coords')
    logger.info("Saving metric coordinates at {}".format(output_path))
    np.savetxt(output_path, marked_metric.get_reduced_metric_coordinates())

    # Save flip sequence
    output_path = os.path.join(output_dir, name + '_flip_seq')
    logger.info("Saving metric coordinates at {}".format(output_path))
    np.savetxt(output_path, flip_seq, fmt="%i")


def constrain_similarity_many(args):
    script_util.run_many(constrain_similarity_one, args)

def add_constrain_similarity_arguments(parser):
    alg_params = penner.NewtonParameters()
    ls_params = penner.LineSearchParameters()
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
                                                     type=float, default=ls_params.lambda0)
    parser.add_argument("--refine",           help="refine spanning triangles if true",
                                                     action="store_true")
    parser.add_argument("--cone_pair_corrections",   help="maximum number of cone pairs to insert",
                                                     type=int, default=0)
    parser.add_argument("--optimization_method",
                        help="optimization method to use",
                        default="metric")
    parser.add_argument("--use_delaunay",      help="use delaunay mesh for optimization",
                                                     action="store_true")
    parser.add_argument("--remove_holonomy_constraints",      help="remove holonomy constraints",
                                                     action="store_true")
    parser.add_argument("--free_interior",      help="remove interior cone constraints",
                                                     action="store_true")
    parser.add_argument("--use_initial_zero",      help="use zero vector for initial metric coordinates",
                                                     action="store_true")
    parser.add_argument("--do_fix_cones",      help="use heuristics to fix cones",
                                                     action="store_true")
    parser.add_argument("--min_cone_index",      help="minimum cone index to allow for cone fixes",
                                                     type=int, default=0)
    parser.add_argument("--regularize",      help="regularize the mesh before optimization",
                                                     action="store_true")
    parser.add_argument("--fit_field",      help="fit intrinsic cross field for rotation form",
                                                     action="store_true")
    parser.add_argument("--skip_output",      help="don't write metric output if true",
                                                     action="store_true")
    parser.add_argument("--max_triangle_quality",      help="maximum triangle quality for regularization",
                                                     type=float, default=1e10)
    parser.add_argument("--min_angle",      help="minimum triangle angle for regularization",
                                                     type=float, default=1)
    parser.add_argument("--solver",      help="solver to use for matrix inversion",
                                                     default=alg_params.solver)
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run optimization method")
    add_constrain_similarity_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    constrain_similarity_many(args)
