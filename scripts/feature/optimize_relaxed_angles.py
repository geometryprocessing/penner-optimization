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

    # skip meshes that are already processed
    try:
        final_metric = np.loadtxt(os.path.join(output_dir, name + "_metric_coords"), dtype=float)
        if (len(final_metric) > 0) and not args['regenerate_metric']:
            print("Skipping processed mesh")
            return
    except:
        pass

    # generate logger
    log_path = os.path.join(output_dir, name+'_optimize_relaxed_angles.log')
    logger = script_util.get_logger(log_path)
    logger.info("Projecting {} to feature alignment constraints".format(name))

    # open mesh
    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['field_dir'], fname))
    except:
        logger.error("Could not open mesh data")
        return

    # load feature edges
    try:
        feature_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_features"), dtype=int)
        spanning_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_hard_features"), dtype=int)
    except:
        logger.error("Could not open feature edge data")
        return

    # load frame field
    try:
        reference_field, theta, kappa, period_jump = penner.load_frame_field(os.path.join(args['field_dir'], name + ".ffield"))
    except:
        logger.error("Could not open frame field")
        return

    # initialize algorithm parameters
    alg_params = penner.NewtonParameters()
    alg_params.error_eps = args['error_eps']
    alg_params.do_reduction = args['do_reduction']
    alg_params.reset_lambda = args['reset_lambda']
    alg_params.lambda0 = args['lambda_init']
    alg_params.solver = args['solver']
    alg_params.max_time = args['max_time']
    alg_params.log_level = args['log_level']
    alg_params.output_dir = output_dir
    alg_params.error_log = True

    # build optimizer
    regularization_factor = 0. if args['use_initial_zero'] else 1.
    aligned_metric_generator = penner.AlignedMetricGenerator(
        V,
        F,
        feature_edges,
        spanning_edges,
        reference_field,
        theta,
        kappa,
        period_jump,
        regularization_factor,
        args['use_minimal_forest'])

    # try full optimization
    logger.info("Solving for full constraints")
    alg_params.max_itr = args['full_itr']
    aligned_metric_generator.optimize_full(alg_params)

    # try relaxed constraints
    logger.info("Solving for relaxed constraints")
    alg_params.max_itr = args['max_itr']
    aligned_metric_generator.optimize_relaxed(alg_params)

    # check if converged
    if (aligned_metric_generator.compute_error() > 100 * args['error_eps']):
        logger.warning("final error {} too high".format(aligned_metric_generator.compute_error()))
        return

    # optionally do parameterization
    if args['do_parametrization']:
        logger.info("Generating parametrized mesh")
        aligned_metric_generator.parameterize(False)
        V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r = aligned_metric_generator.get_parameterization()
        reference_field_r, theta_r, kappa_r, period_jump_r = aligned_metric_generator.get_refined_field()
        feature_face_edges, misaligned_face_edges = aligned_metric_generator.get_refined_features()
        feature_edges = penner.compute_face_edge_endpoints(feature_face_edges, F_r)
        misaligned_edges = penner.compute_face_edge_endpoints(misaligned_face_edges, F_r)
        misaligned_uv_edges = penner.compute_face_edge_endpoints(misaligned_face_edges, FT_r)

        # Write combined refined mesh with uv
        uv_mesh_path = os.path.join(output_dir, name + '_refined_with_uv.obj')
        logger.info("Saving refined uv mesh at {}".format(uv_mesh_path))
        penner.write_obj_with_uv(uv_mesh_path, V_r, F_r, uv_r, FT_r)

        # write misaligned edges to file
        misaligned_path = os.path.join(output_dir, name + '_refined_with_uv_misaligned_edges')
        logger.info("Saving {} misaligned edges at {}".format(len(misaligned_uv_edges), misaligned_path))
        np.savetxt(misaligned_path, misaligned_uv_edges, fmt='%i')

        # write features to file
        logger.info("Saving {} feature edges at {}".format(len(feature_edges), uv_mesh_path))
        with open(uv_mesh_path, "a") as uv_mesh_file:
            for feature_edge in feature_edges:
                vi = feature_edge[0]
                vj = feature_edge[1]
                uv_mesh_file.write("l {} {}\n".format(vi + 1, vj + 1))

        # Write fn_to_f to file
        face_map_path = os.path.join(output_dir, name + '_fn_to_f')
        logger.info("Saving new to old face map at {}".format(face_map_path))
        np.savetxt(face_map_path, fn_to_f_r, fmt='%i')

        # Write endpoints to file
        endpoints_path = os.path.join(output_dir, name + '_endpoints')
        logger.info("Saving endpoints at {}".format(endpoints_path))
        np.savetxt(endpoints_path, endpoints_r, fmt='%i')

        # write refined field to file
        output_filename = os.path.join(output_dir, m + ".ffield")
        logger.info("Saving refined field at {}".format(output_filename))
        penner.write_frame_field(output_filename, reference_field_r, theta_r, kappa_r, period_jump_r)

        # write uv analysis
        analysis_dict = {} 
        uv_length_error, uv_angle_error, uv_length, uv_angle = penner.compute_seamless_error(F_r, uv_r, FT_r)
        feature_error = penner.compute_feature_alignment(F_r, uv_r, FT_r, misaligned_edges)
        analysis_dict['length_error'] = np.max(uv_length_error)
        analysis_dict['seamless_error'] = np.max(uv_angle_error)
        analysis_dict['feature_error'] = np.max(feature_error) if feature_error else 0.
        analysis_df = pd.DataFrame(analysis_dict, index=[name,])
        csv_path = os.path.join(output_dir, 'uv_analysis.csv')
        logger.info(f"Saving analysis table to {csv_path}")
        analysis_df.to_csv(csv_path)

    # Save metric coordinate information
    output_path = os.path.join(output_dir, name + '_metric_coords')
    logger.info("Saving metric coordinates at {}".format(output_path))
    metric_coords = aligned_metric_generator.get_metric()
    np.savetxt(output_path, metric_coords)

    

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = penner.NewtonParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--error_eps",      help="maximum error for projection",
                                                     type=float, default=alg_params.error_eps)
    parser.add_argument("--max_time",      help="maximum time for projection",
                                                     type=float, default=1e10)
    parser.add_argument("-m", "--max_itr",   help="maximum number of iterations for the method",
                                                     type=int, default=alg_params.max_itr)
    parser.add_argument("--full_itr",   help="maximum number of iterations for the full constraint method",
                                                     type=int, default=50)
    parser.add_argument("--do_reduction",      help="do reduction for conformal step",
                                                     type=bool, default=alg_params.do_reduction)
    parser.add_argument("--reset_lambda",      help="reset lambda for each conformal step",
                                                     type=bool, default=alg_params.reset_lambda)
    parser.add_argument("--lambda_init",      help="initial lambda",
                                                     type=bool, default=alg_params.lambda0)
    parser.add_argument("--solver",      help="solver to use for matrix inversion",
                                                     default=alg_params.solver)
    parser.add_argument("--use_initial_zero",      help="use zero vector for initial metric coordinates",
                                                     action="store_true")
    parser.add_argument("--log_level",      help="level of logging: higher for more logging",
                                                     type=int, default=4)
    parser.add_argument("--do_parametrization",      help="make parametrization of overlay mesh",
                                                     type=bool, default=False)
    parser.add_argument("--use_minimal_forest",  help="use minimal spanning forest",
                                                     type=bool, default=False)
    parser.add_argument("--regenerate_metric",  help="don't skip optimization if already complete",
                                                     type=bool, default=False)
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Optimize angles with relaxed alignment")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
