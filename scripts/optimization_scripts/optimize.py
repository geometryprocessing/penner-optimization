# Script to optimize cone metric energy with (usually angle) constraints

import script_util
import pickle
import penner
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def optimize_one(args, fname):
    # Get mesh, lambdas, and parameters
    m, _, _, _, C, opt_energy, _, _ = script_util.generate_mesh(args, fname)
    proj_params, opt_params = script_util.generate_parameters(args)
    opt_params.output_dir = script_util.get_mesh_output_directory(
        args['output_dir'], m)
    proj_params.output_dir = script_util.get_mesh_output_directory(
        args['output_dir'], m)

    # Create output directory for the mesh
    output_dir = os.path.join(args['output_dir'], m + "_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create the logging file handler
    log_path = os.path.join(output_dir, m+'_penner.log')
    logger = script_util.get_logger(log_path)
    logger.info("Optimizing {}".format(m))

    # Save initial conformal lambdas to file
    logger.info("Running conformal method")
    C_conf = penner.project_metric_to_constraint(
        C,
        proj_params,
        opt_params.output_dir
    )
    lambdas_conf = C_conf.get_reduced_metric_coordinates()
    output_lambdas_path = os.path.join(output_dir, 'lambdas_conf')
    logger.info("Saving conformal lambdas to file at {}".format(
        output_lambdas_path))
    np.savetxt(output_lambdas_path, lambdas_conf)

    logger.info("Running implicit metric optimization method")
    C_opt = penner.optimize_metric(
        C,
        opt_energy,
        proj_params,
        opt_params
    )
    lambdas = C_penner.get_reduced_metric_coordinates()

    # Save final lambdas to file
    output_lambdas_path = os.path.join(output_dir, 'lambdas_opt')
    logger.info("Saving final lambdas to file at {}".format(
        output_lambdas_path))
    np.savetxt(output_lambdas_path, lambdas)


def optimize_many(args):
    script_util.run_many(optimize_one, args)


def add_optimize_arguments(parser):
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    parser.add_argument("--checkpoint_dir",
                        help="directory for lambda checkpoints")
    parser.add_argument("--optimization_method",
                        help="optimization method to use",
                        default="metric")
    parser.add_argument("-s", "--steps",
                        help="number of steps for incremental method",
                        type=int, default=0)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run optimization method")
    add_optimize_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    optimize_many(args)
