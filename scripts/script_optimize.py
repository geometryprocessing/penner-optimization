# Script to optimize cone metric energy with (usually angle) constraints
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
# are also exposed for use in other modules.

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import optimization_py as opt
import optimize_impl.optimization as optimization
import pickle
import script_util

def optimize_one(args, fname):
    # Get mesh, lambdas, and parameters
    m, C, lambdas_init, lambdas_target, v3d, f, Th_hat = script_util.generate_mesh(args, fname)
    proj_params, opt_params = script_util.generate_parameters(args)
    opt_params.output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)

    # Create output directory for the mesh
    output_dir = os.path.join(args['output_dir'], m + "_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create the logging file handler
    log_path = os.path.join(output_dir, m+'_opt.log')
    logger = script_util.get_logger(log_path)
    logger.info("Optimizing {}".format(m))

    # Save initial conformal lambdas to file
    logger.info("Running conformal method")
    lambdas_conf, u = opt.project_to_constraint(
        C,
        lambdas_init,
        proj_params,
        opt_params
    )
    output_lambdas_path = os.path.join(output_dir, 'lambdas_conf')
    logger.info("Saving conformal lambdas to file at {}".format(output_lambdas_path))
    np.savetxt(output_lambdas_path, lambdas_conf)

    # Run method of choice
    if args['optimization_method'] == 'conformal':
        logger.info("Running conformal method")
        lambdas, u = opt.project_to_constraint(
            C,
            lambdas_init,
            proj_params
        )
        log = {}
    elif args['optimization_method'] == 'metric':
        logger.info("Running implicit metric optimization method")
        lambdas = opt.optimize_metric(
            C,
            lambdas_target,
            lambdas_init,
            proj_params,
            opt_params
        )
    elif args['optimization_method'] == 'shear_dual':
        logger.info("Running explicit shear dual optimization method")
        lambdas = opt.optimize_shear_dual_coordinates(
            C,
            lambdas_target,
            proj_params,
            opt_params
        )
    elif args['optimization_method'] == 'incremental':
        logger.info("Running incremental method")
        lambdas = optimization.incremental_projection(
            C,
            lambdas_init,
            lambdas_target,
            proj_params=proj_params,
            opt_params=opt_params
        )

        # Save record (including log) to file
        output_path = os.path.join(output_dir, m+'_record.p')
        logger.info("Saving record pickle to file at {}".format(output_path))
        with open(output_path, 'wb') as output_file:
            record = script_util.make_record(v3d, f, Th_hat, lambdas_init, lambdas, log)
            pickle.dump(record, output_file)
    elif args['optimization_method'] == 'implicit_experimental':
        logger.info("Running experimental implicit optimization method")
        log, lambdas = optimization.optimize_lambdas(
            C,
            lambdas_init,
            lambdas_target,
            checkpoint_dir=output_dir,
            proj_params=proj_params,
            opt_params=opt_params
        )

        # Save record (including log) to file
        output_path = os.path.join(output_dir, m+'_record.p')
        logger.info("Saving record pickle to file at {}".format(output_path))
        with open(output_path, 'wb') as output_file:
            record = script_util.make_record(v3d, f, Th_hat, lambdas_init, lambdas, log)
            pickle.dump(record, output_file)

    # Save final lambdas to file
    output_lambdas_path = os.path.join(output_dir, 'lambdas_opt')
    logger.info("Saving final lambdas to file at {}".format(output_lambdas_path))
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
