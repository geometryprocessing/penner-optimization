# Script to optimize cone metric energy with (usually angle) constraints with respect to shear
# coordinates

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)

import script_util
import optimize_impl.optimization as optimization
import penner
import pandas as pd
import numpy as np


def optimize_shear_one(args, fname):
    # Get mesh, lambdas, and parameters
    m, _, _, _, C, opt_energy, _, _ = script_util.generate_mesh(args, fname)
    proj_params, opt_params = script_util.generate_parameters(args)
    opt_params.output_dir = script_util.get_mesh_output_directory(
        args['output_dir'], m)

    # Create output directory for the mesh
    output_dir = os.path.join(args['output_dir'], m + "_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create the logging file handler
    log_path = os.path.join(output_dir, m+'_optimize_shear.log')
    logger = script_util.get_logger(log_path)
    logger.info("Optimizing shear coordinates for {}".format(m))

    # Save initial conformal metric coordinates to file
    logger.info("Running conformal method")
    C_conf = penner.project_metric_to_constraint(
        C,
        proj_params,
        opt_params.output_dir
    )
    lambdas_conf = C_conf.get_reduced_metric_coordinates()
    output_lambdas_path = os.path.join(output_dir, 'lambdas_conf')
    logger.info("Saving conformal lambdas to file at {}".format(output_lambdas_path))
    np.savetxt(output_lambdas_path, lambdas_conf)

    # Get initial shear coordinates
    if args['use_primal_shear']:
        logger.info("Using primal shear")
        shear_basis_matrix, _ = penner.compute_shear_coordinate_basis(C)
    else:
        logger.info("Using dual shear")
        shear_basis_matrix, _ = penner.compute_shear_dual_basis(C)

    # Run shear optimization method
    if (args['optimization_method'] == 'custom'):
        logger.info("Using custom optimization")

        # Custom gradient descent
        reduced_metric_coords = penner.optimize_shear_basis_coordinates(
            C,
            opt_energy,
            shear_basis_matrix,
            proj_params,
            opt_params)
    else:
        # SciPy advanced first order optimization
        reduced_metric_coords, log, _ = optimization.optimize_shear_basis_coordinates(
            C,
            opt_energy,
            shear_basis_matrix,
            proj_params,
            args['optimization_method'],
            args['opt_num_iter']
        )

        # Save iteration log to file
        data_log = {
            "energy": log
        }
        df = pd.DataFrame(data_log)
        output_log_path = os.path.join(output_dir, 'iteration_data_log.csv')
        df.to_csv(output_log_path)

    # Save final metric coordinates to file
    output_lambdas_path = os.path.join(output_dir, 'lambdas_opt')
    logger.info("Saving final metric coordinates to file at {}".format(
        output_lambdas_path))
    np.savetxt(output_lambdas_path, reduced_metric_coords)


def optimize_shear_many(args):
    script_util.run_many(optimize_shear_one, args)


def add_optimize_shear_arguments(parser):
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    parser.add_argument("--optimization_method",
                        help="scipy optimization method to use",
                        default="custom")
    parser.add_argument("--use_primal_shear",
                        help="use primal shear coordinates instead of dual",
                        action="store_true")


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run shear optimization method")
    add_optimize_shear_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    optimize_shear_many(args)
