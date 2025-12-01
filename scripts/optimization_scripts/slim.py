# Script to optimize parameterization with free boundary using SLIM

import penner
import optimize_impl.targets as targets
import script_util
import igl
import os
import sys
import numpy as np
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def add_slim_arguments(parser):
    parser.add_argument("--uv_dir",
                        help="file directory of the uv file")
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )
    parser.add_argument("--slim_energy_choice",
                        help="SLIM energy to optimize",
                        type=int, default=2)
    parser.add_argument("--use_uv_for_slim",
                        help="initialize slim from uv instead of Tutte",
                        type=bool, default=False)
    parser.add_argument("--soft_penalty",
                        help="soft constraint penalty for SLIM",
                        type=float, default=0.0)
    parser.add_argument("--num_iter",
                        help="max number of SLIM iterations",
                        type=int, default=200)
    parser.add_argument("-o", "--output_dir",
                        help="directory to save output pickle file")


def slim_one(args, fname):
    os.makedirs(args['output_dir'], exist_ok=True)

    # Get mesh and uv coordinates
    m, v3d, f, _, _, _, _, _ = script_util.generate_mesh(args, fname)
    os.makedirs(os.path.join(args['output_dir'], m + '_output'), exist_ok=True)
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Get logger
    log_path = os.path.join(
        args['output_dir'], m + '_output', name + '_slim.log')
    logger = script_util.get_logger(log_path)

    # Generate uv coordinates from file or from tutte if non specified
    if (args['use_uv_for_slim']):
        # Load uv
        uv_dir = args['uv_dir']
        logger.info("Loading uv coordinates at {}".format(uv_dir))
        v3d, uv, _, f, fuv, _ = igl.read_obj(
            os.path.join(uv_dir, m + "_output", name + ".obj"))

        # Cut mesh along boundary
        v3d = script_util.cut_mesh(v3d, f, uv, fuv)
        f = fuv
    else:
        logger.info("Using Tutte embedding")
        uv = targets.generate_tutte_param(v3d, f)

    # Save initial layout
    mesh_path = os.path.join(
        args['output_dir'], m + '_output', name+'_slim_init.obj')
    logger.info("Saving initial mesh at {}".format(mesh_path))
    penner.write_obj_with_uv(mesh_path, v3d, f, uv, f)

    # Generate SLIM optimizer using chosen energy
    b = []
    bc = np.zeros((0, 2))
    slim_solver = igl.SLIM(
        v3d,
        f,
        uv[:, :2],
        b,
        bc,
        args['slim_energy_choice'],
        args['soft_penalty']
    )

    # Get initial SLIM energy
    initial_energy = slim_solver.energy()
    logger.info("Initial energy: {}".format(initial_energy))

    # Optimize and log each iteration
    for i in np.arange(args['num_iter']):
        slim_solver.solve(1)
        logger.info("Energy at iteration {}: {}".format(
            i, slim_solver.energy()))

    # Get final energy and parametrization
    final_energy = slim_solver.energy()
    uv_slim = slim_solver.vertices()
    logger.info("Final energy: {}".format(final_energy))

    # Save final layout
    mesh_path = os.path.join(
        args['output_dir'], m + '_output', name+'_slim.obj')
    logger.info("Saving initial mesh at {}".format(mesh_path))
    penner.write_obj_with_uv(mesh_path, v3d, f, uv_slim, f)


def slim_many(args):
    script_util.run_many(slim_one, args)


if __name__ == "__main__":
    # Genereate parser arguments
    parser = script_util.generate_parser("Run SLIM optimization method")
    add_slim_arguments(parser)
    args = parser.parse_args()

    # Run method in parallel
    slim_many(args)
