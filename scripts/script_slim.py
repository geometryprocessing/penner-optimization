import os, sys
import numpy as np
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import igl
import pickle
import script_util
import optimize_impl.targets as targets
import optimization_py as opt

def slim_one(args, fname):
    os.makedirs(args['output_dir'], exist_ok=True)

    # Get mesh and uv coordinates
    m, C, lambdas_init, lambdas_target, v3d, f, Th_hat = script_util.generate_mesh(args, fname)
    os.makedirs(os.path.join(args['output_dir'], m + '_output'), exist_ok=True)
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Get logger
    log_path = os.path.join(args['output_dir'], m + '_output', name + '_slim.log')
    logger = script_util.get_logger(log_path)

    # Generate uv coordinates from file or from tutte if non specified
    if (args['use_uv_for_slim']):
        # Load uv
        uv_dir = args['uv_dir']
        logger.info("Loading uv coordinates at {}".format(uv_dir))
        v3d, uv, _, f, fuv, _ = igl.read_obj(os.path.join(uv_dir, m + "_output", name + ".obj"))

        # Cut mesh along boundary
        v3d = script_util.cut_mesh(v3d, f, uv, fuv)
        f = fuv
    else:
        logger.info("Using Tutte embedding")
        uv = targets.generate_tutte_param(v3d, f)

    # Save initial layout
    mesh_path = os.path.join(args['output_dir'], m + '_output', name+'_slim_init.obj')
    logger.info("Saving initial mesh at {}".format(mesh_path))
    opt.write_obj_with_uv(mesh_path, v3d, f, uv, f)

    # Generate SLIM optimizer using chosen energy
    b = []
    bc = np.zeros((0,2))
    slim_solver = igl.SLIM(
        v3d,
        f,
        uv[:,:2],
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
        logger.info("Energy at iteration {}: {}".format(i, slim_solver.energy()))

    # Get final energy and parametrization
    final_energy = slim_solver.energy()
    uv_slim = slim_solver.vertices()
    logger.info("Final energy: {}".format(final_energy))
    
    # Save final layout
    mesh_path = os.path.join(args['output_dir'], m + '_output', name+'_slim.obj')
    logger.info("Saving initial mesh at {}".format(mesh_path))
    opt.write_obj_with_uv(mesh_path, v3d, f, uv_slim, f)

    # Open cut to sin list
    try:
        cut_to_sin_list_path = os.path.join(args['uv_dir'], m, name+"_cut_to_sin_list.pickle")
        logger.info("Loading cut_to_sin_list at {}".format(cut_to_sin_list_path))
        with open(cut_to_sin_list_path, 'rb') as fp:
            cut_to_sin_list = pickle.load(fp)

        # Save cut to sin list to output directory with new name
        cut_to_sin_list_slim_path = os.path.join(args['output_dir'], m, name+"_slim_cut_to_sin_list.pickle")
        logger.info("Saving cut_to_sin_list at {}".format(cut_to_sin_list_slim_path))
        with open(cut_to_sin_list_slim_path, 'wb') as fp:
            pickle.dump(cut_to_sin_list, fp)
    except:
        logger.error("Could not load cut_to_sin_list")


    # Open endpoints
    try:
        endpoints_path = os.path.join(args['uv_dir'], m, name+"_endpoints")
        logger.info("Loading endpoints at {}".format(endpoints_path))
        with open(endpoints_path, 'rb') as fp:
            endpoints = pickle.load(fp)

        # Save endpoints list to output directory with new name
        endpoints_slim_path = os.path.join(args['output_dir'], m, name+"_slim_endpoints")
        logger.info("Saving endpoints at {}".format(endpoints_slim_path))
        with open(endpoints_slim_path, 'wb') as fp:
            pickle.dump(endpoints, fp)
    except:
        logger.error("Could not load endpoints")


    # Open vn_to_v
    try:
        vn_to_v_path = os.path.join(args['uv_dir'], m, name+"_vn_to_v")
        logger.info("Loading vn_to_v at {}".format(vn_to_v_path))
        with open(vn_to_v_path, 'rb') as fp:
            vn_to_v = pickle.load(fp)

        # Save vn_to_v list to output directory with new name
        vn_to_v_slim_path = os.path.join(args['output_dir'], m, name+"_slim_vn_to_v")
        logger.info("Saving vn_to_v at {}".format(vn_to_v_slim_path))
        with open(vn_to_v_slim_path, 'wb') as fp:
            pickle.dump(vn_to_v, fp)
    except:
        logger.error("Could not load vn_to_v")

    # Write lambdas to global mesh directory
    # FIXME Double check
    lambdas_path = os.path.join(args['output_dir'], m+ "_output", "lambdas_slim")
    logger.info("Saving optimized lambdas at {}".format(lambdas_path))
    lambdas, _ = opt.compute_penner_coordinates(uv_slim, f, np.zeros(len(uv_slim)))
    np.savetxt(lambdas_path, lambdas)


def slim_many(args):
    script_util.run_many(slim_one, args)

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

if __name__ == "__main__":
    # Genereate parser arguments
    parser = script_util.generate_parser("Run SLIM optimization method")
    add_slim_arguments(parser)
    args = parser.parse_args()

    # Run method in parallel
    slim_many(args)