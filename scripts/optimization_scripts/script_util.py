import math
import logging
import multiprocessing
import json
import optimize_impl.targets as targets
import optimization_py as opt
import numpy as np
import igl
import time
import argparse
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def get_mesh_output_directory(output_dir_base, m):
    """
    Make an output directory in a base output directory for a given mesh
    """
    return os.path.join(output_dir_base, m + "_output")


def add_mesh_arguments(parser):
    # Input and output directories
    parser.add_argument("-f", "--fname",         help="filenames of the obj file",
                        nargs='+')
    parser.add_argument("--skip_fname",         help="filenames of the obj files to skip",
                                                     nargs='+', default=[])
    parser.add_argument("-i", "--input_dir",
                        help="input folder that stores obj files and Th_hat")
    parser.add_argument("--lambdas_dir",
                        help="directory for initial lambdas")
    parser.add_argument("--use_edge_lengths",         help="use edge lengths instead of Penner coordinates",
                        type=bool, default=False)
    parser.add_argument("--free_bd_angles",      help="let boundary angles be free",
                        action="store_true")
    parser.add_argument("--free_cones",          help="let cones be free",
                        action="store_true")
    parser.add_argument("--initial_pert_sd",        help="perturbation standard deviation for initial lambdas",
                        type=float, default=0)
    parser.add_argument("--map_to_disk",         help="set target angles to map to a uniform disk",
                        action="store_true")
    parser.add_argument("--use_lengths_from_file", help="set target lambdas at the lambdas dir",
                        action="store_true")
    parser.add_argument("--initial_uv_lengths",      help="use edge lengths from uv coordinates",
                        action="store_true")
    parser.add_argument("--initial_tutte_lengths",      help="use lengths from tutte uv coordinates",
                        type=bool, default=False)


def add_parameter_arguments(parser):
    # Energy parameters
    parser.add_argument("--energy_choice",       help="energy to optimize",
                        type=str, default="p_norm")
    parser.add_argument("-p", "--power",         help="power for p norm energy",
                        type=int, default=2)

    # Conformal parameters
    proj_params = opt.ProjectionParameters()
    parser.add_argument("-m", "--conf_max_itr",   help="maximum number of iterations for the conformal method",
                        type=int, default=proj_params.max_itr)
    parser.add_argument("--conf_error_eps",      help="maximum error for conformal projection",
                        type=float, default=proj_params.error_eps)
    parser.add_argument("--bound_norm_thres",    help="threshold for the norm bound for the conformal method",
                        type=float, default=proj_params.bound_norm_thres)
    parser.add_argument("--do_reduction",        help="Do reduction in conformal projection",
                        type=bool, default=proj_params.do_reduction)

    # Optimization parameters
    opt_params = opt.OptimizationParameters()
    parser.add_argument("-n", "--opt_num_iter",   help="maximum number of iterations for optimization",
                        type=int, default=opt_params.num_iter)
    parser.add_argument("--beta",                help="initial beta value",
                        type=float, default=opt_params.beta_0)
    parser.add_argument("--min_ratio",           help="minimum descent direction ratio",
                        type=float, default=opt_params.min_ratio)
    parser.add_argument("--require_energy_bound",     help="don't require energy to decrease in each iteration",
                        type=bool, default=opt_params.require_energy_decr)
    parser.add_argument("--require_gradient_proj_negative",     help="require gradient projection to be negative",
                        type=bool, default=opt_params.require_gradient_proj_negative)
    parser.add_argument("--max_angle_incr",      help="maximum angle increase for each iteration",
                        type=float, default=opt_params.max_angle_incr)
    parser.add_argument("--max_angle",           help="maximum angle error before attempting projection",
                        type=float, default=opt_params.max_angle)
    parser.add_argument("--max_grad_range",      help="maximum range for the energy gradient",
                        type=float, default=opt_params.max_grad_range)
    parser.add_argument("--max_energy_incr",      help="maximum relative increase for the energy",
                        type=float, default=opt_params.max_energy_incr)
    parser.add_argument("--max_beta",            help="maximum beta value",
                        type=float, default=opt_params.max_beta)
    parser.add_argument("--direction_choice",    help="choose descent direction type",
                        type=str, default=opt_params.direction_choice)
    parser.add_argument("--use_checkpoints",    help="use checkpoints for coordinates",
                                                     type=bool, default=opt_params.use_checkpoints)


def generate_parser(description='Run the optimization method with options.'):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--num_processes",           help="number of processes for parallelism",
                        type=int, default=8)
    add_mesh_arguments(parser)
    add_parameter_arguments(parser)
    return parser


def generate_mesh(args, fname=None):
    # Open mesh
    if not fname:
        fname = args['fname']
    v3d, f = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    dot_index = fname.rfind(".")
    name = fname[:dot_index]

    # Get Th_hat values
    if (args['map_to_disk']):
        Th_hat = targets.map_to_disk(v3d, f)
    else:
        Th_hat = np.loadtxt(os.path.join(
            args['input_dir'], name + "_Th_hat"), dtype=float)
        Th_hat = np.array(opt.correct_cone_angles(Th_hat))

    # Get free cones
    free_cones = []
    if (args['free_cones']):
        free_cones = np.where(np.abs(Th_hat - 2 * np.pi) > 0)[0]

    # Get initial metric embedding mesh
    uv = v3d
    fuv = f
    if (args['initial_uv_lengths']):
        _, uv, _, _, fuv, _ = igl.read_obj(
            os.path.join(args['input_dir'], fname))
    if (args['initial_tutte_lengths']):
        uv = targets.generate_tutte_param(v3d, f)

    # Create halfedge mesh and lambdas
    vtx_reindex = []
    C = opt.generate_initial_mesh(v3d, f, uv, fuv, Th_hat, vtx_reindex,
                                  free_cones, args['free_bd_angles'], args['use_edge_lengths'])
    C_embed = opt.generate_initial_mesh(
        v3d, f, v3d, f, Th_hat, vtx_reindex, free_cones, args['free_bd_angles'], args['use_edge_lengths'])
    C_eucl = opt.generate_initial_mesh(
        v3d, f, v3d, f, Th_hat, vtx_reindex, free_cones, args['free_bd_angles'], True)

    # Build energies (default to 2-norm)
    energy_choice = args['energy_choice']
    if (energy_choice == "log_length"):
        opt_energy = opt.LogLengthEnergy(C_embed, args['power'])
    elif (energy_choice == "log_scale"):
        opt_energy = opt.LogScaleEnergy(C_embed)
    elif (energy_choice == "quadratic_sym_dirichlet"):
        opt_energy = opt.QuadraticSymmetricDirichletEnergy(C_embed, C_eucl)
    elif (energy_choice == "sym_dirichlet"):
        opt_energy = opt.SymmetricDirichletEnergy(C_embed, C_eucl)
    elif (energy_choice == "p_norm"):
        opt_energy = opt.LogLengthEnergy(C_embed, args['power'])
    else:
        opt_energy = opt.LogLengthEnergy(C_embed, 2)

    # Optionally overwrite with lengths from files
    if args['use_lengths_from_file']:
        metric_coords = np.loadtxt(os.path.join(
            args['lambdas_dir'], name+'_output', 'lambdas_opt'), dtype=float)
        C = C.set_metric_coordinates(metric_coords)

    # Optionally perturb lengths
    scale = args['initial_pert_sd']
    if (scale != 0):
        metric_coords = C.get_reduced_metric_coordinates()
        metric_pert = np.random.normal(loc=0,
                                       scale=scale,
                                       size=len(metric_coords))
        metric_coords = metric_coords + metric_pert
        C = C.set_metric_coordinates(metric_coords)

    return name, v3d, f, Th_hat, C, opt_energy, C_embed, C_eucl


def generate_parameters(args):
    # Set projection parameters
    proj_params = opt.ProjectionParameters()
    proj_params.max_itr = args['conf_max_itr']
    proj_params.error_eps = args['conf_error_eps']
    proj_params.bound_norm_thres = args['bound_norm_thres']
    proj_params.do_reduction = args['do_reduction']

    # Use initial Euclidean flips for direct edge length computations
    if args['use_edge_lengths']:
        proj_params.initial_ptolemy = False
        proj_params.use_edge_flips = False

    # Set optimization parameters
    opt_params = opt.OptimizationParameters()
    opt_params.num_iter = args['opt_num_iter']
    opt_params.beta_0 = args['beta']
    opt_params.min_ratio = args['min_ratio']
    opt_params.require_energy_decr = args['require_energy_bound']
    opt_params.require_gradient_proj_negative = args['require_gradient_proj_negative']
    opt_params.max_angle_incr = args['max_angle_incr']
    opt_params.max_angle = args['max_angle']
    opt_params.max_grad_range = args['max_grad_range']
    opt_params.max_energy_incr = args['max_energy_incr']
    opt_params.max_beta = args['max_beta']
    opt_params.direction_choice = args['direction_choice']
    opt_params.use_checkpoints = args['use_checkpoints']

    return proj_params, opt_params


def overwrite_args(args_default, args_overwrite):
    for key in args_overwrite:
        args_default[key] = args_overwrite[key]

    return args_default


def serialize_arguments(args, output_path):
    """
    Save arguments to json file.

    param[in] ArgumentParser args: arguments to save to file
    param[in] path output_path: path to save arguments to
    """
    with open(output_path, 'wt') as f:
        json.dump(args, f, indent=4)


def load_pipeline(pipeline_path):
    with open(pipeline_path, 'rt') as f:
        return json.load(f)


def get_logger(log_path, level=logging.INFO):
    # Create the logging file handler
    logger = logging.getLogger(log_path)
    logger.setLevel(level)
    if os.path.exists(log_path):
        os.remove(log_path)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    return logger


def run_many(method, args):
    args_list = [(args, fname)
                 for fname in args['fname'] if fname not in args['skip_fname']]
    with multiprocessing.Pool(processes=args['num_processes']) as pool:
        pool.starmap(method, args_list, chunksize=1)

    return


def cut_mesh(v, f, uv, fuv):
    """
    Cut the vf mesh along its uv cut lines
    """
    v_cut = np.zeros((len(uv), 3))
    v_cut[fuv] = v[f]
    return v_cut


def get_boundary_edges(v3d, uv, f, fuv, bd_thick):
    v_cut = np.zeros((len(uv), 3))
    v_cut[fuv] = v3d[f]
    bd_list = igl.boundary_facets(fuv)
    v_bd, f_bd = opt.get_edges(v_cut, fuv, bd_list, bd_thick)
    return v_bd, f_bd
