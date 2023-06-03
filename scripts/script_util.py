import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import argparse
import time
import igl
import numpy as np
import optimization_py as opt
import optimize_impl.targets as targets
import json
import multiprocessing
import logging
import math

def make_record(v, f, Th_hat, lambdas_target, lambdas, log):
    record = {}
    record['v'] = v
    record['f'] = f
    record['Th_hat'] = Th_hat
    record['lambdas_target'] = lambdas_target
    record['lambdas'] = lambdas
    record['log'] = log
    
    return record

def get_mesh_output_directory(output_dir_base, m):
    """
    Make an output directory in a base output directory for a given mesh
    """
    return os.path.join(output_dir_base, m + "_output")

def add_mesh_arguments(parser):
    # Input and output directories
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--nonsymmetric",        help="remove symmetry structure from mesh",
                                                     action="store_true")
    parser.add_argument("--lambdas_dir",         help="directory for initial lambdas")
    parser.add_argument("--free_bd_angles",      help="let boundary angles be free",
                                                     action="store_true")
    parser.add_argument("--free_cones",          help="let cones be free",
                                                     action="store_true")
    parser.add_argument("--initial_pert_sd",        help="perturbation standard deviation for initial lambdas",
                                                     type=float, default=0)
    parser.add_argument("--map_to_disk",         help="set target angles to map to a uniform disk",
                                                     action="store_true")
    parser.add_argument("--use_uniform_lengths", help="set target lambdas all to 0",
                                                     action="store_true")
    parser.add_argument("--use_lengths_from_file", help="set target lambdas at the lambdas dir",
                                                     action="store_true")
    parser.add_argument("--initial_uv_lengths",      help="use edge lengths from uv coordinates",
                                                     action="store_true")
    parser.add_argument("--initial_tutte_lengths",      help="use lengths from tutte uv coordinates",
                                                     type=bool, default=False)
    parser.add_argument("--initial_layout_path", help="path to layout for initial lengths")

def add_parameter_arguments(parser):
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
    parser.add_argument("--max_ratio_incr",      help="maximum ratio increase for each iteration",
                                                     type=float, default=opt_params.max_ratio_incr)
    parser.add_argument("--max_angle",           help="maximum angle error before attempting projection",
                                                     type=float, default=opt_params.max_angle)
    parser.add_argument("-p", "--power",         help="power for p norm energy",
                                                     type=int, default=opt_params.p)
    parser.add_argument("--energy_choice",       help="energy to optimize",
                                                     type=str, default=opt_params.energy_choice)
    parser.add_argument("--cone_weight",           help="weight for cones for quadratic energy",
                                                     type=float, default=opt_params.cone_weight)
    parser.add_argument("--bd_weight",           help="weight for boundary edges for weighted energy",
                                                     type=float, default=opt_params.bd_weight)
    parser.add_argument("--max_grad_range",      help="maximum range for the energy gradient",
                                                     type=float, default=opt_params.max_grad_range)
    parser.add_argument("--max_energy_incr",      help="maximum relative increase for the energy",
                                                     type=float, default=opt_params.max_energy_incr)
    parser.add_argument("--fix_bd_lengths",      help="fix boundary lengths instead of angles",
                                                     type=bool, default=opt_params.fix_bd_lengths)
    parser.add_argument("--max_beta",            help="maximum beta value",
                                                     type=float, default=opt_params.max_beta)
    parser.add_argument("--direction_choice",    help="choose descent direction type",
                                                     type=str, default=opt_params.direction_choice)
    parser.add_argument("--reg_factor",          help="regularization factor for energy",
                                                     type=float, default=opt_params.reg_factor)
    parser.add_argument("--use_log",             help="use log edge lengths for optimization",
                                                     type=bool, default=opt_params.use_log)
    parser.add_argument("--use_edge_lengths",    help="use edge lengths instead of Penner coordinates",
                                                     type=bool, default=opt_params.use_edge_lengths)
    parser.add_argument("--use_optimal_projection",    help="use optimal tangent space projection",
                                                     type=bool, default=opt_params.use_optimal_projection)
    parser.add_argument("--use_checkpoints",    help="use checkpoints for coordinates",
                                                     type=bool, default=opt_params.use_checkpoints)

def generate_parser(description='Run the optimization method with options.'):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--num_processes",           help="number of processes for parallelism",
                                                     type=int, default=8)
    add_mesh_arguments(parser)
    add_parameter_arguments(parser)
    return parser
  
def add_render_layout_arguments(parser):
    parser.add_argument("--colormap",
                        help="colormap choice")
    parser.add_argument("-W", "--width",
                        help="image width",
                        type=int, default=1280)
    parser.add_argument("-H", "--height",
                        help="image height",
                        type=int, default=800)
    parser.add_argument("--show_lines",
                        help="show triangulation",
                        action="store_true")
    parser.add_argument("--lighting_factor",
                        help="lighting factor for rendering",
                        type=float, default=0.0)
    parser.add_argument("--scale",
                        help="scale for colormap",
                        type=float, default=None)
    parser.add_argument("--average_per_vertex",
                        help="Use per vertex average value instead of per face functions",
                        action="store_true")
    parser.add_argument("--use_sqrt_scale",
                        help="Use sqrt scale for colormap",
                        action="store_true")
    parser.add_argument("--use_log_scale",
                        help="Use log scale for colormap",
                        action="store_true")
 

def add_render_arguments(parser):
    parser.add_argument("-W", "--width",
                        help="image width",
                        type=int, default=300)
    parser.add_argument("-H", "--height",
                        help="image height",
                        type=int, default=180)
    parser.add_argument("-N", "--grid_lines",
                        help="grid line parameter",
                        type=int, default=10)
    parser.add_argument("--colormap",
                        help="colormap choice",
                        default="scaled_conformal")
    parser.add_argument("--use_log_scale",
                        help="use log scale for colormap",
                        action="store_true")
    parser.add_argument("--scale",
                        help="scale for colormap",
                        type=float, default=1.0)
    parser.add_argument("--no_cones",
                        help="remove cones from image",
                        action="store_true")
    parser.add_argument("--no_shading",
                        help="only render plain mesh",
                        action="store_true")


def double_Th_hat(C):
    """
    Create Th_hat array for C without the symmetry structure. With the symmetry structure,
    the mesh is treated like a tufted cover with angles of adjacent triangles in the doubled
    mesh summed at the original vertex.

    param[in] Mesh C: (possibly symmetric) mesh
    return np.array: array of Th_hat values
    """
    # Get reflection map for vertices
    refl_v = np.array(C.to)[np.array(C.R)[C.out]]

    # Set Th_hat value to half of doubled Th_hat value for interior vertices
    # Note: Must come before boundary vertex assignment
    Th_hat_full = np.array(C.Th_hat)[C.v_rep] / 2.0

    # Fix Th_hat value of boundary vertices (which are the same as in the tufted cover)
    is_bd_v = (refl_v == np.arange(len(refl_v)))
    Th_hat_full[is_bd_v] = np.array(C.Th_hat)[np.array(C.v_rep)[is_bd_v]]

    return Th_hat_full

    
def remove_symmetry(C):
    """
    Remove the symmetry structure from a doubled mesh so that it can be used like a regular
    closed mesh.

    param[in, out] Mesh C: (possibly symmetric) mesh
    """
    # Only change mesh if symmetric
    if (len(C.type) == 0) or (C.type[0] == '\0'):
        return

    # Remove symmetry
    C.Th_hat = double_Th_hat(C)
    C.type = ['\0' for _ in np.arange(len(C.type))]
    C.v_rep = np.arange(len(C.out))


def generate_mesh(args, fname=None):
    # Open mesh
    if not fname:
        fname = args['fname']
    v3d, f = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    
    # Get Th_hat values
    if (args['map_to_disk']):
        Th_hat = targets.map_to_disk(v3d, f)
    else:
        Th_hat = np.loadtxt(os.path.join(args['input_dir'], m + "_Th_hat"), dtype=float)
        print(Th_hat[0])
        Th_hat = np.array(opt.correct_cone_angles(Th_hat))
        print(Th_hat[0])

    # Get cones
    cones = np.where(np.abs(Th_hat - 2 * np.pi) > 0)[0]
    
    # Create halfedge mesh and lambdas
    if (args['free_cones']):
        C, vtx_reindex = opt.fv_to_double(v3d, f, v3d, f, Th_hat, cones, args['free_bd_angles'])
    else:
        C, vtx_reindex = opt.fv_to_double(v3d, f, v3d, f, Th_hat, [], args['free_bd_angles'])

    if (args['use_uniform_lengths']):
        lambdas_target = np.zeros_like(lambdas_target)
    elif (args['use_edge_lengths']):
        lambdas_target = opt.compute_log_edge_lengths(v3d, f, Th_hat)
    else:
        lambdas_target, _ = opt.compute_penner_coordinates(v3d, f, Th_hat)

    # Remove symmetry structure if nonsymmetric
    if (args['nonsymmetric']):
        proj, embed = opt.build_refl_proj(C)
        remove_symmetry(C)
        lambdas_target = lambdas_target[proj]
        
    if args['use_lengths_from_file']:
        lambdas_init = np.loadtxt(os.path.join(args['lambdas_dir'], m+'_output', 'lambdas_opt'), dtype=float)
        print("Using lambdas from file")
    elif (args['initial_uv_lengths']):
        _, uv, _, _, fuv, _ = igl.read_obj(os.path.join(args['input_dir'], fname))
        uv_embed = np.zeros((len(uv), 3))
        uv_embed[:,:2] = uv[:,:2]
        C_uv, _ = opt.fv_to_double(uv_embed, fuv, uv_embed, fuv, Th_hat, [], args['free_bd_angles'])
        lambdas_init = targets.lambdas_from_mesh(C_uv)
    elif (args['initial_tutte_lengths']):
        uv = targets.generate_tutte_param(v3d, f)
        C_uv, _ = opt.fv_to_double(v3d, f, uv, f, Th_hat, [], args['free_bd_angles'])
        lambdas_init = targets.lambdas_from_mesh(C_uv)
        ## FIXME This is all a bit dangerous with different lengths for Penner and edge
    elif (args['initial_layout_path']):
        v3d_layout, f_layout = igl.read_triangle_mesh(args['initial_layout_path'])
        C_layout, _ = opt.fv_to_double(v3d_layout, f_layout, v3d_layout, f_layout, np.zeros(len(v3d_layout)), [], args['free_bd_angles'])
        lambdas_init = targets.lambdas_from_mesh(C_layout)
    else:
        scale = args['initial_pert_sd']
        print("Using scale {}".format(scale))
        lambdas_pert = np.random.normal(loc=0,
                                        scale=scale,
                                        size=len(lambdas_target))
        lambdas_init = lambdas_target + lambdas_pert

    return m, C, lambdas_init, lambdas_target, v3d, f, Th_hat



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
    opt_params.max_ratio_incr = args['max_ratio_incr']
    opt_params.max_angle = args['max_angle']
    opt_params.p = args['power']
    opt_params.fix_bd_lengths = args['fix_bd_lengths']
    opt_params.energy_choice = args['energy_choice']
    opt_params.max_grad_range = args['max_grad_range']
    opt_params.max_energy_incr = args['max_energy_incr']
    opt_params.bd_weight = args['bd_weight']
    opt_params.cone_weight = args['cone_weight']
    opt_params.max_beta = args['max_beta']
    opt_params.direction_choice = args['direction_choice']
    opt_params.reg_factor = args['reg_factor']
    opt_params.use_edge_lengths = args['use_edge_lengths']
    opt_params.use_optimal_projection = args['use_optimal_projection']
    opt_params.use_log = args['use_log']
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
    logger = multiprocessing.get_logger()
    logger.setLevel(level)
    if os.path.exists(log_path):
        os.remove(log_path)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    return logger

def count_running_processes(p_list):
    count = 0
    for p in p_list:
        if p.is_alive():
            count += 1
    return count

def run_many(method, args):
    p_list = [
        multiprocessing.Process(target=method, args=(args,fname))
        for fname in args['fname']
    ]
    for p in p_list:
        p.start()
        print("Running processes:", count_running_processes(p_list))
        while (count_running_processes(p_list) >= args['num_processes']):
            time.sleep(0.01)
    for p in p_list:
        p.join()

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
