# Script to create histograms for mesh optimization results.
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
# are also exposed for use in other modules.

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import seaborn as sns
import optimization_py as opt
import optimize_impl.energies as energies
import optimize_impl.analysis as analysis
from conformal_impl.halfedge import *
import script_util

def add_histogram_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images"
    )
    #parser.add_argument(
    #    "--lambdas_dir",
    #    help="directory for lambdas"
    #)
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )
    parser.add_argument(
        "--histogram_choice",
        help="histogram value to plot",
        default="compare_scale_factors"
    )
    parser.add_argument(
        "--bin_min",
        help="minimum value for bin",
        type=float
    )
    parser.add_argument(
        "--bin_max",
        help="maximum value for bin",
        type=float
    )
    parser.add_argument(
        "--ylim",
        help="y limit for the histogram",
        type=float,
        default=50
    )
    parser.add_argument(
        "--histogram_width",
        help="histogram width",
        type=float,
        default=7
    )
    parser.add_argument(
        "--comparison_label",
        help="label for comparison figures",
        default="optimized"
    )
    parser.add_argument(
        "--color",
        help="color for histogram",
        default='red'
    )
    parser.add_argument(
        "--second_color",
        help="second color for comparison histograms",
        default='blue'
    )

def histogram_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']
    os.makedirs(os.path.join(args['output_dir'],  m + "_output"), exist_ok=True)

    # Get logger
    log_path = os.path.join(args['output_dir'], m + "_output", m+'_histogram.log')
    logger = script_util.get_logger(log_path)
    logger.info("Generating histograms for {}".format(name))

    # Get mesh and lambdas
    logger.info("Loading mesh")
    try:
        m, C, lambdas, lambdas_target, v3d, f, Th_hat = script_util.generate_mesh(args, fname)
        proj, embed = opt.build_refl_proj(C)
        he2e, e2he = opt.build_edge_maps(C)
        proj = np.array(proj)
        he2e = np.array(he2e)
    except:
        return

    # Get final lambdas
    try:
        lambdas_path = os.path.join(args['lambdas_dir'], m + "_output", 'lambdas_'+args['suffix'])
        logger.info("Loading lambdas from {}".format(lambdas_path))
        lambdas = np.loadtxt(lambdas_path)
    except:
        logger.error('Could not load lambdas')
        return

    # Get conformal lambdas (if they exist)
    try:
        lambdas_conf_path = os.path.join(args['lambdas_dir'], m + "_output", 'lambdas_conf')
        logger.info("Loading conformal lambdas from {}".format(lambdas_conf_path))
        lambdas_conf = np.loadtxt(lambdas_conf_path)
    except:
        lambdas_conf = None

    # Get bin range (or None if no range values provided)
    if (args['bin_min'] or args['bin_max']):
        binrange = (args['bin_min'], args['bin_max'])
    else:
        binrange = None

    # Get histogram color
    color_dict = {
        'red': "#b90f29",
        'blue': "#3c4ac8"
    }
    first_color = args['color']
    if first_color in color_dict:
        first_color = color_dict[first_color]
    colors = [first_color,]

    # Add second color if it exists
    second_color = args['second_color']
    if second_color:
        if second_color in color_dict:
            second_color = color_dict[second_color]
        colors.append(second_color)
    
    # Set palette
    sns.set_palette(colors)

    os.makedirs(os.path.join(args['output_dir'], 'histograms'), exist_ok=True)
    if (args['histogram_choice'] == 'scale_factors'):
        X = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])

        label = 'scale factors'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_scale_factors.png"
        )
        analysis.generate_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'scale_residuals'):
        B = opt.conformal_scaling_matrix(C)
        u = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])
        X = (lambdas[proj] - lambdas_target[proj]) - B @ u

        label = 'scale residuals'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_scale_residuals.png"
        )
        analysis.generate_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'stretch_factors'):
        X = energies.symmetric_stretches(lambdas, lambdas_target)

        label = 'stretch factors'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_stretch_factors.png"
        )
        analysis.generate_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'sym_dirichlet'):
        X, _ = opt.symmetric_dirichlet_energy(C, lambdas_target[proj], lambdas[proj], False)
        X = np.array(X) - 4

        label = 'symmetric dirichlet'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_sym_dirichlet.png"
        )
        analysis.generate_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'del_sym_dirichlet'):
        C_del, lambdas_init_del_full, _, flip_seq = opt.make_delaunay_with_jacobian(C, lambdas_target[proj], False)
        _, lambdas_del_full = opt.flip_edges(C, lambdas[proj], flip_seq)


        X, _ = opt.symmetric_dirichlet_energy(C, lambdas_init_del_full, lambdas_del_full, False)
        X = np.array(X) - 4

        label = 'delaunay symmetric dirichlet'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_del_sym_dirichlet.png"
        )
        analysis.generate_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'compare_scale_factors'):
        X_conf = opt.best_fit_conformal(C, lambdas_target[proj], lambdas_conf[proj])
        X_opt = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])
        X = {"conformal": X_conf, args['comparison_label']: X_opt}

        label = 'scale factors'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_compare_scale_factors.png"
        )
        analysis.generate_comparison_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'compare_scale_residuals'):
        B = opt.conformal_scaling_matrix(C)
        u_conf = opt.best_fit_conformal(C, lambdas_target[proj], lambdas_conf[proj])
        X_conf = (lambdas_conf[proj] - lambdas_target[proj]) - B @ u_conf
        u_opt = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])
        X_opt = (lambdas[proj] - lambdas_target[proj]) - B @ u_opt
        X = {"conformal": X_conf, args['comparison_label']: X_opt}

        label = 'scale residuals'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_compare_scale_residuals.png"
        )
        analysis.generate_comparison_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    if (args['histogram_choice'] == 'compare_stretch_factors'):
        X_conf = energies.symmetric_stretches(lambdas_conf, lambdas_target)
        X_opt = energies.symmetric_stretches(lambdas, lambdas_target)
        X = {"conformal": X_conf, args['comparison_label']: X_opt}

        label = 'stretch factors'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_compare_stretch_factors.png"
        )
        analysis.generate_comparison_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])

def histogram_many(args):
    script_util.run_many(histogram_one, args)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate histograms for mesh")
    add_histogram_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    histogram_many(args)
