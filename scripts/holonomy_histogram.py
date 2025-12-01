# Script to create histograms for mesh optimization results.
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
# are also exposed for use in other modules.

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import igl
import numpy as np
import seaborn as sns
import penner
import optimize_impl.energies as energies
import optimize_impl.analysis as analysis
from conformal_impl.halfedge import *
import optimization_scripts.script_util as script_util

def add_similarity_histogram_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images"
    )
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

def similarity_histogram_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']
    V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    Th_hat = np.loadtxt(os.path.join(args['input_dir'], m + "_Th_hat"), dtype=float)
    rotation_form = np.loadtxt(os.path.join(args['input_dir'], m + "_kappa_hat"), dtype=float)
    
    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(args['output_dir'], 'histograms'), exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_similarity_histogram.log')
    logger = script_util.get_logger(log_path)
    logger.info("Generating histograms for {}".format(name))

    # Generate initial similarity metric
    free_cones = []
    marked_metric_params = penner.MarkedMetricParameters()
    marked_metric, _ = penner.generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params)

    # get target metric
    num_homology_loops = marked_metric.n_homology_basis_loops()
    if (args['use_delaunay']):
        logger.info("Using Delaunay connectivity")
        marked_target = marked_metric.clone_cone_metric()
        marked_target.make_discrete_metric()
        flip_seq = np.array(marked_target.get_flip_sequence())
        penner_target = marked_target.get_reduced_metric_coordinates()
    else:
        penner_target = marked_metric.get_reduced_metric_coordinates()

    # get final metric coordinates
    try:
        lambdas_path = os.path.join(args['lambdas_dir'], m + "_output", name + '_metric_coords')
        logger.info("Loading metric coordinates from {}".format(lambdas_path))
        reduced_metric_coords = np.loadtxt(lambdas_path)
    except:
        logger.error('Could not load metric coordinates')
        return

    # ensure coordinates are defined on same connectivity
    marked_metric = marked_metric.set_metric_coordinates(reduced_metric_coords)
    if (args['use_delaunay']):
        logger.info("Flipping to Delaunay connectivity")
        for h in flip_seq:
            marked_metric.flip_ccw(h, True)
    penner_coords = marked_metric.get_reduced_metric_coordinates()



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

    if (args['histogram_choice'] == 'stretch_factors'):
        logger.info("Computing symmetric stretches")
        X = energies.symmetric_stretches(penner_coords, penner_target)
        print("Average stretch factor:", np.average(X))
        print("Max stretch factor:", np.max(X))
        print("Stretch factors above 2:", len(np.where(X > 2)[0]), "/", len(X))

        label = 'stretch factors'
        output_path = os.path.join(
            args['output_dir'],
            'histograms',
            name+"_stretch_factors.png"
        )
        analysis.generate_histogram(X, label, binrange, output_path, ylim=args['ylim'], width=args['histogram_width'])
    else:
        logger.info("No histogram selected")

def similarity_histogram_many(args):
    script_util.run_many(similarity_histogram_one, args)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate histograms for mesh")
    add_similarity_histogram_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    similarity_histogram_many(args)
