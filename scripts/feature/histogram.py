# Script to create histograms for mesh optimization results.
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
# are also exposed for use in other modules.

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)

import optimize_aligned_angles
import penner
import seaborn as sns
import numpy as np
import igl
import optimization_scripts.script_util as script_util
import optimize_impl.analysis as analysis

def add_histogram_arguments(parser):
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
        "--color",
        help="color for histogram",
        default='red'
    )
    parser.add_argument(
        "--second_color",
        help="second color for comparison histograms",
        default='blue'
    )


def add_arguments(parser):
    optimize_aligned_angles.add_arguments(parser)
    add_histogram_arguments(parser)
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )
    parser.add_argument(
        "--histogram_choice",
        help="histogram value to plot",
        default="relaxed_feature_alignment"
    )


def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_feature_histogram.log')
    logger = script_util.get_logger(log_path)
    logger.info(f"Generating {args['histogram_choice']} histogram for {name}")

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['field_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # load feature edges
    try:
        feature_edges = np.loadtxt(os.path.join(args['field_dir'], m + "_features"), dtype=int)
        spanning_edges = np.loadtxt(os.path.join(args['field_dir'], m + "_hard_features"), dtype=int)
    except:
        logger.info("Could not open feature edge data")
        return

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        uv_path = os.path.join(uv_dir, m + "_output", name + ".obj")
        logger.info("Loading uv coordinates at {}".format(uv_path))
        v3d, uv, _, f, fuv, _ = igl.read_obj(uv_path)
    except:
        logger.error("Could not load uv coordinates")
        return

    # Load endpoints map
    try:
        endpoints_path = os.path.join(uv_dir, m + "_output", m + '_endpoints')
        logger.info("Loading endpoints at {}".format(endpoints_path))
        endpoints = np.loadtxt(endpoints_path, dtype=int)
    except:
        logger.error("Could not load endpoints")
        endpoints = np.full((len(v3d), 2), -1)

    # generate hard feature constraints
    logger.info("Creating feature findere")
    feature_finder = penner.FeatureFinder(V, F)
    feature_finder.mark_features(feature_edges)
    V_cut, F_cut, V_map, F_is_feature = feature_finder.generate_feature_cut_mesh()

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

    if (args['histogram_choice'] == 'feature_alignment'):
        logger.info("Computing feature alignment of relaxed edges")
        #F_is_relaxed = penner.compute_relaxed_features(F_is_feature, F_is_cut)
        #F_o_is_feature = penner.generate_overlay_cut_mask(f, endpoints, F_cut, F_is_feature)
        F_o_is_feature = penner.generate_overlay_cut_mask(f, endpoints, F, F_is_feature)
        cut_corners = penner.compute_mask_corners(F_o_is_feature)
        X = penner.compute_uv_alignment(uv, fuv, cut_corners)

        ylim=None
        binrange = (0, 1)
        binrange = None
        label = 'alignment'
        output_dir = os.path.join(args['output_dir'], 'histograms')
        output_path = os.path.join(output_dir, name+"_feature_alignment.png")
        os.makedirs(output_dir, exist_ok=True)
        analysis.generate_histogram(
            X, label, binrange, output_path, ylim=ylim, width=args['histogram_width'], use_percentage=True, bins=51, logy=False)
        logger.info("Max value: {}".format(np.max(X)))
    else:
        logger.info("No histogram selected")


def run_many(args):
    script_util.run_many(run_one, args)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser(
        "Generate feature histograms for mesh")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    run_many(args)
