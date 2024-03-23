# Script to generate histograms for per-vertex colormaps used for rendering meshes

import script_util
import igl
import seaborn as sns
import optimize_impl.analysis as analysis
import optimize_impl.energies as energies
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def add_colormap_histogram_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images"
    )
    parser.add_argument(
        "--uv_dir",
        help="directory for uv coordinates"
    )
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )
    parser.add_argument(
        "--label",
        help="x axis label",
    )
    parser.add_argument(
        "--colormap",
        help="histogram value to plot",
        default="scale_factors"
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
        help="y limit for the histogam",
        type=float,
        default=50
    )
    parser.add_argument(
        "--color",
        help="color for histogram",
        default='red'
    )
    parser.add_argument(
        "-H", "--height",
        help="image height",
        type=int,
        default=800
    )
    parser.add_argument(
        "-W", "--width",
        help="image width",
        type=int,
        default=1280
    )


def colormap_histogram_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Get logger
    log_path = os.path.join(
        args['output_dir'], m + "_output", m+'_colormap_histogram.log')
    logger = script_util.get_logger(log_path)
    logger.info("Generating colormap histograms for {}".format(name))

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        logger.info("Loading uv coordinates at {}".format(uv_dir))
        v3d, uv, _, f, fuv, _ = igl.read_obj(
            os.path.join(uv_dir, m + "_output", name + ".obj"))
    except:
        logger.error("Could not load uv coordinates")
        return

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
    color = args['color']
    if color in color:
        color = color_dict[color]
    colors = [color,]
    sns.set_palette(colors)

    # Compute chosen vertex energy
    X = energies.get_vertex_energy(v3d, f, uv, fuv, args['colormap'])

    # Create a label for the plot
    if args['label']:
        label = args['label']
    else:
        label = args['colormap']

    # Generate histogram and save to file
    os.makedirs(os.path.join(args['output_dir'],
                'colormap_histograms'), exist_ok=True)
    output_path = os.path.join(
        args['output_dir'],
        'colormap_histograms',
        name+"_"+args["colormap"]+".png"
    )
    analysis.generate_histogram(
        X, label, binrange, output_path, ylim=args['ylim'],)


def colormap_histogram_many(args):
    script_util.run_many(colormap_histogram_one, args)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser(
        "Generate colormap histograms for mesh")
    add_colormap_histogram_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    colormap_histogram_many(args)
