# Script to render mesh layout with a colormap

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import script_util
import penner
import igl
import optimize_impl.render as render

def add_render_layout_arguments(parser):
    # Parse arguments for the script
    parser.add_argument("--uv_dir",
                        help="directory of the uv coordinates")
    parser.add_argument("-o", "--output_dir",
                        help="directory to save rendered layout")
    parser.add_argument(
        "--colormap_scale",
        help="scale for colormap",
        type=float,
        default=1
    )
    parser.add_argument(
        "--lighting_factor",
        help="lighting factor for rendering",
        type=float,
        default=1
    )
    parser.add_argument(
        "--colormap",
        help="energy for colormap",
        default="sym_dirichlet"
    )
    parser.add_argument(
        "--average_per_vertex",
        help="use per vertex energy",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--use_sqrt_scale",
        help="use square root scale for energy",
        action="store_true"
    )
    parser.add_argument(
        "--use_log_scale",
        help="use log scale for energy",
        action="store_true"
    )
    parser.add_argument(
        "--show_lines",
        help="show wireframe for layout",
        action="store_true"
    )
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
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
    
def render_layout_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Get logger
    log_path = os.path.join(args['output_dir'], m + "_output", name+'_render_layout.log')
    logger = script_util.get_logger(log_path)
    logger.info("Rendering layout for {}".format(name))

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        logger.info("Loading uv coordinates at {}".format(uv_dir))
        v3d, uv, _, f, fuv, _ = igl.read_obj(os.path.join(uv_dir, m + "_output", name + ".obj"))
    except:
        logger.error("Could not load uv coordinates")
        return
    
    output_dir = os.path.join(args['output_dir'], m + "_output")
    os.makedirs(output_dir, exist_ok=True)

    # Cut mesh along along uv lines
    v_cut = script_util.cut_mesh(v3d, f, uv, fuv)

    # Generate energy colormap
    logger.info("Generating layout colormap {} with scale {}".format(
        args['colormap'],
        args['colormap_scale']
    ))
    c = render.get_layout_colormap(
        v_cut,
        fuv,
        uv,
        fuv,
        args['colormap'],
        args['colormap_scale'],
        args['use_sqrt_scale'],
        args['use_log_scale'],
        args['average_per_vertex']
    )

    # Render layout
    logger.info("Rendering layout")
    viewer = render.render_layout(
        v_cut,
        fuv,
        uv,
        c,
        args['show_lines'],
        args['lighting_factor'],
        args['average_per_vertex']
    )

    # Save mesh viewer to file
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    image_path = os.path.join(output_dir, 'images', name+'_layout.png')
    logger.info("Saving layout to {}".format(image_path))
    penner.save_mesh_screen_capture(viewer, image_path, args['height'], args['width'])

    # Render layout
    logger.info("Rendering layout")
    viewer = render.render_layout(
        v_cut,
        fuv,
        uv,
        c,
        args['show_lines'],
        args['lighting_factor'],
        args['average_per_vertex']
    )

    # Save mesh viewer to file in global directory
    os.makedirs(os.path.join(args['output_dir'], 'images'), exist_ok=True)
    image_path = os.path.join(args['output_dir'], 'images', name+'_layout.png')
    logger.info("Saving layout to {}".format(image_path))
    penner.save_mesh_screen_capture(viewer, image_path, args['height'], args['width'])

    logger.info("Done")
    

def render_layout_many(args):
    script_util.run_many(render_layout_one, args)

if __name__ == "__main__":
    # Parse arguments for the script 
    parser = script_util.generate_parser("Render layout")
    add_render_layout_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    render_layout_many(args)
