# Script to render a quad mesh
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
# are also exposed for use in other modules.

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import igl
import script_util
import polyscope as ps


def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], 'images')
    os.makedirs(output_dir, exist_ok=True)

    # Skip meshes that are already processed
    output_file = os.path.join(output_dir, name+".png")
    if os.path.isfile(output_file):
        print("Skipping processed mesh")
        return

    # Load uv information
    print("Loading mesh ", name)
    try:
        mesh_dir = os.path.join(args['output_dir'], 'quad_meshes')
        v3d, _, _, f, _, _ = igl.read_obj(os.path.join(mesh_dir, name + "_quad.obj"))
    except:
        print("Could not load quad mesh at ", mesh_dir)
        return
    
    # get colors based on mesh size
    off_white = np.array([1., 1., 0.95])

    # register mesh
    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", v3d, f, smooth_shade=False, material="wax", color=off_white, edge_width=1.)
    ps.set_ground_plane_mode("none")
    ps.reset_camera_to_home_view()

    # Save image to file in global directory
    image_path = os.path.join(output_dir, name+".png")
    ps.screenshot(image_path)


def run_many(args):
    script_util.run_many(run_one, args)


def add_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images"
    )
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Render mesh")
    add_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    run_many(args)
