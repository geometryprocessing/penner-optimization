# Script to render mesh
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


def render_mesh_one(args, fname):
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

    # Load mesh information
    try:
        input_dir = args['input_dir']
        v3d_orig, f_orig = igl.read_triangle_mesh(os.path.join(input_dir, m+'.obj'))
    except:
        return

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        v3d, uv, _, f, fuv, _ = igl.read_obj(os.path.join(uv_dir, name + ".obj"))
    except:
        print("Could not load uv coordinates")
        return
    
    # get colors based on mesh size
    teal = np.array([0.290,0.686,0.835])
    orange = np.array([0.906,0.639,0.224])
    sage = np.array([0.569,0.694,0.529])
    if (len(f_orig) >= 50000):
        color = orange 
    elif (len(f_orig) > 5000):
        color = teal
    else:
        color = sage

    # register mesh
    ps.init()
    uv_flat = uv[fuv.flatten()]
    ps_mesh = ps.register_surface_mesh("mesh", v3d, f, smooth_shade=False, material="wax")
    ps_mesh.add_parameterization_quantity("uv", uv_flat, defined_on='corners',
                                        coords_type='world', viz_style='grid',
                                        grid_colors=(0.25 * color, color), enabled=True)

    ps.set_ground_plane_mode("none")
    ps.reset_camera_to_home_view()

    # Save image to file in global directory
    image_path = os.path.join(output_dir, name+".png")
    ps.screenshot(image_path)


def render_mesh_many(args):
    script_util.run_many(render_mesh_one, args)


def add_render_mesh_arguments(parser):
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
    add_render_mesh_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    render_mesh_many(args)
